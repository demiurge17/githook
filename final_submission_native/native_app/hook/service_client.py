"""Private Unix-socket client for the lazy local inference process."""
from __future__ import annotations

import json
import fcntl
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import time

from secret_detector import DetectorError, PROJECT_ROOT


MAX_MESSAGE_BYTES = 16 * 1024 * 1024


def socket_path(config: dict) -> Path:
    explicit = os.environ.get("CODEBERT_SECRET_HOOK_SOCKET")
    if explicit:
        return Path(explicit)
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime) if runtime else Path.home() / ".cache" / "codebert-secret-hook"
    return base / "cbsh.sock" if runtime else base / "s.sock"


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    data = bytearray()
    while len(data) < count:
        part = sock.recv(count - len(data))
        if not part:
            raise DetectorError("Local inference service closed an incomplete response; commit blocked.")
        data.extend(part)
    return bytes(data)


def _exchange(path: Path, request: dict, timeout: float, timings: dict | None = None,
              phase: str = "prediction") -> dict:
    timings = timings if timings is not None else {}
    started = time.monotonic()
    payload = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    timings["serialization_seconds"] = timings.get("serialization_seconds", 0.0) + time.monotonic() - started
    if len(payload) > MAX_MESSAGE_BYTES:
        raise DetectorError("Staged request exceeds the local service limit; commit blocked.")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        stage_started = time.monotonic()
        client.connect(str(path))
        duration = time.monotonic() - stage_started
        timings["connection_seconds"] = timings.get("connection_seconds", 0.0) + duration
        timings[phase + "_connection_seconds"] = timings.get(phase + "_connection_seconds", 0.0) + duration
        stage_started = time.monotonic()
        client.sendall(len(payload).to_bytes(4, "big") + payload)
        size = int.from_bytes(_recv_exact(client, 4), "big")
        if size > MAX_MESSAGE_BYTES:
            raise DetectorError("Local inference service returned an invalid response; commit blocked.")
        raw = _recv_exact(client, size)
        duration = time.monotonic() - stage_started
        timings["request_response_seconds"] = timings.get("request_response_seconds", 0.0) + duration
        timings[phase + "_request_response_seconds"] = timings.get(phase + "_request_response_seconds", 0.0) + duration
    stage_started = time.monotonic()
    try:
        result = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise DetectorError("Local inference service returned an invalid response; commit blocked.") from None
    if not isinstance(result, dict):
        raise DetectorError("Local inference service returned an invalid response; commit blocked.")
    duration = time.monotonic() - stage_started
    timings["response_handling_seconds"] = timings.get("response_handling_seconds", 0.0) + duration
    timings[phase + "_response_handling_seconds"] = timings.get(phase + "_response_handling_seconds", 0.0) + duration
    return result


def _prepare_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if info.st_uid != os.getuid() or not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
        raise DetectorError("Local service directory is not private to this user; commit blocked.")


def _probe_ready(path: Path, timeout: float = 0.2) -> bool:
    try:
        return _exchange(path, {"op": "ping"}, timeout).get("status") == "ready"
    except (OSError, DetectorError):
        return False


def _start_service(path: Path, config_path: Path, start_timeout: float = 10.0) -> None:
    _prepare_private_directory(path.parent)
    # Serialize startup across repositories. Keep the lock inode; flock releases
    # automatically if a client exits, so there is no stale-lock recovery race.
    lock_path = path.parent / ".service-start.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        lock_fd = os.open(lock_path, flags, 0o600)
        lock_info = os.fstat(lock_fd)
        if (not stat.S_ISREG(lock_info.st_mode) or lock_info.st_uid != os.getuid()
                or lock_info.st_mode & 0o077):
            os.close(lock_fd)
            raise DetectorError("Local service startup lock is unsafe; commit blocked.")
        with os.fdopen(lock_fd, "r+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if _probe_ready(path):
                return
            # Only remove an owned socket that does not answer the protocol.
            if path.exists() or path.is_symlink():
                info = path.lstat()
                if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
                    raise DetectorError("Local service socket path is occupied or unsafe; commit blocked.")
                path.unlink(missing_ok=True)
            server = PROJECT_ROOT / "hook" / "inference_service.py"
            try:
                subprocess.Popen(
                    [sys.executable, str(server), "--socket", str(path), "--config", str(config_path)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True, close_fds=True,
                )
            except OSError:
                raise DetectorError("Could not start the local inference service; commit blocked.") from None
            deadline = time.monotonic() + min(max(float(start_timeout), 1.0), 30.0)
            while time.monotonic() < deadline:
                if _probe_ready(path):
                    return
                time.sleep(0.025)
            raise DetectorError("Local inference service did not become available; commit blocked.")
    except DetectorError:
        raise
    except OSError:
        raise DetectorError("Could not coordinate local inference service startup; commit blocked.") from None


class ServiceDetector:
    """Detector proxy; the model is loaded by the service only when first needed."""
    def __init__(self, config: dict):
        self.config = config
        self.inference_seconds = 0.0
        self.model_load_seconds = 0.0
        self.tokenization_seconds = 0.0
        self.encoder_seconds = 0.0
        self.classifier_seconds = 0.0
        self.service_peak_rss_mib = None
        self.service_pid = None
        self.service_startup_seconds = 0.0
        self.connection_seconds = 0.0
        self.request_response_seconds = 0.0
        self.response_handling_seconds = 0.0
        self.service_processing_seconds = 0.0
        self.service_started = False
        self.path = socket_path(config)
        self.config_path = Path(os.environ.get("CODEBERT_SECRET_HOOK_CONFIG", str(PROJECT_ROOT / "hook" / "config.json")))

    def predict(self, examples: list[str]) -> list[int]:
        started = time.monotonic()
        self.service_started = False
        self.service_startup_seconds = 0.0
        timeout = float(self.config.get("service_request_timeout_seconds", 300))
        exchange_timings: dict[str, float] = {}
        try:
            _prepare_private_directory(self.path.parent)
            try:
                result = _exchange(self.path, {"op": "predict", "examples": examples}, timeout, exchange_timings)
            except (FileNotFoundError, ConnectionRefusedError, ConnectionResetError):
                startup_started = time.monotonic()
                start_timeout = float(self.config.get("service_start_timeout_seconds", 10))
                _start_service(self.path, self.config_path, start_timeout)
                deadline = time.monotonic() + min(max(start_timeout, 1), 30)
                last_error = None
                while time.monotonic() < deadline:
                    try:
                        readiness = _exchange(self.path, {"op": "ping"}, min(timeout, 2), exchange_timings, "startup")
                        if readiness.get("status") != "ready":
                            raise DetectorError("Local inference service returned an invalid readiness response; commit blocked.")
                        break
                    except (FileNotFoundError, ConnectionRefusedError, ConnectionResetError, socket.timeout) as error:
                        last_error = error
                        time.sleep(0.025)
                else:
                    raise DetectorError("Local inference service did not become available; commit blocked.") from last_error
                self.service_started = True
                self.service_startup_seconds = time.monotonic() - startup_started
                result = _exchange(self.path, {"op": "predict", "examples": examples}, timeout, exchange_timings)
            if result.get("status") != "ok":
                raise DetectorError("Local inference service failed; commit blocked.")
            predictions = result.get("predictions")
            if not isinstance(predictions, list) or len(predictions) != len(examples) or any(
                type(label) is not int or label not in (0, 1) for label in predictions
            ):
                raise DetectorError("Local inference service returned incompatible predictions; commit blocked.")
            service_pid = result.get("service_pid")
            if type(service_pid) is not int or service_pid <= 1:
                raise DetectorError("Local inference service returned an invalid process identity; commit blocked.")
            self.service_pid = service_pid
            self.inference_seconds = float(result.get("inference_seconds", 0.0))
            self.model_load_seconds = float(result.get("model_load_seconds", 0.0))
            self.tokenization_seconds = float(result.get("tokenization_seconds", 0.0))
            self.encoder_seconds = float(result.get("encoder_seconds", 0.0))
            self.classifier_seconds = float(result.get("classifier_seconds", 0.0))
            self.service_processing_seconds = float(result.get("service_processing_seconds", 0.0))
            peak = result.get("service_peak_rss_mib")
            self.service_peak_rss_mib = float(peak) if peak is not None else None
            self.connection_seconds = exchange_timings.get("connection_seconds", 0.0)
            self.request_response_seconds = exchange_timings.get("request_response_seconds", 0.0)
            self.response_handling_seconds = exchange_timings.get("response_handling_seconds", 0.0)
            for key, value in exchange_timings.items():
                setattr(self, key, value)
            return predictions
        except DetectorError:
            raise
        except Exception:
            raise DetectorError("Local inference service could not complete the scan; commit blocked.") from None
        finally:
            self.client_seconds = time.monotonic() - started
