#!/usr/bin/env python3
"""On-demand local inference server. It holds the model only until idle timeout."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import socket
import stat
import time

from secret_detector import DetectorError, load_detector, read_config

MAX_MESSAGE_BYTES = 16 * 1024 * 1024


def recv_exact(client: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        part = client.recv(size - len(data))
        if not part:
            raise ValueError("truncated request")
        data.extend(part)
    return bytes(data)


def send(client: socket.socket, response: dict) -> None:
    payload = json.dumps(response, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        payload = b'{"status":"error"}'
    client.sendall(len(payload).to_bytes(4, "big") + payload)


def request(client: socket.socket, detector, config: dict):
    size = int.from_bytes(recv_exact(client, 4), "big")
    if size <= 0 or size > MAX_MESSAGE_BYTES:
        send(client, {"status": "error"})
        return detector
    try:
        payload = json.loads(recv_exact(client, size).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError
        if payload.get("op") == "ping":
            send(client, {"status": "ready", "service_pid": os.getpid()})
            return detector
        if payload.get("op") != "predict":
            raise ValueError
        processing_started = time.monotonic()
        examples = payload.get("examples")
        if not isinstance(examples, list) or len(examples) > int(config["max_examples"]):
            raise ValueError
        if any(not isinstance(item, str) or not item.strip() for item in examples):
            raise ValueError
        model_load_seconds = 0.0
        if detector is None:
            load_started = time.monotonic()
            detector = load_detector(config)
            model_load_seconds = time.monotonic() - load_started
        prior_inference_seconds = detector.inference_seconds
        prior_tokenization_seconds = detector.tokenization_seconds
        prior_encoder_seconds = detector.encoder_seconds
        prior_classifier_seconds = detector.classifier_seconds
        predictions = detector.predict(examples)
        send(client, {"status": "ok", "predictions": predictions,
                      "service_pid": os.getpid(),
                      "inference_seconds": detector.inference_seconds - prior_inference_seconds,
                      "tokenization_seconds": detector.tokenization_seconds - prior_tokenization_seconds,
                      "encoder_seconds": detector.encoder_seconds - prior_encoder_seconds,
                      "classifier_seconds": detector.classifier_seconds - prior_classifier_seconds,
                      "model_load_seconds": model_load_seconds,
                      "service_processing_seconds": time.monotonic() - processing_started,
                      "service_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0})
    except Exception:
        # Never serialize exception details; they can contain staged source text.
        try:
            send(client, {"status": "error"})
        except OSError:
            pass
    return detector


def serve(path: Path, config_path: Path, idle_timeout: float) -> int:
    config = read_config(config_path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = path.parent.stat()
    if directory.st_uid != os.getuid() or directory.st_mode & 0o077:
        return 2
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
            return 2
        # Do not steal a live listener's pathname.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            try:
                probe.connect(str(path))
                return 0
            except OSError:
                path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    os.chmod(path, 0o600)
    server.listen(8)
    server.settimeout(min(max(idle_timeout, 1.0), 5.0))
    socket_info = path.lstat()
    detector = None
    last_request = time.monotonic()
    stop = False

    def stop_handler(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    try:
        while not stop:
            try:
                client, _ = server.accept()
            except socket.timeout:
                if time.monotonic() - last_request >= idle_timeout:
                    break
                continue
            with client:
                client.settimeout(30.0)
                detector = request(client, detector, config)
            last_request = time.monotonic()
    finally:
        # Dropping the model reference before process exit makes the lifecycle
        # explicit; process termination below releases the remaining RSS.
        detector = None
        server.close()
        try:
            current = path.lstat()
            if current.st_ino == socket_info.st_ino and stat.S_ISSOCK(current.st_mode):
                path.unlink()
        except FileNotFoundError:
            pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = read_config(args.config)
        return serve(args.socket, args.config, float(config.get("service_idle_seconds", 600)))
    except Exception:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
