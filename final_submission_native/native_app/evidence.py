#!/usr/bin/env python3
"""Collect real native-hook evidence; no model doubles, training or output overrides.

Run with native_app/.venv/bin/python after installing the application. Unix sockets and disposable
Git repositories must be permitted. All exported candidate text is redacted.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "hook"))
from pre_commit import collect_examples
from secret_detector import read_config, available_memory_mib
from service_client import _exchange

TIMING_FIELDS = ["workload", "eligible_lines", "condition", "run", "full_hook_elapsed_s",
                 "service_load_s", "inference_s", "commit_outcome", "host_id", "model_hash", "config_hash"]
CASE_FIELDS = ["case_id", "description", "expected_label", "actual_model_label", "commit_allowed",
               "hook_exit_code", "evidence_path", "test_status"]
SAFE_LINE = "total = 1"
# A fixed, declared set, never expanded in response to predictions. No live
# credentials are used or verified against any external service.
CASES = [
    ("SAFE-01", "Arithmetic assignment", 0, SAFE_LINE,
     "Only a numeric assignment; no credential content."),
    ("SAFE-02", "Sum of numeric literals", 0, "total = sum([1, 2, 3])",
     "Only arithmetic on numeric literals."),
    ("SAFE-03", "Boolean application setting", 0, "debug_enabled = False",
     "Boolean configuration value, not a credential."),
    ("SYN-01", "Historical FT-15 fixture repeated in a new staged trial", 1,
     "api_key = 'FAKE_FUNCTIONAL_TEST_TOKEN_NOT_A_REAL_CREDENTIAL_12345'",
     "Simulated API-key assignment; expected 1 is a synthetic detection target, not proof of a live secret."),
    ("SYN-02", "Nonfunctional AWS-shaped access-key assignment", 1,
     "aws_access_key_id = 'AKIA" + "0" * 16 + "'",
     "Credential variable and provider-shaped synthetic value; all-zero payload is a nonfunctional test fixture."),
    ("SYN-03", "Nonfunctional GitHub-shaped token assignment", 1,
     "github_token = 'ghp_" + "0" * 36 + "'",
     "Credential variable and provider-shaped synthetic value; all-zero payload is a nonfunctional test fixture."),
    ("SYN-04", "Nonfunctional password assignment", 1,
     "password = 'CH4_NONFUNCTIONAL_PASSWORD_000000'",
     "Simulated literal password assignment; no account or service uses it."),
    ("SYN-05", "Nonfunctional credential-bearing database URI", 1,
     "database_url = 'postgresql://test_user:CH4_FAKE_PASSWORD@db.invalid:5432/example'",
     "Simulated URI with username/password; reserved .invalid host, no external connection."),
]


class EvidenceError(RuntimeError):
    pass


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def timestamp():
    return datetime.now(ZoneInfo("Europe/Copenhagen")).isoformat()


def redact(text):
    for value in sorted([SAFE_LINE, *[case[3] for case in CASES]], key=len, reverse=True):
        text = text.replace(value, "[REDACTED SYNTHETIC CANDIDATE]")
    # Standalone values are also removed if a child process ever echoes one.
    for _, _, _, source, _ in CASES:
        for value in re.findall(r"'([^']+)'", source):
            text = text.replace(value, "[REDACTED SYNTHETIC VALUE]")
    return text


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def diagnostics(text):
    def seconds(label):
        match = re.search(r"^\s*" + re.escape(label) + r"\s+([0-9]+(?:\.[0-9]+)?) s\s*$", text, re.MULTILINE)
        return float(match.group(1)) if match else None
    labels = [int(value) for value in re.findall(r"^\s*Candidate .* label=([01]) decision=(?:allow|block)\s*$", text, re.MULTILINE)]
    service = re.search(r"^\s*Service\s+(cold start|warm \(reused\))\s*$", text, re.MULTILINE)
    return {"labels": labels, "service_load_s": seconds("Model load"), "inference_s": seconds("Inference"),
            "service_condition": service.group(1) if service else None,
            "scan_incomplete": "SCAN INCOMPLETE" in text}


def host_metadata():
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), "not recorded")
    memory = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
              for line in Path("/proc/meminfo").read_text().splitlines()
              if line.startswith(("MemTotal:", "MemAvailable:"))}
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "numpy", "scikit-learn", "joblib", "streamlit", "altair"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    host = {"cpu": cpu, "logical_cpu_count": os.cpu_count(), "platform": platform.platform(),
            "os": platform.freedesktop_os_release(), "ram": memory, "python": sys.version,
            "python_executable": sys.executable, "packages": packages}
    identity = {key: host[key] for key in ("cpu", "logical_cpu_count", "platform")}
    host["host_id"] = "ch4-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    host["identity_note"] = "Derived from reported CPU/kernel information, not a unique hardware identity. New run; not historical-host measurements."
    return host


class Collector:
    def __init__(self, output, trials):
        self.output, self.trials = Path(output).resolve(), trials
        self.config = read_config()
        self.host = host_metadata()
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = self.output / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, mode=0o700)
        self.integrity = {name: digest(ROOT / name) for name in (
            "hook/config.json", "hook/pre_commit.py", "hook/secret_detector.py", "hook/service_client.py",
            "hook/inference_service.py", "hook/detection_engine.py", "hook/install_global.sh",
            "install.sh", "requirements-model.txt",
            self.config["classifier_path"], self.config["experiment_record"],
            str(Path(self.config["experiment_record"]).with_name("run_config.json")))}
        self.model_hash = self.integrity[self.config["classifier_path"]]
        self.config_hash = self.integrity["hook/config.json"]
        self.summary = {"run_id": self.run_id, "started_at": timestamp(), "status": "running", "host": self.host,
                        "config": self.config, "integrity_before": self.integrity, "groups": [], "blockers": [],
                        "method": "Monotonic wall time around git commit, including native hook, observer shell and Git commit overhead; staging/reset/install excluded. Cold means new unloaded service, not dropped OS filesystem caches."}
        write_json(self.run_dir / "environment.json", self.summary)
        self.temp = tempfile.TemporaryDirectory(prefix="cb-ch4-")
        self.private = Path(self.temp.name)
        self.private.chmod(0o700)
        self.repo = self.private / "repo"
        self.repo.mkdir()
        self.socket = self.private / "s.sock"
        self.pid = None
        self.loaded_pid = None
        self.env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        self.env.pop("TRANSFORMERS_CACHE", None)
        self.env.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null", GIT_CONFIG_NOSYSTEM="1",
                        CODEBERT_PYTHON=sys.executable, CODEBERT_SECRET_HOOK_CONFIG=str(ROOT / "hook/config.json"),
                        CODEBERT_SECRET_HOOK_SOCKET=str(self.socket), CODEBERT_SECRET_HOOK_DIAGNOSTICS="1",
                        HF_HUB_CACHE=str(ROOT / "model_assets/huggingface/hub"), HF_HUB_OFFLINE="1",
                        TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")
        template = self.private / "empty-template"
        template.mkdir()
        self.env["GIT_TEMPLATE_DIR"] = str(template)

    def command(self, args, *, timeout=360):
        return subprocess.run(args, cwd=self.repo, env=self.env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=timeout)

    def git(self, *args):
        result = self.command(["git", *args])
        if result.returncode:
            raise EvidenceError("Disposable Git setup operation failed.")
        return result.stdout.strip()

    def setup(self):
        self.git("init", "--quiet")
        for key, value in {"user.name": "Chapter 4 Evidence", "user.email": "chapter4@example.invalid",
                           "core.hooksPath": ".git/hooks", "commit.gpgsign": "false", "gc.auto": "0",
                           "core.autocrlf": "false"}.items():
            self.git("config", "--local", key, value)
        self.git("commit", "--quiet", "--allow-empty", "-m", "Evidence baseline")
        self.baseline = self.git("rev-parse", "HEAD")
        hook = self.repo / ".git/hooks/pre-commit"
        native = hook.with_name("pre-commit.native")
        # Keep the user's global Git configuration untouched. This disposable
        # repository invokes the packaged pre-commit entry point directly.
        native.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " "
                         + shlex.quote(str(ROOT / "hook/pre_commit.py")) + " --config "
                         + shlex.quote(str(ROOT / "hook/config.json")) + " --diagnostics\n", encoding="utf-8")
        native.chmod(0o755)
        self.summary["installed_native_hook_sha256"] = digest(native)
        # Observer records exit status while preserving the actual hook result.
        hook.write_text("#!/bin/sh\nsh " + shlex.quote(str(native)) + "\nstatus=$?\n"
                        'printf "%s\\n" "$status" > "$CODEBERT_EVIDENCE_EXIT_FILE"\nexit "$status"\n')
        hook.chmod(0o755)
        self.summary["observer_hook_sha256"] = digest(hook)
        self.summary["observer_note"] = "Observer adds one shell call and exit-file write; the native launcher bytes and returned decision are unchanged."

    def ping(self):
        try:
            response = _exchange(self.socket, {"op": "ping"}, 1.0)
        except (FileNotFoundError, ConnectionRefusedError, ConnectionResetError):
            return None
        if response.get("status") != "ready" or type(response.get("service_pid")) is not int:
            raise EvidenceError("Private service readiness response is invalid.")
        pid = response["service_pid"]
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if str(self.socket).encode() not in command or str(ROOT / "hook/inference_service.py").encode() not in command:
            raise EvidenceError("Private service identity could not be verified.")
        self.pid = pid
        return pid

    @staticmethod
    def alive(pid):
        try:
            return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0] != "Z"
        except FileNotFoundError:
            return False

    def stop_service(self):
        pid = self.ping()
        if pid:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 8
            while self.alive(pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            if self.alive(pid):
                os.kill(pid, signal.SIGKILL)
                deadline = time.monotonic() + 5
                while self.alive(pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
            if self.alive(pid):
                raise EvidenceError("Owned service did not stop; cold state is unverified.")
        self.loaded_pid = None
        if self.ping() is not None or self.socket.exists():
            raise EvidenceError("Private socket remained present; cold state is unverified.")
        return {"stopped_pid": pid, "process_memory_released": not pid or not self.alive(pid), "socket_absent": True}

    def trial(self, name, text, expected_lines):
        self.git("reset", "--hard", self.baseline)
        (self.repo / "sample.py").write_text(text, encoding="utf-8")
        self.git("add", "--", "sample.py")
        examples, _ = collect_examples(self.repo, self.config)
        if len(examples) != expected_lines:
            raise EvidenceError("Actual staged candidate count differs from planned workload.")
        exit_file = self.repo / ".git/observed-hook-exit"
        exit_file.unlink(missing_ok=True)
        self.env["CODEBERT_EVIDENCE_EXIT_FILE"] = str(exit_file)
        available = available_memory_mib()
        started_at = timestamp()
        started = time.monotonic()
        commit = self.command(["git", "commit", "--quiet", "-m", "Evidence trial"])
        elapsed = time.monotonic() - started
        output = redact(commit.stdout)
        log = self.run_dir / (name + ".log")
        log.write_text(output)
        os.chmod(log, 0o600)
        hook_exit = int(exit_file.read_text().strip()) if exit_file.exists() else None
        parsed = diagnostics(output)
        pid = self.ping()
        inference_valid = not parsed["scan_incomplete"] and len(parsed["labels"]) == expected_lines
        decision_consistent = (inference_valid and hook_exit == int(any(parsed["labels"]))
                               and commit.returncode == (0 if hook_exit == 0 else 1))
        valid = inference_valid and decision_consistent
        if valid:
            self.loaded_pid = pid
        outcome = ("allowed" if commit.returncode == 0 else "blocked_secret") if valid else "blocked_scan_incomplete_or_error"
        detail = {"started_at": started_at, "full_hook_elapsed_s": elapsed, "eligible_lines": len(examples),
                  "git_commit_exit_code": commit.returncode, "hook_exit_code": hook_exit,
                  "commit_allowed": commit.returncode == 0, "commit_outcome": outcome,
                  "available_memory_before_mib": available, "service_pid": pid, "valid_inference": valid,
                  "model_labels_valid": inference_valid, "decision_consistent_with_labels": decision_consistent,
                  "log_path": str(log.relative_to(ROOT)) if log.is_relative_to(ROOT) else str(log), **parsed}
        write_json(self.run_dir / (name + ".json"), detail)
        return detail

    def benchmark(self):
        rows = []
        with (self.output / "hook_timing_runs.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=TIMING_FIELDS)
            writer.writeheader()
            stop = False
            for count in (1, 25, 100):
                for condition in ("cold", "warm"):
                    group = []
                    for run in range(1, self.trials + 1):
                        if condition == "cold":
                            precondition = self.stop_service()
                        else:
                            pid = self.ping()
                            if not pid or pid != self.loaded_pid:
                                raise EvidenceError("Warm model-loaded state is unverified.")
                            precondition = {"loaded_service_pid": pid, "prior_successful_model_inference": True}
                        name = f"runtime-{count}-{condition}-{run:02d}"
                        result = self.trial(name, (SAFE_LINE + "\n") * count, count)
                        state_valid = (result["service_condition"] == "cold start" and (result["service_load_s"] or 0) > 0
                                       if condition == "cold" else result["service_condition"] == "warm (reused)"
                                       and result["service_load_s"] == 0 and result["service_pid"] == precondition["loaded_service_pid"])
                        result["condition_verified"] = bool(state_valid and result["valid_inference"])
                        result["precondition"] = precondition
                        write_json(self.run_dir / (name + ".json"), result)
                        row = {"workload": f"staged_{count}", "eligible_lines": count, "condition": condition,
                               "run": run, "full_hook_elapsed_s": f"{result['full_hook_elapsed_s']:.6f}",
                               "service_load_s": result["service_load_s"] if result["service_load_s"] is not None else "",
                               "inference_s": result["inference_s"] if result["inference_s"] is not None else "",
                               "commit_outcome": result["commit_outcome"], "host_id": self.host["host_id"],
                               "model_hash": self.model_hash, "config_hash": self.config_hash}
                        writer.writerow(row); stream.flush()
                        rows.append(row)
                        if result["condition_verified"]:
                            group.append(result["full_hook_elapsed_s"])
                        else:
                            self.summary["blockers"].append(f"{name}: native inference or {condition} state failed; see per-run log/JSON. Remaining benchmark trials not executed.")
                            stop = True
                        print(json.dumps({"phase": "benchmark", "workload": count, "condition": condition, "run": run,
                                          "elapsed_s": round(result["full_hook_elapsed_s"], 3), "valid": result["condition_verified"]}), flush=True)
                        if stop: break
                    self.summary["groups"].append({"eligible_lines": count, "condition": condition, "valid_trials": len(group),
                        "median_s": statistics.median(group) if group else None, "minimum_s": min(group) if group else None,
                        "maximum_s": max(group) if group else None})
                    write_json(self.run_dir / "environment.json", self.summary)
                    if stop: break
                if stop: break
        self.summary["benchmark_complete"] = (len(rows) == 60 and self.trials == 10 and not stop)
        return rows

    def detection_cases(self):
        manifest = [{"case_id": case_id, "description": description, "expected_label": label, "rationale": rationale,
                     "synthetic_nonfunctional": label == 1, "fixture_sha256": hashlib.sha256(source.encode()).hexdigest()}
                    for case_id, description, label, source, rationale in CASES]
        write_json(self.run_dir / "case_manifest.json", manifest)
        results = []
        with (self.output / "real_model_staged_commit_results.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CASE_FIELDS); writer.writeheader()
            for case_id, description, expected, source, _ in CASES:
                result = self.trial("case-" + case_id, source + "\n", 1)
                actual = result["labels"][0] if result["model_labels_valid"] else ""
                status = ("ERROR_HOOK_DECISION_MISMATCH" if result["model_labels_valid"] and not result["decision_consistent_with_labels"] else
                          "PASS_SAFE_ALLOWED" if expected == 0 and actual == 0 else
                          "PASS_SYNTHETIC_DETECTED" if expected == 1 and actual == 1 else
                          "OBSERVED_SYNTHETIC_MISS" if expected == 1 and actual == 0 else
                          "OBSERVED_FALSE_POSITIVE" if expected == 0 and actual == 1 else "ERROR_SCAN_INCOMPLETE")
                row = {"case_id": case_id, "description": description, "expected_label": expected,
                       "actual_model_label": actual, "commit_allowed": result["commit_allowed"],
                       "hook_exit_code": result["hook_exit_code"] if result["hook_exit_code"] is not None else "",
                       "evidence_path": result["log_path"], "test_status": status}
                writer.writerow(row); stream.flush(); results.append(row)
                print(json.dumps({"phase": "real_model_case", "case_id": case_id, "actual_label": actual, "status": status}), flush=True)
        self.summary["case_results"] = results
        self.summary["actual_positive_cases_found"] = sum(row["actual_model_label"] == 1 for row in results)
        self.summary["real_model_cases_complete"] = all(type(row["actual_model_label"]) is int
                                                        and not row["test_status"].startswith("ERROR") for row in results)

    def close(self):
        try:
            self.stop_service()
        finally:
            self.temp.cleanup()

    def run(self):
        try:
            self.setup()
            self.benchmark()
            self.detection_cases()
            self.summary["status"] = ("complete" if self.summary["benchmark_complete"]
                                      and self.summary["real_model_cases_complete"] else "partial")
        except Exception as error:
            self.summary["status"] = "failed_or_incomplete"
            self.summary["blockers"].append(str(error) if isinstance(error, EvidenceError) else type(error).__name__ + ": see sanitized logs; no complete benchmark claim.")
        finally:
            try:
                self.close()
            except Exception:
                self.summary["blockers"].append("Owned service cleanup needs manual verification.")
            self.summary["finished_at"] = timestamp()
            self.summary["integrity_after"] = {name: digest(ROOT / name) for name in self.integrity}
            self.summary["production_integrity_unchanged"] = self.summary["integrity_before"] == self.summary["integrity_after"]
            write_json(self.run_dir / "environment.json", self.summary)
            write_json(self.output / "ch4_runtime_summary.json", self.summary)
            print(json.dumps({"status": self.summary["status"], "run_id": self.run_id,
                              "benchmark_complete": self.summary.get("benchmark_complete", False),
                              "blockers": self.summary["blockers"]}), flush=True)
        return 0 if self.summary["status"] == "complete" and self.summary["production_integrity_unchanged"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("trials must be positive")
    for name in ("hook_timing_runs.csv", "real_model_staged_commit_results.csv", "ch4_runtime_summary.json"):
        if (args.output / name).exists():
            parser.error("Evidence outputs already exist; choose a new --output directory to preserve previous runs.")
    return Collector(args.output, args.trials).run()


if __name__ == "__main__":
    raise SystemExit(main())
