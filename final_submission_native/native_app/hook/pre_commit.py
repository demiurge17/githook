#!/usr/bin/env python3
"""Scan staged source additions with the frozen local secret detector."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

from secret_detector import DEFAULT_CONFIG, DetectorError, read_config
from service_client import ServiceDetector
from detection_engine import Example, is_selected, line_examples, classify_examples


HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
SCAN_STATUSES = {"A", "M", "T"}


def _git(repo: Path, args: list[str]) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        raise RuntimeError("Git could not be started; staged scanning is incomplete.") from None
    if result.returncode:
        raise RuntimeError("Git could not read the staged index; scanning is incomplete.")
    return result.stdout


def repository_root(cwd: Path) -> Path:
    try:
        raw = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
    except OSError:
        raise RuntimeError("Git could not be started.") from None
    if raw.returncode:
        raise RuntimeError("Current directory is not inside a Git repository.")
    return Path(os.fsdecode(raw.stdout.rstrip(b"\n"))).resolve()


def staged_paths(repo: Path) -> list[tuple[str, str]]:
    unmerged = _git(repo, ["ls-files", "-u", "-z"])
    if unmerged:
        raise RuntimeError("The index has unresolved merge entries; resolve them before committing.")
    raw = _git(repo, ["diff", "--cached", "--name-status", "-z", "--no-renames"])
    fields = raw.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    changes: list[tuple[str, str]] = []
    index = 0
    while index < len(fields):
        status = os.fsdecode(fields[index])
        index += 1
        needed = 2 if status[:1] in {"C", "R"} else 1
        if index + needed > len(fields):
            raise RuntimeError("Git returned an incomplete staged path list.")
        names = [os.fsdecode(item) for item in fields[index:index + needed]]
        index += needed
        path = names[-1]
        if status[:1] in SCAN_STATUSES:
            changes.append((status[:1], path))
    return changes


def _staged_blob(repo: Path, path: str, config: dict) -> bytes | None:
    # :./ avoids revision/path ambiguity for unusual names; literal pathspec
    # ensures path punctuation is never interpreted as Git path magic.
    literal = f":(literal){path}"
    staged = _git(repo, ["ls-files", "--stage", "-z", "--", literal])
    entries = [item for item in staged.split(b"\0") if item]
    if len(entries) != 1 or b"\t" not in entries[0]:
        raise RuntimeError("An eligible staged file could not be resolved from the index.")
    metadata = entries[0].split(b"\t", 1)[0].split()
    if len(metadata) != 3 or metadata[2] != b"0":
        raise RuntimeError("An eligible staged file has an unresolved index state.")
    mode, object_id = metadata[0], metadata[1]
    if mode not in {b"100644", b"100755"}:
        raise RuntimeError("An eligible staged file is not a regular text-file entry.")
    size_raw = _git(repo, ["cat-file", "-s", os.fsdecode(object_id)]).strip()
    try:
        size = int(size_raw)
    except ValueError:
        raise RuntimeError("Git returned an invalid staged file size.") from None
    if size > int(config["max_file_bytes"]):
        raise RuntimeError("An eligible staged file exceeds the configured size limit; scan incomplete.")
    return _git(repo, ["cat-file", "blob", os.fsdecode(object_id)])


def added_lines(diff_bytes: bytes) -> list[tuple[int, str]]:
    try:
        diff = diff_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise RuntimeError("A staged text diff is not valid UTF-8; scan incomplete.") from None
    line_number = None
    additions: list[tuple[int, str]] = []
    for line in diff.split("\n"):
        line = line.removesuffix("\r")
        match = HUNK_RE.match(line)
        if match:
            line_number = int(match.group(1))
            continue
        if line_number is None or not line:
            continue
        if line.startswith("+"):
            code = line[1:]
            if code.strip():
                additions.append((line_number, code))
            line_number += 1
        elif line.startswith(" "):
            line_number += 1
        # Deletions do not advance the new-file line number.
    return additions


def collect_examples(repo: Path, config: dict) -> tuple[list[Example], dict]:
    discovery_started = time.monotonic()
    changes = staged_paths(repo)
    staged_discovery_seconds = time.monotonic() - discovery_started
    selected = [(status, path) for status, path in changes if is_selected(path, config)]
    if len(selected) > int(config["max_files"]):
        raise RuntimeError("Staged source-file count exceeds the configured limit; scan incomplete.")
    examples: list[Example] = []
    stats = {"files_considered": len(selected), "files_scanned": 0, "binary_files_skipped": 0}
    parse_started = time.monotonic()
    for _status, path in selected:
        blob = _staged_blob(repo, path, config)
        if blob is None:
            continue
        if b"\0" in blob:
            stats["binary_files_skipped"] += 1
            continue
        try:
            blob.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError:
            raise RuntimeError("An eligible staged file has an unsupported encoding; scan incomplete.") from None
        diff = _git(repo, [
            "diff", "--cached", "--no-ext-diff", "--no-color", "--no-renames",
            "--no-textconv", "--unified=0", "--no-prefix", "--", f":(literal){path}",
        ])
        if b"Binary files " in diff or b"GIT binary patch" in diff:
            stats["binary_files_skipped"] += 1
            continue
        lines = added_lines(diff)
        if lines:
            stats["files_scanned"] += 1
            examples.extend(line_examples(path, lines, config))
            if len(examples) > int(config["max_examples"]):
                raise RuntimeError("Staged example count exceeds the configured limit; scan incomplete.")
    stats["staged_discovery_seconds"] = round(staged_discovery_seconds, 4)
    stats["candidate_extraction_seconds"] = round(time.monotonic() - parse_started, 4)
    return examples, stats


def peak_rss_mib() -> float | None:
    try:
        # Linux reports ru_maxrss in KiB.
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except (AttributeError, OSError):
        return None


def scan(repo: Path, config: dict, detector_factory=ServiceDetector) -> tuple[int, dict, list[Example]]:
    examples, stats = collect_examples(repo, config)
    stats["examples_processed"] = len(examples)
    stats["model_load_seconds"] = 0.0
    stats["inference_seconds"] = 0.0
    stats["tokenization_seconds"] = 0.0
    stats["encoder_seconds"] = 0.0
    stats["classifier_seconds"] = 0.0
    stats["service_roundtrip_seconds"] = 0.0
    stats["peak_rss_mib"] = None
    stats["findings"] = []
    stats["candidate_classifications"] = []
    stats["prediction_counts"] = {"0": 0, "1": 0}
    for timing in ("service_started", "service_startup_seconds", "connection_seconds",
                   "startup_connection_seconds", "prediction_connection_seconds",
                   "request_response_seconds", "startup_request_response_seconds",
                   "prediction_request_response_seconds", "response_handling_seconds",
                   "startup_response_handling_seconds", "prediction_response_handling_seconds",
                   "service_processing_seconds"):
        stats[timing] = False if timing == "service_started" else 0.0
    stats["service_peak_rss_mib"] = None
    if not examples:
        return 0, stats, examples
    init_started = time.monotonic()
    detector = detector_factory(config)
    classifications = classify_examples(examples, detector)
    predictions = [row["label"] for row in classifications]
    stats["model_load_seconds"] = round(float(getattr(detector, "model_load_seconds", time.monotonic() - init_started)), 3)
    stats["inference_seconds"] = round(float(detector.inference_seconds), 3)
    stats["service_roundtrip_seconds"] = round(float(getattr(detector, "client_seconds", 0.0)), 3)
    for timing in ("tokenization", "encoder", "classifier"):
        stats[timing + "_seconds"] = round(float(getattr(detector, timing + "_seconds", 0.0)), 4)
    stats["peak_rss_mib"] = peak_rss_mib()
    stats["service_peak_rss_mib"] = getattr(detector, "service_peak_rss_mib", None)
    if len(predictions) != len(examples):
        raise RuntimeError("Prediction count did not match scanned examples; scan incomplete.")
    if any(type(label) is not int or label not in (0, 1) for label in predictions):
        raise RuntimeError("Classifier returned an invalid label; scan incomplete.")
    label_counts = Counter(int(label) for label in predictions)
    stats["prediction_counts"] = {str(label): label_counts.get(label, 0) for label in (0, 1)}
    for timing in ("service_started", "service_startup_seconds", "connection_seconds",
                   "startup_connection_seconds", "prediction_connection_seconds",
                   "request_response_seconds", "startup_request_response_seconds",
                   "prediction_request_response_seconds", "response_handling_seconds",
                   "startup_response_handling_seconds", "prediction_response_handling_seconds",
                   "service_processing_seconds"):
        stats[timing] = getattr(detector, timing, False if timing == "service_started" else 0.0)
    findings = [example for example, label in zip(examples, predictions) if int(label) == 1]
    stats["findings"] = [{"path": item.path, "line": item.line} for item in findings]
    stats["candidate_classifications"] = [
        {"path": item.path, "line": item.line, "label": label}
        for item, label in zip(examples, predictions)
    ]
    return (1 if findings else 0), stats, examples


def print_scan_report(result: int, stats: dict, elapsed: float, diagnostics: bool = False) -> None:
    """Show the scan outcome and safe metadata without printing candidate text."""
    count = stats["examples_processed"]
    status = "NO CANDIDATES" if not count else ("COMMIT BLOCKED" if result else "COMMIT ALLOWED")
    print(f"CodeBERT Secret Scan | {status}")
    print("-" * 60)
    if not count:
        print("  No eligible staged added lines were found.")
        print("  Inference not started. Commit allowed.")
    else:
        line_word = "line" if count == 1 else "lines"
        file_word = "file" if stats["files_scanned"] == 1 else "files"
        print(f"  {'Scope':<13}{count} added {line_word} in {stats['files_scanned']} staged {file_word}")
        positives = stats["prediction_counts"]["1"]
        secret_word = "secret" if positives == 1 else "secrets"
        print(f"  {'Predictions':<13}{stats['prediction_counts']['0']} non-secret | "
              f"{positives} potential {secret_word}")
    print(f"  {'Files':<13}{stats['files_scanned']} scanned / {stats['files_considered']} eligible | "
          f"{stats['binary_files_skipped']} binary skipped")
    print(f"  {'Duration':<13}{elapsed:.2f} s")
    if count:
        print()
        if result:
            print(f"  Findings ({len(stats['findings'])})")
            for finding in stats["findings"]:
                safe_path = json.dumps(finding["path"], ensure_ascii=True)[1:-1]
                print(f"  Potential secret: {safe_path}:{finding['line']}")
            print()
            print("  Review these staged lines and remove or replace any real credential.")
        else:
            print("  No potential secrets detected in eligible staged additions.")

    if diagnostics:
        print("\n  Diagnostics")
        print("  " + "-" * 58)
        for candidate in stats["candidate_classifications"]:
            safe_path = json.dumps(candidate["path"], ensure_ascii=True)[1:-1]
            outcome = "block" if candidate["label"] == 1 else "allow"
            print(f"  Candidate {safe_path}:{candidate['line']} label={candidate['label']} decision={outcome}")
        service = "not started" if not count else ("cold start" if stats["service_started"] else "warm (reused)")
        print(f"  {'Service':<24}{service}")
        for label, key, precision in (
            ("Staged discovery", "staged_discovery_seconds", 4),
            ("Candidate extraction", "candidate_extraction_seconds", 4),
            ("Startup / readiness", "service_startup_seconds", 3),
            ("Model load", "model_load_seconds", 3),
            ("Connection", "connection_seconds", 4),
            ("Request / response", "request_response_seconds", 3),
            ("Server processing", "service_processing_seconds", 3),
            ("Response handling", "response_handling_seconds", 4),
            ("Inference", "inference_seconds", 3),
            ("Tokenization", "tokenization_seconds", 4),
            ("Encoder", "encoder_seconds", 4),
            ("Classifier", "classifier_seconds", 4),
            ("Service roundtrip", "service_roundtrip_seconds", 3),
        ):
            print(f"  {label:<24}{stats.get(key, 0.0):.{precision}f} s")
        for label, key in (("Peak RSS", "peak_rss_mib"), ("Service peak RSS", "service_peak_rss_mib")):
            memory = stats.get(key)
            value = "not measured" if memory is None else f"{memory:.1f} MiB"
            print(f"  {label:<24}{value}")


def main(argv: list[str] | None = None, detector_factory=ServiceDetector) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-model", action="store_true", help="load and validate local frozen assets, without scanning a repository")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--diagnostics", action="store_true", help="show candidate labels, detailed timings and memory, never source text")
    args = parser.parse_args(argv)
    hook_started = time.monotonic()
    try:
        config = read_config(args.config)
        if args.verify_model:
            from secret_detector import load_detector
            load_detector(config)
            print("CodeBERT Secret Scan | MODEL VERIFIED")
            print("-" * 60)
            print("Frozen classifier and local pinned CodeBERT checkpoint verified.")
            return 0
        repo = repository_root(Path.cwd())
        result, stats, _examples = scan(repo, config, detector_factory)
        diagnostics_enabled = args.diagnostics or os.environ.get("CODEBERT_SECRET_HOOK_DIAGNOSTICS") == "1"
        print_scan_report(result, stats, time.monotonic() - hook_started, diagnostics_enabled)
        if result:
            return 1
        return 0
    except Exception:
        # Exception text is intentionally withheld to prevent leaking source values.
        print("CodeBERT Secret Scan | SCAN INCOMPLETE", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        if args.verify_model:
            print("ERROR: frozen classifier or local encoder validation failed.", file=sys.stderr)
        else:
            print("ERROR: staged secret scan could not be completed; commit blocked.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
