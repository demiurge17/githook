from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOOK_DIR = PROJECT_ROOT / "hook"
sys.path.insert(0, str(HOOK_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import pre_commit
import secret_detector
import inference_service
import service_client


class FakeDetector:
    def __init__(self, labels=None, error=None):
        self.labels = labels or []
        self.error = error
        self.inference_seconds = 0.001
        self.received = []

    def predict(self, examples):
        self.received.extend(examples)
        if self.error:
            raise RuntimeError(self.error)
        return self.labels[:len(examples)]


class PreCommitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        # Isolate scanner unit tests from the user's installed global hook.
        subprocess.run(["git", "-C", str(self.repo), "config", "core.hooksPath", ".git/hooks"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.email", "hook-test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.name", "Hook Test"], check=True)
        self.config = secret_detector.read_config()
        self.config["max_file_bytes"] = 1024
        self.config["max_files"] = 20
        self.config["max_examples"] = 20

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def commit_file(self, path, content):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
        self.git("add", "--", path)
        self.git("commit", "-qm", "fixture")
        return target

    def scan(self, detector):
        return pre_commit.scan(self.repo, self.config, lambda _config: detector)

    def test_clean_staged_file_scans_staged_additions_and_passes(self):
        detector = FakeDetector([0])
        (self.repo / "safe.py").write_text("safe = True\n", encoding="utf-8")
        self.git("add", "--", "safe.py")
        code, stats, _ = self.scan(detector)
        self.assertEqual(code, 0)
        self.assertEqual(detector.received, ["safe = True"])
        self.assertEqual(stats["examples_processed"], 1)

    def test_empty_scan_allows_clean_non_source_commit_and_reports_not_started(self):
        (self.repo / "README.md").write_text("documentation only\n", encoding="utf-8")
        self.git("add", "README.md")
        output = io.StringIO()
        detector_factory = mock.Mock(side_effect=AssertionError("Empty scans must not start inference."))
        with mock.patch.object(pre_commit, "repository_root", return_value=self.repo), \
                mock.patch.dict(os.environ, {"CODEBERT_SECRET_HOOK_DIAGNOSTICS": "0"}), \
                contextlib.redirect_stdout(output):
            result = pre_commit.main([], detector_factory)
        self.assertEqual(result, 0)
        detector_factory.assert_not_called()
        report = output.getvalue()
        self.assertIn("CodeBERT Secret Scan | NO CANDIDATES", report)
        self.assertIn("No eligible staged added lines were found.", report)
        self.assertIn("Inference not started. Commit allowed.", report)
        self.assertNotIn("Predictions", report)
        self.assertNotIn("Diagnostics", report)

    def test_default_clean_report_is_concise_and_shows_scan_counts(self):
        (self.repo / "clean.py").write_text("first = 1\nsecond = 2\n", encoding="utf-8")
        self.git("add", "clean.py")
        output = io.StringIO()
        with mock.patch.object(pre_commit, "repository_root", return_value=self.repo), \
                mock.patch.dict(os.environ, {"CODEBERT_SECRET_HOOK_DIAGNOSTICS": "0"}), \
                contextlib.redirect_stdout(output):
            result = pre_commit.main([], lambda _config: FakeDetector([0, 0]))
        self.assertEqual(result, 0)
        report = output.getvalue()
        self.assertIn("CodeBERT Secret Scan | COMMIT ALLOWED", report)
        self.assertIn("No potential secrets detected in eligible staged additions.", report)
        self.assertRegex(report, r"Scope\s*:?\s+2 added line(?:s|\(s\))? in 1 staged file(?:s|\(s\))?")
        self.assertRegex(report, r"Predictions\s*:?\s+2 non-secret \| 0 potential secret")
        self.assertRegex(report, r"Files\s*:?\s+1 scanned / 1 eligible \| 0 binary skipped")
        self.assertRegex(report, r"Duration\s*:?\s+\d+(?:\.\d+)?\s*s")
        for detail in ("Diagnostics", "Model load", "Encoder", "Classifier", "Peak RSS", "Candidate "):
            self.assertNotIn(detail, report)

    def test_blocked_report_preserves_mixed_predictions_and_hides_source(self):
        canary = "BLOCKED_REPORT_PRIVATE_CANARY"
        (self.repo / "mixed.py").write_text(f"safe = True\ntoken = '{canary}'\n", encoding="utf-8")
        self.git("add", "mixed.py")
        output = io.StringIO()
        with mock.patch.object(pre_commit, "repository_root", return_value=self.repo), \
                mock.patch.dict(os.environ, {"CODEBERT_SECRET_HOOK_DIAGNOSTICS": "0"}), \
                contextlib.redirect_stdout(output):
            result = pre_commit.main([], lambda _config: FakeDetector([0, 1]))
        self.assertEqual(result, 1)
        report = output.getvalue()
        self.assertIn("CodeBERT Secret Scan | COMMIT BLOCKED", report)
        self.assertRegex(report, r"Predictions\s*:?\s+1 non-secret \| 1 potential secret")
        self.assertIn("Potential secret: mixed.py:2", report)
        self.assertNotIn("Potential secret: mixed.py:1", report)
        self.assertNotIn("Model load", report)
        self.assertNotIn("Peak RSS", report)
        self.assertNotIn(canary, report)

    def test_opt_in_candidate_diagnostics_show_location_and_decision_only(self):
        content_marker = "SYNTHETIC_DIAGNOSTIC_CANARY"
        (self.repo / "src.py").write_text(f"token = '{content_marker}'\n", encoding="utf-8")
        self.git("add", "src.py")
        detector = FakeDetector([1])
        output = io.StringIO()
        with mock.patch.object(pre_commit, "repository_root", return_value=self.repo), \
                contextlib.redirect_stdout(output):
            result = pre_commit.main(["--diagnostics"], lambda _config: detector)
        self.assertEqual(result, 1)
        report = output.getvalue()
        self.assertIn("CodeBERT Secret Scan | COMMIT BLOCKED", report)
        self.assertRegex(report, r"Predictions\s*:?\s+0 non-secret \| 1 potential secret")
        self.assertIn("Diagnostics", report)
        for metric in ("Service", "Model load", "Encoder", "Classifier", "Peak RSS"):
            self.assertIn(metric, report)
        self.assertIn("Candidate src.py:1 label=1 decision=block", report)
        self.assertNotIn("reason=", report)
        self.assertNotIn(content_marker, report)

    def test_findings_and_diagnostics_escape_control_characters_in_paths(self):
        path = "source\n\x1b[31m.py"
        (self.repo / path).write_text("value = 'SYNTHETIC_PATH_TEST'\n", encoding="utf-8")
        self.git("add", "--", path)
        output = io.StringIO()
        with mock.patch.object(pre_commit, "repository_root", return_value=self.repo), \
                contextlib.redirect_stdout(output):
            result = pre_commit.main(["--diagnostics"], lambda _config: FakeDetector([1]))
        self.assertEqual(result, 1)
        report = output.getvalue()
        escaped_path = json.dumps(path, ensure_ascii=True)[1:-1]
        self.assertIn(f"Potential secret: {escaped_path}:1", report)
        self.assertIn(f"Candidate {escaped_path}:1 label=1 decision=block", report)
        self.assertNotIn(path, report)
        self.assertNotIn("\x1b", report)
        self.assertNotIn("SYNTHETIC_PATH_TEST", report)

    def test_verify_model_reports_success_without_scanning(self):
        output = io.StringIO()
        with mock.patch.object(secret_detector, "load_detector") as load_detector, \
                mock.patch.object(pre_commit, "repository_root") as repository_root, \
                mock.patch.object(pre_commit, "scan") as scan, \
                contextlib.redirect_stdout(output):
            result = pre_commit.main(["--verify-model"])
        self.assertEqual(result, 0)
        load_detector.assert_called_once()
        repository_root.assert_not_called()
        scan.assert_not_called()
        self.assertIn("CodeBERT Secret Scan | MODEL VERIFIED", output.getvalue())

    def test_scan_uses_raw_binary_predictions_for_every_candidate(self):
        examples = [
            "API_KEYS = []",
            "API_KEY = None",
            'API_KEY = ""',
            'API_KEY = os.getenv("API_KEY")',
            "API_KEYS",
            "API_KEY: str",
            'API_KEY = "SYNTHETIC_TEST_TOKEN"',
        ]
        (self.repo / "examples.py").write_text("\n".join(examples) + "\n", encoding="utf-8")
        self.git("add", "examples.py")
        for label in (0, 1):
            with self.subTest(label=label):
                code, stats, _ = self.scan(FakeDetector([label] * len(examples)))
                self.assertEqual(code, label)
                self.assertEqual(len(stats["findings"]), len(examples) if label else 0)
                self.assertEqual([item["label"] for item in stats["candidate_classifications"]],
                                 [label] * len(examples))
                self.assertEqual(stats["prediction_counts"][str(label)], len(examples))

    def test_invalid_or_incomplete_predictions_fail_closed_without_content(self):
        canary = "INVALID_PREDICTION_CANARY"
        (self.repo / "invalid.py").write_text(f"token = '{canary}'\n", encoding="utf-8")
        self.git("add", "invalid.py")
        for labels in ([], [0, 1], [9], [-1], [True], [0.5], ["1"], [None]):
            with self.subTest(labels=labels):
                output, errors = io.StringIO(), io.StringIO()
                detector = FakeDetector(labels)
                detector.predict = mock.Mock(return_value=labels)
                with mock.patch.object(pre_commit, "repository_root", return_value=self.repo), \
                        contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                    result = pre_commit.main([], lambda _config: detector)
                self.assertEqual(result, 2)
                self.assertIn("CodeBERT Secret Scan | SCAN INCOMPLETE", errors.getvalue())
                self.assertIn("commit blocked", errors.getvalue())
                self.assertNotIn(canary, output.getvalue() + errors.getvalue())

    def test_service_predictions_reach_git_decision_unchanged(self):
        (self.repo / "examples.py").write_text("API_KEY = None\nAPI_KEYS = []\n", encoding="utf-8")
        self.git("add", "examples.py")
        for labels in ([0, 0], [1, 1], [0, 1], [1, 0]):
            with self.subTest(labels=labels):
                with mock.patch.object(service_client, "_prepare_private_directory"), \
                        mock.patch.object(service_client, "_exchange", return_value={
                            "status": "ok", "predictions": labels, "service_pid": 12345,
                        }):
                    code, stats, _ = pre_commit.scan(self.repo, self.config)
                self.assertEqual(code, int(any(labels)))
                self.assertEqual([item["line"] for item in stats["findings"]],
                                 [index + 1 for index, label in enumerate(labels) if label == 1])
                self.assertEqual([item["label"] for item in stats["candidate_classifications"]], labels)

    def test_saved_forest_predictions_reach_scanner_without_changes(self):
        import joblib
        import numpy as np

        # Synthetic feature vectors exercise the saved forest without loading
        # CodeBERT, opening datasets, or fitting any classifier. This checks the
        # prediction-to-decision path, not the accuracy of the feature extractor.
        classifier = joblib.load(PROJECT_ROOT / self.config["classifier_path"])
        features = np.vstack((np.zeros(768, dtype=np.float32), np.ones(768, dtype=np.float32)))
        labels = [int(label) for label in classifier.predict(features)]
        (self.repo / "examples.py").write_text("API_KEY = None\nAPI_KEYS = []\n", encoding="utf-8")
        self.git("add", "examples.py")
        with mock.patch.object(service_client, "_prepare_private_directory"), \
                mock.patch.object(service_client, "_exchange", return_value={
                    "status": "ok", "predictions": labels, "service_pid": 12345,
                }):
            code, stats, _ = pre_commit.scan(self.repo, self.config)
        self.assertEqual(code, int(any(labels)))
        self.assertEqual([item["label"] for item in stats["candidate_classifications"]], labels)
        self.assertEqual([item["line"] for item in stats["findings"]],
                         [index + 1 for index, label in enumerate(labels) if label == 1])

    def test_positive_prediction_blocks_without_printing_content(self):
        canary = "SYNTHETIC_CANARY_DO_NOT_DISPLAY"
        (self.repo / "src.py").write_text(f"token = '{canary}'\n", encoding="utf-8")
        self.git("add", "--", "src.py")
        detector = FakeDetector([1])
        code, stats, _ = self.scan(detector)
        self.assertEqual(code, 1)
        self.assertEqual(stats["findings"], [{"path": "src.py", "line": 1}])
        stream = io.StringIO()
        with mock.patch.object(pre_commit, "repository_root", return_value=self.repo), contextlib.redirect_stdout(stream):
            pre_commit.main([], lambda _config: detector)
        self.assertNotIn(canary, stream.getvalue())

    def test_staged_blob_is_used_when_worktree_changes_after_staging(self):
        path = self.commit_file("module.py", "old = 1\n")
        path.write_text("staged_value = 'STAGED_CANARY'\n", encoding="utf-8")
        self.git("add", "--", "module.py")
        path.write_text("worktree_only = 'UNSTAGED_CANARY'\n", encoding="utf-8")
        detector = FakeDetector([0])
        code, _, _ = self.scan(detector)
        self.assertEqual(code, 0)
        self.assertEqual(detector.received, ["staged_value = 'STAGED_CANARY'"])
        self.assertNotIn("UNSTAGED_CANARY", detector.received[0])

    def test_partial_staging_only_scans_index_additions(self):
        path = self.commit_file("partial.py", "first = 1\nsecond = 2\n")
        path.write_text("first = 1\nstaged_second = 2\n", encoding="utf-8")
        self.git("add", "--", "partial.py")
        path.write_text("first = 1\nstaged_second = 2\nunstaged_third = 3\n", encoding="utf-8")
        detector = FakeDetector([0])
        self.scan(detector)
        self.assertEqual(detector.received, ["staged_second = 2"])

    def test_filename_with_spaces_is_handled(self):
        (self.repo / "source file.py").write_text("x = 1\n", encoding="utf-8")
        self.git("add", "--", "source file.py")
        detector = FakeDetector([0])
        self.scan(detector)
        self.assertEqual(detector.received, ["x = 1"])

    def test_filename_with_newline_and_punctuation_is_handled(self):
        path = "line\nbreak;[x].py"
        (self.repo / path).write_text("odd_name = 1\n", encoding="utf-8")
        self.git("add", "--", path)
        detector = FakeDetector([0])
        self.scan(detector)
        self.assertEqual(detector.received, ["odd_name = 1"])

    def test_renamed_file_is_scanned_under_new_path(self):
        self.commit_file("old_name.py", "original = 1\n")
        self.git("mv", "--", "old_name.py", "new_name.py")
        (self.repo / "new_name.py").write_text("renamed_addition = 2\n", encoding="utf-8")
        self.git("add", "--", "new_name.py")
        detector = FakeDetector([0])
        self.scan(detector)
        self.assertEqual(detector.received, ["renamed_addition = 2"])

    def test_binary_file_is_skipped(self):
        (self.repo / "binary.py").write_bytes(b"x = 1\x00\xff")
        self.git("add", "--", "binary.py")
        detector = FakeDetector([])
        code, stats, _ = self.scan(detector)
        self.assertEqual(code, 0)
        self.assertEqual(stats["binary_files_skipped"], 1)
        self.assertEqual(detector.received, [])

    def test_deleted_file_is_not_scanned(self):
        path = self.commit_file("gone.py", "x = 1\n")
        self.git("rm", "-q", "--", "gone.py")
        detector = FakeDetector([])
        code, stats, _ = self.scan(detector)
        self.assertEqual(code, 0)
        self.assertEqual(stats["examples_processed"], 0)
        self.assertEqual(detector.received, [])

    def test_oversize_file_fails_closed(self):
        self.config["max_file_bytes"] = 5
        (self.repo / "large.py").write_text("longer = True\n", encoding="utf-8")
        self.git("add", "--", "large.py")
        with self.assertRaisesRegex(RuntimeError, "size limit"):
            self.scan(FakeDetector([]))

    def test_invalid_utf8_fails_closed(self):
        (self.repo / "bad_encoding.py").write_bytes(b"x = '\xff'\n")
        self.git("add", "--", "bad_encoding.py")
        with self.assertRaisesRegex(RuntimeError, "unsupported encoding"):
            self.scan(FakeDetector([]))

    def test_too_many_examples_fails_instead_of_partial_clean_result(self):
        self.config["max_examples"] = 1
        (self.repo / "many.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
        self.git("add", "--", "many.py")
        with self.assertRaisesRegex(RuntimeError, "example count"):
            self.scan(FakeDetector([0]))

    def test_inference_exception_is_sanitized(self):
        canary = "INFERENCE_EXCEPTION_CANARY"
        (self.repo / "failure.py").write_text(f"x = '{canary}'\n", encoding="utf-8")
        self.git("add", "--", "failure.py")
        detector = FakeDetector(error=f"raw input leaked: {canary}")
        with mock.patch.object(pre_commit, "repository_root", return_value=self.repo), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            result = pre_commit.main([], lambda _config: detector)
        self.assertEqual(result, 2)
        self.assertIn("CodeBERT Secret Scan | SCAN INCOMPLETE", err.getvalue())
        self.assertIn("commit blocked", err.getvalue())
        self.assertNotIn(canary, err.getvalue())

    def test_missing_or_corrupt_classifier_fails_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "artifacts/run/models").mkdir(parents=True)
            (root / "artifacts/run/experiment.json").write_text(json.dumps({
                "embedding_model": "microsoft/codebert-base",
                "embedding_model_revision": "rev",
                "max_length_tokens": 512,
                "pooling": "pool",
                "hidden_size": 768,
                "label_mapping": {"1": "secret"},
                "selected_random_forest": {},
            }), encoding="utf-8")
            (root / "artifacts/run/run_config.json").write_text(json.dumps({
                "model": "microsoft/codebert-base",
                "model_revision": "rev",
                "tokenizer_revision": "rev",
            }), encoding="utf-8")
            model_path = root / "artifacts/run/models/rf.joblib"
            model_path.write_bytes(b"corrupt synthetic artifact")
            config = {
                "experiment_record": "artifacts/run/experiment.json",
                "classifier_path": "artifacts/run/models/rf.joblib",
                "classifier_sha256": "0" * 64,
                "model": "microsoft/codebert-base", "model_revision": "rev",
                "tokenizer_revision": "rev", "max_length": 512, "pooling": "pool",
                "embedding_size": 768, "positive_label": 1, "batch_size": 2,
                "torch_threads": 1, "min_memavailable_mib": 1024,
                "min_memavailable_during_mib": 512,
            }
            # Isolate checksum validation from host RAM and heavyweight imports.
            # Memory protection has its own test and remains active in production.
            modules = {name: mock.MagicMock() for name in ("joblib", "numpy", "torch", "transformers")}
            with mock.patch.object(secret_detector, "PROJECT_ROOT", root), \
                    mock.patch.object(secret_detector, "available_memory_mib", return_value=2048.0), \
                    mock.patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(secret_detector.DetectorError, "checksum"):
                    secret_detector.load_detector(config)
                model_path.unlink()
                with self.assertRaises(secret_detector.DetectorError):
                    secret_detector.load_detector(config)

    def test_memory_guard_stops_before_loading(self):
        with mock.patch.object(secret_detector, "available_memory_mib", return_value=128.0):
            with self.assertRaisesRegex(secret_detector.DetectorError, "memory"):
                secret_detector.require_memory(self.config)
            with self.assertRaisesRegex(secret_detector.DetectorError, "memory"):
                secret_detector.load_detector(self.config)
        with mock.patch.object(secret_detector, "available_memory_mib", return_value=None):
            with self.assertRaisesRegex(secret_detector.DetectorError, "could not be checked"):
                secret_detector.require_memory(self.config)

    def test_unselected_file_types_are_not_scanned(self):
        (self.repo / "notes.md").write_text("text = 'not source-selected'\n", encoding="utf-8")
        self.git("add", "--", "notes.md")
        detector = FakeDetector([])
        code, stats, _ = self.scan(detector)
        self.assertEqual(code, 0)
        self.assertEqual(stats["files_considered"], 0)
        self.assertEqual(detector.received, [])

    def install_test_hook(self, labels):
        hook = self.repo / ".git/hooks/pre-commit"
        hook.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            f"sys.path.insert(0, {str(HOOK_DIR)!r})\n"
            "import pre_commit\n"
            "class TestDetector:\n"
            "    inference_seconds = 0.001\n"
            "    def predict(self, examples):\n"
            f"        return {labels!r}[:len(examples)]\n"
            "raise SystemExit(pre_commit.main(detector_factory=lambda _config: TestDetector()))\n",
            encoding="utf-8",
        )
        hook.chmod(0o755)

    def test_native_hook_allows_clean_commit(self):
        self.install_test_hook([0])
        (self.repo / "clean.py").write_text("number = 3\n", encoding="utf-8")
        self.git("add", "--", "clean.py")
        result = subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "clean test"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))

    def test_service_request_returns_predictions_without_echoing_input(self):
        class TestDetector:
            inference_seconds = 0.003
            tokenization_seconds = 0.001
            encoder_seconds = 0.001
            classifier_seconds = 0.001
            def predict(self, examples):
                self.examples = examples
                return [0] * len(examples)
        class MemorySocket:
            def __init__(self):
                self.incoming = bytearray()
                self.outgoing = bytearray()
            def recv(self, count):
                result = bytes(self.incoming[:count])
                del self.incoming[:count]
                return result
            def sendall(self, data):
                self.outgoing.extend(data)
        server = MemorySocket()
        client = MemorySocket()
        detector = TestDetector()
        request_payload = json.dumps({"op": "predict", "examples": ["PRIVATE_CANARY"]}).encode()
        server.incoming.extend(len(request_payload).to_bytes(4, "big") + request_payload)
        inference_service.request(server, detector, {"max_examples": 5})
        size = int.from_bytes(server.outgoing[:4], "big")
        response = bytes(server.outgoing[4:4 + size])
        self.assertNotIn(b"PRIVATE_CANARY", response)
        self.assertEqual(json.loads(response)["predictions"], [0])

    def test_service_response_validation_rejects_bad_labels(self):
        proxy = service_client.ServiceDetector(self.config)
        for labels in ([9], [-1], [True], [0.5], ["1"], [None], [], [0, 1], None):
            with self.subTest(labels=labels):
                with mock.patch.object(service_client, "_prepare_private_directory"), \
                        mock.patch.object(service_client, "_exchange", return_value={
                            "status": "ok", "predictions": labels, "service_pid": 12345,
                        }):
                    with self.assertRaisesRegex(secret_detector.DetectorError, "incompatible predictions"):
                        proxy.predict(["synthetic test input"])

    def test_service_client_preserves_raw_labels_for_all_source_forms(self):
        proxy = service_client.ServiceDetector(self.config)
        examples = ["API_KEY = None", "API_KEYS = []", 'API_KEY = "SYNTHETIC_TEST_TOKEN"']
        for labels in ([1, 1, 1], [0, 0, 0], [1, 0, 1]):
            with self.subTest(labels=labels):
                with mock.patch.object(service_client, "_prepare_private_directory"), \
                        mock.patch.object(service_client, "_exchange", return_value={
                            "status": "ok", "predictions": labels, "service_pid": 12345,
                        }):
                    self.assertEqual(proxy.predict(examples), labels)

    def test_unavailable_service_fails_closed_quickly(self):
        config = dict(self.config)
        config["service_request_timeout_seconds"] = 0.2
        config["service_start_timeout_seconds"] = 1
        proxy = service_client.ServiceDetector(config)
        with mock.patch.object(service_client, "_start_service"), \
                mock.patch.object(service_client.time, "sleep"), \
                mock.patch.object(service_client, "_exchange", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(secret_detector.DetectorError, "did not become available"):
                proxy.predict(["synthetic test input"])

    def test_cold_start_readiness_and_warm_request_are_separated(self):
        proxy = service_client.ServiceDetector(self.config)
        prediction_response = {
            "status": "ok", "predictions": [1], "model_load_seconds": 3.2,
            "inference_seconds": 0.1, "service_pid": 12345,
        }
        with mock.patch.object(service_client, "_prepare_private_directory"), \
                mock.patch.object(service_client, "_start_service"), \
                mock.patch.object(service_client, "_exchange", side_effect=[
                    FileNotFoundError(), {"status": "ready"}, prediction_response,
                    {**prediction_response, "model_load_seconds": 0.0},
                ]):
            self.assertEqual(proxy.predict(["synthetic input"]), [1])
            self.assertTrue(proxy.service_started)
            self.assertEqual(proxy.model_load_seconds, 3.2)
            self.assertEqual(proxy.predict(["synthetic input"]), [1])
            self.assertFalse(proxy.service_started)
            self.assertEqual(proxy.model_load_seconds, 0.0)

    def test_configured_service_idle_timeout_defaults_to_ten_minutes(self):
        self.assertEqual(self.config["service_idle_seconds"], 600)

    def test_service_idle_shutdown_releases_model_and_removes_socket(self):
        class DisposableDetector:
            inference_seconds = 0.0
            tokenization_seconds = 0.0
            encoder_seconds = 0.0
            classifier_seconds = 0.0
            def __init__(self, released):
                self.released = released
            def predict(self, examples):
                return [0] * len(examples)
            def __del__(self):
                self.released.set()

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            root.chmod(0o700)
            capability_probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                capability_probe.bind(str(root / "capability.sock"))
            except OSError as error:
                self.skipTest(f"Unix socket binding unavailable in this environment: {error}")
            finally:
                capability_probe.close()
                (root / "capability.sock").unlink(missing_ok=True)
            sock_path = root / "service.sock"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"max_examples": 5}), encoding="utf-8")
            released = threading.Event()
            result = []
            # Signal handlers can only be registered on Python's main thread;
            # this lifecycle test runs the real socket server in a worker thread.
            with mock.patch.object(inference_service, "load_detector",
                                   side_effect=lambda _config: DisposableDetector(released)), \
                    mock.patch.object(inference_service.signal, "signal"):
                server_thread = threading.Thread(
                    target=lambda: result.append(inference_service.serve(sock_path, config_path, 1.0)),
                    daemon=True,
                )
                server_thread.start()
                deadline = time.monotonic() + 2
                while not sock_path.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(sock_path.exists(), "service did not bind its socket")
                response = service_client._exchange(sock_path, {
                    "op": "predict", "examples": ["synthetic safe example"]
                }, 3)
                self.assertEqual(response["predictions"], [0])
                server_thread.join(timeout=3)
                self.assertFalse(server_thread.is_alive(), "idle service did not exit")
                self.assertEqual(result, [0])
                self.assertTrue(released.wait(timeout=1), "service retained its model after shutdown")
                self.assertFalse(sock_path.exists(), "service left a stale socket")

    def test_concurrent_service_startup_launches_one_process(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            root.chmod(0o700)
            sock_path = root / "service.sock"
            config_path = root / "config.json"
            launches = []
            launched = threading.Event()

            def launch(*_args, **_kwargs):
                launches.append(True)
                time.sleep(0.05)
                launched.set()
                return mock.Mock()

            def probe(_path, _timeout=0.2):
                return launched.is_set()

            errors = []
            with mock.patch.object(service_client.subprocess, "Popen", side_effect=launch), \
                    mock.patch.object(service_client, "_probe_ready", side_effect=probe):
                callers = [threading.Thread(target=lambda: self._start_service_capturing(
                    sock_path, config_path, errors)) for _ in range(2)]
                for caller in callers:
                    caller.start()
                for caller in callers:
                    caller.join(timeout=3)
                self.assertTrue(all(not caller.is_alive() for caller in callers))
            self.assertEqual(errors, [])
            self.assertEqual(len(launches), 1)

    @staticmethod
    def _start_service_capturing(path, config_path, errors):
        try:
            service_client._start_service(path, config_path, start_timeout=2)
        except Exception as error:
            errors.append(error)


    def test_native_hook_blocks_positive_commit_without_content_output(self):
        self.install_test_hook([1])
        canary = "NATIVE_HOOK_CANARY_NEVER_PRINT"
        (self.repo / "candidate.py").write_text(f"token = '{canary}'\n", encoding="utf-8")
        self.git("add", "--", "candidate.py")
        result = subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "blocked test"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(canary.encode(), combined)
        self.assertIn(b"Potential secret: candidate.py:1", combined)
        self.assertNotEqual(subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           check=False).returncode, 0)


if __name__ == "__main__":
    unittest.main()
