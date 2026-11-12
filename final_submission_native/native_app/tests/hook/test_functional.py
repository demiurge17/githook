"""FT-01–FT-15. Reuse repository fixtures from the established unittest suite.

Use functional-testing/run_tests.py to preserve per-case evidence and reports.
Ordinary unittest discovery also works; real inference requires --real in the runner
or CODEBERT_FUNCTIONAL_REAL=1. No classifier is fitted or artifact modified.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from test_pre_commit import PreCommitTests, FakeDetector, HOOK_DIR, PROJECT_ROOT
import pre_commit
import secret_detector
import service_client


SCENARIOS = [
    ("New staged file", "Eligible additions, paths and staged line numbers reach the detector.", "Python file with two nonblank additions and a blank line."),
    ("Modified tracked file", "Only added lines reach the detector; unchanged context is excluded.", "One replacement between two unchanged lines."),
    ("Partial staging", "Index and worktree differ; only staged additions and their index line numbers determine the commit decision.", "Two separated replacements; hash-object/update-index stages only the first; both decision polarities checked."),
    ("Unstaged modifications", "No unstaged candidate is classified and inference is not started.", "Tracked file modified without git add."),
    ("Multiple staged files", "All eligible additions have correct paths and staged line numbers.", "Two Python files in different directories."),
    ("Binary file", "Eligible binary content is skipped without a scan failure or inference call.", "A .py file containing NUL and invalid UTF-8 bytes."),
    ("Negative predictions", "All-zero predictions permit an actual Git commit.", "Two staged lines; controlled labels [0, 0]."),
    ("Positive prediction", "A positive prediction blocks Git; HEAD is unchanged.", "One staged line; controlled label [1]."),
    ("Mixed predictions", "At least one positive prediction blocks Git; HEAD is unchanged.", "Two staged lines; controlled labels [0, 1]."),
    ("Invalid inference response", "Missing/extra/invalid labels and malformed wire responses block Git; HEAD is unchanged.", "Missing, extra, bool, float, string, null, out-of-range labels; invalid JSON/non-object/truncated frames."),
    ("Inference service unavailable", "An unavailable service blocks Git and reports an incomplete scan.", "Controlled missing-socket connection; service launch deliberately fails."),
    ("Model loading failure", "Simulated CodeBERT and Random Forest load errors block Git without disclosing exception content.", "Controlled loader errors through actual service request handling and client validation."),
    ("Safe warning messages", "Warning shows staged file and line, without canary or unnecessary source content.", "Synthetic canary on staged line 3; controlled positive prediction."),
    ("Harmless synthetic code", "Actual offline CodeBERT + Random Forest inference returns a binary label, recorded without assuming 0.", "total = sum([1, 2, 3])"),
    ("Synthetic credential-like code", "Actual offline CodeBERT + Random Forest inference returns a binary label, recorded without assuming 1.", "api_key = 'FAKE_FUNCTIONAL_TEST_TOKEN_NOT_A_REAL_CREDENTIAL_12345'"),
]


class FunctionalTests(PreCommitTests):
    def git(self, *args):
        print("$", shlex.join(["git", "-C", str(self.repo), *args]))
        result = super().git(*args)
        print("returncode:", result.returncode)
        for output in (result.stdout, result.stderr):
            if output:
                print(output.decode("utf-8", "replace"), end="")
        return result

    def candidates(self, expected, labels):
        detector = FakeDetector(labels)
        code, stats, examples = self.scan(detector)
        observed = [(e.path, e.line, e.code) for e in examples]
        print("extracted_candidates:", json.dumps(observed))
        print("detector_received:", json.dumps(detector.received))
        print("scan_exit_status:", code)
        self.assertEqual(observed, expected)
        self.assertEqual(detector.received, [item[2] for item in expected])
        return code, stats

    def prepare_commit(self, text="first = 1\nsecond = 2\n"):
        self.commit_file("baseline.md", "synthetic baseline\n")
        (self.repo / "candidate.py").write_text(text, encoding="utf-8")
        self.git("add", "--", "candidate.py")

    def hook_commit(self, labels=None, setup="", assert_block=True):
        # This wrapper replaces only the detector boundary. Git still invokes
        # the application's actual main(), extraction and reporting code.
        hook = self.repo / ".git/hooks/pre-commit"
        private_runtime = self.repo / ".git/functional-service"
        private_runtime.mkdir(mode=0o700, exist_ok=True)
        hook.write_text(
            f"#!{sys.executable}\nimport sys, json\n"
            f"sys.path.insert(0, {str(HOOK_DIR)!r})\n"
            "import pre_commit, service_client, inference_service\n"
            "from secret_detector import DetectorError\n"
            "from unittest import mock\n"
            "from pathlib import Path\n"
            "import os\n"
            "os.environ['CODEBERT_SECRET_HOOK_SOCKET'] = str(Path.cwd() / '.git/functional-service/missing.sock')\n"
            + setup + "\n"
            + ("class ControlledDetector:\n"
               "    inference_seconds = 0.0\n"
               "    def predict(self, examples):\n"
               "        Path('.git/detector-input.json').write_text(json.dumps(examples))\n"
               f"        return {labels!r}\n"
               "raise SystemExit(pre_commit.main(detector_factory=lambda c: ControlledDetector()))\n"
               if labels is not None else "raise SystemExit(pre_commit.main())\n"),
            encoding="utf-8",
        )
        hook.chmod(0o755)
        before = self.git("rev-parse", "HEAD").stdout
        command = ["git", "-C", str(self.repo), "commit", "-qm", "functional fixture"]
        print("$", shlex.join(command))
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        print("returncode:", result.returncode)
        print(result.stdout + result.stderr)
        after = self.git("rev-parse", "HEAD").stdout
        if assert_block:
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(before, after, "Blocked attempt created a commit")
        else:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotEqual(before, after)
        return result

    def test_ft_01_new_staged_file(self):
        (self.repo / "new.py").write_text("first = 1\n\nsecond = 2\n")
        self.git("add", "new.py")
        self.candidates([("new.py", 1, "first = 1"), ("new.py", 3, "second = 2")], [0, 0])

    def test_ft_02_modified_tracked_file(self):
        path = self.commit_file("tracked.py", "first = 1\nold = 2\nlast = 3\n")
        path.write_text("first = 1\nreplacement = 2\nlast = 3\n")
        self.git("add", "tracked.py")
        self.candidates([("tracked.py", 2, "replacement = 2")], [0])

    def test_ft_03_partial_staging(self):
        original = [f"context_{i} = {i}\n" for i in range(1, 31)]
        path = self.commit_file("partial.py", "".join(original))
        for staged_positive in (False, True):
            with self.subTest(staged_positive=staged_positive):
                # Restore the temporary index baseline for the second polarity.
                self.git("read-tree", "HEAD")
                indexed = original.copy()
                indexed[2] = "staged_value = '" + ("POSITIVE" if staged_positive else "NEGATIVE") + "'\n"
                worktree = indexed.copy()
                worktree[25] = "unstaged_value = '" + ("NEGATIVE" if staged_positive else "POSITIVE") + "'\n"
                path.write_text("".join(worktree))
                staged_blob = self.repo / ".git/staged-fixture"
                staged_blob.write_text("".join(indexed))
                oid = self.git("hash-object", "-w", str(staged_blob)).stdout.decode().strip()
                self.git("update-index", "--cacheinfo", "100644", oid, "partial.py")
                self.assertEqual(self.git("show", ":partial.py").stdout.decode(), "".join(indexed))
                self.assertEqual(path.read_text(), "".join(worktree))
                cached = self.git("diff", "--cached").stdout.decode()
                unstaged = self.git("diff").stdout.decode()
                self.assertIn("+" + indexed[2].strip(), cached)
                self.assertNotIn("+unstaged_value", cached)
                self.assertIn("+" + worktree[25].strip(), unstaged)
                self.assertNotIn("+staged_value", unstaged)
                detector = FakeDetector()
                detector.predict = lambda examples: (detector.received.extend(examples) or
                                                     [int("POSITIVE" in e) for e in examples])
                decision, _, examples = self.scan(detector)
                self.assertEqual([(e.path, e.line, e.code) for e in examples],
                                 [("partial.py", 3, indexed[2].strip())])
                self.assertEqual(detector.received, [indexed[2].strip()])
                self.assertEqual(decision, int(staged_positive))
                print("detector_received:", json.dumps(detector.received))
                print("scan_exit_status:", decision)
                # Test actual Git decisions while preserving the index for both polarities.
                result = self.hook_commit([int(staged_positive)], assert_block=staged_positive)
                self.assertIn("COMMIT BLOCKED" if staged_positive else "COMMIT ALLOWED",
                              result.stdout + result.stderr)
                received = json.loads((self.repo / ".git/detector-input.json").read_text())
                self.assertEqual(received, [indexed[2].strip()])
                if not staged_positive:
                    # HEAD now contains the negative staged change; restore the
                    # original committed fixture in this temporary repo only.
                    self.git("update-ref", "HEAD", "HEAD~1")

    def test_ft_04_unstaged_modifications(self):
        path = self.commit_file("tracked.py", "original = 1\n")
        path.write_text("unstaged = 'SYNTHETIC_ONLY'\n")
        self.git("diff")
        factory = mock.Mock(side_effect=AssertionError("Inference must not start"))
        code, _, examples = pre_commit.scan(self.repo, self.config, factory)
        self.assertEqual(code, 0)
        self.assertEqual(examples, [])
        factory.assert_not_called()
        print("No candidates; detector factory not called; scan exit status 0.")

    def test_ft_05_multiple_staged_files(self):
        (self.repo / "src").mkdir()
        (self.repo / "a.py").write_text("a = 1\n\nb = 2\n")
        (self.repo / "src/b.py").write_text("\nc = 3\n")
        self.git("add", "a.py", "src/b.py")
        _, stats = self.candidates([("a.py", 1, "a = 1"), ("a.py", 3, "b = 2"),
                                    ("src/b.py", 2, "c = 3")], [0, 0, 0])
        self.assertEqual(stats["files_scanned"], 2)

    def test_ft_06_binary_file(self):
        (self.repo / "binary.py").write_bytes(b"synthetic\x00\xff\x01")
        self.git("add", "binary.py")
        factory = mock.Mock(side_effect=AssertionError("Binary must not reach inference"))
        code, stats, examples = pre_commit.scan(self.repo, self.config, factory)
        self.assertEqual((code, examples, stats["binary_files_skipped"]), (0, [], 1))
        factory.assert_not_called()
        print("Binary skipped: 1; no candidates; detector not started; scan exit status 0.")

    def test_ft_07_negative_predictions(self):
        self.prepare_commit()
        self.hook_commit([0, 0], assert_block=False)

    def test_ft_08_positive_prediction(self):
        self.prepare_commit("value = 1\n")
        self.hook_commit([1])

    def test_ft_09_mixed_predictions(self):
        self.prepare_commit()
        self.hook_commit([0, 1])

    def test_ft_10_invalid_inference_response(self):
        self.prepare_commit("value = 'SYNTHETIC_INVALID_RESPONSE'\n")
        responses = [{"status": "ok", "service_pid": 12345, "predictions": labels}
                     for labels in ([], [0, 1], [True], [0.5], ["1"], [None], [9], [-1], None)]
        responses += [{"status": "ok", "service_pid": 12345}, {"status": "error"}]
        for response in responses:
            with self.subTest(response=response):
                print("controlled_response:", json.dumps(response))
                marker = self.repo / ".git/response-exercised"
                marker.unlink(missing_ok=True)
                setup = f'''
def controlled_exchange(*args):
    Path('.git/response-exercised').write_text('response used')
    return {response!r}
service_client._exchange = controlled_exchange
'''
                result = self.hook_commit(setup=setup)
                self.assertIn("SCAN INCOMPLETE", result.stderr)
                self.assertTrue(marker.exists(), "Controlled service response was not exercised")
        # Exercise the actual client framing/JSON decoder with controlled wire bytes.
        # No _exchange replacement here; socket transport alone is substituted.
        for raw in (b"not-json", b"[]", b"null", b'{"status":'):
            with self.subTest(wire=raw):
                marker = self.repo / ".git/wire-exercised"
                marker.unlink(missing_ok=True)
                setup = f'''
class WireSocket:
    def __init__(self, *args):
        payload = {raw!r}
        self.data = len(payload).to_bytes(4, 'big') + payload
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def settimeout(self, *args): pass
    def connect(self, *args): Path('.git/wire-exercised').write_text('wire used')
    def sendall(self, *args): pass
    def recv(self, n):
        part, self.data = self.data[:n], self.data[n:]
        return part
service_client.socket.socket = WireSocket
'''
                print("controlled_wire_payload:", repr(raw))
                result = self.hook_commit(setup=setup)
                self.assertIn("SCAN INCOMPLETE", result.stderr)
                self.assertTrue(marker.exists(), "Malformed wire decoder was not exercised")
        setup = '''
class TruncatedSocket:
    def __init__(self, *args): self.data = (100).to_bytes(4, 'big') + b'{}'
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def settimeout(self, *args): pass
    def connect(self, *args): Path('.git/truncated-exercised').write_text('truncated frame used')
    def sendall(self, *args): pass
    def recv(self, n):
        part, self.data = self.data[:n], self.data[n:]
        return part
service_client.socket.socket = TruncatedSocket
'''
        self.assertIn("SCAN INCOMPLETE", self.hook_commit(setup=setup).stderr)
        self.assertTrue((self.repo / ".git/truncated-exercised").exists())

    def test_ft_11_service_unavailable(self):
        self.prepare_commit("value = 1\n")
        setup = '''
import os
def fail_connect(*args, **kwargs):
    Path('.git/unavailable-connection-exercised').write_text('controlled missing-socket connection')
    raise FileNotFoundError('Controlled absent inference socket')
service_client._exchange = fail_connect
def fail_start(*args, **kwargs):
    Path('.git/unavailable-exercised').write_text('startup attempted after missing socket')
    raise DetectorError('Controlled service launch unavailable')
service_client._start_service = fail_start
'''
        result = self.hook_commit(setup=setup)
        self.assertIn("SCAN INCOMPLETE", result.stderr)
        self.assertIn("commit blocked", result.stderr)
        self.assertTrue((self.repo / ".git/unavailable-connection-exercised").exists())
        self.assertTrue((self.repo / ".git/unavailable-exercised").exists())
        print("Missing-socket connection and failed service launch exercised; commit blocked.")

    def test_ft_12_model_loading_failure(self):
        self.prepare_commit("value = 1\n")
        for component in ("CodeBERT", "Random Forest"):
            with self.subTest(component=component):
                marker = self.repo / ".git/loader-exercised"
                marker.unlink(missing_ok=True)
                setup = f'''
inference_service.load_detector = mock.Mock(side_effect=DetectorError({component!r} + ' synthetic load failure'))
class MemoryConnection:
    def __init__(self, request):
        payload = json.dumps(request).encode()
        self.incoming = len(payload).to_bytes(4, 'big') + payload
        self.outgoing = b''
    def recv(self, n):
        result, self.incoming = self.incoming[:n], self.incoming[n:]
        return result
    def sendall(self, data): self.outgoing += data
def exchange(path, request, *args):
    conn = MemoryConnection(request)
    inference_service.request(conn, None, {{'max_examples': 500}})
    inference_service.load_detector.assert_called_once()
    response = json.loads(conn.outgoing[4:])
    assert response == {{'status': 'error'}}
    Path('.git/loader-exercised').write_text('loader error handled by service')
    return response
service_client._exchange = exchange
'''
                result = self.hook_commit(setup=setup)
                self.assertIn("SCAN INCOMPLETE", result.stderr)
                self.assertNotIn("synthetic load failure", result.stdout + result.stderr)
                self.assertTrue(marker.exists(), "Loader failure did not reach service handler")
                print("Simulated loader failure:", component, "; service emitted error; Git blocked.")

    def test_ft_13_safe_warning_messages(self):
        canary = "FT13_CANARY_SYNTHETIC_NEVER_REAL_7392"
        source = f"token = '{canary}'"
        self.prepare_commit("safe = True\n\n" + source + "\n")
        result = self.hook_commit([0, 1])
        combined = result.stdout + result.stderr
        self.assertIn("Potential secret: candidate.py:3", combined)
        for forbidden in (canary, source, "safe = True"):
            self.assertNotIn(forbidden, combined)
        print("Warning location verified; synthetic canary and source absent from stdout/stderr.")


class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CODEBERT_FUNCTIONAL_REAL") != "1":
            raise unittest.SkipTest("Real-model tests require --real or CODEBERT_FUNCTIONAL_REAL=1")
        missing = [name for name in ("torch", "transformers", "numpy", "joblib", "sklearn")
                   if importlib.util.find_spec(name) is None]
        if missing:
            raise unittest.SkipTest("Missing real-model dependencies: " + ", ".join(missing))
        cls.temp = tempfile.TemporaryDirectory(prefix="cb-ft-real-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.root.chmod(0o700)
        cls.path = cls.root / "s.sock"
        cls.config = secret_detector.read_config()
        cls.config["service_idle_seconds"] = 15
        cls.config_path = cls.root / "config.json"
        cls.config_path.write_text(json.dumps(cls.config))
        cls.env = os.environ.copy()
        cls.env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")
        # Reuse the already prepared pinned snapshot without copying or modifying it.
        cache = PROJECT_ROOT / "model_assets/huggingface/hub"
        if cache.is_dir():
            cls.env["HF_HUB_CACHE"] = str(cache)
        cls.service_log = (cls.root / "service.log").open("w+")
        cls.addClassCleanup(cls.service_log.close)
        command = [sys.executable, str(HOOK_DIR / "inference_service.py"),
                   "--socket", str(cls.path), "--config", str(cls.config_path)]
        print("$", shlex.join(command))
        cls.server = subprocess.Popen(command, env=cls.env, stdout=cls.service_log,
                                      stderr=subprocess.STDOUT)
        cls.addClassCleanup(cls.stop_service)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if service_client._probe_ready(cls.path):
                return
            if cls.server.poll() is not None:
                break
            time.sleep(0.05)
        raise RuntimeError("Dedicated inference service did not become ready")

    @classmethod
    def stop_service(cls):
        cls.server.terminate()
        try:
            cls.server.wait(timeout=7)
        except subprocess.TimeoutExpired:
            cls.server.kill()
            cls.server.wait(timeout=5)
        print("Dedicated service final returncode:", cls.server.returncode)
        cls.service_log.flush()
        print("Dedicated service log:", (cls.root / "service.log").read_text())
        if cls.path.exists():
            raise AssertionError("Service left its socket behind")

    def infer(self, snippet):
        print("Synthetic input:", snippet)
        proxy = service_client.ServiceDetector(self.config)
        proxy.path = self.path
        # Readiness was checked above. Real transport, service, tokenizer,
        # encoder, pooling and saved classifier run with no detector doubles.
        predictions = proxy.predict([snippet])
        self.assertEqual(len(predictions), 1)
        self.assertIs(type(predictions[0]), int)
        self.assertIn(predictions[0], (0, 1))
        print("actual_binary_prediction:", predictions[0])
        print("service_pid:", proxy.service_pid)
        print("model_revision:", self.config["model_revision"])
        for metric in ("model_load_seconds", "tokenization_seconds", "encoder_seconds",
                       "classifier_seconds", "inference_seconds", "service_peak_rss_mib"):
            print(metric + ":", getattr(proxy, metric))

    def test_ft_14_harmless_real_model(self):
        self.infer("total = sum([1, 2, 3])")

    def test_ft_15_credential_like_real_model(self):
        self.infer("api_key = 'FAKE_FUNCTIONAL_TEST_TOKEN_NOT_A_REAL_CREDENTIAL_12345'")


def load_tests(loader, tests, pattern):
    # Inherited tests remain in the original suite; do not run/count them twice.
    suite = unittest.TestSuite()
    for cls in (FunctionalTests, RealModelTests):
        for name in loader.getTestCaseNames(cls):
            if name.startswith("test_ft_"):
                suite.addTest(cls(name))
    return suite
