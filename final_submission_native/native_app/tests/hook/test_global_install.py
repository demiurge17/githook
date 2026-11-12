from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HOOK_DIR = Path(__file__).resolve().parents[2] / "hook"


class GlobalInstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.home = self.root / "home"
        self.home.mkdir(mode=0o700)
        self.config_home = self.home / ".config"
        self.config_home.mkdir(mode=0o700)
        self.app = self.root / "app"
        self.hook_dir = self.app / "hook"
        self.hook_dir.mkdir(parents=True)
        (self.hook_dir / "pre_commit.py").write_text("# test stub\n", encoding="utf-8")
        shutil.copy2(HOOK_DIR / "install_global.sh", self.hook_dir)
        shutil.copy2(HOOK_DIR / "uninstall_global.sh", self.hook_dir)
        self.fake_python = self.root / "fake-python"
        self.scan_log = self.root / "scan.log"
        self.fake_python.write_text(
            "#!/bin/sh\n"
            "if [ \"${1:-}\" = -c ]; then exec python3 \"$@\"; fi\n"
            "if [ \"${2:-}\" = --verify-model ]; then exit 0; fi\n"
            "printf '%s\\n' scanned >> \"$TEST_SCAN_LOG\"\n"
            "[ \"${TEST_SCAN_BLOCK:-0}\" = 0 ]\n",
            encoding="utf-8",
        )
        self.fake_python.chmod(0o755)
        self.repo = self.root / "repo"
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        self.env = {
            **os.environ,
            "HOME": str(self.home),
            "XDG_CONFIG_HOME": str(self.config_home),
            "GIT_CONFIG_GLOBAL": str(self.home / ".gitconfig"),
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "CODEBERT_PYTHON": str(self.fake_python),
            "TEST_SCAN_LOG": str(self.scan_log),
        }
        subprocess.run(["git", "-C", str(self.repo), "config", "user.name", "Hook Test"],
                       env=self.env, check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.email", "hook@example.invalid"],
                       env=self.env, check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "core.hooksPath=/dev/null",
                        "commit", "--allow-empty", "-qm", "baseline"], env=self.env, check=True)

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args, check=False):
        return subprocess.run(["git", "-C", str(self.repo), *args], env=self.env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, check=check)

    def test_global_hook_install_chain_decisions_and_uninstall(self):
        installer = subprocess.run(["sh", str(self.hook_dir / "install_global.sh")], env=self.env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(installer.returncode, 0, installer.stderr)
        managed_hooks = self.config_home / "git" / "codebert-secret-hook" / "hooks"
        configured = subprocess.check_output(["git", "config", "--global", "--get", "core.hooksPath"],
                                             env=self.env, text=True).strip()
        self.assertEqual(configured, str(managed_hooks))

        local_hook = self.repo / ".git" / "hooks" / "pre-commit"
        local_log = self.root / "local.log"
        local_hook.write_text("#!/bin/sh\nprintf '%s\\n' local >> \"$LOCAL_HOOK_LOG\"\nexit 0\n",
                              encoding="utf-8")
        local_hook.chmod(0o755)
        self.env["LOCAL_HOOK_LOG"] = str(local_log)

        source = self.repo / "safe.py"
        source.write_text("safe = True\n", encoding="utf-8")
        self.git("add", "safe.py", check=True)
        allowed = self.git("commit", "-qm", "allowed")
        self.assertEqual(allowed.returncode, 0, allowed.stderr)
        self.assertEqual(self.scan_log.read_text(encoding="utf-8").splitlines(), ["scanned"])
        self.assertEqual(local_log.read_text(encoding="utf-8").splitlines(), ["local"])

        source.write_text("blocked = True\n", encoding="utf-8")
        self.git("add", "safe.py", check=True)
        self.env["TEST_SCAN_BLOCK"] = "1"
        blocked = self.git("commit", "-qm", "blocked")
        self.assertNotEqual(blocked.returncode, 0,
                            f"stdout={blocked.stdout!r}, stderr={blocked.stderr!r}, env={self.env.get('TEST_SCAN_BLOCK')!r}, "
                            f"scan_log={self.scan_log.read_text(encoding='utf-8')!r}, "
                            f"status={self.git('status', '--short').stdout!r}")
        self.assertEqual(len(self.scan_log.read_text(encoding="utf-8").splitlines()), 2)
        self.assertEqual(len(local_log.read_text(encoding="utf-8").splitlines()), 2)

        uninstall = subprocess.run(["sh", str(self.hook_dir / "uninstall_global.sh")], env=self.env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(uninstall.returncode, 0, uninstall.stderr)
        self.assertFalse((managed_hooks / "pre-commit").exists())
        self.assertNotEqual(subprocess.run(["git", "config", "--global", "--get", "core.hooksPath"],
                                           env=self.env, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE).returncode, 0)

    def test_existing_global_hooks_path_is_refused_unchanged(self):
        existing = self.root / "existing-hooks"
        existing.mkdir()
        subprocess.run(["git", "config", "--global", "core.hooksPath", str(existing)],
                       env=self.env, check=True)
        installer = subprocess.run(["sh", str(self.hook_dir / "install_global.sh")], env=self.env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertNotEqual(installer.returncode, 0)
        configured = subprocess.check_output(["git", "config", "--global", "--get", "core.hooksPath"],
                                             env=self.env, text=True).strip()
        self.assertEqual(configured, str(existing))
        self.assertFalse((self.config_home / "git" / "codebert-secret-hook" / "hooks" /
                          "pre-commit").exists())


if __name__ == "__main__":
    unittest.main()
