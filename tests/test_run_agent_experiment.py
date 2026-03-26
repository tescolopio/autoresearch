from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_agent_experiment.py"
HELPER = REPO_ROOT / "scripts" / "git_agent.py"


def run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise AssertionError(result.stderr)
    return result


def run_script(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python", str(SCRIPT), "--repo-root", str(repo), "--git-helper-script", str(HELPER), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class RunAgentExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name)
        run_git(self.repo, "init")
        run_git(self.repo, "config", "user.name", "Test User")
        run_git(self.repo, "config", "user.email", "test@example.com")
        (self.repo / "README.md").write_text("seed\n", encoding="utf-8")
        run_git(self.repo, "add", "README.md")
        run_git(self.repo, "commit", "-m", "initial commit")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_run_agent_experiment_creates_branch_and_log(self) -> None:
        result = run_script(
            self.repo,
            "--branch-tag",
            "smoke-run",
            "--log-path",
            ".ternary_lab/runs/smoke.log",
            "--",
            "python",
            "-c",
            "print('ok')",
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["branch"]["branch"], "autoresearch/smoke-run")
        self.assertTrue((self.repo / ".ternary_lab" / "runs" / "smoke.log").exists())
        self.assertEqual(run_git(self.repo, "branch", "--show-current").stdout.strip(), "autoresearch/smoke-run")

    def test_run_agent_experiment_reverts_failed_commit(self) -> None:
        target = self.repo / "train.py"
        target.write_text("print('candidate')\n", encoding="utf-8")

        result = run_script(
            self.repo,
            "--commit-message",
            "experiment: candidate",
            "--commit-paths",
            "train.py",
            "--revert-on-failure",
            "--log-path",
            ".ternary_lab/runs/fail.log",
            "--",
            "python",
            "-c",
            "import sys; sys.exit(3)",
        )

        self.assertEqual(result.returncode, 3, msg=result.stderr)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertIsNotNone(payload["commit"])
        self.assertIsNotNone(payload["reverted"])
        self.assertFalse(target.exists())
        self.assertIn("Revert \"experiment: candidate\"", run_git(self.repo, "log", "-1", "--pretty=%s").stdout.strip())