from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "git_agent.py"


def run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stderr)
    return result


def run_script(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python", str(SCRIPT), "--repo", str(repo), *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


class GitAgentScriptTest(unittest.TestCase):
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

    def test_status_reports_dirty_worktree(self) -> None:
        (self.repo / "train.py").write_text("print('dirty')\n", encoding="utf-8")

        result = run_script(self.repo, "status")

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["dirty"])
        self.assertEqual(payload["branch"], run_git(self.repo, "branch", "--show-current").stdout.strip())
        changed_paths = {row["path"] for row in payload["changes"]}
        self.assertIn("train.py", changed_paths)

    def test_ensure_branch_creates_prefixed_branch(self) -> None:
        result = run_script(self.repo, "ensure-branch", "--tag", "cpu-baseline")

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["action"], "created")
        self.assertEqual(payload["branch"], "autoresearch/cpu-baseline")
        self.assertEqual(run_git(self.repo, "branch", "--show-current").stdout.strip(), "autoresearch/cpu-baseline")

    def test_commit_stages_requested_paths_only(self) -> None:
        target = self.repo / "train.py"
        target.write_text("print('candidate')\n", encoding="utf-8")

        result = run_script(self.repo, "commit", "--message", "experiment: candidate", "--paths", "train.py")

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["staged_paths"], ["train.py"])
        self.assertEqual(payload["message"], "experiment: candidate")
        log_message = run_git(self.repo, "log", "-1", "--pretty=%s").stdout.strip()
        self.assertEqual(log_message, "experiment: candidate")

    def test_revert_creates_revert_commit(self) -> None:
        target = self.repo / "train.py"
        target.write_text("print('candidate')\n", encoding="utf-8")
        commit_result = run_script(self.repo, "commit", "--message", "experiment: candidate", "--paths", "train.py")
        commit_payload = json.loads(commit_result.stdout)

        revert_result = run_script(self.repo, "revert", "--commit", commit_payload["commit"])

        self.assertEqual(revert_result.returncode, 0, msg=revert_result.stderr)
        revert_payload = json.loads(revert_result.stdout)
        self.assertEqual(revert_payload["reverted_commit"], commit_payload["commit"])
        self.assertFalse(target.exists())
        self.assertIn("Revert \"experiment: candidate\"", run_git(self.repo, "log", "-1", "--pretty=%s").stdout.strip())

    def test_publish_fails_without_remote(self) -> None:
        result = run_script(self.repo, "publish")

        self.assertNotEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertIn("remote 'origin' is not configured", payload["error"])