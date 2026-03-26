from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "loop_status.py"
REPORT_SCRIPT = REPO_ROOT / "scripts" / "loop_report.py"
LAUNCHER_SCRIPT = REPO_ROOT / "scripts" / "supervised_local_run.py"
DOCTOR_SCRIPT = REPO_ROOT / "scripts" / "roadmap_doctor.py"


def run_script(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python", str(SCRIPT), "--repo-root", str(repo), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class LoopStatusScriptTests(unittest.TestCase):
    def test_loop_status_initializes_control_and_reports_empty_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            result = run_script(repo)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["history_length"], 0)
            self.assertEqual(payload["control"]["desired_state"], "running")

    def test_loop_status_can_pause_and_override_objective(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            result = run_script(
                repo,
                "--set-state",
                "paused",
                "--objective-override",
                "Focus on reliability regression recovery.",
                "--note",
                "pause here for review",
                "--stop-after-iteration",
                "--updated-by",
                "tester",
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["control"]["desired_state"], "paused")
            self.assertTrue(payload["control"]["stop_after_iteration"])
            self.assertIn("reliability regression recovery", payload["control"]["objective_override"].lower())

    def test_loop_status_supports_text_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            result = run_script(repo, "--format", "text")
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("history_length:", result.stdout)
            self.assertIn("desired_state:", result.stdout)

    def test_loop_report_writes_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            state_path = repo / ".ternary_lab" / "state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "created_at": 100,
                        "last_activity_at": 125,
                        "history": [
                            {
                                "run_id": "run-1",
                                "metrics": {"status": "keep", "task_eval_score": 0.75, "val_bpb": 1.0, "reliability_score": 0.9, "capability_score": 0.7, "tokens_per_second": 5.5},
                                "started_at": 110,
                                "finished_at": 125,
                                "iteration_duration_seconds": 15.0,
                            }
                        ],
                        "best": {
                            "run_id": "run-1",
                            "metrics": {"status": "keep", "task_eval_score": 0.75, "val_bpb": 1.0, "reliability_score": 0.9, "capability_score": 0.7, "tokens_per_second": 5.5},
                        },
                        "objective": "demo",
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                ["python", str(REPORT_SCRIPT), "--repo-root", str(repo)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            report_path = Path(result.stdout.strip())
            self.assertTrue(report_path.exists())
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("Ternary Lab Loop Report", report_text)
            self.assertIn("average_iteration_seconds", report_text)
            self.assertIn("Recent Activity", report_text)
            self.assertIn("1970-01-01 00:01:40 UTC", report_text)
            self.assertIn("1970-01-01 00:01:50 UTC", report_text)
            self.assertIn("1970-01-01 00:02:05 UTC", report_text)

    def test_supervised_local_run_writes_report_after_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=repo, capture_output=True, text=True, check=False)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, capture_output=True, text=True, check=False)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, capture_output=True, text=True, check=False)
            (repo / "README.md").write_text("seed\n", encoding="utf-8")
            (repo / "environment.yml").write_text("name: demo\n", encoding="utf-8")
            (repo / "scripts").mkdir(parents=True, exist_ok=True)
            (repo / "scripts" / "bootstrap_bitnet_research.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=repo, capture_output=True, text=True, check=False)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, capture_output=True, text=True, check=False)

            for file_name in ["ternary_lab.py", "deploy_cpu_agent.py", "scripts/git_agent.py", "scripts/loop_status.py", "scripts/loop_report.py", "scripts/roadmap_doctor.py"]:
                target = repo / file_name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text((REPO_ROOT / file_name).read_text(encoding="utf-8"), encoding="utf-8")

            agent_dir = repo / ".ternary_lab" / "agents" / "cpu-agent-1"
            agent_dir.mkdir(parents=True, exist_ok=True)
            (agent_dir / "agent.json").write_text("{}\n", encoding="utf-8")
            (agent_dir / "deployment_report.json").write_text("{}\n", encoding="utf-8")

            result = subprocess.run(
                [
                    "python",
                    str(LAUNCHER_SCRIPT),
                    "--repo-root",
                    str(repo),
                    "--branch-tag",
                    "demo-run",
                    "--trainer-backend",
                    "mock",
                    "--max-iterations",
                    "1",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["iterations_completed"], 1)
            self.assertTrue(payload["report_paths"])
            self.assertTrue(payload["snapshot_paths"])
            self.assertEqual(payload["milestone_gate"], "local-agent-lifecycle")
            report_path = Path(payload["report_paths"][0])
            self.assertTrue(report_path.exists())
            snapshot_path = Path(payload["snapshot_paths"][0])
            self.assertTrue(snapshot_path.exists())

    def test_supervised_local_run_can_commit_snapshot_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=repo, capture_output=True, text=True, check=False)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, capture_output=True, text=True, check=False)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, capture_output=True, text=True, check=False)
            (repo / "README.md").write_text("seed\n", encoding="utf-8")
            (repo / "environment.yml").write_text("name: demo\n", encoding="utf-8")
            (repo / "scripts").mkdir(parents=True, exist_ok=True)
            (repo / "scripts" / "bootstrap_bitnet_research.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=repo, capture_output=True, text=True, check=False)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, capture_output=True, text=True, check=False)

            for file_name in ["ternary_lab.py", "deploy_cpu_agent.py", "scripts/git_agent.py", "scripts/loop_status.py", "scripts/loop_report.py", "scripts/roadmap_doctor.py"]:
                target = repo / file_name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text((REPO_ROOT / file_name).read_text(encoding="utf-8"), encoding="utf-8")

            agent_dir = repo / ".ternary_lab" / "agents" / "cpu-agent-1"
            agent_dir.mkdir(parents=True, exist_ok=True)
            (agent_dir / "agent.json").write_text("{}\n", encoding="utf-8")
            (agent_dir / "deployment_report.json").write_text("{}\n", encoding="utf-8")

            result = subprocess.run(
                [
                    "python",
                    str(LAUNCHER_SCRIPT),
                    "--repo-root",
                    str(repo),
                    "--branch-tag",
                    "audit-run",
                    "--trainer-backend",
                    "mock",
                    "--max-iterations",
                    "1",
                    "--commit-snapshots",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(len(payload["snapshot_commits"]), 1)
            log_message = subprocess.run(["git", "log", "-1", "--pretty=%s"], cwd=repo, capture_output=True, text=True, check=False).stdout.strip()
            self.assertIn("audit: supervised snapshot 1 (audit-run)", log_message)

    def test_roadmap_doctor_reports_current_milestone(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "environment.yml").write_text("name: demo\n", encoding="utf-8")
            (repo / "scripts").mkdir(parents=True, exist_ok=True)
            (repo / "scripts" / "bootstrap_bitnet_research.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (repo / "deploy_cpu_agent.py").write_text((REPO_ROOT / "deploy_cpu_agent.py").read_text(encoding="utf-8"), encoding="utf-8")
            result = subprocess.run(
                ["python", str(DOCTOR_SCRIPT), "--repo-root", str(repo)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["current_milestone"], "local-agent-lifecycle")

    def test_roadmap_doctor_enforces_required_milestone(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "environment.yml").write_text("name: demo\n", encoding="utf-8")
            (repo / "scripts").mkdir(parents=True, exist_ok=True)
            (repo / "scripts" / "bootstrap_bitnet_research.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (repo / "deploy_cpu_agent.py").write_text((REPO_ROOT / "deploy_cpu_agent.py").read_text(encoding="utf-8"), encoding="utf-8")
            result = subprocess.run(
                [
                    "python",
                    str(DOCTOR_SCRIPT),
                    "--repo-root",
                    str(repo),
                    "--require-complete-through",
                    "local-agent-lifecycle",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)