import json
import pathlib
import subprocess
import tempfile
import unittest

from ternary_lab import (
    ExperimentCandidate,
    LabConfig,
    build_knowledge_graph,
    candidate_policy_violations,
    compute_agent_task_scores,
    derive_machine_secret,
    execute_loop,
    execute_supervisor,
    is_better_candidate,
    load_control,
    save_control,
    propose_next_candidate,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GIT_HELPER = REPO_ROOT / "scripts" / "git_agent.py"


def run_git(repo, *args):
    completed = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    return completed


class TernaryLabTests(unittest.TestCase):
    def test_machine_secret_is_stable_for_same_fingerprint(self):
        fingerprint = "lab-host:12345"
        first = derive_machine_secret("seed", fingerprint=fingerprint)
        second = derive_machine_secret("seed", fingerprint=fingerprint)
        self.assertEqual(first, second)

    def test_policy_rejects_gpu_and_network_terms(self):
        candidate = ExperimentCandidate(
            name="bad-candidate",
            description="Try sending metrics over http://localhost after a GPU run.",
            device="cuda",
        )
        violations = candidate_policy_violations(candidate, "Use the internet for tuning", ["stay local"])
        self.assertTrue(any("device must remain cpu" in violation for violation in violations))
        self.assertTrue(any("forbidden term detected" in violation for violation in violations))

    def test_propose_next_candidate_recovers_capacity_after_regression(self):
        state = {
            "best": {
                "candidate": {
                    "name": "best",
                    "description": "best so far",
                    "depth": 4,
                    "device_batch_size": 8,
                    "total_batch_size": 16384,
                    "window_pattern": "L",
                    "bitlinear_scaling": "mean",
                    "bitlinear_threshold": 0.5,
                    "use_subln": True,
                    "device": "cpu",
                    "linear_impl": "bitlinear",
                    "avg_power_watts": 15.0,
                },
                "metrics": {"val_bpb": 1.0, "status": "keep"},
            },
            "history": [
                {
                    "candidate": {"name": "best", "description": "best so far", "depth": 4, "device_batch_size": 8, "total_batch_size": 16384, "window_pattern": "L", "bitlinear_scaling": "mean", "bitlinear_threshold": 0.5, "use_subln": True, "device": "cpu", "linear_impl": "bitlinear", "avg_power_watts": 15.0},
                    "metrics": {"val_bpb": 1.0, "status": "keep"},
                },
                {
                    "candidate": {"name": "worse", "description": "worse run", "depth": 4, "device_batch_size": 8, "total_batch_size": 16384, "window_pattern": "L", "bitlinear_scaling": "mean", "bitlinear_threshold": 0.5, "use_subln": True, "device": "cpu", "linear_impl": "bitlinear", "avg_power_watts": 15.0},
                    "metrics": {"val_bpb": 1.05, "status": "discard"},
                },
            ],
        }
        candidate = propose_next_candidate([], state)
        self.assertGreaterEqual(candidate.depth, 6)
        self.assertIn("recover capacity", candidate.description.lower())

    def test_build_knowledge_graph_keeps_salient_nodes(self):
        state = {
            "history": [
                {
                    "run_id": "run-1",
                    "candidate": {"name": "baseline", "description": "cpu baseline", "depth": 4, "device_batch_size": 8, "total_batch_size": 16384, "window_pattern": "L", "bitlinear_scaling": "mean", "bitlinear_threshold": 0.5, "use_subln": True, "device": "cpu", "linear_impl": "bitlinear", "avg_power_watts": 15.0},
                    "metrics": {"val_bpb": 1.1, "status": "keep", "device": "cpu", "linear_impl": "bitlinear", "energy_j_per_token": 0.1, "tokens_per_second": 5.0},
                    "state": "gamma",
                },
                {
                    "run_id": "run-2",
                    "candidate": {"name": "novel", "description": "median scaling threshold shift", "depth": 5, "device_batch_size": 8, "total_batch_size": 16384, "window_pattern": "L", "bitlinear_scaling": "median", "bitlinear_threshold": 0.45, "use_subln": True, "device": "cpu", "linear_impl": "bitlinear", "avg_power_watts": 15.0},
                    "metrics": {"val_bpb": 1.2, "status": "discard", "device": "cpu", "linear_impl": "bitlinear", "energy_j_per_token": 0.1, "tokens_per_second": 5.0},
                    "state": "theta",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "knowledge_graph.json"
            graph = build_knowledge_graph(state, output_path)
            saved = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(graph["nodes"], saved["nodes"])
        self.assertEqual(saved["nodes"][0]["label"], "baseline")
        self.assertTrue(any(node["label"] == "novel" for node in saved["nodes"]))

    def test_compute_agent_task_scores_rewards_stable_cpu_bitnet_runs(self):
        scores = compute_agent_task_scores(
            {
                "status": "keep",
                "signature_verified": True,
                "device": "cpu",
                "linear_impl": "bitlinear",
                "log_path": ".ternary_lab/runs/run.log",
                "val_bpb": 1.0,
                "tokens_per_second": 6.0,
                "energy_j_per_token": 0.03,
            }
        )
        self.assertGreater(scores["reliability_score"], 0.9)
        self.assertGreater(scores["task_eval_score"], 0.7)

    def test_is_better_candidate_prefers_task_eval_score_over_val_bpb(self):
        best = {
            "status": "keep",
            "task_eval_score": 0.61,
            "val_bpb": 0.95,
            "energy_j_per_token": 0.04,
        }
        challenger = {
            "status": "keep",
            "task_eval_score": 0.74,
            "val_bpb": 0.99,
            "energy_j_per_token": 0.05,
        }
        self.assertTrue(is_better_candidate(challenger, best))

    def test_execute_loop_uses_git_helper_on_clean_repo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = pathlib.Path(tmpdir)
            run_git(repo, "init")
            run_git(repo, "config", "user.name", "Test User")
            run_git(repo, "config", "user.email", "test@example.com")
            (repo / "README.md").write_text("seed\n", encoding="utf-8")
            run_git(repo, "add", "README.md")
            run_git(repo, "commit", "-m", "initial commit")
            state = execute_loop(
                LabConfig(
                    repo_root=str(repo),
                    trainer_backend="mock",
                    iterations=1,
                    git_helper_script=str(GIT_HELPER),
                    git_branch_tag="agent-reliability",
                )
            )
        self.assertTrue(state["git"]["enabled"])
        self.assertEqual(state["git"]["branch"]["branch"], "autoresearch/agent-reliability")
        self.assertIn("reliability and capability", state["objective"].lower())
        self.assertIn("task_eval_score", state["best"]["metrics"])

    def test_execute_loop_respects_paused_control_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = pathlib.Path(tmpdir)
            control_path = repo / ".ternary_lab" / "control.json"
            save_control(
                control_path,
                {
                    "desired_state": "paused",
                    "objective_override": "",
                    "human_note": "pause before next run",
                    "updated_at": 1,
                    "updated_by": "tester",
                    "stop_after_iteration": False,
                    "showcase_loop_enabled": False,
                    "showcase_note": "Frontier comparison is separate from main-loop decisions.",
                    "default_objective": "default",
                },
            )
            state = execute_loop(
                LabConfig(
                    repo_root=str(repo),
                    trainer_backend="mock",
                    iterations=2,
                    manage_git=False,
                )
            )
        self.assertEqual(state.get("history", []), [])
        self.assertEqual(state["control"]["desired_state"], "paused")
        self.assertIn("paused", state["last_control_event"])

    def test_execute_loop_can_pause_after_iteration_boundary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = pathlib.Path(tmpdir)
            control_path = repo / ".ternary_lab" / "control.json"
            control = load_control(control_path)
            control["stop_after_iteration"] = True
            save_control(control_path, control)
            state = execute_loop(
                LabConfig(
                    repo_root=str(repo),
                    trainer_backend="mock",
                    iterations=3,
                    manage_git=False,
                )
            )
            updated_control = load_control(control_path)
        self.assertEqual(len(state.get("history", [])), 1)
        self.assertEqual(updated_control["desired_state"], "paused")
        self.assertFalse(updated_control["stop_after_iteration"])

    def test_execute_supervisor_stops_at_requested_boundary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = pathlib.Path(tmpdir)
            control_path = repo / ".ternary_lab" / "control.json"
            control = load_control(control_path)
            control["stop_after_iteration"] = True
            save_control(control_path, control)
            state = execute_supervisor(
                LabConfig(
                    repo_root=str(repo),
                    trainer_backend="mock",
                    run_until_stopped=True,
                    max_supervisor_iterations=10,
                    manage_git=False,
                )
            )
            updated_control = load_control(control_path)
        self.assertEqual(len(state.get("history", [])), 1)
        self.assertEqual(updated_control["desired_state"], "paused")
        self.assertIn("paused", state.get("last_control_event", ""))

    def test_execute_loop_records_iteration_timing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = pathlib.Path(tmpdir)
            state = execute_loop(
                LabConfig(
                    repo_root=str(repo),
                    trainer_backend="mock",
                    iterations=1,
                    manage_git=False,
                )
            )
        entry = state["history"][0]
        self.assertGreaterEqual(entry["finished_at"], entry["started_at"])
        self.assertGreaterEqual(entry["iteration_duration_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()