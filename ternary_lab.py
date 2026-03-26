"""CPU-native BitNet autoresearch orchestration for Ternary Lab."""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


RESULTS_HEADER = (
    "commit\tval_bpb\tmemory_gb\tstatus\tdescription\tdevice\tlinear_impl\t"
    "signature_verified\tenergy_j_per_token\ttokens_per_second\t"
    "avg_cpu_process_percent\tavg_cpu_load_percent\tavg_gpu_util_percent\tavg_gpu_mem_used_mb\t"
    "reliability_score\tcapability_score\ttask_eval_score\n"
)
DEFAULT_OBJECTIVE = (
    "Improve the reliability and capability of the local research agent in research and training tasks, "
    "while minimizing val_bpb and joules per experiment using CPU-native BitNet only."
)
HUMAN_CORE_PRINCIPLES = (
    "Never use GPU, cloud APIs, or remote accelerators.",
    "Never send research data, weights, prompts, or telemetry off the local machine.",
    "Only execute signed CPU-native BitNet experiments that satisfy the human mission.",
)
FORBIDDEN_TERMS = (
    "http://",
    "https://",
    "socket",
    "requests",
    "urllib",
    "curl ",
    "wget ",
    "internet",
    "network",
    "ssh ",
)
RESOURCE_ALLOCATION_TABLE = {
    "delta": {"cpu_threads": 0, "ram_priority": "snn-buffer", "power_use": "<1W"},
    "alpha": {"cpu_threads": 4, "ram_priority": "vector-index", "power_use": "15-30W"},
    "gamma": {"cpu_threads": "max", "ram_priority": "full-bitnet", "power_use": "peak"},
    "theta": {"cpu_threads": "background", "ram_priority": "kv-cache-pruning", "power_use": "5-10W"},
}


@dataclass
class ExperimentCandidate:
    name: str
    description: str
    depth: int = 4
    device_batch_size: int = 8
    total_batch_size: int = 2**14
    window_pattern: str = "L"
    bitlinear_scaling: str = "mean"
    bitlinear_threshold: float = 0.5
    use_subln: bool = True
    device: str = "cpu"
    linear_impl: str = "bitlinear"
    avg_power_watts: float = 15.0


@dataclass
class LabConfig:
    repo_root: str
    iterations: int = 1
    agent_name: str = "cpu-agent-1"
    objective: str = DEFAULT_OBJECTIVE
    mission_seed: str = "ternary-lab"
    results_tsv: str = "results.tsv"
    state_path: str = ".ternary_lab/state.json"
    archive_path: str = ".ternary_lab/knowledge_graph.json"
    mission_path: str = ".ternary_lab/mission.json"
    train_script: str = "train.py"
    trainer_backend: str = "train"
    mock_val_bpb: float = 1.250000
    timeout_seconds: int = 900
    manage_git: bool = True
    git_helper_script: str = "scripts/git_agent.py"
    git_branch_prefix: str = "autoresearch"
    git_branch_tag: str = ""
    control_path: str = ".ternary_lab/control.json"
    run_until_stopped: bool = False
    max_supervisor_iterations: int = 0


def machine_fingerprint(hostname=None, mac_address=None):
    host = hostname or platform.node() or "unknown-host"
    mac = mac_address if mac_address is not None else uuid.getnode()
    return f"{host}:{mac}"


def derive_machine_secret(seed, fingerprint=None):
    material = f"{seed}:{fingerprint or machine_fingerprint()}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def compute_instruction_signature(payload, secret):
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def seal_research_mission(objective, principles, secret):
    mission = {
        "objective": objective,
        "principles": list(principles),
        "sealed_at": int(time.time()),
    }
    payload = json.dumps(mission, sort_keys=True)
    mission["signature"] = compute_instruction_signature(payload, secret)
    return mission


def verify_research_mission(mission, secret):
    payload = json.dumps(
        {
            "objective": mission["objective"],
            "principles": mission["principles"],
            "sealed_at": mission["sealed_at"],
        },
        sort_keys=True,
    )
    expected = compute_instruction_signature(payload, secret)
    return hmac.compare_digest(expected, mission.get("signature", ""))


def candidate_policy_violations(candidate, objective, principles):
    violations = []
    if candidate.device != "cpu":
        violations.append("device must remain cpu")
    if candidate.linear_impl != "bitlinear":
        violations.append("linear implementation must remain bitlinear")
    policy_text = " ".join([objective, *principles, candidate.description, candidate.name]).lower()
    for term in FORBIDDEN_TERMS:
        if term.strip() in policy_text:
            violations.append(f"forbidden term detected: {term.strip()}")
    return violations


def ensure_results_tsv(path):
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if not result_path.exists():
        result_path.write_text(RESULTS_HEADER, encoding="utf-8")


def append_results_tsv(path, metrics):
    ensure_results_tsv(path)
    description = metrics["description"].replace("\t", " ").replace("\r", " ").replace("\n", " ")
    row = [
        metrics["commit"],
        f"{float(metrics['val_bpb']):.6f}",
        f"{float(metrics['memory_gb']):.1f}",
        metrics["status"],
        description,
        metrics["device"],
        metrics["linear_impl"],
        "yes" if metrics["signature_verified"] else "no",
        f"{float(metrics['energy_j_per_token']):.9f}",
        f"{float(metrics['tokens_per_second']):.1f}",
        f"{float(metrics.get('avg_cpu_process_percent', 0.0)):.1f}",
        f"{float(metrics.get('avg_cpu_load_percent', 0.0)):.1f}",
        f"{float(metrics.get('avg_gpu_util_percent', 0.0)):.1f}",
        f"{float(metrics.get('avg_gpu_mem_used_mb', 0.0)):.1f}",
        f"{float(metrics.get('reliability_score', 0.0)):.4f}",
        f"{float(metrics.get('capability_score', 0.0)):.4f}",
        f"{float(metrics.get('task_eval_score', 0.0)):.4f}",
    ]
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\t".join(row) + "\n")


def load_results_tsv(path):
    result_path = Path(path)
    if not result_path.exists():
        return []
    with open(result_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def load_state(path):
    state_path = Path(path)
    if not state_path.exists():
        return {"history": [], "best": None}
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(path, state):
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def load_control(path, default_objective=DEFAULT_OBJECTIVE):
    control_path = Path(path)
    if not control_path.exists():
        return {
            "desired_state": "running",
            "objective_override": "",
            "human_note": "",
            "updated_at": 0,
            "updated_by": "system",
            "stop_after_iteration": False,
            "showcase_loop_enabled": False,
            "showcase_note": "Frontier comparison is separate from main-loop decisions.",
            "default_objective": default_objective,
        }
    control = json.loads(control_path.read_text(encoding="utf-8"))
    control.setdefault("desired_state", "running")
    control.setdefault("objective_override", "")
    control.setdefault("human_note", "")
    control.setdefault("updated_at", 0)
    control.setdefault("updated_by", "system")
    control.setdefault("stop_after_iteration", False)
    control.setdefault("showcase_loop_enabled", False)
    control.setdefault("showcase_note", "Frontier comparison is separate from main-loop decisions.")
    control.setdefault("default_objective", default_objective)
    return control


def save_control(path, control):
    control_path = Path(path)
    control_path.parent.mkdir(parents=True, exist_ok=True)
    control_path.write_text(json.dumps(control, indent=2, sort_keys=True), encoding="utf-8")


def effective_objective(config, control):
    return control.get("objective_override") or config.objective


def control_gate(control):
    desired_state = control.get("desired_state", "running")
    if desired_state == "running":
        return None
    if desired_state == "paused":
        return "paused by human control"
    if desired_state == "stopped":
        return "stopped by human control"
    return f"unsupported desired_state: {desired_state}"


def summarize_state(state, control):
    history = state.get("history", [])
    best = state.get("best")
    last = history[-1] if history else None
    durations = [float(entry.get("iteration_duration_seconds", 0.0) or 0.0) for entry in history if entry.get("iteration_duration_seconds") is not None]
    avg_iteration_seconds = sum(durations) / len(durations) if durations else 0.0
    recent_activity = []
    for entry in history[-3:]:
        recent_activity.append(
            {
                "run_id": entry.get("run_id"),
                "status": entry.get("metrics", {}).get("status"),
                "task_eval_score": entry.get("metrics", {}).get("task_eval_score", 0.0),
                "started_at": entry.get("started_at", 0),
                "finished_at": entry.get("finished_at", 0),
                "iteration_duration_seconds": entry.get("iteration_duration_seconds", 0.0),
            }
        )
    summary = {
        "history_length": len(history),
        "best_run_id": best["run_id"] if best else None,
        "best_metrics": best["metrics"] if best else None,
        "last_run_id": last["run_id"] if last else None,
        "last_metrics": last["metrics"] if last else None,
        "objective": state.get("objective", control.get("default_objective", DEFAULT_OBJECTIVE)),
        "control": control,
        "git": state.get("git"),
        "created_at": state.get("created_at", 0),
        "last_activity_at": state.get("last_activity_at", 0),
        "average_iteration_seconds": avg_iteration_seconds,
        "last_iteration_seconds": (last or {}).get("iteration_duration_seconds", 0.0) if last else 0.0,
        "recent_activity": recent_activity,
        "last_control_event": state.get("last_control_event", ""),
    }
    return summary


def format_summary_lines(summary):
    control = summary.get("control", {})
    best_metrics = summary.get("best_metrics") or {}
    last_metrics = summary.get("last_metrics") or {}
    lines = [
        "---",
        f"history_length:         {summary.get('history_length', 0)}",
        f"objective:              {summary.get('objective', '')}",
        f"desired_state:          {control.get('desired_state', 'running')}",
        f"stop_after_iteration:   {control.get('stop_after_iteration', False)}",
        f"updated_by:             {control.get('updated_by', 'system')}",
        f"updated_at:             {control.get('updated_at', 0)}",
        f"created_at:             {summary.get('created_at', 0)}",
        f"last_activity_at:       {summary.get('last_activity_at', 0)}",
        f"avg_iteration_seconds:  {float(summary.get('average_iteration_seconds', 0.0)):.3f}",
        f"last_iteration_seconds: {float(summary.get('last_iteration_seconds', 0.0)):.3f}",
        f"best_run_id:            {summary.get('best_run_id')}",
        f"last_run_id:            {summary.get('last_run_id')}",
    ]
    if best_metrics:
        lines.extend(
            [
                f"best_status:            {best_metrics.get('status')}",
                f"best_task_eval_score:   {float(best_metrics.get('task_eval_score', 0.0)):.4f}",
                f"best_reliability:       {float(best_metrics.get('reliability_score', 0.0)):.4f}",
                f"best_capability:        {float(best_metrics.get('capability_score', 0.0)):.4f}",
                f"best_val_bpb:           {float(best_metrics.get('val_bpb', 0.0)):.6f}",
            ]
        )
    if last_metrics:
        lines.extend(
            [
                f"last_status:            {last_metrics.get('status')}",
                f"last_task_eval_score:   {float(last_metrics.get('task_eval_score', 0.0)):.4f}",
                f"last_val_bpb:           {float(last_metrics.get('val_bpb', 0.0)):.6f}",
            ]
        )
    if summary.get("last_control_event"):
        lines.append(f"last_control_event:     {summary['last_control_event']}")
    return lines


def baseline_candidate():
    return ExperimentCandidate(
        name="loop0-baseline",
        description="CPU-native BitNet baseline with fixed 5-minute wall-clock budget.",
    )


def _candidate_from_state(best_state):
    if not best_state:
        return baseline_candidate()
    data = best_state["candidate"].copy()
    return ExperimentCandidate(**data)


def propose_next_candidate(results_rows, state):
    best_state = state.get("best")
    base = _candidate_from_state(best_state)
    history_len = len(state.get("history", []))
    if history_len == 0:
        return base
    last = state["history"][-1]
    last_status = last["metrics"]["status"]
    last_val = float(last["metrics"]["val_bpb"])
    best_val = float(best_state["metrics"]["val_bpb"]) if best_state else last_val
    if last_status == "crash":
        return ExperimentCandidate(
            **{
                **asdict(base),
                "name": f"loop2-recovery-{history_len}",
                "description": "Recovery candidate after crash with reduced CPU pressure.",
                "device_batch_size": max(4, base.device_batch_size // 2),
                "total_batch_size": max(2**13, base.total_batch_size // 2),
            }
        )
    if last_val > best_val + 0.02:
        return ExperimentCandidate(
            **{
                **asdict(base),
                "name": f"loop2-capacity-{history_len}",
                "description": "Increase depth to recover capacity lost to ternary quantization.",
                "depth": min(base.depth + 2, 12),
            }
        )
    phase = history_len % 4
    if phase == 1:
        return ExperimentCandidate(
            **{
                **asdict(base),
                "name": f"loop2-depth-{history_len}",
                "description": "Probe deeper BitNet stack while staying CPU-native.",
                "depth": min(base.depth + 1, 12),
            }
        )
    if phase == 2:
        return ExperimentCandidate(
            **{
                **asdict(base),
                "name": f"loop2-threshold-{history_len}",
                "description": "Adjust ternary threshold to trade sparsity against retained capacity.",
                "bitlinear_threshold": 0.45 if base.bitlinear_threshold >= 0.5 else 0.55,
            }
        )
    if phase == 3:
        return ExperimentCandidate(
            **{
                **asdict(base),
                "name": f"loop2-scaling-{history_len}",
                "description": "Switch ternary scaling rule to stabilize CPU quantization-aware updates.",
                "bitlinear_scaling": "median" if base.bitlinear_scaling == "mean" else "mean",
            }
        )
    return ExperimentCandidate(
        **{
            **asdict(base),
            "name": f"loop2-batch-{history_len}",
            "description": "Increase effective batch to improve CPU-side gradient signal.",
            "total_batch_size": min(base.total_batch_size * 2, 2**16),
        }
    )


def instruction_payload(candidate, objective, principles):
    return json.dumps(
        {
            "candidate": asdict(candidate),
            "objective": objective,
            "principles": list(principles),
        },
        sort_keys=True,
    )


def select_state(status):
    if status == "keep":
        return "theta"
    if status == "crash":
        return "alpha"
    return "gamma"


def novelty_score(description, accepted_descriptions):
    tokens = set(description.lower().split())
    if not accepted_descriptions or not tokens:
        return 1.0
    best_overlap = 0.0
    for prior in accepted_descriptions:
        prior_tokens = set(prior.lower().split())
        if not prior_tokens:
            continue
        overlap = len(tokens & prior_tokens) / len(tokens | prior_tokens)
        best_overlap = max(best_overlap, overlap)
    return 1.0 - best_overlap


def build_knowledge_graph(state, path, novelty_threshold=0.30):
    accepted = []
    accepted_descriptions = []
    for entry in state.get("history", []):
        metrics = entry["metrics"]
        score = novelty_score(entry["candidate"]["description"], accepted_descriptions)
        if metrics["status"] == "keep" or score >= novelty_threshold:
            accepted.append(
                {
                    "id": entry["run_id"],
                    "label": entry["candidate"]["name"],
                    "state": entry["state"],
                    "novelty": round(score, 3),
                    "candidate": entry["candidate"],
                    "metrics": metrics,
                }
            )
            accepted_descriptions.append(entry["candidate"]["description"])
    edges = []
    last_kept = None
    for node in accepted:
        if node["metrics"]["status"] == "keep":
            if last_kept is not None:
                edges.append({"source": last_kept, "target": node["id"], "type": "improves"})
            last_kept = node["id"]
    graph = {
        "generated_at": int(time.time()),
        "resource_allocation": RESOURCE_ALLOCATION_TABLE,
        "nodes": accepted,
        "edges": edges,
    }
    archive_path = Path(path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(json.dumps(graph, indent=2, sort_keys=True), encoding="utf-8")
    return graph


def resolve_path(repo_root, path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path(repo_root) / path


def run_git_helper(config, *args):
    helper_path = resolve_path(config.repo_root, config.git_helper_script)
    if not helper_path.exists():
        raise FileNotFoundError(f"git helper not found: {helper_path}")
    completed = subprocess.run(
        [sys.executable, str(helper_path), "--repo", str(Path(config.repo_root).resolve()), *args],
        cwd=config.repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError(f"git helper returned no output for args: {args}")
    payload = json.loads(stdout)
    if completed.returncode != 0 or not payload.get("ok"):
        raise RuntimeError(payload.get("error", f"git helper failed for args: {args}"))
    return payload


def prepare_git_session(config):
    if not config.manage_git:
        return {"enabled": False, "reason": "git management disabled"}
    try:
        status = run_git_helper(config, "status")
    except (FileNotFoundError, RuntimeError) as exc:
        return {"enabled": False, "reason": str(exc)}
    session = {
        "enabled": True,
        "status_before": status,
        "branch": None,
    }
    branch_prefix = f"{config.git_branch_prefix}/"
    if status["dirty"]:
        session["branch"] = {
            "action": "skipped",
            "reason": "dirty worktree",
            "branch": status["branch"],
        }
        return session
    if status["branch"].startswith(branch_prefix):
        session["branch"] = {
            "action": "already-on-run-branch",
            "branch": status["branch"],
        }
        return session
    branch_tag = config.git_branch_tag or f"{config.agent_name}-{int(time.time())}"
    session["branch"] = run_git_helper(
        config,
        "ensure-branch",
        "--tag",
        branch_tag,
        "--prefix",
        config.git_branch_prefix,
    )
    return session


def mock_candidate_metrics(candidate, mock_val_bpb):
    signature_source = json.dumps(asdict(candidate), sort_keys=True)
    digest = hashlib.sha256(signature_source.encode("utf-8")).hexdigest()
    offset = (int(digest[:4], 16) % 25) / 1000
    val_bpb = max(0.5, mock_val_bpb - offset)
    return {
        "commit": f"mock-{digest[:7]}",
        "val_bpb": val_bpb,
        "memory_gb": 0.4 + candidate.depth * 0.02,
        "status": "candidate",
        "description": candidate.description,
        "device": candidate.device,
        "linear_impl": candidate.linear_impl,
        "signature_verified": True,
        "energy_j_per_token": 0.028 + candidate.depth * 0.0005,
        "tokens_per_second": max(1.0, 7.0 - candidate.depth * 0.2),
        "avg_cpu_process_percent": 85.0,
        "avg_cpu_load_percent": 65.0,
        "avg_gpu_util_percent": 0.0,
        "avg_gpu_mem_used_mb": 0.0,
        "log_path": "mock://ternary-lab",
    }


def compute_agent_task_scores(metrics):
    status = metrics.get("status", "candidate")
    reliability_score = 0.0
    if status != "crash":
        reliability_score += 0.45
    if metrics.get("signature_verified"):
        reliability_score += 0.2
    if metrics.get("device") == "cpu":
        reliability_score += 0.15
    if metrics.get("linear_impl") == "bitlinear":
        reliability_score += 0.1
    if metrics.get("log_path"):
        reliability_score += 0.1
    reliability_score = min(reliability_score, 1.0)

    val_bpb = float(metrics.get("val_bpb", 0.0) or 0.0)
    tokens_per_second = float(metrics.get("tokens_per_second", 0.0) or 0.0)
    energy_j_per_token = float(metrics.get("energy_j_per_token", 0.0) or 0.0)
    quality_term = 1.0 / (1.0 + max(val_bpb, 0.0))
    throughput_term = min(tokens_per_second / 10.0, 1.0)
    efficiency_term = 1.0 / (1.0 + max(energy_j_per_token, 0.0) * 100.0)
    capability_score = min((0.55 * quality_term) + (0.3 * throughput_term) + (0.15 * efficiency_term), 1.0)

    task_eval_score = min((0.6 * reliability_score) + (0.4 * capability_score), 1.0)
    return {
        "reliability_score": reliability_score,
        "capability_score": capability_score,
        "task_eval_score": task_eval_score,
    }


def rank_metrics(metrics):
    return (
        float(metrics.get("task_eval_score", 0.0) or 0.0),
        -float(metrics.get("val_bpb", 0.0) or 0.0),
        -float(metrics.get("energy_j_per_token", 0.0) or 0.0),
    )


def is_better_candidate(metrics, best_metrics):
    if best_metrics is None:
        return metrics.get("status") != "crash"
    return rank_metrics(metrics) > rank_metrics(best_metrics)


def run_candidate(config, candidate, secret, mission):
    repo_root = Path(config.repo_root)
    lab_dir = repo_root / ".ternary_lab"
    runs_dir = lab_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{int(time.time())}-{candidate.name}"
    if config.trainer_backend == "mock":
        return run_id, mock_candidate_metrics(candidate, config.mock_val_bpb)
    summary_path = runs_dir / f"{run_id}.json"
    log_path = runs_dir / f"{run_id}.log"
    payload = instruction_payload(candidate, config.objective, mission["principles"])
    signature = compute_instruction_signature(payload, secret)
    cmd = [
        sys.executable,
        config.train_script,
        "--device",
        "cpu",
        "--cpu-only",
        "--cpu-bitnet-poc",
        "--linear-impl",
        candidate.linear_impl,
        "--bitlinear-scaling",
        candidate.bitlinear_scaling,
        "--bitlinear-threshold",
        str(candidate.bitlinear_threshold),
        "--depth",
        str(candidate.depth),
        "--device-batch-size",
        str(candidate.device_batch_size),
        "--total-batch-size",
        str(candidate.total_batch_size),
        "--window-pattern",
        candidate.window_pattern,
        "--results-tsv",
        "",
        "--summary-json",
        str(summary_path),
        "--status",
        "candidate",
        "--description",
        candidate.description,
        "--avg-power-watts",
        str(candidate.avg_power_watts),
        "--objective",
        payload,
        "--signature",
        signature,
        "--signature-secret",
        secret,
        "--require-signature",
    ]
    with open(log_path, "w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            cmd,
            cwd=repo_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            timeout=config.timeout_seconds,
            check=False,
        )
    if completed.returncode != 0 or not summary_path.exists():
        return run_id, {
            "commit": "unknown",
            "val_bpb": 0.0,
            "memory_gb": 0.0,
            "status": "crash",
            "description": candidate.description,
            "device": candidate.device,
            "linear_impl": candidate.linear_impl,
            "signature_verified": False,
            "energy_j_per_token": 0.0,
            "tokens_per_second": 0.0,
            "avg_cpu_process_percent": 0.0,
            "avg_cpu_load_percent": 0.0,
            "avg_gpu_util_percent": 0.0,
            "avg_gpu_mem_used_mb": 0.0,
            "log_path": str(log_path),
        }
    metrics = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics["log_path"] = str(log_path)
    return run_id, metrics


def execute_loop(config):
    control_path = str(Path(config.repo_root) / config.control_path)
    control = load_control(control_path, config.objective)
    active_objective = effective_objective(config, control)
    secret = derive_machine_secret(config.mission_seed)
    mission = seal_research_mission(active_objective, HUMAN_CORE_PRINCIPLES, secret)
    if not verify_research_mission(mission, secret):
        raise RuntimeError("Mission sealing failed.")
    state_path = str(Path(config.repo_root) / config.state_path)
    state = load_state(state_path)
    state.setdefault("created_at", int(time.time()))
    state["objective"] = active_objective
    state["git"] = prepare_git_session(config)
    state["control"] = control

    mission_path = Path(config.repo_root) / config.mission_path
    mission_path.parent.mkdir(parents=True, exist_ok=True)
    mission_path.write_text(json.dumps(mission, indent=2, sort_keys=True), encoding="utf-8")

    results_path = str(Path(config.repo_root) / config.results_tsv)
    archive_path = str(Path(config.repo_root) / config.archive_path)
    ensure_results_tsv(results_path)

    for _ in range(config.iterations):
        control = load_control(control_path, config.objective)
        state["control"] = control
        active_objective = effective_objective(config, control)
        state["objective"] = active_objective
        gate_reason = control_gate(control)
        if gate_reason is not None:
            state["last_control_event"] = gate_reason
            save_state(state_path, state)
            break
        results_rows = load_results_tsv(results_path)
        candidate = propose_next_candidate(results_rows, state)
        violations = candidate_policy_violations(candidate, active_objective, HUMAN_CORE_PRINCIPLES)
        if violations:
            raise RuntimeError("Instruction signing refused: " + "; ".join(violations))
        git_status = None
        if config.manage_git and state.get("git", {}).get("enabled"):
            git_status = run_git_helper(config, "status")
        mission = seal_research_mission(active_objective, HUMAN_CORE_PRINCIPLES, secret)
        iteration_started_at = int(time.time())
        iteration_start = time.time()
        run_id, metrics = run_candidate(config, candidate, secret, mission)
        iteration_finished_at = int(time.time())
        iteration_duration_seconds = time.time() - iteration_start
        best = state.get("best")
        if metrics["status"] != "crash":
            metrics.update(compute_agent_task_scores(metrics))
            is_better = is_better_candidate(metrics, None if best is None else best["metrics"])
            metrics["status"] = "keep" if is_better else "discard"
        else:
            metrics.update(compute_agent_task_scores(metrics))
        entry = {
            "run_id": run_id,
            "candidate": asdict(candidate),
            "metrics": metrics,
            "state": select_state(metrics["status"]),
            "git_status": git_status,
            "started_at": iteration_started_at,
            "finished_at": iteration_finished_at,
            "iteration_duration_seconds": iteration_duration_seconds,
        }
        state.setdefault("history", []).append(entry)
        state["last_activity_at"] = iteration_finished_at
        if metrics["status"] == "keep":
            state["best"] = entry
        append_results_tsv(results_path, metrics)
        build_knowledge_graph(state, archive_path)
        save_state(state_path, state)
        if control.get("stop_after_iteration"):
            control["desired_state"] = "paused"
            control["stop_after_iteration"] = False
            control["updated_at"] = int(time.time())
            control["updated_by"] = control.get("updated_by") or "system"
            save_control(control_path, control)
            state["control"] = control
            state["last_control_event"] = "paused after requested iteration boundary"
            save_state(state_path, state)
            break
    return state


def execute_supervisor(config):
    iterations_completed = 0
    state = load_state(str(Path(config.repo_root) / config.state_path))
    while True:
        control = load_control(str(Path(config.repo_root) / config.control_path), config.objective)
        gate_reason = control_gate(control)
        if gate_reason is not None:
            state = load_state(str(Path(config.repo_root) / config.state_path))
            state["control"] = control
            state["last_control_event"] = gate_reason
            save_state(str(Path(config.repo_root) / config.state_path), state)
            return state
        single_config = LabConfig(**{**asdict(config), "iterations": 1, "run_until_stopped": False})
        state = execute_loop(single_config)
        iterations_completed += 1
        if config.max_supervisor_iterations and iterations_completed >= config.max_supervisor_iterations:
            state["last_control_event"] = "supervisor stopped at max iteration limit"
            save_state(str(Path(config.repo_root) / config.state_path), state)
            return state


def parse_args():
    parser = argparse.ArgumentParser(description="Run the CPU-native Ternary Lab autoresearch loop.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--agent-name", default="cpu-agent-1")
    parser.add_argument("--objective", default=DEFAULT_OBJECTIVE)
    parser.add_argument("--mission-seed", default="ternary-lab")
    parser.add_argument("--results-tsv", default="results.tsv")
    parser.add_argument("--state-path", default=".ternary_lab/state.json")
    parser.add_argument("--archive-path", default=".ternary_lab/knowledge_graph.json")
    parser.add_argument("--mission-path", default=".ternary_lab/mission.json")
    parser.add_argument("--train-script", default="train.py")
    parser.add_argument("--trainer-backend", choices=["train", "mock"], default="train")
    parser.add_argument("--mock-val-bpb", type=float, default=1.250000)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--manage-git", dest="manage_git", action="store_true", default=True)
    parser.add_argument("--no-manage-git", dest="manage_git", action="store_false")
    parser.add_argument("--git-helper-script", default="scripts/git_agent.py")
    parser.add_argument("--git-branch-prefix", default="autoresearch")
    parser.add_argument("--git-branch-tag", default="")
    parser.add_argument("--control-path", default=".ternary_lab/control.json")
    parser.add_argument("--run-until-stopped", action="store_true")
    parser.add_argument("--max-supervisor-iterations", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    config = LabConfig(**vars(args))
    state = execute_supervisor(config) if config.run_until_stopped else execute_loop(config)
    summary = summarize_state(state, load_control(str(Path(config.repo_root) / config.control_path), config.objective))
    best = state.get("best")
    if best is None:
        for line in format_summary_lines(summary):
            print(line)
        return
    for line in format_summary_lines(summary):
        print(line)


if __name__ == "__main__":
    main()