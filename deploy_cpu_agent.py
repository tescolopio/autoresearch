"""Deploy the first local CPU-native BitNet agent for development testing."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ternary_lab import DEFAULT_OBJECTIVE, LabConfig, execute_loop


@dataclass
class DeploymentConfig:
    repo_root: str
    agent_name: str = "cpu-agent-1"
    objective: str = DEFAULT_OBJECTIVE
    iterations: int = 1
    mode: str = "auto"
    mission_seed: str = "ternary-lab"
    timeout_seconds: int = 900
    mock_val_bpb: float = 1.250000


def cache_status(home=None):
    home_path = Path(home).expanduser() if home else Path.home()
    cache_root = home_path / ".cache" / "autoresearch"
    data_dir = cache_root / "data"
    tokenizer_dir = cache_root / "tokenizer"
    data_ready = data_dir.exists() and any(data_dir.glob("*.parquet"))
    tokenizer_ready = (
        tokenizer_dir.exists()
        and (tokenizer_dir / "tokenizer.pkl").exists()
        and (tokenizer_dir / "token_bytes.pt").exists()
    )
    return {
        "cache_root": str(cache_root),
        "data_dir": str(data_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "data_ready": data_ready,
        "tokenizer_ready": tokenizer_ready,
        "ready": data_ready and tokenizer_ready,
    }


def resolve_mode(requested_mode, readiness):
    if requested_mode == "auto":
        return "train" if readiness["ready"] else "mock"
    return requested_mode


def agent_paths(repo_root, agent_name):
    base = Path(repo_root) / ".ternary_lab" / "agents" / agent_name
    return {
        "base": base,
        "manifest": base / "agent.json",
        "report": base / "deployment_report.json",
        "results": base / "results.tsv",
        "state": base / "state.json",
        "archive": base / "knowledge_graph.json",
        "mission": base / "mission.json",
        "control": base / "control.json",
    }


def build_manifest(config, readiness, backend):
    return {
        "agent_name": config.agent_name,
        "objective": config.objective,
        "iterations": config.iterations,
        "deployment_mode": backend,
        "cpu_only": True,
        "bitnet": {
            "linear_impl": "bitlinear",
            "quantization": "W1.58A8",
            "window_pattern": "L",
        },
        "local_readiness": readiness,
        "next_step": "Run uv run prepare.py to enable real training mode." if backend == "mock" else "Ready for local CPU training.",
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def deploy_agent(config):
    readiness = cache_status()
    backend = resolve_mode(config.mode, readiness)
    paths = agent_paths(config.repo_root, config.agent_name)
    manifest = build_manifest(config, readiness, backend)
    write_json(paths["manifest"], manifest)

    lab_config = LabConfig(
        repo_root=config.repo_root,
        iterations=config.iterations,
        agent_name=config.agent_name,
        objective=config.objective,
        mission_seed=config.mission_seed,
        results_tsv=str(paths["results"].relative_to(config.repo_root)),
        state_path=str(paths["state"].relative_to(config.repo_root)),
        archive_path=str(paths["archive"].relative_to(config.repo_root)),
        mission_path=str(paths["mission"].relative_to(config.repo_root)),
        control_path=str(paths["control"].relative_to(config.repo_root)),
        trainer_backend=backend,
        mock_val_bpb=config.mock_val_bpb,
        timeout_seconds=config.timeout_seconds,
    )
    state = execute_loop(lab_config)
    report = {
        "manifest": manifest,
        "best": state.get("best"),
        "history_length": len(state.get("history", [])),
        "control": state.get("control"),
    }
    write_json(paths["report"], report)
    return paths, report


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy a local CPU-native BitNet agent for development testing.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--agent-name", default="cpu-agent-1")
    parser.add_argument("--objective", default=DEFAULT_OBJECTIVE)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--mode", choices=["auto", "mock", "train"], default="auto")
    parser.add_argument("--mission-seed", default="ternary-lab")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--mock-val-bpb", type=float, default=1.250000)
    return parser.parse_args()


def main():
    args = parse_args()
    config = DeploymentConfig(**vars(args))
    paths, report = deploy_agent(config)
    best = report["best"]
    print("---")
    print(f"agent_name:       {config.agent_name}")
    print(f"deployment_mode:  {report['manifest']['deployment_mode']}")
    print(f"manifest:         {paths['manifest']}")
    print(f"report:           {paths['report']}")
    if best:
        metrics = best["metrics"]
        print(f"best_val_bpb:     {float(metrics['val_bpb']):.6f}")
        print(f"device:           {metrics['device']}")
        print(f"linear_impl:      {metrics['linear_impl']}")
        print(f"tokens_per_sec:   {float(metrics['tokens_per_second']):.1f}")


if __name__ == "__main__":
    main()