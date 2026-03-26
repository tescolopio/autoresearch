#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PRACTICAL_MILESTONE_ORDER = [
    "environment-reproducibility",
    "local-agent-lifecycle",
    "data-and-tokenizer-readiness",
    "real-cpu-training-baseline",
    "task-specific-evaluation",
    "single-iteration-research-loop",
    "short-unattended-burn-in",
    "comparative-measurement-layer",
    "overnight-readiness-gate",
]


def cache_status(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    from deploy_cpu_agent import cache_status as _cache_status  # pylint: disable=import-outside-toplevel

    return _cache_status()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_environment(repo_root: Path, run_focused_tests: bool):
    evidence = []
    missing = []
    if (repo_root / "environment.yml").exists():
        evidence.append("environment.yml present")
    else:
        missing.append("environment.yml missing")
    if (repo_root / "scripts" / "bootstrap_bitnet_research.sh").exists():
        evidence.append("bootstrap script present")
    else:
        missing.append("bootstrap script missing")
    if run_focused_tests:
        command = [
            sys.executable,
            "-m",
            "unittest",
            "tests/test_bitnet_cpu_poc.py",
            "tests/test_ternary_lab.py",
            "tests/test_deploy_cpu_agent.py",
            "tests/test_compare_agents.py",
            "tests/test_benchmark_compare.py",
            "tests/test_provider_benchmark.py",
        ]
        completed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, check=False)
        if completed.returncode == 0:
            evidence.append("focused validation suite passed")
        else:
            missing.append("focused validation suite failed")
    else:
        evidence.append("focused validation suite not executed by roadmap_doctor; run with --run-focused-tests for stronger verification")
    return {"complete": not missing, "evidence": evidence, "missing": missing}


def check_local_agent_lifecycle(repo_root: Path):
    evidence = []
    missing = []
    agent_root = repo_root / ".ternary_lab" / "agents"
    manifests = list(agent_root.glob("*/agent.json"))
    reports = list(agent_root.glob("*/deployment_report.json"))
    if manifests:
        evidence.append(f"found {len(manifests)} agent manifest(s)")
    else:
        missing.append("no agent manifest found under .ternary_lab/agents/*/agent.json")
    if reports:
        evidence.append(f"found {len(reports)} deployment report(s)")
    else:
        missing.append("no deployment report found under .ternary_lab/agents/*/deployment_report.json")
    return {"complete": not missing, "evidence": evidence, "missing": missing}


def check_data_ready(repo_root: Path):
    readiness = cache_status(repo_root)
    evidence = []
    missing = []
    if readiness["data_ready"]:
        evidence.append("data parquet shard detected")
    else:
        missing.append("no cached parquet shard detected")
    if readiness["tokenizer_ready"]:
        evidence.append("tokenizer artifacts detected")
    else:
        missing.append("tokenizer artifacts missing")
    return {"complete": not missing, "evidence": evidence, "missing": missing}


def _summary_has_cpu_metrics(summary_path: Path):
    if not summary_path.exists():
        return False
    payload = read_json(summary_path)
    required = ["val_bpb", "tokens_per_second", "energy_j_per_token", "memory_gb"]
    return (
        payload.get("device") == "cpu"
        and payload.get("linear_impl") == "bitlinear"
        and all(key in payload for key in required)
    )


def check_cpu_baseline(repo_root: Path):
    evidence = []
    missing = []
    candidates = [
        repo_root / "comparison_reports" / "cpu_bitnet.json",
        repo_root / "comparison_reports" / "benchmark" / "cpu_bitnet.json",
    ]
    found = next((path for path in candidates if _summary_has_cpu_metrics(path)), None)
    if found is not None:
        evidence.append(f"cpu baseline summary found at {found.relative_to(repo_root)}")
    else:
        missing.append("no valid cpu baseline summary json found")
    return {"complete": not missing, "evidence": evidence, "missing": missing}


def check_task_specific_evaluation(repo_root: Path):
    evidence = []
    missing = []
    expected = [
        repo_root / "provider_reports" / "latest" / "responses.jsonl",
        repo_root / "provider_reports" / "latest" / "provider_summary.csv",
        repo_root / "provider_reports" / "latest" / "provider_summary.md",
    ]
    for path in expected:
        if path.exists():
            evidence.append(f"found {path.relative_to(repo_root)}")
        else:
            missing.append(f"missing {path.relative_to(repo_root)}")
    return {"complete": not missing, "evidence": evidence, "missing": missing}


def load_state_and_results(repo_root: Path):
    state_path = repo_root / ".ternary_lab" / "state.json"
    state = read_json(state_path) if state_path.exists() else {"history": [], "best": None}
    results_path = repo_root / "results.tsv"
    rows = []
    if results_path.exists():
        lines = results_path.read_text(encoding="utf-8").strip().splitlines()
        rows = lines[1:] if len(lines) > 1 else []
    return state, rows


def check_single_iteration_loop(repo_root: Path):
    evidence = []
    missing = []
    state, rows = load_state_and_results(repo_root)
    if state.get("history"):
        evidence.append("state history contains at least one iteration")
    else:
        missing.append("state history has no completed iterations")
    if (repo_root / ".ternary_lab" / "knowledge_graph.json").exists():
        evidence.append("knowledge graph exists")
    else:
        missing.append("knowledge graph missing")
    if rows:
        evidence.append("results.tsv contains at least one result row")
    else:
        missing.append("results.tsv contains no result rows")
    return {"complete": not missing, "evidence": evidence, "missing": missing}


def check_burn_in(repo_root: Path):
    evidence = []
    missing = []
    state, _ = load_state_and_results(repo_root)
    history = state.get("history", [])
    successful = [entry for entry in history if entry.get("metrics", {}).get("status") in {"keep", "discard"}]
    if len(successful) >= 3:
        evidence.append(f"found {len(successful)} non-crash iterations")
    else:
        missing.append("fewer than 3 non-crash iterations recorded")
    return {"complete": not missing, "evidence": evidence, "missing": missing}


def check_comparative_layer(repo_root: Path):
    evidence = []
    missing = []
    expected = [
        repo_root / "comparison_reports" / "latest" / "comparison.md",
        repo_root / "comparison_reports" / "latest" / "skeptic_summary.md",
        repo_root / "comparison_reports" / "latest" / "throughput.png",
    ]
    for path in expected:
        if path.exists():
            evidence.append(f"found {path.relative_to(repo_root)}")
        else:
            missing.append(f"missing {path.relative_to(repo_root)}")
    return {"complete": not missing, "evidence": evidence, "missing": missing}


def check_overnight_gate(statuses):
    prior = PRACTICAL_MILESTONE_ORDER[:-1]
    incomplete = [name for name in prior if not statuses[name]["complete"]]
    if incomplete:
        return {"complete": False, "evidence": [], "missing": [f"prior milestones incomplete: {', '.join(incomplete)}"]}
    return {"complete": True, "evidence": ["all prior milestones complete"], "missing": []}


def evaluate_milestones(repo_root: Path, run_focused_tests: bool):
    statuses = {}
    statuses["environment-reproducibility"] = check_environment(repo_root, run_focused_tests)
    statuses["local-agent-lifecycle"] = check_local_agent_lifecycle(repo_root)
    statuses["data-and-tokenizer-readiness"] = check_data_ready(repo_root)
    statuses["real-cpu-training-baseline"] = check_cpu_baseline(repo_root)
    statuses["task-specific-evaluation"] = check_task_specific_evaluation(repo_root)
    statuses["single-iteration-research-loop"] = check_single_iteration_loop(repo_root)
    statuses["short-unattended-burn-in"] = check_burn_in(repo_root)
    statuses["comparative-measurement-layer"] = check_comparative_layer(repo_root)
    statuses["overnight-readiness-gate"] = check_overnight_gate(statuses)
    current = next((name for name in PRACTICAL_MILESTONE_ORDER if not statuses[name]["complete"]), None)
    return {
        "milestones": statuses,
        "current_milestone": current,
        "all_complete": current is None,
    }


def build_text_report(report):
    lines = ["---"]
    lines.append(f"all_complete:          {report['all_complete']}")
    lines.append(f"current_milestone:    {report['current_milestone']}")
    for name in PRACTICAL_MILESTONE_ORDER:
        status = report["milestones"][name]
        lines.append(f"{name}: {'complete' if status['complete'] else 'incomplete'}")
        for item in status["missing"][:3]:
            lines.append(f"  missing: {item}")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Check roadmap milestones and identify the current gating milestone.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-focused-tests", action="store_true")
    parser.add_argument("--format", choices=["json", "text"], default="json")
    parser.add_argument("--require-complete-through", choices=PRACTICAL_MILESTONE_ORDER)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    report = evaluate_milestones(repo_root, args.run_focused_tests)
    exit_code = 0
    if args.require_complete_through is not None:
        target_index = PRACTICAL_MILESTONE_ORDER.index(args.require_complete_through)
        required = PRACTICAL_MILESTONE_ORDER[: target_index + 1]
        failed = [name for name in required if not report["milestones"][name]["complete"]]
        if failed:
            report["required_failed"] = failed
            exit_code = 1
    if args.format == "text":
        print(build_text_report(report))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()