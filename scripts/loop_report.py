#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from ternary_lab import DEFAULT_OBJECTIVE, load_control, load_state, summarize_state


def format_timestamp(value):
    timestamp = int(value or 0)
    if timestamp <= 0:
        return "n/a"
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def build_markdown(summary):
    control = summary.get("control", {})
    best = summary.get("best_metrics") or {}
    last = summary.get("last_metrics") or {}
    lines = [
        "# Ternary Lab Loop Report",
        "",
        "## Current State",
        f"- history_length: {summary.get('history_length', 0)}",
        f"- objective: {summary.get('objective', '')}",
        f"- desired_state: {control.get('desired_state', 'running')}",
        f"- stop_after_iteration: {control.get('stop_after_iteration', False)}",
        f"- updated_by: {control.get('updated_by', 'system')}",
        f"- updated_at: {format_timestamp(control.get('updated_at', 0))}",
        f"- created_at: {format_timestamp(summary.get('created_at', 0))}",
        f"- last_activity_at: {format_timestamp(summary.get('last_activity_at', 0))}",
        f"- average_iteration_seconds: {float(summary.get('average_iteration_seconds', 0.0)):.3f}",
        f"- last_iteration_seconds: {float(summary.get('last_iteration_seconds', 0.0)):.3f}",
        f"- human_note: {control.get('human_note', '')}",
        "",
        "## Best Run",
        f"- run_id: {summary.get('best_run_id')}",
        f"- status: {best.get('status')}",
        f"- task_eval_score: {float(best.get('task_eval_score', 0.0)):.4f}",
        f"- reliability_score: {float(best.get('reliability_score', 0.0)):.4f}",
        f"- capability_score: {float(best.get('capability_score', 0.0)):.4f}",
        f"- val_bpb: {float(best.get('val_bpb', 0.0)):.6f}",
        f"- tokens_per_second: {float(best.get('tokens_per_second', 0.0)):.2f}",
        "",
        "## Last Run",
        f"- run_id: {summary.get('last_run_id')}",
        f"- status: {last.get('status')}",
        f"- task_eval_score: {float(last.get('task_eval_score', 0.0)):.4f}",
        f"- val_bpb: {float(last.get('val_bpb', 0.0)):.6f}",
        "",
        "## Recent Activity",
    ]
    recent = summary.get("recent_activity", [])
    if recent:
        lines.extend([
            "| run_id | status | task_eval_score | started_at | finished_at | duration_s |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for item in recent:
            lines.append(
                f"| {item.get('run_id')} | {item.get('status')} | {float(item.get('task_eval_score', 0.0)):.4f} | {format_timestamp(item.get('started_at', 0))} | {format_timestamp(item.get('finished_at', 0))} | {float(item.get('iteration_duration_seconds', 0.0)):.3f} |"
            )
    else:
        lines.append("- No iterations have completed yet.")
    lines.extend([
        "",
        "## Control Guidance",
        "- Use `python scripts/loop_status.py --set-state paused` to pause before the next iteration.",
        "- Use `python scripts/loop_status.py --stop-after-iteration` to pause at the next iteration boundary.",
        "- Use `python scripts/loop_status.py --set-state running --clear-stop-after-iteration` to resume.",
        "- The frontier showcase loop remains separate and should not drive main-loop keep/discard decisions.",
        "",
    ])
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Write a human-readable markdown report for the current Ternary Lab loop state.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--state-path", default=".ternary_lab/state.json")
    parser.add_argument("--control-path", default=".ternary_lab/control.json")
    parser.add_argument("--output", default=".ternary_lab/loop_report.md")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root)
    state = load_state(repo_root / args.state_path)
    control = load_control(repo_root / args.control_path, state.get("objective", DEFAULT_OBJECTIVE))
    summary = summarize_state(state, control)
    report_path = repo_root / args.output
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_markdown(summary) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()