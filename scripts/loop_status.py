#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ternary_lab import DEFAULT_OBJECTIVE, load_control, load_state, save_control, summarize_state


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect or update the main Ternary Lab loop state and human control file.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--state-path", default=".ternary_lab/state.json")
    parser.add_argument("--control-path", default=".ternary_lab/control.json")
    parser.add_argument("--set-state", choices=["running", "paused", "stopped"])
    parser.add_argument("--objective-override", default=None)
    parser.add_argument("--clear-objective-override", action="store_true")
    parser.add_argument("--note", default=None)
    parser.add_argument("--stop-after-iteration", action="store_true")
    parser.add_argument("--clear-stop-after-iteration", action="store_true")
    parser.add_argument("--updated-by", default="human")
    parser.add_argument("--format", choices=["json", "text"], default="json")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root)
    state_path = repo_root / args.state_path
    control_path = repo_root / args.control_path

    state = load_state(state_path)
    control = load_control(control_path, state.get("objective", DEFAULT_OBJECTIVE))

    changed = False
    if args.set_state is not None:
        control["desired_state"] = args.set_state
        changed = True
    if args.objective_override is not None:
        control["objective_override"] = args.objective_override
        changed = True
    if args.clear_objective_override:
        control["objective_override"] = ""
        changed = True
    if args.note is not None:
        control["human_note"] = args.note
        changed = True
    if args.stop_after_iteration:
        control["stop_after_iteration"] = True
        changed = True
    if args.clear_stop_after_iteration:
        control["stop_after_iteration"] = False
        changed = True

    if changed:
        control["updated_at"] = int(time.time())
        control["updated_by"] = args.updated_by
        save_control(control_path, control)

    summary = summarize_state(state, control)
    summary["repo_root"] = str(repo_root.resolve())
    if args.format == "text":
        print("---")
        print(f"history_length:         {summary['history_length']}")
        print(f"objective:              {summary['objective']}")
        print(f"desired_state:          {summary['control']['desired_state']}")
        print(f"stop_after_iteration:   {summary['control']['stop_after_iteration']}")
        print(f"best_run_id:            {summary['best_run_id']}")
        print(f"last_run_id:            {summary['last_run_id']}")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()