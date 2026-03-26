#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def run_json_command(command, cwd):
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"command failed: {' '.join(command)}")
    stdout = completed.stdout.strip()
    payload = json.loads(stdout) if stdout.startswith("{") else stdout
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object output from command: {' '.join(command)}")
    return payload


def run_command(command, cwd):
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"command failed: {' '.join(command)}")
    return completed.stdout.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a supervised local BitNet run with branch setup and report refresh after each iteration.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--branch-tag", required=True)
    parser.add_argument("--branch-prefix", default="autoresearch")
    parser.add_argument("--git-helper-script", default="scripts/git_agent.py")
    parser.add_argument("--loop-status-script", default="scripts/loop_status.py")
    parser.add_argument("--loop-report-script", default="scripts/loop_report.py")
    parser.add_argument("--trainer-backend", choices=["train", "mock"], default="mock")
    parser.add_argument("--agent-name", default="cpu-agent-1")
    parser.add_argument("--objective-override", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--no-manage-git", action="store_true")
    parser.add_argument("--snapshot-dir", default=".ternary_lab/snapshots")
    parser.add_argument("--snapshot-prefix", default="iteration")
    parser.add_argument("--commit-snapshots", action="store_true")
    parser.add_argument("--commit-message-prefix", default="audit")
    parser.add_argument("--roadmap-doctor-script", default="scripts/roadmap_doctor.py")
    parser.add_argument("--skip-milestone-gate", action="store_true")
    return parser.parse_args()


def write_snapshot(repo_root, report_path, iteration_index, snapshot_dir, snapshot_prefix):
    snapshot_root = repo_root / snapshot_dir
    snapshot_root.mkdir(parents=True, exist_ok=True)
    stamped_name = f"{snapshot_prefix}-{iteration_index:03d}.md"
    snapshot_path = snapshot_root / stamped_name
    shutil.copyfile(report_path, snapshot_path)
    return snapshot_path


def commit_snapshot(git_helper, repo_root, snapshot_path, commit_message):
    return run_json_command(
        [
            sys.executable,
            str(git_helper),
            "--repo",
            str(repo_root),
            "commit",
            "--message",
            commit_message,
            "--paths",
            str(snapshot_path.relative_to(repo_root)),
        ],
        cwd=repo_root,
    )


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    git_helper = repo_root / args.git_helper_script
    loop_status = repo_root / args.loop_status_script
    loop_report = repo_root / args.loop_report_script
    roadmap_doctor = repo_root / args.roadmap_doctor_script
    ternary_lab = repo_root / "ternary_lab.py"

    if not args.skip_milestone_gate:
        required_milestone = "single-iteration-research-loop" if args.trainer_backend == "train" else "local-agent-lifecycle"
        subprocess.run(
            [
                sys.executable,
                str(roadmap_doctor),
                "--repo-root",
                str(repo_root),
                "--require-complete-through",
                required_milestone,
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )

    branch = None
    if not args.no_manage_git:
        branch_result: dict[str, Any] = run_json_command(
            [sys.executable, str(git_helper), "--repo", str(repo_root), "ensure-branch", "--tag", args.branch_tag, "--prefix", args.branch_prefix],
            cwd=repo_root,
        )
        branch = branch_result.get("branch")

    status_command = [sys.executable, str(loop_status), "--repo-root", str(repo_root), "--set-state", "running", "--clear-stop-after-iteration"]
    if args.objective_override:
        status_command.extend(["--objective-override", args.objective_override])
    if args.note:
        status_command.extend(["--note", args.note])
    run_command(status_command, cwd=repo_root)

    reports = []
    snapshots = []
    snapshot_commits = []
    iterations_completed = 0
    last_summary = None
    while iterations_completed < args.max_iterations:
        state_before: dict[str, Any] = run_json_command([sys.executable, str(loop_status), "--repo-root", str(repo_root)], cwd=repo_root)
        desired_state = state_before["control"]["desired_state"]
        if desired_state != "running":
            last_summary = state_before
            break

        run_command(
            [
                sys.executable,
                str(ternary_lab),
                "--repo-root",
                str(repo_root),
                "--iterations",
                "1",
                "--trainer-backend",
                args.trainer_backend,
                "--agent-name",
                args.agent_name,
                "--git-branch-tag",
                args.branch_tag,
                "--timeout-seconds",
                str(args.timeout_seconds),
                *( ["--no-manage-git"] if args.no_manage_git else [] ),
            ],
            cwd=repo_root,
        )
        report_path = run_command([sys.executable, str(loop_report), "--repo-root", str(repo_root)], cwd=repo_root)
        reports.append(report_path)
        iterations_completed += 1
        snapshot_path = write_snapshot(
            repo_root,
            Path(report_path),
            iterations_completed,
            args.snapshot_dir,
            args.snapshot_prefix,
        )
        snapshots.append(str(snapshot_path))
        if args.commit_snapshots and not args.no_manage_git:
            commit_result = commit_snapshot(
                git_helper,
                repo_root,
                snapshot_path,
                f"{args.commit_message_prefix}: supervised snapshot {iterations_completed} ({args.branch_tag})",
            )
            snapshot_commits.append(commit_result)
        last_summary = run_json_command([sys.executable, str(loop_status), "--repo-root", str(repo_root)], cwd=repo_root)
        if last_summary["control"]["desired_state"] != "running":
            break

    payload = {
        "ok": True,
        "repo_root": str(repo_root),
        "branch": branch,
        "iterations_completed": iterations_completed,
        "report_paths": reports,
        "snapshot_paths": snapshots,
        "snapshot_commits": snapshot_commits,
        "last_summary": last_summary,
        "finished_at": int(time.time()),
        "milestone_gate": None if args.skip_milestone_gate else required_milestone,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()