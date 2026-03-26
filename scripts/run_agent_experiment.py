#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def json_output(payload, exit_code=0):
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


def run_git_helper(helper_script, repo_root, *args):
    completed = subprocess.run(
        [sys.executable, str(helper_script), "--repo", str(repo_root), *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(completed.stdout or "{}")
    if completed.returncode != 0 or not payload.get("ok"):
        raise RuntimeError(payload.get("error", f"git helper failed: {' '.join(args)}"))
    return payload


def run_command(repo_root, command, log_path):
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return completed.returncode, time.time() - start


def parse_args():
    parser = argparse.ArgumentParser(description="Run one safe agent experiment with git, log, and optional revert handling.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--git-helper-script", default="scripts/git_agent.py")
    parser.add_argument("--branch-tag", default="")
    parser.add_argument("--branch-prefix", default="autoresearch")
    parser.add_argument("--allow-dirty-branch-switch", action="store_true")
    parser.add_argument("--commit-message", default="")
    parser.add_argument("--commit-paths", nargs="*", default=[])
    parser.add_argument("--commit-all", action="store_true")
    parser.add_argument("--log-path", default=".ternary_lab/runs/manual-experiment.log")
    parser.add_argument("--revert-on-failure", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    helper_script = Path(args.git_helper_script)
    if not helper_script.is_absolute():
        helper_script = (repo_root / helper_script).resolve()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        return json_output({"ok": False, "error": "no command provided"}, exit_code=1)

    try:
        status_before = run_git_helper(helper_script, repo_root, "status")
        branch_result = None
        warnings = []
        if args.branch_tag:
            if status_before["dirty"] and not args.allow_dirty_branch_switch:
                branch_result = {
                    "action": "skipped",
                    "reason": "dirty worktree",
                    "branch": status_before["branch"],
                }
                warnings.append("branch switch skipped because the worktree is dirty")
            else:
                branch_result = run_git_helper(
                    helper_script,
                    repo_root,
                    "ensure-branch",
                    "--tag",
                    args.branch_tag,
                    "--prefix",
                    args.branch_prefix,
                )

        commit_result = None
        if args.commit_message:
            commit_args = ["commit", "--message", args.commit_message]
            if args.commit_all:
                commit_args.append("--all")
            elif args.commit_paths:
                commit_args.extend(["--paths", *args.commit_paths])
            else:
                raise RuntimeError("commit requested but no --commit-paths or --commit-all was provided")
            commit_result = run_git_helper(helper_script, repo_root, *commit_args)

        log_path = Path(args.log_path)
        if not log_path.is_absolute():
            log_path = repo_root / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        exit_code, duration_seconds = run_command(repo_root, command, log_path)

        revert_result = None
        if exit_code != 0 and args.revert_on_failure and commit_result is not None:
            revert_result = run_git_helper(
                helper_script,
                repo_root,
                "revert",
                "--commit",
                commit_result["commit"],
                "--allow-dirty",
            )

        payload = {
            "ok": exit_code == 0,
            "status_before": status_before,
            "branch": branch_result,
            "commit": commit_result,
            "command": command,
            "log_path": str(log_path),
            "duration_seconds": duration_seconds,
            "exit_code": exit_code,
            "reverted": revert_result,
            "warnings": warnings,
        }
        return json_output(payload, exit_code=0 if exit_code == 0 else exit_code)
    except RuntimeError as exc:
        return json_output({"ok": False, "error": str(exc)}, exit_code=1)


if __name__ == "__main__":
    sys.exit(main())