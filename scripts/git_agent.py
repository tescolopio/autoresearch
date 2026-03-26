#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


BRANCH_NAME_RE = re.compile(r"^[A-Za-z0-9._/-]+$")


class GitCommandError(RuntimeError):
    def __init__(self, message: str, *, command: list[str], returncode: int, stderr: str) -> None:
        super().__init__(message)
        self.command = command
        self.returncode = returncode
        self.stderr = stderr


def git(repo: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = ["git", *args]
    result = subprocess.run(
        command,
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.strip()
        raise GitCommandError(
            stderr or "git command failed",
            command=command,
            returncode=result.returncode,
            stderr=stderr,
        )
    return result


def json_output(payload: dict[str, object], *, exit_code: int = 0) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


def resolve_repo(path_value: str) -> Path:
    repo = Path(path_value).resolve()
    if not repo.exists():
        raise FileNotFoundError(f"repository path does not exist: {repo}")
    git(repo, ["rev-parse", "--is-inside-work-tree"])
    root = git(repo, ["rev-parse", "--show-toplevel"]).stdout.strip()
    return Path(root)


def validate_branch_name(branch: str) -> str:
    if not BRANCH_NAME_RE.match(branch):
        raise ValueError(f"invalid branch name: {branch}")
    if ".." in branch or branch.startswith("/") or branch.endswith("/"):
        raise ValueError(f"invalid branch name: {branch}")
    return branch


def parse_status_lines(repo: Path) -> list[dict[str, str]]:
    status = git(repo, ["status", "--short"]).stdout.splitlines()
    rows: list[dict[str, str]] = []
    for line in status:
        if not line:
            continue
        code = line[:2]
        path_text = line[3:]
        rows.append(
            {
                "index_status": code[0],
                "worktree_status": code[1],
                "path": path_text,
            }
        )
    return rows


def current_branch(repo: Path) -> str:
    return git(repo, ["branch", "--show-current"]).stdout.strip()


def has_remote(repo: Path, remote_name: str) -> bool:
    result = git(repo, ["remote", "get-url", remote_name], check=False)
    return result.returncode == 0


def ensure_clean_worktree(repo: Path) -> None:
    if parse_status_lines(repo):
        raise RuntimeError("worktree is not clean")


def command_status(args: argparse.Namespace) -> int:
    repo = resolve_repo(args.repo)
    branch = current_branch(repo)
    rows = parse_status_lines(repo)
    payload = {
        "ok": True,
        "repo_root": str(repo),
        "branch": branch,
        "dirty": bool(rows),
        "changes": rows,
        "has_origin": has_remote(repo, "origin"),
    }
    return json_output(payload)


def command_ensure_branch(args: argparse.Namespace) -> int:
    repo = resolve_repo(args.repo)
    branch = args.branch
    if branch is None:
        if not args.tag:
            raise ValueError("either --branch or --tag is required")
        branch = f"{args.prefix}/{args.tag}"
    branch = validate_branch_name(branch)

    exists = git(repo, ["show-ref", "--verify", f"refs/heads/{branch}"], check=False).returncode == 0
    if exists:
        git(repo, ["checkout", branch])
        action = "switched"
    else:
        git(repo, ["checkout", "-b", branch])
        action = "created"

    published = False
    if args.publish:
        if not has_remote(repo, args.remote):
            raise RuntimeError(f"remote '{args.remote}' is not configured")
        git(repo, ["push", "-u", args.remote, branch])
        published = True

    payload = {
        "ok": True,
        "repo_root": str(repo),
        "branch": branch,
        "action": action,
        "published": published,
        "remote": args.remote if published else None,
    }
    return json_output(payload)


def command_commit(args: argparse.Namespace) -> int:
    repo = resolve_repo(args.repo)
    if not args.all and not args.paths:
        raise ValueError("provide --paths or use --all")

    if args.all:
        git(repo, ["add", "-A"])
    else:
        git(repo, ["add", "--", *args.paths])

    staged = git(repo, ["diff", "--cached", "--name-only"]).stdout.splitlines()
    if not staged:
        raise RuntimeError("no staged changes to commit")

    git(repo, ["commit", "-m", args.message])
    commit_sha = git(repo, ["rev-parse", "HEAD"]).stdout.strip()
    payload = {
        "ok": True,
        "repo_root": str(repo),
        "branch": current_branch(repo),
        "commit": commit_sha,
        "staged_paths": staged,
        "message": args.message,
    }
    return json_output(payload)


def command_publish(args: argparse.Namespace) -> int:
    repo = resolve_repo(args.repo)
    branch = args.branch or current_branch(repo)
    branch = validate_branch_name(branch)
    if not has_remote(repo, args.remote):
        raise RuntimeError(f"remote '{args.remote}' is not configured")
    git(repo, ["push", "-u", args.remote, branch])
    payload = {
        "ok": True,
        "repo_root": str(repo),
        "branch": branch,
        "remote": args.remote,
        "published": True,
    }
    return json_output(payload)


def command_revert(args: argparse.Namespace) -> int:
    repo = resolve_repo(args.repo)
    if not args.allow_dirty:
        ensure_clean_worktree(repo)
    git(repo, ["revert", "--no-edit", args.commit])
    revert_sha = git(repo, ["rev-parse", "HEAD"]).stdout.strip()
    payload = {
        "ok": True,
        "repo_root": str(repo),
        "branch": current_branch(repo),
        "reverted_commit": args.commit,
        "revert_commit": revert_sha,
    }
    return json_output(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe non-interactive git helpers for autoresearch agents.")
    parser.add_argument("--repo", default=".", help="Path inside the target git repository.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="Emit current branch and worktree status as JSON.")
    status_parser.set_defaults(func=command_status)

    branch_parser = subparsers.add_parser("ensure-branch", help="Create or switch to a branch for an experiment run.")
    branch_parser.add_argument("--tag", help="Run tag that becomes <prefix>/<tag>.")
    branch_parser.add_argument("--branch", help="Explicit branch name to create or switch to.")
    branch_parser.add_argument("--prefix", default="autoresearch", help="Branch prefix used with --tag.")
    branch_parser.add_argument("--publish", action="store_true", help="Push the branch to the remote after switching.")
    branch_parser.add_argument("--remote", default="origin", help="Remote name to use when publishing.")
    branch_parser.set_defaults(func=command_ensure_branch)

    commit_parser = subparsers.add_parser("commit", help="Stage changes and create a non-interactive experiment commit.")
    commit_parser.add_argument("--message", required=True, help="Commit message.")
    commit_parser.add_argument("--paths", nargs="*", default=[], help="Specific paths to stage before committing.")
    commit_parser.add_argument("--all", action="store_true", help="Stage all tracked and untracked changes.")
    commit_parser.set_defaults(func=command_commit)

    publish_parser = subparsers.add_parser("publish", help="Push the current or requested branch to a remote.")
    publish_parser.add_argument("--branch", help="Branch to publish. Defaults to current branch.")
    publish_parser.add_argument("--remote", default="origin", help="Remote name to use when publishing.")
    publish_parser.set_defaults(func=command_publish)

    revert_parser = subparsers.add_parser("revert", help="Create a revert commit for a prior experiment commit.")
    revert_parser.add_argument("--commit", required=True, help="Commit SHA to revert.")
    revert_parser.add_argument("--allow-dirty", action="store_true", help="Allow revert even if the worktree is dirty.")
    revert_parser.set_defaults(func=command_revert)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except (FileNotFoundError, GitCommandError, RuntimeError, ValueError) as exc:
        payload = {
            "ok": False,
            "error": str(exc),
        }
        if isinstance(exc, GitCommandError):
            payload["command"] = exc.command
            payload["returncode"] = exc.returncode
            payload["stderr"] = exc.stderr
        return json_output(payload, exit_code=1)


if __name__ == "__main__":
    sys.exit(main())