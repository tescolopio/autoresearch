"""Run repeatable CPU/GPU baseline benchmarks and generate a comparison report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from compare_agents import collect_rows, generate_report, to_frame


def run_command(command, cwd, timeout):
    subprocess.run(command, cwd=cwd, check=True, timeout=timeout)


def benchmark_paths(output_dir):
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    return {
        "cpu_json": base / "cpu_bitnet.json",
        "cpu_tsv": base / "cpu_bitnet.tsv",
        "gpu_json": base / "gpu_dense.json",
        "gpu_tsv": base / "gpu_dense.tsv",
    }


def write_summary(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def build_mock_gpu_summary(label="gpu dense baseline"):
    return {
        "commit": "mock-gpu",
        "val_bpb": 0.982,
        "memory_gb": 14.6,
        "status": "keep",
        "description": label,
        "device": "cuda",
        "linear_impl": "dense",
        "signature_verified": False,
        "energy_j_per_token": 0.194,
        "tokens_per_second": 21.4,
        "avg_cpu_process_percent": 8.0,
        "avg_cpu_load_percent": 11.0,
        "avg_gpu_util_percent": 82.0,
        "avg_gpu_mem_used_mb": 14950.0,
    }


def run_cpu_baseline(repo_root, output_dir, timeout, extra_args=None):
    paths = benchmark_paths(output_dir)
    cmd = [
        sys.executable,
        "train.py",
        "--device",
        "cpu",
        "--cpu-only",
        "--cpu-bitnet-poc",
        "--summary-json",
        str(paths["cpu_json"]),
        "--results-tsv",
        str(paths["cpu_tsv"]),
        "--description",
        "cpu bitnet baseline",
    ]
    cmd.extend(extra_args or [])
    run_command(cmd, repo_root, timeout)
    return paths


def run_gpu_baseline(repo_root, output_dir, timeout, extra_args=None):
    paths = benchmark_paths(output_dir)
    cmd = [
        sys.executable,
        "train.py",
        "--device",
        "cuda",
        "--allow-accelerator",
        "--linear-impl",
        "dense",
        "--summary-json",
        str(paths["gpu_json"]),
        "--results-tsv",
        str(paths["gpu_tsv"]),
        "--description",
        "gpu dense baseline",
    ]
    cmd.extend(extra_args or [])
    run_command(cmd, repo_root, timeout)
    return paths


def materialize_mock_gpu_baseline(output_dir, description="gpu dense baseline"):
    paths = benchmark_paths(output_dir)
    write_summary(paths["gpu_json"], build_mock_gpu_summary(description))
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description="Run skeptic-grade CPU/GPU benchmark comparisons.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="comparison_reports/benchmark")
    parser.add_argument("--report-dir", default="comparison_reports/latest")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--cpu-summary-json", default="")
    parser.add_argument("--cpu-agent-dir", default="")
    parser.add_argument("--gpu-summary-json", default="")
    parser.add_argument("--mock-gpu", action="store_true")
    parser.add_argument("--cpu-arg", action="append", default=[])
    parser.add_argument("--gpu-arg", action="append", default=[])
    parser.add_argument("--title", default="CPU BitNet vs GPU agent comparison")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root)
    benchmark_output = benchmark_paths(args.output_dir)
    summary_paths = []
    agent_dirs = []

    if args.cpu_summary_json:
        summary_paths.append(Path(args.cpu_summary_json))
    elif args.cpu_agent_dir:
        agent_dirs.append(args.cpu_agent_dir)
    elif not args.skip_cpu:
        run_cpu_baseline(repo_root, args.output_dir, args.timeout_seconds, args.cpu_arg)

    if args.gpu_summary_json:
        summary_paths.append(Path(args.gpu_summary_json))
    elif args.mock_gpu:
        materialize_mock_gpu_baseline(args.output_dir)
    elif not args.skip_gpu:
        run_gpu_baseline(repo_root, args.output_dir, args.timeout_seconds, args.gpu_arg)

    if benchmark_output["cpu_json"].exists():
        summary_paths.append(benchmark_output["cpu_json"])
    if benchmark_output["gpu_json"].exists():
        summary_paths.append(benchmark_output["gpu_json"])
    rows = collect_rows(summary_paths=summary_paths, agent_dirs=agent_dirs)
    frame = to_frame(rows)
    if frame.empty:
        raise SystemExit("No benchmark results available to compare.")
    outputs = generate_report(frame, args.report_dir, args.title)
    print("---")
    print(f"cpu_json:         {benchmark_output['cpu_json']}")
    print(f"gpu_json:         {benchmark_output['gpu_json']}")
    print(f"comparison_md:    {outputs['markdown']}")
    print(f"skeptic_md:       {outputs['skeptic_markdown']}")
    print(f"throughput_png:   {outputs['throughput_png']}")


if __name__ == "__main__":
    main()