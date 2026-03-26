"""Generate comparison tables and plots for CPU BitNet and GPU agent runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


NUMERIC_COLUMNS = [
    "val_bpb",
    "memory_gb",
    "energy_j_per_token",
    "tokens_per_second",
    "avg_cpu_process_percent",
    "avg_cpu_load_percent",
    "avg_gpu_util_percent",
    "avg_gpu_mem_used_mb",
    "reliability_score",
    "capability_score",
    "task_eval_score",
]


def load_results_tsv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            row["source_path"] = str(path)
            row["source_kind"] = "results_tsv"
            rows.append(row)
    return rows


def load_summary_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["source_path"] = str(path)
    payload["source_kind"] = "summary_json"
    return [payload]


def collect_rows(result_paths=None, summary_paths=None, agent_dirs=None):
    rows = []
    for path in result_paths or []:
        rows.extend(load_results_tsv(path))
    for path in summary_paths or []:
        rows.extend(load_summary_json(path))
    for agent_dir in agent_dirs or []:
        agent_path = Path(agent_dir)
        results_path = agent_path / "results.tsv"
        report_path = agent_path / "deployment_report.json"
        if results_path.exists():
            rows.extend(load_results_tsv(results_path))
        elif report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("best"):
                best = report["best"]["metrics"].copy()
                best["description"] = report["best"]["candidate"]["description"]
                best["source_path"] = str(report_path)
                best["source_kind"] = "deployment_report"
                rows.append(best)
    return rows


def to_frame(rows):
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        else:
            frame[column] = 0.0
    if "device" not in frame.columns:
        frame["device"] = "unknown"
    if "linear_impl" not in frame.columns:
        frame["linear_impl"] = "unknown"
    if "description" not in frame.columns:
        frame["description"] = ""
    frame["label"] = frame.apply(
        lambda row: f"{row['device']}:{row['linear_impl']}:{Path(row['source_path']).stem}",
        axis=1,
    )
    return frame


def build_markdown(frame, title):
    columns = [
        "label",
        "device",
        "linear_impl",
        "task_eval_score",
        "reliability_score",
        "capability_score",
        "tokens_per_second",
        "energy_j_per_token",
        "avg_cpu_process_percent",
        "avg_gpu_util_percent",
        "memory_gb",
        "val_bpb",
    ]
    view = frame[columns].copy().sort_values(["device", "task_eval_score", "tokens_per_second"], ascending=[True, False, False])
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in view.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return f"# {title}\n\n" + "\n".join([header, divider, *rows])


def save_bar_chart(frame, output_path):
    ordered = frame.sort_values("tokens_per_second", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ordered["label"], ordered["tokens_per_second"], color=["#2f855a" if d == "cpu" else "#2b6cb0" for d in ordered["device"]])
    ax.set_ylabel("Tokens / second")
    ax.set_title("Agent throughput")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_resource_chart(frame, output_path):
    ordered = frame.sort_values("label")
    x = range(len(ordered))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, ordered["avg_cpu_process_percent"], marker="o", label="CPU process %")
    ax.plot(x, ordered["avg_gpu_util_percent"], marker="o", label="GPU util %")
    ax.plot(x, ordered["avg_gpu_mem_used_mb"], marker="o", label="GPU mem MB")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ordered["label"], rotation=35, ha="right")
    ax.set_title("Resource utilization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_efficiency_chart(frame, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = frame["device"].map({"cpu": "#2f855a", "cuda": "#2b6cb0"}).fillna("#4a5568")
    ax.scatter(frame["energy_j_per_token"], frame["tokens_per_second"], c=colors)
    for _, row in frame.iterrows():
        ax.annotate(row["label"], (row["energy_j_per_token"], row["tokens_per_second"]), fontsize=8)
    ax.set_xlabel("Energy / token (J)")
    ax.set_ylabel("Tokens / second")
    ax.set_title("Efficiency vs throughput")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_skeptic_summary(frame, title):
    lines = [f"# {title}", "", "## Measured Locally", ""]
    cpu_rows = frame[frame["device"] == "cpu"]
    gpu_rows = frame[frame["device"] == "cuda"]
    if not cpu_rows.empty:
        best_cpu = cpu_rows.sort_values(["task_eval_score", "tokens_per_second"], ascending=[False, False]).iloc[0]
        lines.extend(
            [
                "### CPU BitNet evidence",
                f"- label: {best_cpu['label']}",
                f"- task_eval_score: {best_cpu['task_eval_score']:.4f}",
                f"- reliability_score: {best_cpu['reliability_score']:.4f}",
                f"- capability_score: {best_cpu['capability_score']:.4f}",
                f"- tokens_per_second: {best_cpu['tokens_per_second']:.2f}",
                f"- energy_j_per_token: {best_cpu['energy_j_per_token']:.6f}",
                f"- memory_gb: {best_cpu['memory_gb']:.2f}",
                f"- avg_cpu_process_percent: {best_cpu['avg_cpu_process_percent']:.2f}",
                f"- val_bpb: {best_cpu['val_bpb']:.6f}",
                "",
            ]
        )
    if not gpu_rows.empty:
        best_gpu = gpu_rows.sort_values("tokens_per_second", ascending=False).iloc[0]
        lines.extend(
            [
                "### GPU baseline evidence",
                f"- label: {best_gpu['label']}",
                f"- tokens_per_second: {best_gpu['tokens_per_second']:.2f}",
                f"- energy_j_per_token: {best_gpu['energy_j_per_token']:.6f}",
                f"- memory_gb: {best_gpu['memory_gb']:.2f}",
                f"- avg_gpu_util_percent: {best_gpu['avg_gpu_util_percent']:.2f}",
                f"- avg_gpu_mem_used_mb: {best_gpu['avg_gpu_mem_used_mb']:.2f}",
                f"- val_bpb: {best_gpu['val_bpb']:.6f}",
                "",
            ]
        )
    if not cpu_rows.empty and not gpu_rows.empty:
        cpu_best = cpu_rows.sort_values(["task_eval_score", "tokens_per_second"], ascending=[False, False]).iloc[0]
        gpu_best = gpu_rows.sort_values("tokens_per_second", ascending=False).iloc[0]
        throughput_ratio = cpu_best["tokens_per_second"] / max(gpu_best["tokens_per_second"], 1e-9)
        energy_ratio = cpu_best["energy_j_per_token"] / max(gpu_best["energy_j_per_token"], 1e-9)
        memory_ratio = cpu_best["memory_gb"] / max(gpu_best["memory_gb"], 1e-9)
        lines.extend(
            [
                "## Relative comparison",
                f"- cpu_task_eval_score: {cpu_best['task_eval_score']:.4f}",
                f"- cpu_vs_gpu_throughput_ratio: {throughput_ratio:.4f}",
                f"- cpu_vs_gpu_energy_ratio: {energy_ratio:.4f}",
                f"- cpu_vs_gpu_memory_ratio: {memory_ratio:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## What this does prove",
            "- The exact throughput, process memory, and runtime utilization observed on this machine for the recorded runs.",
            "- The local research-loop task evaluation score, reliability score, and capability score for runs that include those fields.",
            "- Whether the CPU BitNet path can complete the fixed 5-minute loop and log measurable outputs locally.",
            "- Whether the GPU baseline consumes materially different memory, utilization, and energy-per-token on the same repository code path.",
            "",
            "## What this does not prove by itself",
            "- Frontier-model parity against proprietary systems such as GPT-4 or Claude.",
            "- GSM8K, WinoGrande, or ARC-Challenge reasoning parity unless those evaluations are run and recorded separately.",
            "- Claims about 70B or 100B deployment unless those exact models are measured on the target hardware.",
            "- That the separate frontier showcase loop should override main-loop keep or discard decisions.",
            "",
            "## Skeptic check",
            "- Re-run the commands in the benchmark protocol and compare the regenerated CSV, markdown table, and plots.",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_report(frame, output_dir, title):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "comparison.csv"
    md_path = output_path / "comparison.md"
    skeptic_md_path = output_path / "skeptic_summary.md"
    throughput_png = output_path / "throughput.png"
    resource_png = output_path / "resource_utilization.png"
    efficiency_png = output_path / "efficiency.png"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(build_markdown(frame, title), encoding="utf-8")
    skeptic_md_path.write_text(build_skeptic_summary(frame, title), encoding="utf-8")
    save_bar_chart(frame, throughput_png)
    save_resource_chart(frame, resource_png)
    save_efficiency_chart(frame, efficiency_png)
    return {
        "csv": csv_path,
        "markdown": md_path,
        "skeptic_markdown": skeptic_md_path,
        "throughput_png": throughput_png,
        "resource_png": resource_png,
        "efficiency_png": efficiency_png,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare CPU BitNet and GPU agent runs.")
    parser.add_argument("--results-tsv", nargs="*", default=[])
    parser.add_argument("--summary-json", nargs="*", default=[])
    parser.add_argument("--agent-dir", nargs="*", default=[])
    parser.add_argument("--output-dir", default="comparison_reports/latest")
    parser.add_argument("--title", default="CPU BitNet vs GPU agent comparison")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = collect_rows(args.results_tsv, args.summary_json, args.agent_dir)
    frame = to_frame(rows)
    if frame.empty:
        raise SystemExit("No comparison inputs found.")
    outputs = generate_report(frame, args.output_dir, args.title)
    print("---")
    print(f"rows:             {len(frame)}")
    print(f"markdown:         {outputs['markdown']}")
    print(f"skeptic_md:       {outputs['skeptic_markdown']}")
    print(f"csv:              {outputs['csv']}")
    print(f"throughput_png:   {outputs['throughput_png']}")
    print(f"resource_png:     {outputs['resource_png']}")
    print(f"efficiency_png:   {outputs['efficiency_png']}")


if __name__ == "__main__":
    main()