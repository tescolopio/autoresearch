"""Run the same task suite against frontier APIs and a local BitNet-compatible provider."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests


@dataclass
class ProviderConfig:
    name: str
    type: str
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    command: str = ""
    system_prompt: str = ""
    temperature: float = 0.0
    max_tokens: int = 512
    mock_responses: dict | None = None


def load_config(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    providers = [ProviderConfig(**provider) for provider in payload["providers"]]
    tasks = payload["tasks"]
    return providers, tasks


def estimate_tokens(text):
    return max(len(text.split()), 1) if text else 0


def task_passes(task, response_text):
    text = (response_text or "").lower()
    required = [item.lower() for item in task.get("must_contain", [])]
    forbidden = [item.lower() for item in task.get("must_not_contain", [])]
    if any(item not in text for item in required):
        return False
    if any(item in text for item in forbidden):
        return False
    return True


def call_openai_like(provider, prompt, system_prompt=None):
    api_key = os.getenv(provider.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env: {provider.api_key_env}")
    base_url = provider.base_url.rstrip("/")
    payload = {
        "model": provider.model,
        "messages": [
            *([{"role": "system", "content": system_prompt or provider.system_prompt}] if (system_prompt or provider.system_prompt) else []),
            {"role": "user", "content": prompt},
        ],
        "temperature": provider.temperature,
        "max_tokens": provider.max_tokens,
    }
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", estimate_tokens(content))


def call_anthropic(provider, prompt, system_prompt=None):
    api_key = os.getenv(provider.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env: {provider.api_key_env}")
    response = requests.post(
        provider.base_url or "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": provider.model,
            "system": system_prompt or provider.system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": provider.temperature,
            "max_tokens": provider.max_tokens,
        },
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()
    content = "".join(block.get("text", "") for block in body.get("content", []))
    usage = body.get("usage", {})
    return content, usage.get("input_tokens", 0), usage.get("output_tokens", estimate_tokens(content))


def call_gemini(provider, prompt, system_prompt=None):
    api_key = os.getenv(provider.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env: {provider.api_key_env}")
    base_url = provider.base_url or "https://generativelanguage.googleapis.com/v1beta/models"
    url = f"{base_url}/{provider.model}:generateContent?key={api_key}"
    parts = []
    if system_prompt or provider.system_prompt:
        parts.append({"text": system_prompt or provider.system_prompt})
    parts.append({"text": prompt})
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": provider.temperature, "maxOutputTokens": provider.max_tokens},
        },
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()
    candidate = body["candidates"][0]
    content = "".join(part.get("text", "") for part in candidate.get("content", {}).get("parts", []))
    usage = body.get("usageMetadata", {})
    return content, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", estimate_tokens(content))


def call_local_command(provider, prompt, system_prompt=None):
    command = shlex.split(provider.command)
    full_prompt = prompt if not (system_prompt or provider.system_prompt) else f"{system_prompt or provider.system_prompt}\n\n{prompt}"
    completed = subprocess.run(command, input=full_prompt, text=True, capture_output=True, check=True, timeout=120)
    content = completed.stdout.strip()
    return content, estimate_tokens(full_prompt), estimate_tokens(content)


def call_mock(provider, task_id):
    responses = provider.mock_responses or {}
    content = responses.get(task_id, f"mock response for {task_id}")
    return content, estimate_tokens(task_id), estimate_tokens(content)


def run_provider_task(provider, task):
    start = time.time()
    if provider.type in {"openai", "openai-compatible"}:
        content, prompt_tokens, completion_tokens = call_openai_like(provider, task["prompt"], task.get("system_prompt"))
    elif provider.type == "anthropic":
        content, prompt_tokens, completion_tokens = call_anthropic(provider, task["prompt"], task.get("system_prompt"))
    elif provider.type == "gemini":
        content, prompt_tokens, completion_tokens = call_gemini(provider, task["prompt"], task.get("system_prompt"))
    elif provider.type == "local-command":
        content, prompt_tokens, completion_tokens = call_local_command(provider, task["prompt"], task.get("system_prompt"))
    elif provider.type == "mock":
        content, prompt_tokens, completion_tokens = call_mock(provider, task["id"])
    else:
        raise RuntimeError(f"Unsupported provider type: {provider.type}")
    latency = time.time() - start
    return {
        "provider": provider.name,
        "provider_type": provider.type,
        "model": provider.model,
        "task_id": task["id"],
        "latency_seconds": latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_second": completion_tokens / max(latency, 1e-9),
        "pass": task_passes(task, content),
        "response": content,
    }


def run_suite(providers, tasks):
    rows = []
    for provider in providers:
        for task in tasks:
            rows.append(run_provider_task(provider, task))
    return pd.DataFrame(rows)


def save_outputs(frame, output_dir, title):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_jsonl = output / "responses.jsonl"
    csv_path = output / "provider_summary.csv"
    md_path = output / "provider_summary.md"
    chart_path = output / "provider_latency.png"
    with open(raw_jsonl, "w", encoding="utf-8") as handle:
        for record in frame.to_dict(orient="records"):
            handle.write(json.dumps(record) + "\n")
    frame.to_csv(csv_path, index=False)
    summary = frame.groupby(["provider", "provider_type", "model"], dropna=False).agg(
        avg_latency_seconds=("latency_seconds", "mean"),
        avg_tokens_per_second=("tokens_per_second", "mean"),
        pass_rate=("pass", "mean"),
    ).reset_index()
    md_lines = [f"# {title}", "", "| provider | provider_type | model | avg_latency_seconds | avg_tokens_per_second | pass_rate |", "| --- | --- | --- | --- | --- | --- |"]
    for _, row in summary.iterrows():
        md_lines.append(
            f"| {row['provider']} | {row['provider_type']} | {row['model']} | {row['avg_latency_seconds']:.3f} | {row['avg_tokens_per_second']:.3f} | {row['pass_rate']:.3f} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary["provider"], summary["avg_tokens_per_second"], color="#2f855a")
    ax.set_ylabel("Average output tokens / second")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    return {"jsonl": raw_jsonl, "csv": csv_path, "markdown": md_path, "chart": chart_path}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a separate showcase loop comparing frontier API providers with a local BitNet-compatible provider on the same tasks."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="provider_reports/latest")
    parser.add_argument("--title", default="Showcase loop: frontier API vs local BitNet task comparison")
    return parser.parse_args()


def main():
    args = parse_args()
    providers, tasks = load_config(args.config)
    frame = run_suite(providers, tasks)
    outputs = save_outputs(frame, args.output_dir, args.title)
    print("---")
    print(f"rows:             {len(frame)}")
    print(f"jsonl:            {outputs['jsonl']}")
    print(f"csv:              {outputs['csv']}")
    print(f"markdown:         {outputs['markdown']}")
    print(f"chart:            {outputs['chart']}")


if __name__ == "__main__":
    main()