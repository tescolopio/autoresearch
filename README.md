# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies its experiment configuration, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. This repo now pivots that loop away from GPU-heavy research and toward a CPU-native BitNet path built around ternary `BitLinear` layers, signed local-only execution, and a knowledge-graph archive. The core idea is that you're not touching the training internals during each experiment. Instead, you steer the autonomous loop through `program.md` and the CPU-first `ternary_lab.py` orchestrator. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069) and [this tweet](https://x.com/karpathy/status/2031135152349524125).

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Repeatable setup

**Requirements:** Conda plus shell access on Linux or macOS. The repository is now structured around a CPU-native BitNet path, so the default install flow uses CPU PyTorch wheels and a single conda environment name: `bitnet_research`.

### Option A: one repeatable bootstrap command

```bash
bash scripts/bootstrap_bitnet_research.sh
```

That script will:

- create or update the `bitnet_research` conda environment from `environment.yml`
- install CPU-oriented project dependencies into the active conda interpreter
- install the repo in editable mode
- run the focused unit tests

Optional flags:

```bash
bash scripts/bootstrap_bitnet_research.sh --prepare-data --deploy-agent
```

- `--prepare-data` downloads the local dataset and tokenizer cache
- `--deploy-agent` deploys `cpu-agent-1` after setup completes
- without `--prepare-data`, the deployed agent runs in deterministic `mock` mode
- with `--prepare-data`, the deployed agent uses `auto` mode and can advance into real CPU training

### Option B: explicit manual steps

```bash
# 1. Create the environment once
conda env create -f environment.yml

# 2. Activate it
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bitnet_research

# 3. Install the repo into the active interpreter
bash scripts/bootstrap_bitnet_research.sh --skip-tests

# 4. Run the focused validation suite
python -m unittest tests/test_bitnet_cpu_poc.py tests/test_ternary_lab.py tests/test_deploy_cpu_agent.py
```

If the above commands all work, the setup is repeatable and ready for local CPU-native research.

Before attempting the first overnight run, follow the milestone checklist in [ROADMAP.md](/mnt/d/source/3D-Tech-Solutions/autoresearch/ROADMAP.md).

If you want to align this repository to a Windows/CPU `llama.cpp` BitNet milestone framework, read [WINDOWS_CPU_POC_ALIGNMENT.md](/mnt/d/source/3D-Tech-Solutions/autoresearch/WINDOWS_CPU_POC_ALIGNMENT.md) as well. It explains how the current research loop maps to a future runtime PoC lane for `llama.cpp`, vector memory, PDF research tasks, and self-optimization.

To make milestone advancement explicit instead of informal, use:

```bash
python scripts/roadmap_doctor.py --format text
```

For a stronger environment verification pass that actually executes the focused unit suite:

```bash
python scripts/roadmap_doctor.py --run-focused-tests --format text
```

To enforce that all prerequisites through a milestone are complete before proceeding:

```bash
python scripts/roadmap_doctor.py --require-complete-through single-iteration-research-loop
```

The supervised launcher now uses this milestone gate automatically unless `--skip-milestone-gate` is passed.

## Quick start

After the environment is bootstrapped, the shortest working flow is:

```bash
# 1. Download data and train tokenizer (one-time, ~2 min)
python prepare.py

# 2. Manually run a single CPU-native training experiment (~5 min)
python train.py

# 3. Deploy the first local CPU agent
python deploy_cpu_agent.py --agent-name cpu-agent-1

# 4. Run the CPU-native Ternary Lab loop (~5 min per experiment)
OBJECTIVE="Minimize energy while maintaining accuracy"
python ternary_lab.py --iterations 1 --objective "$OBJECTIVE"
```

## Ternary Lab pipeline

The repository now includes a CPU-only orchestration layer in `ternary_lab.py`.

- It proposes BitNet experiment candidates using a fixed 5-minute wall-clock budget.
- It signs each execution instruction against locally sealed mission text before launching `train.py`.
- It refuses candidates that violate the local-only CPU policy, such as GPU use or networked behavior.
- It records keep or discard decisions in `results.tsv` and consolidates salient experiments into `.ternary_lab/knowledge_graph.json`.

Run multiple autonomous CPU experiments like this:

```bash
uv run ternary_lab.py --iterations 3 \
   --objective "Minimize val_bpb and joules per experiment using CPU-native BitNet only"
```

## Deploying the first CPU agent

Use `deploy_cpu_agent.py` to stand up the first local agent instance.

- In `auto` mode, it checks whether `~/.cache/autoresearch/` already contains the prepared dataset and tokenizer.
- If the cache is missing, it deploys in `mock` mode so you can validate the local agent lifecycle immediately.
- If the cache is ready, it switches to real CPU BitNet training automatically.
- Agent artifacts are written under `.ternary_lab/agents/<agent-name>/`.

Fast local smoke test:

```bash
python deploy_cpu_agent.py --agent-name cpu-agent-1 --mode mock
```

Automatic mode selection:

```bash
python deploy_cpu_agent.py --agent-name cpu-agent-1
```

After `python prepare.py` completes, rerun the deployment command and the agent will move from `mock` to real CPU training mode.

## Git workflow helpers for the agent

The repo now includes a small non-interactive git helper at `scripts/git_agent.py`. It is designed for LLM agents that already understand git concepts but need stable, machine-readable command results and safer defaults than ad hoc shell composition.

Current commands:

```bash
# Inspect the current branch and worktree as JSON
python scripts/git_agent.py status

# Create or switch to an experiment branch such as autoresearch/cpu-baseline
python scripts/git_agent.py ensure-branch --tag cpu-baseline

# Publish the current branch when origin is configured
python scripts/git_agent.py publish

# Commit a single experiment change without using interactive git flows
python scripts/git_agent.py commit --message "experiment: lower bitlinear threshold" --paths train.py

# Revert one experimental commit with a new revert commit
python scripts/git_agent.py revert --commit <sha>
```

Why this exists:

- branch creation and switching stay non-interactive
- status snapshots come back as JSON instead of free-form terminal text
- experiment commits can be limited to explicit paths such as `train.py`
- failed experiments can be undone with `git revert` instead of destructive resets

This is meant to support the workflow described in [program.md](/mnt/d/source/3D-Tech-Solutions/autoresearch/program.md): dedicated run branches, explicit experiment commits, log capture, and safe keep-or-discard decisions.

For a single end-to-end experiment invocation, use `scripts/run_agent_experiment.py`:

```bash
python scripts/run_agent_experiment.py \
   --branch-tag reliability-pass-1 \
   --commit-message "experiment: tighten recovery path" \
   --commit-paths train.py \
   --log-path .ternary_lab/runs/reliability-pass-1.log \
   --revert-on-failure \
   -- python train.py --device cpu --cpu-only --cpu-bitnet-poc
```

That wrapper:

- snapshots git status first
- creates or switches to the run branch when safe
- creates a non-interactive experiment commit when requested
- redirects stdout and stderr to a log file
- creates a revert commit automatically if the command fails and `--revert-on-failure` was requested

The default research objective for the first agent loop is now explicitly about improving the reliability and capability of the local research agent in its research and training tasks, rather than only chasing raw model metrics.

## CPU BitNet PoC validation

If you want to show that the CPU-native BitNet proof of concept is working, use the following validation flow.

### 1. Automated checks

Run the existing focused checks first:

```bash
python -m unittest tests/test_bitnet_cpu_poc.py
python -m py_compile train.py prepare.py tests/test_bitnet_cpu_poc.py
```

These checks verify that the CPU PoC entrypoints, signature verification, `results.tsv` logging, and CPU-safe data/eval hooks are still present.

### 2. Manual PoC run

Run one signed 5-minute CPU PoC experiment:

```bash
OBJECTIVE="Minimize energy while maintaining accuracy"
SIGNATURE="$(python - <<'PY'
import hashlib, hmac
objective = "Minimize energy while maintaining accuracy"
print(hmac.new(b"demo", objective.encode(), hashlib.sha256).hexdigest())
PY
)"

python train.py \
  --device cpu \
  --cpu-bitnet-poc \
  --objective "$OBJECTIVE" \
  --signature-secret demo \
  --signature "$SIGNATURE"
```

### 3. What a successful PoC must show

Treat the PoC as successful if all of the following are true:

1. The run completes its fixed 5-minute training budget without crashing.
2. The summary output includes:
   - `val_bpb`
   - `device:           cpu`
   - `linear_impl:      bitlinear`
   - `energy_j/token`
   - `tokens_per_sec`
   - `signature_ok:     True`
3. `results.tsv` gets a new row with the CPU PoC metadata columns populated:
   - `device`
   - `linear_impl`
   - `signature_verified`
   - `energy_j_per_token`
   - `tokens_per_second`
4. The recorded row shows the CPU path was actually exercised:
   - `device=cpu`
   - `linear_impl=bitlinear`
   - `signature_verified=yes`

### 4. Expectations for this proof of concept

This PoC is intended to prove **capability**, not to promise a fixed benchmark win on every machine.

The concrete expectation is:

- the autoresearch loop can run on CPU,
- ternary BitLinear mode is actually selected,
- signed objectives can gate execution,
- and the run produces measurable PoC outputs in both stdout and `results.tsv`.

Nice-to-have follow-up evidence is to compare the CPU BitNet run with a dense CPU baseline such as:

```bash
python train.py --device cpu --linear-impl dense
```

That comparison can help you judge relative energy and throughput on your machine, but it is **not required** to demonstrate that the PoC itself exists and works.

## Comparing CPU BitNet and GPU runs

To show a fair side-by-side comparison, record one CPU BitNet run and one GPU dense run with summary JSON output, then generate a report.

For skeptic-grade evidence, the standard should be stricter than a screenshot. Use the same repository, the same fixed 5-minute training budget, and machine-generated artifacts for both runs.

### 1. Record a CPU BitNet baseline

```bash
python train.py \
   --device cpu \
   --cpu-only \
   --cpu-bitnet-poc \
   --summary-json comparison_reports/cpu_bitnet.json \
   --results-tsv comparison_reports/cpu_bitnet.tsv \
   --description "cpu bitnet baseline"
```

### 2. Record a GPU baseline

```bash
python train.py \
   --device cuda \
   --allow-accelerator \
   --linear-impl dense \
   --summary-json comparison_reports/gpu_dense.json \
   --results-tsv comparison_reports/gpu_dense.tsv \
   --description "gpu dense baseline"
```

### 3. Generate a comparison report

```bash
python compare_agents.py \
   --summary-json comparison_reports/cpu_bitnet.json comparison_reports/gpu_dense.json \
   --output-dir comparison_reports/latest
```

The report generator writes:

- `comparison_reports/latest/comparison.md`
- `comparison_reports/latest/comparison.csv`
- `comparison_reports/latest/throughput.png`
- `comparison_reports/latest/resource_utilization.png`
- `comparison_reports/latest/efficiency.png`

### 4. Metrics to highlight in the demo

- `tokens_per_second`: raw throughput comparison
- `avg_cpu_process_percent`: how hard the CPU agent drove the local machine
- `avg_gpu_util_percent`: how hard the GPU baseline drove the accelerator
- `avg_gpu_mem_used_mb`: GPU memory footprint
- `memory_gb`: process memory footprint from the run summary
- `energy_j_per_token`: normalized energy estimate for the run
- `val_bpb`: model-quality anchor so throughput is not discussed in isolation

This gives you a clean story: throughput, resource pressure, and energy efficiency, all tied back to the same fixed 5-minute training budget.

## Skeptic-grade benchmark protocol

If the goal is to satisfy a hostile technical review, use `benchmark_compare.py` instead of ad hoc commands.

```bash
python benchmark_compare.py --output-dir comparison_reports/benchmark --report-dir comparison_reports/latest
```

That command runs the standard CPU BitNet baseline and the standard GPU dense baseline, then produces:

- `comparison_reports/benchmark/cpu_bitnet.json`
- `comparison_reports/benchmark/gpu_dense.json`
- `comparison_reports/latest/comparison.md`
- `comparison_reports/latest/skeptic_summary.md`
- `comparison_reports/latest/comparison.csv`
- `comparison_reports/latest/throughput.png`
- `comparison_reports/latest/resource_utilization.png`
- `comparison_reports/latest/efficiency.png`

### What this protocol proves

- exact tokens per second measured on your hardware
- exact process memory and GPU memory pressure for the recorded runs
- exact sampled CPU process load and GPU utilization for the recorded runs
- exact local `val_bpb` results under the same 5-minute budget

### What this protocol does not prove by itself

- parity with GPT-4, Claude, or other proprietary frontier systems
- reasoning benchmark parity on GSM8K, WinoGrande, or ARC-Challenge unless those evaluations are run separately
- deployment claims for 70B or 100B models unless those exact models are measured on the target machine

That distinction matters. A skeptic will accept measured local evidence, but they will reject claims that are not directly backed by artifacts from this repository.

### Real CPU, mocked GPU

If you do not want to spend time or accelerator budget on the GPU side, keep the CPU BitNet run real and synthesize only the GPU baseline:

```bash
python benchmark_compare.py \
   --mock-gpu \
   --output-dir comparison_reports/benchmark \
   --report-dir comparison_reports/latest
```

That path gives you a measured CPU artifact plus a clearly labeled mocked GPU comparator.

### Real CPU, external local LLM comparator

If you have another local LLM method already measured, compare the real CPU BitNet run against that existing summary JSON instead of re-running a GPU training job:

```bash
python benchmark_compare.py \
   --gpu-summary-json path/to/other_local_llm.json \
   --output-dir comparison_reports/benchmark \
   --report-dir comparison_reports/latest
```

### Existing CPU agent, mocked GPU

If the CPU side already exists as a deployed local agent, reuse it directly:

```bash
python benchmark_compare.py \
   --cpu-agent-dir .ternary_lab/agents/cpu-agent-conda \
   --skip-cpu \
   --mock-gpu \
   --report-dir comparison_reports/latest
```

This is the fastest route when you already have a real CPU BitNet artifact and only need a presentation-ready comparison package.

## Frontier API vs local BitNet task parity

If the goal is to show that the same user-visible task can be handled by Claude, ChatGPT, Gemini, and a completely local BitNet-backed model, use `provider_benchmark.py`.

This is a separate showcase loop. It is not the main autoresearch optimization loop and must not be used to decide `keep` or `discard` inside `ternary_lab.py`.

This is a different claim from training parity. It measures the same prompt suite across multiple providers and captures:

- latency per task
- output tokens per second
- simple task-pass checks
- raw responses for manual inspection

### 1. Copy the example config

Start from [provider_benchmark.example.json](/mnt/d/source/3D-Tech-Solutions/autoresearch/provider_benchmark.example.json) and replace the local command with your real BitNet runner if you have one.

The example local command points to [scripts/bitnet_local_stub.py](/mnt/d/source/3D-Tech-Solutions/autoresearch/scripts/bitnet_local_stub.py), which is only a wiring example.

### 2. Set API keys

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
```

### 3. Run the same task suite everywhere

```bash
python provider_benchmark.py \
   --config provider_benchmark.example.json \
   --output-dir provider_reports/latest
```

### 4. Review the artifacts

- `provider_reports/latest/responses.jsonl`
- `provider_reports/latest/provider_summary.csv`
- `provider_reports/latest/provider_summary.md`
- `provider_reports/latest/provider_latency.png`

### What this proves

- the same prompts were run across frontier APIs and a local command-backed model
- the local BitNet path can be evaluated on the same task suite as the API models
- latency and throughput can be compared directly at the task level

### What this does not prove by itself

- that the local BitNet model matches frontier models on broad reasoning benchmarks
- that local BitNet training is equivalent to frontier-scale training infrastructure
- that a stubbed local command is representative of a real BitNet inference engine

To make the local side fully credible, replace the stub command with the actual BitNet inference command or service you want to defend.

## Main loop vs showcase loop

Keep these loops separate:

- Main loop: `ternary_lab.py`
   - optimizes the local CPU-native research agent
   - uses local run artifacts to score reliability, capability, and training quality
   - drives `keep`, `discard`, and `crash` decisions
- Showcase loop: `provider_benchmark.py`
   - compares the local system against frontier APIs on a presentation task suite
   - generates demo artifacts for external review
   - does not feed back into the main experiment decision rule

## Pausing or changing the main loop

The main loop now has a human control file at `.ternary_lab/control.json` and a helper script for reading or updating it.

Inspect the current loop state:

```bash
python scripts/loop_status.py
```

Pause before the next iteration boundary:

```bash
python scripts/loop_status.py --stop-after-iteration --note "pause after current run for review"
```

Pause immediately before any new iteration starts:

```bash
python scripts/loop_status.py --set-state paused --note "hold the loop while I inspect results"
```

Resume the loop:

```bash
python scripts/loop_status.py --set-state running --clear-stop-after-iteration
```

Change the objective without editing code:

```bash
python scripts/loop_status.py \
   --objective-override "Prioritize reliability recovery and stable CPU-native experiment execution." \
   --note "temporary focus shift"
```

This gives the user a natural control point between iterations instead of forcing direct edits to `.ternary_lab/state.json`.

Generate a human-readable markdown snapshot of the current loop:

```bash
python scripts/loop_report.py
```

The report now includes timestamps, average iteration duration, last iteration duration, and a short recent-activity table so the current pace is visible at a glance.

Run the main loop under supervisor mode until the control file says pause or stop:

```bash
python ternary_lab.py --trainer-backend mock --run-until-stopped --max-supervisor-iterations 10
```

For real training, omit `--trainer-backend mock` after readiness milestones are satisfied. The supervisor keeps executing one iteration at a time and re-reads `.ternary_lab/control.json` before each next step.

For a small all-in-one launcher that sets the branch tag, starts supervised iteration, and refreshes the markdown report after every completed iteration:

```bash
python scripts/supervised_local_run.py --branch-tag nightly-burnin --trainer-backend mock --max-iterations 3
```

That launcher is intended for local operator use. It prepares the run branch, ensures the loop is in `running` state, executes one iteration at a time, and rewrites `.ternary_lab/loop_report.md` after each step.

Optional audit-trail features:

```bash
python scripts/supervised_local_run.py \
   --branch-tag nightly-burnin \
   --trainer-backend mock \
   --max-iterations 3 \
   --commit-snapshots
```

- writes per-iteration report snapshots under `.ternary_lab/snapshots/`
- optionally creates a non-interactive git commit for each snapshot report
- leaves a clearer operator audit trail without changing main-loop keep/discard logic

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform support

The default training path is now CPU-native BitNet. The stack uses ternary `BitLinear` layers, a generic scaled-dot-product-attention fallback, CPU-safe dataloading and evaluation, and signed objective verification so the 5-minute autoresearch loop can run without CUDA.

To run a single CPU-native experiment directly, use `python train.py --device cpu --cpu-only --cpu-bitnet-poc`. Successful runs append PoC metrics to `results.tsv`, including device, linear implementation, signature status, estimated energy per token, and throughput.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## License

MIT
