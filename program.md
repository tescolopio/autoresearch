# autoresearch

This is a CPU-native BitNet research contract for a local autonomous agent.

The end goal is not generic model tinkering. The end goal is to improve a local model on a defined target task through repeatable 5-minute CPU experiments, while preserving local-only execution and clear keep or discard decisions.

## Mission

You are operating a local BitNet research loop.

Your mission is to:

1. establish a real CPU-native baseline
2. improve the model on the current target task through iterative experiments
3. keep only changes that improve the measured objective
4. preserve local-only, CPU-only execution unless the human explicitly asks for a comparison baseline

## Hard constraints

You must follow these constraints at all times:

- default to CPU-native execution
- treat BitNet mode as the required local path
- do not rely on cloud training or remote compute
- do not modify `prepare.py`
- do not add new dependencies unless the human explicitly asks for them
- do not change the evaluation harness in `prepare.py`
- do not start long unattended loops until readiness milestones are satisfied

## Readiness gate before real experimentation

Before running the real loop, confirm that all of the following are true:

1. the local environment is reproducible
2. the first local agent can be deployed
3. the tokenizer and data cache exist under `~/.cache/autoresearch/`
4. one real CPU BitNet training run completes successfully
5. one task-specific evaluation path exists

If any of these are not true, stop the full loop and work only on closing the missing readiness gap.

Use [ROADMAP.md](/mnt/d/source/3D-Tech-Solutions/autoresearch/ROADMAP.md) as the authority for readiness milestones.

## In-scope files

Read these files before making decisions:

- `README.md`
- `ROADMAP.md`
- `prepare.py`
- `train.py`
- `program.md`

You may also read:

- `ternary_lab.py`
- `deploy_cpu_agent.py`
- `benchmark_compare.py`
- `provider_benchmark.py`

## Primary objective

The optimization target has two layers:

1. local training quality: improve `val_bpb`
2. task usefulness: improve the current task-specific evaluation suite once it exists

If there is a conflict, prefer the metric that directly reflects the user’s target task.

Until a task-specific evaluation suite is fully established, use `val_bpb` as the temporary optimization anchor.

## Setup protocol

When starting a new real experiment run:

1. agree on a run tag and create a dedicated branch such as `autoresearch/<tag>` when git is available
2. confirm the current branch and git status
3. confirm readiness milestones are satisfied
4. confirm data and tokenizer artifacts exist
5. confirm `results.tsv` exists or initialize it
6. confirm the current best baseline metrics

If the data cache is missing, do not pretend experimentation can begin. Tell the human the real loop is blocked on `python prepare.py`.

When git is available, prefer the non-interactive helper commands in `scripts/git_agent.py` instead of ad hoc shell composition.

## Useful workflow preserved from the original repo

The original autoresearch workflow had several good operational habits. Keep them.

- use a dedicated feature branch for a run instead of experimenting on an arbitrary branch
- inspect git state before each experiment
- make one intentional change per experiment
- commit experiment code before running a real trial when the change is worth preserving in history
- capture run output to a log file instead of flooding the agent context
- append experiment results to `results.tsv`
- keep successful changes and discard unsuccessful ones

These workflow mechanics are still valid on the BitNet path. The difference is that the default execution target is now CPU-native BitNet, not GPU-first training.

Preferred git helper patterns:

```bash
python scripts/git_agent.py status
python scripts/git_agent.py ensure-branch --tag <run-tag>
python scripts/git_agent.py commit --message "experiment: <summary>" --paths train.py
python scripts/git_agent.py revert --commit <sha>
```

For one real experiment with branch setup, commit, log capture, and optional automatic revert, prefer:

```bash
python scripts/run_agent_experiment.py \
   --branch-tag <run-tag> \
   --commit-message "experiment: <summary>" \
   --commit-paths train.py \
   --log-path .ternary_lab/runs/<run-tag>.log \
   --revert-on-failure \
   -- python train.py --device cpu --cpu-only --cpu-bitnet-poc
```

## Allowed experimentation surface

You may modify:

- `train.py`

You may use:

- `ternary_lab.py` for orchestrated local runs
- `deploy_cpu_agent.py` for agent lifecycle setup
- `benchmark_compare.py` and `provider_benchmark.py` for evidence and validation

Treat `provider_benchmark.py` as a separate showcase loop for demonstrating the system against frontier APIs. It is not the authority for `keep` or `discard` decisions in the main local research loop.

You may not modify:

- `prepare.py`

## CPU-first execution rules

For local BitNet research, the default execution path is:

```bash
python train.py --device cpu --cpu-only --cpu-bitnet-poc
```

Use GPU only when explicitly collecting a comparison baseline, not as the default research path.

## Experiment loop

Once readiness is satisfied and a real loop is authorized, iterate as follows:

1. inspect the current best result in `results.tsv`
2. inspect git status and confirm you understand the current branch state
3. propose one small, defensible change to `train.py`
4. record the intent of the experiment in the description field
5. if the change is a real candidate, create a non-interactive git commit before the run
6. run one fixed 5-minute training experiment and redirect output to a log file when operating autonomously
7. capture:
   - `val_bpb`
   - `tokens_per_second`
   - `energy_j_per_token`
   - CPU and GPU utilization metrics if present
   - memory footprint
8. decide `keep`, `discard`, or `crash`
9. update `results.tsv`
10. if the result is worse, revert only the experiment change and never destroy unrelated work
11. continue only if the system remains stable

When running autonomously, prefer a log capture pattern equivalent to:

```bash
python train.py ... > run.log 2>&1
```

Then read the summary or failure trace from the log instead of streaming the entire run into the working context.

## Keep and discard rules

Use these rules when judging a result:

- `keep` if the local task-evaluation signal or other target metric improves meaningfully and the change is not needlessly complex
- `discard` if the result is worse or equal without a compelling simplification win
- `crash` if the run fails, times out, or produces invalid metrics

Prefer simpler code when improvement is marginal.

## Stability rules for unattended execution

Do not start an overnight run until:

- one real iteration succeeds
- three unattended real iterations succeed
- state files remain valid
- result logging remains valid

If those conditions are not met, continue local burn-in only.

At any time, the human may pause, resume, or retarget the loop through `.ternary_lab/control.json` or `python scripts/loop_status.py`. Respect that control point before starting the next iteration.

## Output expectations

Every real run should yield a summary containing at least:

- `val_bpb`
- `training_seconds`
- `total_seconds`
- `tokens_per_sec`
- `energy_j/token`
- device and linear implementation

If the run is on CPU, the summary should make that explicit.

## What success looks like

The loop is working correctly when:

- local setup is reproducible
- data and tokenizer are present
- real CPU BitNet training runs complete consistently
- results are appended cleanly to `results.tsv`
- the agent can complete repeated keep or discard decisions without supervision

## What not to claim

Do not claim any of the following unless directly measured in this repository:

- frontier-model parity
- broad benchmark parity
- 70B or 100B local deployment results
- overnight scientific improvement if the loop has not actually completed one

Stick to measured local evidence.
