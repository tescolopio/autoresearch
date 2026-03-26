# Local Readiness Roadmap

This roadmap defines the minimum local milestones required before running the full CPU-native research loop overnight.

If you are aligning this repo to a Windows/CPU `llama.cpp`-centric runtime milestone framework, also read [WINDOWS_CPU_POC_ALIGNMENT.md](/mnt/d/source/3D-Tech-Solutions/autoresearch/WINDOWS_CPU_POC_ALIGNMENT.md). That document separates the existing research-loop milestones from the future runtime PoC milestones for inference, peripherals, and vector memory.

The intent is simple:

- remove ambiguity about what “ready” means
- make every stage locally testable
- avoid starting an overnight loop before the machine, data, metrics, and agent controls are proven stable

## Milestone 0: Environment Reproducibility

Objective:
- anyone on the team can create the same local environment and run the focused validation suite

Required evidence:
- `bash scripts/bootstrap_bitnet_research.sh` completes successfully
- editable install works in `bitnet_research`
- focused unit suite passes

Exit criteria:
- `environment.yml` creates the environment cleanly
- bootstrap script finishes without manual repair
- these tests pass:

```bash
python -m unittest \
  tests/test_bitnet_cpu_poc.py \
  tests/test_ternary_lab.py \
  tests/test_deploy_cpu_agent.py \
  tests/test_compare_agents.py \
  tests/test_benchmark_compare.py \
  tests/test_provider_benchmark.py
```

Why it matters:
- if setup is not repeatable, every later benchmark or overnight result is suspect

## Milestone 1: Local Agent Lifecycle

Objective:
- prove the first CPU agent can be created, tracked, and resumed locally

Required evidence:
- `deploy_cpu_agent.py` creates an agent manifest and deployment report
- `.ternary_lab/agents/<agent-name>/` contains stable artifacts

Exit criteria:
- this command succeeds:

```bash
python deploy_cpu_agent.py --agent-name cpu-agent-1 --mode mock
```

- these files exist:
  - `.ternary_lab/agents/cpu-agent-1/agent.json`
  - `.ternary_lab/agents/cpu-agent-1/deployment_report.json`

Why it matters:
- the overnight loop needs a stable local identity, state path, and result sink before it can be trusted unattended

## Milestone 2: Data And Tokenizer Readiness

Objective:
- prove the training path has the local data needed to run a real experiment

Required evidence:
- `prepare.py` completes successfully
- local cache contains at least one parquet shard and tokenizer artifacts

Exit criteria:
- this command succeeds:

```bash
python prepare.py
```

- these artifacts exist:
  - `~/.cache/autoresearch/data/*.parquet`
  - `~/.cache/autoresearch/tokenizer/tokenizer.pkl`
  - `~/.cache/autoresearch/tokenizer/token_bytes.pt`

Why it matters:
- without local data readiness, the overnight loop will fail or degrade into a mock-only demo

## Milestone 3: Real CPU Training Baseline

Objective:
- prove the CPU BitNet path can complete one real 5-minute training run and write metrics

Required evidence:
- `train.py` completes in CPU mode
- summary JSON and TSV contain throughput, memory, energy, and utilization metrics

Exit criteria:
- this command succeeds:

```bash
python train.py \
  --device cpu \
  --cpu-only \
  --cpu-bitnet-poc \
  --summary-json comparison_reports/cpu_bitnet.json \
  --results-tsv comparison_reports/cpu_bitnet.tsv \
  --description "cpu bitnet baseline"
```

- the summary output includes:
  - `val_bpb`
  - `tokens_per_sec`
  - `energy_j/token`
  - `cpu_proc_%`
  - `memory_gb` or `peak_mem_mb`

Why it matters:
- until one real CPU run is stable, there is no basis for autonomous iteration

## Milestone 4: Task-Specific Evaluation

Objective:
- define the target behavior the research loop is supposed to improve

Required evidence:
- a local task suite exists
- pass or fail can be scored automatically
- the same suite can be run before and after experiments

Recommended first target:
- local research and planning assistance

Minimum deliverables:
- a prompt suite with 8 to 20 tasks
- expected answer checks such as `must_contain` or exact answers
- a summary artifact with pass rate and latency

Exit criteria:
- this command succeeds with a real local provider command or a temporary stub:

```bash
python provider_benchmark.py \
  --config provider_benchmark.example.json \
  --output-dir provider_reports/latest
```

- output artifacts exist:
  - `provider_reports/latest/responses.jsonl`
  - `provider_reports/latest/provider_summary.csv`
  - `provider_reports/latest/provider_summary.md`

Why it matters:
- `val_bpb` is not enough if the real goal is improving a specific task

## Milestone 5: Comparative Measurement Layer

Objective:
- prove the CPU path is measurable against another method, even if only locally

Required evidence:
- comparison report can be produced from real CPU artifacts and either a mocked GPU baseline or another local comparator

Exit criteria:
- one of these commands succeeds:

```bash
python benchmark_compare.py --mock-gpu
```

or

```bash
python benchmark_compare.py --gpu-summary-json path/to/other_local_llm.json
```

- report artifacts exist:
  - `comparison_reports/latest/comparison.md`
  - `comparison_reports/latest/skeptic_summary.md`
  - `comparison_reports/latest/throughput.png`

Why it matters:
- the overnight loop needs a trustworthy measurement layer so improvements can be defended later

## Milestone 6: Single Iteration Research Loop

Objective:
- prove the autonomous loop can complete one real experiment cycle locally

Required evidence:
- candidate proposal
- signed execution
- training run
- result logging
- keep or discard decision

Exit criteria:
- this command succeeds without mock mode:

```bash
python ternary_lab.py --iterations 1 --objective "Improve the local target task under a fixed 5-minute CPU budget"
```

- these artifacts update:
  - `results.tsv`
  - `.ternary_lab/state.json`
  - `.ternary_lab/knowledge_graph.json`

Why it matters:
- this is the first proof that the “scientist loop” works end to end on local hardware

## Milestone 7: Short Unattended Burn-In

Objective:
- prove the loop is stable enough to run unattended for a short local window before committing to overnight operation

Required evidence:
- at least 3 consecutive real iterations complete without manual intervention
- no corrupted state
- no stuck process
- metrics continue to append correctly

Exit criteria:
- this command completes successfully:

```bash
python ternary_lab.py --iterations 3 --objective "Improve the local target task under a fixed 5-minute CPU budget"
```

- logs and summaries show:
  - no crash rows unless the loop correctly recovered
  - valid summary JSON per run
  - stable state progression

Why it matters:
- if 3 local iterations are unstable, an overnight run will waste the night

## Milestone 8: Overnight Readiness Gate

Objective:
- define the minimum conditions required before launching the full overnight suite

Required gate:
- Milestones 0 through 7 are complete
- real CPU training baseline is stable
- task-specific benchmark exists
- comparison reporting works
- at least one short unattended burn-in succeeded

Recommended overnight launch command:

```bash
python ternary_lab.py --iterations 12 --objective "Improve the local target task under a fixed 5-minute CPU budget"
```

Suggested first overnight target:
- 12 iterations, not 100

Why:
- 12 iterations is enough to test the unattended workflow without betting the entire night on immature instrumentation

## Practical milestone order for this repo

If the goal is to get to a real overnight run as fast as possible, do the milestones in this order:

1. Environment Reproducibility
2. Local Agent Lifecycle
3. Data And Tokenizer Readiness
4. Real CPU Training Baseline
5. Task-Specific Evaluation
6. Single Iteration Research Loop
7. Short Unattended Burn-In
8. Comparative Measurement Layer
9. Overnight Readiness Gate

The comparison layer is useful earlier for demos, but it is not the gating dependency for the first real overnight loop.

## Recommended local definition of done

You are ready for the first overnight run when all of the following are true:

- setup is reproducible from scratch
- one real CPU BitNet run completes and records metrics
- one task-specific evaluation suite exists
- one real research iteration completes and logs a keep or discard decision
- three unattended real iterations complete without intervention
- artifacts are understandable the next morning