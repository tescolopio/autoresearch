# Windows/CPU PoC Alignment

This document aligns the current `autoresearch` repository with the Windows/CPU milestone framework for an "Auto-Research Cornerstone" built without GPUs.

The key point is that there are **two related but different tracks**:

1. `autoresearch` main loop
   - local research orchestration
   - experiment control, audit trail, keep/discard logic
   - CPU-native training and local task evaluation
2. Windows CPU runtime PoC
   - `llama.cpp` BitNet or ternary inference runtime
   - local peripherals and long-term memory
   - user-facing research assistant behavior

The current repository is already strong on track 1. Your milestone framework introduces major requirements from track 2. Alignment means we should connect them deliberately instead of pretending the current `train.py` path already satisfies `llama.cpp` runtime milestones.

## Current Status

What the repo already covers well:

- CPU-only local research loop in `ternary_lab.py`
- branch-safe and non-destructive experiment workflow
- pause/resume/change control surface through `.ternary_lab/control.json`
- human-readable loop reports and supervised launcher
- measurable local metrics for reliability, capability, throughput, memory, and energy
- separate showcase loop for frontier comparisons

What the repo does **not** yet cover:

- `llama.cpp` runtime integration
- Windows or WSL2-specific runtime validation
- 7B BitNet inference throughput benchmarking such as `>15 tokens/second`
- webcam or microphone event bridge
- local vector database integration such as ChromaDB or Qdrant
- PDF ingestion and research-paper summarization loop
- self-rewriting prompt loop attached to a real local inference runtime

## Boundary Decision

To align cleanly, keep this separation:

- `train.py` and `ternary_lab.py` remain the **main research loop**
- `llama.cpp` + vector DB + peripherals become the **runtime PoC layer**
- frontier-model comparison remains a **showcase loop**, never the authority for the main loop

That means the Windows/CPU milestone framework should be added as a **parallel readiness track**, not as a replacement for the existing research loop controls.

## Milestone Mapping

### Milestone 1: The Hollow Agent

Target from your framework:

- load a 1.58-bit 7B model on CPU
- achieve `>15 tokens/second`

Alignment decision:

- this is **not** a `train.py` milestone
- this is a new **runtime benchmark milestone**
- recommended runtime: `llama.cpp` BitNet-capable path under Windows with WSL2 support

What should be added next:

- a small runtime adapter script that can invoke a local `llama.cpp` binary
- a benchmark script that records prompt latency and tokens per second
- a result artifact stored separately from `results.tsv`, for example under `.ternary_lab/runtime/`

Success evidence:

- one command loads the 7B model locally
- records measured tokens/second on the target Windows/WSL2 machine
- writes summary JSON for the operator report

### Milestone 2: Peripheral Integration

Target from your framework:

- Python connects to webcam and microphone
- sensory agent detects presence and signals the 7B agent

Alignment decision:

- this is outside the current training loop
- this should be treated as an **event-ingestion milestone** for the runtime PoC layer

What should be added next:

- `sensor_bridge.py` or similar event producer
- a simple local event contract such as `user_present=true`
- a mock-first implementation before any hardware capture path is considered stable

Success evidence:

- the runtime layer can receive a local presence event
- the event is logged and can trigger a local inference call

### Milestone 3: The First Research Task

Target from your framework:

- search a local directory of PDF papers
- summarize them
- update a local knowledge base in a vector DB

Alignment decision:

- this should become the **first real task-specific evaluation suite** for the runtime PoC
- the current repo already has the right control loop shape, but no PDF/vector runtime layer yet

What should be added next:

- local PDF ingestion tool
- chunking and embedding path
- ChromaDB or Qdrant local integration
- task harness that scores retrieval + summary quality

Success evidence:

- local documents are ingested without network use
- summaries are written locally
- the vector DB is updated deterministically
- the task can be run again for before/after comparison

### Milestone 4: Self-Optimization

Target from your framework:

- agent proposes a new system prompt for itself
- loop restarts and evaluates whether summary quality improves

Alignment decision:

- this should be the **runtime self-optimization milestone**, not the first optimization milestone overall
- the main research loop already supports iterative optimization; this milestone connects that mechanism to a local runtime prompt layer

What should be added next:

- prompt registry file under source control or runtime state
- automatic prompt candidate generation
- before/after summary-quality scoring for the PDF research task
- revert path when prompt changes do not improve the measured result

Success evidence:

- prompt change is versioned
- old and new prompts are comparable on the same local task suite
- system can keep or discard prompt edits using measured local results

## Hardware And Stack Interpretation

Your proposed stack is reasonable as a PoC target, but it should be treated as a **host/runtime milestone set**, not a claim that the current repo already satisfies it.

Recommended interpretation:

- CPU:
  - modern Ryzen 7/9 or Intel i7/i9 class is appropriate
  - AVX-512 is a bonus, not a hard repo requirement
- RAM:
  - 32 GB minimum for practical local experimentation
  - 64 GB preferred for larger multi-model runtime experiments
- Storage:
  - NVMe SSD is appropriate for local model swapping and document indexing
- Runtime:
  - `llama.cpp` should be introduced as an inference/runtime layer, not as a replacement for this repo's training instrumentation
- Environment:
  - WSL2 should be treated as a supported host mode for the runtime PoC track
- Vector DB:
  - ChromaDB is likely the lightest first step
  - Qdrant is a reasonable second option if persistence and filtering needs grow

## Recommended Repo Alignment Plan

Phase A: keep the current repo honest

- keep `ternary_lab.py` as the main local research loop
- keep `provider_benchmark.py` as showcase-only
- do not blur runtime PoC claims into `train.py` metrics

Phase B: add a runtime PoC lane

- add a `llama.cpp` adapter
- add a runtime throughput benchmark
- add vector DB integration
- add PDF research task harness
- add optional peripheral event bridge

Phase C: connect both lanes

- allow the research loop to optimize prompt/runtime config for the PDF research task
- keep training metrics and runtime task metrics separate but jointly visible in reports

## Practical Next Milestones For This Repo

If we align to your framework pragmatically, the next implementation milestones should be:

1. `llama.cpp` runtime benchmark harness for Windows/WSL2
2. local vector DB adapter and document-ingestion prototype
3. PDF research task suite with measurable summary checks
4. prompt self-optimization loop using the existing keep/discard control model

## What Not To Claim Yet

Until the runtime PoC lane is actually implemented, this repo should **not** claim:

- that `llama.cpp` BitNet runtime is already integrated here
- that webcam or microphone event handling exists
- that local vector memory is already wired
- that the Windows/CPU milestone framework has been completed

What we can claim now is narrower and accurate:

- the repo already has a CPU-first autonomous research loop
- it already has operator control, audit trail, and pause/resume support
- it is ready to become the orchestration backbone for the Windows/CPU runtime PoC track