# Provider-agnostic kernel compute CLI

**Status:** implementation proposal, 2026-11-21. This document plans the tool; it does not claim that the CLI or a new provider has been implemented.

**Implementation follow-up:** the Waterloo-first CLI is now implemented in [`compute/`](../../compute/README.md). See the [implementation and live-test report](compute-cli-validation.md) for observed behavior, test evidence, and differences from this original proposal. The proposal below is preserved as design history.

## Decision

Build a small Python package with a reusable execution service and a thin CLI, provisionally named **`kcompute`**. Its first provider is **Waterloo Data Science Cluster**, with identifier **`waterloo-datasci`**. Providers control where work runs; a shared GPU worker controls how a kernel is executed and measured.

The first useful release should let us submit several kernel tasks, close the terminal, and later retrieve each task's status, logs, and a consistent JSON result containing basic timings and memory measurements. Slurm is Waterloo's durable queue. We do not need to build a second scheduler or deploy a server to accomplish this.

Baseten and Velocity are future provider targets, not implementations or confirmed execution capabilities. Adding them should normally add a provider adapter and environment packaging, without changing task definitions, measurement semantics, or the CLI's job lifecycle.

## What exists, and what we can reuse

Repository observations below refer to commit `d29ffbcf4ed7ee45a7add0baaff8483a95454a7f`. Cluster observations refer to the separately dated investigation artifact. External API documentation is supporting design guidance, not evidence that a particular dependency works on this cluster.

- [Cluster investigation](datasci-gpu-investigation.md), “GPU and CUDA configuration” and evidence E3–E5: successful scheduler-managed access to `gpu-8-vm` and `gpu-1-vm` as full GPUs, and a `2g.48gb` MIG allocation on `gpu-6-vm`. `gpu-8-vm` has CUDA Toolkit 13.2.86; `gpu-1-vm` has 13.3.73. These are dated observations, not permanent node capabilities.
- [`pygpubench/evaluate_pygpubench.py`](../../pygpubench/evaluate_pygpubench.py), `_precheck`, `_bench`, `evaluate_pygpubench`, and the Modal setup: useful candidate loading, failure reporting, and correctness/timing patterns. Provider-specific image setup and evaluation currently live in the same script. Do not make importing the new core construct Modal images or import GPU packages on the user's laptop.
- [`KernelBench/src/kernelbench/timing.py`](../../KernelBench/src/kernelbench/timing.py), `time_execution_with_cuda_event`: existing CUDA-event timing code explicitly assumes work executes on the current stream. Its cold-cache protocol is not automatically the desired v1 protocol.
- [`KernelBench/pyproject.toml`](../../KernelBench/pyproject.toml) requires Python 3.10 and includes substantial model/provider dependencies. The root README mentions a root `pyproject.toml` and scripts that are not at those locations in the current tree. Create a self-contained package with accurate setup instructions instead of building on those stale paths.
- [`data/README.md`](../../data/README.md): the existing database contains projects, threads, and messages, not compute runs. Leave its schema unchanged for the CLI MVP.
- [Harness axioms](axioms.md): execution success, tested correctness, and performance improvement are separate claims. Basic profiling without an oracle must be labeled unvalidated, not correct.

## MVP scope

Implement one provider, one shared worker, one task interface, and one result schema:

- A Mac/Linux coordinator that does not require a local GPU or PyTorch.
- A Python GPU worker supporting CUDA/Triton/PyTorch operations exposed through a small Python task adapter.
- One GPU device or explicitly selected MIG instance per run; one workload configuration per run.
- Persistent submission, status, logs, result retrieval, and cancellation.
- Multiple task manifests submitted as separate jobs; an interrupted batch can resume without silently duplicating already-submitted runs.
- Warm-up, repeated CUDA-event measurements, basic allocator-memory measurements, and environment capture.
- Optional reference checks with explicit comparison criteria. No reference means `validation.status=not_requested` and no correctness claim.
- A simple CUDA vector-add example with a reference, plus a project-style `ModelNew` example to demonstrate that the interface fits our kernels.

Defer additional real providers, a web API, automatic provider selection, billing optimization, cross-provider migration, multi-GPU jobs, persistent warm GPU pools, task DAGs, Slurm arrays, Nsight hardware counters, and adversarial benchmarking. A fake provider is included for tests, not presented as another production compute service.

## Architecture and responsibilities

### CLI

Parses commands, loads a task and provider profile, calls the execution service, and formats output. It does not generate Slurm commands or calculate kernel timings.

### Execution service

Validates requests, creates stable run IDs, builds immutable bundles, persists metadata, delegates to a provider, reconciles state, and retrieves results. Both a future server and the CLI use this service.

### Compute provider

Owns credentials/configuration, resource discovery, file transport, environment selection, submission, scheduler-state translation, cancellation, and artifact retrieval. It must reject unsupported requirements instead of silently changing the requested GPU, MIG policy, or environment.

### GPU worker

Runs within the granted allocation. Verifies the actual device/environment, prepares the task, performs any requested validation, measures execution, and writes versioned results and logs. The worker has no knowledge of SSH, Slurm job submission, Baseten API calls, or CLI rendering.

### Local run store

Use SQLite under the user's application state directory for metadata, and ordinary per-run directories for downloaded artifacts. Do not put run logs, credentials, or large outputs into Git. Expose a small store interface so a later server can use PostgreSQL and object storage without changing the provider contract.

Keep this as one Python package initially; these are module boundaries, not separate services.

## Proposed command surface

```bash
# Installed from the new compute/ package; no local CUDA required.
kcompute providers list
kcompute doctor --provider waterloo-datasci

# Submit and return a stable run ID immediately after durable acceptance.
kcompute submit examples/vector-add/task.toml --provider waterloo-datasci

# Convenience: submit, wait, retrieve, and print the result.
kcompute run examples/vector-add/task.toml --provider waterloo-datasci

# Queue several independent tasks; each receives its own run ID.
kcompute submit tasks/add.toml tasks/matmul.toml --provider waterloo-datasci

kcompute status RUN_ID
kcompute logs RUN_ID --follow
kcompute result RUN_ID --json
kcompute cancel RUN_ID
kcompute list --provider waterloo-datasci
```

All commands support machine-readable JSON. JSON goes to stdout; progress and human diagnostics go to stderr. `submit` success means the job was accepted, not that the kernel passed. `run` exits nonzero for execution failure or a requested validation failure. `result` success means a valid result was retrieved; callers must inspect its outcome fields. Waiting interruption detaches without cancelling; `cancel` is explicit. A wait timeout also detaches and returns the run ID, rather than pretending the job stopped.

The CLI must distinguish a single-run JSON object from a batch-submission response. A batch response lists accepted, failed, and ambiguous submissions individually; it cannot report whole-batch success when only part was submitted.

## Task contract: a kernel needs an executable workload

A `.cu` file alone does not say how to allocate inputs, choose shapes, launch the kernel, or check outputs. V1 accepts a task folder containing an explicit manifest, Python adapter, and allowlisted source files. The provider never interprets the kernel's signature.

Proposed manifest:

```toml
schema_version = 1
name = "vector-add-fp32"
entrypoint = "task.py:build"
files = ["task.py", "kernel.cu"]
environment = "cuda-pytorch-v1"

[case]
n = 1048576
dtype = "float32"
seed = 42

[resources]
gpu_count = 1
min_gpu_memory_gib = 80
mig = "forbid"
cpu_cores = 4
host_memory_gib = 8
wall_time_s = 600

[measurement]
warmup_iterations = 10
sample_count = 100
cache_policy = "warm"
timing_scope = "callable_current_stream"

[validation]
mode = "reference"
seeds = [42, 43, 44]
rtol = 0.00001
atol = 0.000001
equal_nan = false
```

The 80 GiB requirement demonstrates a whole 96 GB-class GPU request; it is not a vector-add memory requirement or a universal default. A default run requests one full GPU, four CPU cores, 8 GiB host RAM, and ten minutes; tasks may override these within provider limits. Provider-neutral resources describe desired capabilities, not Slurm GRES counts. The Waterloo profile translates them to node eligibility and Slurm options.

The Python adapter exposes `build(case, device) -> PreparedTask`. The worker supplies the selected device and a reproducible case/seed. A `PreparedTask` supplies:

- `make_inputs(seed)`: creates inputs before a validation or timing trial.
- `run(*inputs)`: the operation being measured; may call a compiled CUDA extension or Triton kernel.
- Optional `reference(*inputs)`: separate expected-result computation.
- Input/output metadata and a declared current-stream, single-device execution contract.

V1 output checking supports a tensor or a fixed tuple of tensors, with explicit shape/dtype and floating-point tolerances; integer/Boolean outputs compare exactly. General nested structures and custom checkers can follow later. V1 repeated timing requires read-only inputs and stateless behavior; reject tasks declaring input mutation or persistent state until reset semantics are implemented. Use separate, equivalent input objects for candidate/reference validation. The worker never invents numerical tolerances from a candidate's observed errors.

A KernelBench adapter can construct this interface from `Model`, `get_init_inputs`, `get_inputs`, and `ModelNew`, preserving model initialization/weights and the task's reference/check policy. Do not automatically import all KernelBench dependencies or change its Python-version restriction. The first project-style example can be self-contained; a general KernelBench importer is a follow-up.

## Provider interface

Keep the initial interface small:

```python
class ComputeProvider(Protocol):
    def describe(self) -> ProviderCapabilities: ...
    def preflight(self, request: RunRequest) -> PreflightReport: ...
    def submit(self, run_id: str, bundle: Bundle, request: RunRequest) -> JobHandle: ...
    def observe(self, handle: JobHandle) -> JobObservation: ...
    def read_logs(self, handle: JobHandle, cursor: LogCursor | None) -> LogChunk: ...
    def cancel(self, handle: JobHandle) -> CancelAcknowledgement: ...
    def fetch(self, handle: JobHandle, destination: Path) -> ArtifactManifest: ...
```

These are conceptual signatures, not an implemented API. `submit` includes staging required files; `JobHandle` carries a provider ID, opaque native job ID, cluster identity where relevant, submission time, and artifact locator. The service persists it. A job ID alone is not a globally stable identity. `observe` returns normalized state, raw provider state, timestamps when available, and structured failure reasons.

`read_logs` returns bounded stdout/stderr chunks and an opaque continuation cursor. The CLI implements `--follow` by polling; a provider may report that logs are available only after completion. Final artifact retrieval remains independent of live log delivery.

Capabilities describe supported execution/runtime types, full-device versus MIG selection, cancellation, and profiling availability. Do not require future providers to emulate SSH or Slurm. `preflight` checks accessible capabilities; actual allocation constraints are rechecked by the worker before loading task code. Successful preflight is not a resource reservation.

## Waterloo implementation

Provider display name: **Waterloo Data Science Cluster**. Adapter ID: **`waterloo-datasci`**.

User-local provider configuration supplies the SSH host/username, known-host verification, key reference, remote workspace, environment profile path, and optional provider-specific node restrictions. Never bundle private keys or account tokens with a task. Use the existing SSH agent/config; unattended use fails clearly on authentication or host-key problems rather than prompting indefinitely.

Initial named profile:

- Host `datasci-gpu.cs.uwaterloo.ca`; partition `ALL`.
- Pin to `gpu-8-vm` for the first implementation and acceptance tests. Default to non-MIG; do not silently fall back to a slice when the node is busy.
- Toolkit path `/usr/local/cuda-13.2`, validated on the allocated node. Explicitly set `CUDA_HOME` and the job's PATH; the current noninteractive environment does not expose `nvcc` by default.
- Pre-created, versioned Python environment under the user's remote cache. Record an environment lock/hash and worker version. Reuse it across jobs, without modifying it concurrently or installing dependencies on every run.
- Run a small setup/compatibility job before accepting this environment profile. Pin a PyTorch build and compiler configuration proven to work with compute capability 12.0. The cluster inventory did not test PyTorch or compile a kernel; do not assume that compatibility is already established.

Submission path:

1. Validate and snapshot the manifest and allowlisted files. Hash contents and resolved environment settings. Editing a local file afterward cannot change a submitted run.
2. Persist the run ID and request locally before network side effects.
3. Upload a bundle to a unique user-owned remote run directory through SSH/SFTP. Validate paths, sizes, and checksums; do not upload the entire repo, `.git`, environments, or secrets.
4. Create a fixed batch-script template that runs the shared worker with a request-file path. Avoid interpolating arbitrary shell fragments from manifests. Schedule compute and compilation on compute nodes, not the login node.
5. Submit with `sbatch --parsable`, explicit wall time/CPU/RAM/GPU requests, job-specific working directory, and separate stdout/stderr files. Record the native response remotely and locally.
6. Return the run ID. Slurm continues scheduling and executing when the user's laptop disconnects. Slurm does not transfer task files itself; staging is the adapter's responsibility. [Slurm `sbatch`, Description and `--parsable`](https://slurm.schedmd.com/sbatch.html).
7. Poll bounded, batched `squeue` queries for active jobs and `sacct` for finished jobs; retain pending reasons. Initial polling interval: 15 seconds with backoff on transport errors. Account for accounting delay and job-step records; use the parent job for lifecycle and the relevant steps for diagnostics. [Slurm `sacct`, Description](https://slurm.schedmd.com/sacct.html).
8. Fetch the finalized result manifest and its checksummed artifacts. A failed download retries retrieval, not kernel execution. Explicit `cancel` calls `scancel` for this run's verified job ID and waits for observed terminal state when requested.

Do not parse the human-formatted `sresources` table as the only machine API. Use stable Slurm fields/JSON where supported by the installed 24.11.6 version, and test parsing against captured output. The node's generic `gres/gpu` count is insufficient to establish physical GPU count, model, usable memory, or MIG status; those require profile constraints plus allocated-device verification.

V1 submits one batch job per run and lets Slurm queue it. It does not promise FIFO order, queue-start deadlines, a cluster-wide concurrency cap, or uninterrupted execution. A per-request execution wall time is enforced by the batch allocation; a CLI wait timeout has different semantics. Add durable orchestration/concurrency policies later only if the server needs them.

## Run lifecycle and failure handling

Separate local transfer/submission state, provider job state, and result availability. Suggested normalized lifecycle states are `staging`, `submitting`, `queued`, `running`, `succeeded`, `failed`, `timed_out`, and `cancelled`; `unknown` is an observation condition, not a terminal result. A run may be remotely completed while its artifacts are not yet downloaded.

- Persist stage transitions and native identifiers. On reconnect, reconcile remote metadata and scheduler state instead of launching a replacement job.
- Guard submission with a remote per-run lock and persist a submission intent keyed by run ID and request hash. Repeating a run ID with different content is an error.
- If the connection drops after Slurm may have accepted the job, mark submission ambiguous. Recover using the remote receipt or a matching run identifier in owner-scoped scheduler/accounting records. Absence from a delayed accounting response is not proof of non-submission. Do not automatically resubmit an ambiguous attempt; v1 does not promise exactly-once execution across all failure windows.
- `squeue` disappearance alone is not success. Combine accounting state with a validated final worker result. A completed job with no valid result is an incomplete/protocol failure, not a successful benchmark.
- Candidate compile errors, numerical failures, OOM, CUDA runtime errors, worker crashes, allocation timeouts, provider failures, and cancellation retain distinct reasons. Missing measurements are `null` with a reason, never fabricated zeroes.
- Cancellation acknowledges a request first; actual terminal state determines the outcome if completion raced with cancellation. Preserve any already-produced diagnostics.
- Automatic retries are limited to safe control-plane operations and artifact transfers. Re-running a kernel is an explicit new attempt linked to the previous run; do not overwrite original evidence.

Remote artifacts stay in the run directory until explicitly cleaned up. Bound log/output sizes. Automatic retention/deletion policy is a later decision because the user's storage quota remains unverified.

## Basic profiling protocol

The shared worker, not the provider, owns the versioned measurement protocol. V1 measures a **callable invocation**, which may launch one or several kernels; do not label it as instruction-level or pure device-kernel duration.

1. Capture allocated device, MIG profile, available device memory, driver, driver-reported CUDA version, actual toolkit/compiler, Python/PyTorch versions, task and environment hashes, and worker version.
2. Verify requirements and prepare/compile the task. Record preparation wall time separately. Lazy compilation/autotuning triggered by initial calls belongs to warm-up, not steady-state latency.
3. If reference validation is requested, run the specified cases/seeds and comparison rule first. Stop on failure and return diagnostics without a valid performance score. A passing check describes only the tested cases, not a proof over all inputs.
4. Warm up ten calls by default. Synchronize, then collect 100 valid samples using CUDA timing events on the declared current stream. Keep input allocation, reference execution, validation, and host/device transfers outside the timed region. Cache policy is explicitly warm; do not mix with existing cold-cache results.
5. Synchronize completion before reading elapsed time. Convert event milliseconds to microseconds explicitly. Store raw samples plus mean, median, minimum, maximum, population standard deviation, and nearest-rank p95. Record sample/warm-up counts and protocol version. CUDA-event timing and synchronization follow [PyTorch Event documentation](https://docs.pytorch.org/docs/2.15/generated/torch.cuda.Event.html).
6. Measure an empty event pair separately as a timing-resolution diagnostic; do not blindly subtract it. Flag samples near that overhead. Candidate work on unmanaged CUDA streams is unsupported by this v1 protocol; a full-device synchronization after the end event alone does not include work outside that event interval.
7. Separately measure PyTorch allocator peak memory after warm-up, outside latency sampling: recreate inputs, reset peak stats before one representative invocation, and record baseline/peak allocated and reserved bytes. These include resident inputs/model memory and exclude allocations invisible to PyTorch. Label them allocator metrics, not total VRAM usage. See [PyTorch CUDA memory implementation and API descriptions](https://github.com/pytorch/pytorch/blob/main/torch/cuda/memory.py).

Report `prepare_wall_ms`, `warmup_wall_ms`, worker wall time, and scheduler queue/allocation times separately when available. Compilation time is only a separate metric when a build hook can actually isolate it. Do not use `nvidia-smi` polling to claim per-microsecond kernel utilization, energy, or accurate peak memory. SM occupancy, bandwidth, FLOPs, instruction counts, and Nsight traces are later profiling modes.

An optional baseline timing uses the same inputs, allocation, precision, and timing protocol after successful validation. Define speedup as baseline median divided by candidate median. Missing validation or incomparable protocols mean no speedup claim. Baseline timing can follow after the first basic-latency release.

## Result contract

Return a versioned JSON document, plus separate stdout/stderr, request snapshot, raw timing samples, and artifact checksums. Required groups:

- **Identity:** schema version, run ID, task name/hash, request hash, environment hash, protocol and worker versions, optional future project/task/candidate IDs.
- **Execution:** provider and native job identity, node/device, lifecycle timestamps, status, exit code, structured error stage/reason, and result availability.
- **Validation:** `passed`, `failed`, `not_requested`, or `not_run`; reference/case identity, seeds and tolerances, diagnostics, and tested domain. Do not represent skipped checks as a pass.
- **Latency:** units (`us`), timing scope, warm-up count, sample count, summary statistics, and raw-sample artifact.
- **Memory:** baseline/peak PyTorch allocated/reserved bytes and explicit scope; unavailable measurements carry reasons.
- **Environment:** GPU model/compute capability/usable memory/MIG configuration, host CPU/RAM allocation, driver, CUDA driver support, toolkit and framework versions, power limit when readable.
- **Artifacts and warnings:** log paths/checksums, truncated-log flags, hardware mismatch, unsupported measurement, or low timing-resolution warnings.

The worker writes results to a temporary file and atomically renames the finalized file, then writes a completion manifest. Task stdout is never parsed as the authoritative result channel. A supervisor converts abrupt worker termination into a structured failure when possible; provider timeout or crash records cover cases where no worker result survives. Result validation verifies schema, run/request identity, and finite nonnegative measurements.

## Execution trust boundary

This MVP is a personal/team tool for code run under the submitting user's cluster identity. A separate process and Slurm allocation help contain resource consumption and crashes, but they are not a hostile-code security boundary and do not prevent a malicious task from forging local files. Basic correctness checks and profiling are not an adversarial benchmark evaluator.

Use per-run work/build directories and bounded processes; keep connection credentials on the coordinator. Do not call inherited shared caches immutable if candidate code can write them. Before a server accepts arbitrary uploads or serves multiple users, design stronger isolation, separate authorization and execution identities, quotas, artifact access control, and trusted evaluation. That is a prerequisite to that future deployment, not an additional approval flow for this CLI.

## Repository layout and implementation steps

Create `compute/` alongside `data/`, `frontend/`, and `pygpubench/`:

```text
compute/
  pyproject.toml
  uv.lock
  README.md
  src/kernel_compute/
    cli.py
    models.py
    service.py
    store.py
    bundle.py
    providers/base.py
    providers/waterloo_datasci.py
    worker/main.py
    worker/tasks.py
    worker/measure.py
  examples/vector-add/
  examples/model-new/
  tests/
```

Proposed stack: Python 3.12 for the new package and worker, `argparse` for the small CLI, Pydantic for versioned request/result validation, standard-library SQLite for local state, system OpenSSH/SFTP for Waterloo transport, and GPU dependencies installed only in the worker environment. Validate and lock exact dependency versions during implementation; do not inherit KernelBench's Python 3.10-only packaging accidentally.

### Step 1 — Define contracts and a standalone worker

Implement request/result models, bundle allowlists/hashes, the task adapter, the worker, and a CUDA vector-add task. Establish a pinned remote environment with a tiny correctness/timing smoke test on `gpu-8-vm`. This is the first execution milestone; no production queueing feature depends on an untested environment.

Acceptance: a single local-on-GPU worker invocation emits a valid result with correct units, expected output checks, raw timings, actual hardware/software metadata, and scoped memory metrics. Compile failure, wrong output, and a worker timeout emit distinct outcomes.

### Step 2 — Waterloo submission and retrieval

Add SSH preflight, remote staging, batch-script generation, `sbatch` submission, durable handle storage, status reconciliation, logs, fetch, and cancellation. Implement `submit`, `status`, `result`, and `cancel` through the service; add `run` as composition of those operations.

Acceptance: submit from the Mac, terminate the client, then retrieve the completed run from a new process. The kernel ran on the expected full GPU; waiting/cancelling has documented semantics; no inspection jobs or stray processes remain afterward.

### Step 3 — Several queued kernels and failure recovery

Add multi-manifest submission, run listing, resumable result downloads, idempotency/reconciliation handling, and a fake provider for lifecycle tests. Add the self-contained `ModelNew` example. Document installation and provider configuration in `compute/README.md`; link it from the root README without claiming to have repaired unrelated legacy workflows.

Acceptance: queue several tasks, observe independent results, cancel one pending job, recover from a dropped SSH connection, and distinguish partial-batch submission from success. Repeated retrieval or ambiguous submission must not accidentally launch another kernel.

**The MVP ends here.** All three steps deliver the requested provider-agnostic CLI and basic profiling. Exact timing of delivery should be estimated after the environment smoke test, since cluster dependencies and SDK support are not yet validated.

### Step 4 — Later provider/server integration

For each of Baseten and Velocity, first verify the actual product/API, authentication, arbitrary CUDA-code execution, environment control, scheduling, cancellation and artifact model. An inference-only endpoint is not automatically a general kernel-execution provider. If compatible execution is available, implement the same provider interface and run the same contract tests; otherwise record the missing capability instead of forcing it into the abstraction.

A Python server can call `ExecutionService` directly. A TypeScript server can initially invoke the versioned CLI JSON protocol through an argument-array subprocess (not shell strings), or call a small Python service later. The server provides its own identity/authorization and job tracking. Add run/artifact database tables only when that integration is implemented; do not overload conversation messages as the durable compute queue.

## Verification plan

- Unit tests for manifest validation, unit conversion, statistics from fixed samples, archive path safety, immutable bundle hashing, and result identity checks.
- Fake-provider lifecycle tests for delayed accounting, lost submission acknowledgement, pending/running cancellation races, provider failure, timeout, malformed/missing result, and interrupted downloads. Exercise actual recovery behavior rather than mirroring implementation details.
- Waterloo parser tests using the captured 24.11.6 outputs from the cluster investigation, including generic GRES counts and absent user association rows.
- Small live GPU tests: valid vector add, deliberate wrong result, compilation error, bounded hang/timeout, unavailable full-GPU requirement, and a queued job cancelled before start. Use short limits and verify cleanup.
- Timing sanity: known CUDA operation with correct units and nonzero finite timings; verify setup/validation is outside the interval and memory measurement is outside latency trials. Do not assert a fragile fixed runtime in tests.
- Provider independence: the fake provider runs the same lifecycle/result contract without SSH/Slurm-specific fields entering the task or worker API. This tests the seam without implementing a speculative second cloud provider.

## Decisions to revisit only when needed

- Exact pinned Python/PyTorch/compiler environment after the Blackwell smoke test.
- Whether formal benchmark evaluation should use a separately validated PyGPUBench adapter after support for this GPU and cluster isolation features is established.
- Storage retention and quotas once administrators confirm them.
- Server language and stronger isolation model before multi-user or untrusted-code deployment.
- Baseten/Velocity execution capabilities when those providers are actually added.

The provider-neutral boundary, Waterloo-first scope, asynchronous Slurm submission, one-GPU workload adapter, and basic metrics are concrete enough to implement without resolving those later choices first.

Return to the [artifacts index](README.md).
