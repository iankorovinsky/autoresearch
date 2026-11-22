# Kernel compute CLI

`kcompute` queues kernel workloads on a compute provider and retrieves versioned profiling results. The default provider is **Waterloo Data Science Cluster** (`waterloo-datasci`), pinned to **`gpu-8-vm`**. **Modal** (`modal`) provides selectable GPU types. A shared Python worker performs execution and measurement. The provider handles scheduling and transport. The coordinator runs on Mac/Linux without PyTorch or a GPU.

## Install and configure

From this directory:

```bash
uv sync --locked --python 3.12
uv run kcompute providers list
```

For a standalone executable available outside the checkout, run `uv tool install .` from this directory. Use Python 3.12 or newer. The package includes the worker, remote helper, and environment dependency lock.

Create `~/.config/kcompute/config.toml`:

```toml
[providers.waterloo-datasci]
user = "YOUR_WATERLOO_USERNAME"
# Defaults; normally omit these.
host = "datasci-gpu.cs.uwaterloo.ca"
partition = "ALL"
node = "gpu-8-vm"
cuda_home = "/usr/local/cuda-13.2"
remote_root = ".cache/kcompute"
```

Alternatively set `KCOMPUTE_SSH_USER`. `KCOMPUTE_CONFIG` or the command's `--config` option selects a different configuration file. Existing OpenSSH configuration/agent credentials are used. Establish and verify SSH access first: unattended commands require a trusted known-host entry and never disable host-key verification. No passwords or keys are copied into task bundles.

```bash
uv run kcompute doctor
uv run kcompute setup --wait
uv run kcompute doctor
```

`setup` submits a bounded, 20-minute Slurm environment setup job. It creates a user-owned environment addressed by the SHA-256 of the dependency lock, installs hash-checked wheels, checks PyTorch GPU execution, and compiles/runs a small CUDA program on the selected node. Dependencies include PyTorch **2.9.1+cu130**, Triton **3.5.1**, NumPy **2.3.5**, and Ninja **1.13.0**. CUDA Toolkit **13.2** is supplied by the cluster; framework CUDA **13.0** and driver-supported CUDA **13.2** are different version fields. The tested Python on the node is **3.12.3**.

Setup returns an existing ready environment or the existing setup receipt when repeated; it does not silently resubmit an ambiguous/failed setup. Failed setup output identifies its job and remote logs. Inspect that attempt before manually clearing a failed setup receipt/intent and retrying. Ordinary runs do not install packages. Workers verify the installed package inventory against the setup record before loading task code.

## Use Modal

Authenticate with `uv run modal token new` or use an existing Modal SDK profile.
Modal requires a payment method for GPU allocations. The optional profile below
sets the default GPU and selects a Modal environment. The CLI `--gpu` option
overrides the profile for that invocation.

```toml
[providers.modal]
gpu = "T4"
# environment_name = "main"
# app_name = "kcompute"
# volume_name = "kcompute-artifacts-v2"
```

```bash
uv run kcompute providers list --provider modal
uv run kcompute setup --provider modal
uv run kcompute probe --provider modal --gpu T4
uv run kcompute run examples/vector-add/task.toml --provider modal --gpu T4
uv run kcompute submit examples/vector-add/task.toml --provider modal --gpu L4
```

Modal setup prepares a CUDA 13.0 development image and a Modal App and Volume.
Runs and probes can also prepare their dependencies on first use. `T4` is the
default and the lowest-priced GPU in [Modal's published pricing](https://modal.com/pricing).
The provider lists every documented [Modal GPU selector](https://modal.com/docs/guide/gpu),
including aliases and variants. Each task requests one full GPU. Its manifest
sets CPU, host memory, minimum GPU memory, and wall time (up to 24 hours on
Modal). Keep the same Modal workspace and environment active when retrieving or
cancelling a run.

`probe` launches one bounded GPU Sandbox and returns JSON with the requested
and actual GPU and a result for each check. It measures a CUDA event interval,
runs a compiled CUDA kernel and a Triton kernel, checks NVIDIA management data,
and looks for CUDA activity in PyTorch Profiler and Nsight Systems. Nsight
Compute must return a numeric GPU hardware counter to pass. It runs with GPU
clock locking disabled. Tool results can
vary across GPU types or workspaces. The probe is separate from task timing
and reports tool failures in its JSON.

On the Modal T4 with PyTorch 2.9.1 and CUDA 13.0, Kineto disables CUPTI's
default per-thread activity buffering before it flushes the first trace,
which loses CUDA kernel events. The probe completes a separate warm-up
profiler session in the same process, then checks the exported trace from the
next session for CUDA kernel events. Use the same two-session pattern when
profiling your own workload on this image. See the [Kineto buffer-flush fix](https://github.com/pytorch/kineto/commit/19283706e915).

Modal submissions survive CLI exit. The scheduler reports a Sandbox as pending
or running until it exits. Final worker results, samples, and logs are retained
on the configured Modal Volume. The CLI reads worker logs after they have been
published and can fetch the artifacts in a separate process.

## Run or queue tasks

```bash
# Submit, wait, and print a complete result.
uv run kcompute run examples/vector-add/task.toml --provider waterloo-datasci

# Submit and return immediately; the job survives terminal/laptop disconnection.
uv run kcompute submit examples/model-new/task.toml --provider waterloo-datasci

# Each manifest becomes an independent scheduled job.
uv run kcompute submit examples/vector-add/task.toml examples/model-new/task.toml

uv run kcompute status RUN_ID
uv run kcompute logs RUN_ID --follow
uv run kcompute result RUN_ID --json
uv run kcompute cancel RUN_ID --wait
uv run kcompute list
uv run kcompute resume BATCH_ID
```

`run` is submission plus waiting and retrieval. Ctrl-C or `--wait-timeout SECONDS` detaches; it does **not** cancel the remote job. Use `cancel` explicitly. Waiting returns exit code 124 on timeout, and interruption returns 130. A submitted run's provider configuration is saved with it, so later profile edits do not redirect status/cancellation to a different cluster.

`submit` records a batch ID and all run IDs before remote submission. Single-task output is one run object with its batch ID; multiple tasks return a batch object with `runs`. Partial or ambiguous submissions produce a nonzero exit and per-run outcomes. `resume` submits only the still-prepared members, reconciles ambiguous members, and never silently launches another attempt for an already-submitted run. `--attempt-of RUN_ID` on a new `submit`/`run` explicitly links a new attempt to an earlier one.

All commands support `--json`. Results/status/submission are JSON by default; progress goes to stderr. Logs are text unless `--json` is requested; following JSON logs emits JSON Lines with continuation cursors. One non-following log read is bounded to 64 KiB per stream; `--follow` drains subsequent chunks. `list` reads cached local state; use `status` to refresh a run. Polling defaults to 15 seconds, with transport-error backoff.

`submit` exit 0 means durable acceptance. `run` exit 0 requires successful execution and any requested validation. `result` exit 0 means retrieval/validation of the result document succeeded, even when that document describes a failed kernel. Automation must inspect `execution.status` and `validation.status`.

## Workload contract

Use a folder with a TOML manifest, a Python adapter, and explicitly listed source files. A CUDA source file alone does not specify inputs or launch arguments. See [CUDA vector addition](examples/vector-add/task.toml) and [ModelNew adapter](examples/model-new/task.py).

```toml
schema_version = 1
name = "my-kernel"
entrypoint = "task.py:build"
files = ["task.py", "kernel.cu"]
environment = "cuda-pytorch-v1"

[case]
n = 1048576
seed = 42

[resources]
gpu_count = 1
cpu_cores = 4
host_memory_gib = 8
wall_time_s = 600
min_gpu_memory_gib = 0
mig = "forbid"

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

The adapter defines `build(case, device)` returning an object with:

- `make_inputs(seed)`: a tuple of input tensors, allocated before timing.
- `run(*inputs)`: the operation to measure; returns a tensor or fixed tuple of tensors.
- `reference(*inputs)` when reference validation is requested.
- Declarations `mutates_inputs = False`, `stateful = False`, and `timing_scope = "callable_current_stream"`.

V1 supports one device, read-only inputs, stateless repeated invocations, and current-stream GPU work. Use the provided device/current stream when launching CUDA or Triton. These are task contracts for trusted code, not enforcement against dishonest adapters. Input reset/stateful tasks, nested outputs, and unmanaged multi-stream execution are unsupported. Reference inputs are cloned before candidate execution; shape, dtype, device, and comparison tolerances are checked. Integer/Boolean outputs compare exactly.

Omitting `[validation]` profiles without an oracle and returns `validation.status = "not_requested"`. It never means correct. Validation failure stops performance measurement. Passing the selected cases is not a proof over other inputs.

Bundles snapshot manifest values and allowlisted source bytes before upload. They reject escaping paths, symlinks, hidden files, credential extensions, and common environment/cache folders; the complete archive is limited to 32 MiB. Bundle allowlisting is not a secret scanner: explicitly review the source files you include. Changing local files after submission does not change an existing run.

## Metrics and artifacts

The versioned `cuda-event-warm-v1` protocol reports:

- CUDA-event **callable latency** in microseconds: mean, median, minimum, maximum, population standard deviation, nearest-rank p95, raw samples, warm-up/sample counts, and event-pair overhead.
- PyTorch allocator baseline/peak allocated and reserved **bytes**, measured separately after warm-up. Includes resident inputs/model memory; does not capture external CUDA allocations or total physical VRAM use.
- Preparation, warm-up, and worker wall times; scheduler queue/allocation wall times when UTC accounting timestamps are available (second-resolution source).
- Actual GPU/compute capability/MIG status/device memory, CPU affinity count, CUDA compiler and framework versions, driver/support version, package inventory, source/environment hashes, protocol version, and requested CPU/RAM/time allocation.

Timing includes everything between two events around the Python callable on its current stream, including GPU idle gaps from host dispatch. It is not isolated instruction/device-kernel duration. Warm cache is explicit. The worker does not subtract overhead; short measurements near the empty-event interval receive a warning. Compilation, input generation, reference checks, and host/device transfers belong outside the timed region. Memory sampling is separate. Speedup and Nsight counters are not part of this release.

Local state defaults to `~/.local/state/kcompute/` (`KCOMPUTE_STATE_DIR` or `--state-dir` overrides it). SQLite stores lifecycle metadata; `runs/RUN_ID/` contains the frozen bundle, final service result, and downloaded `artifacts/` with request, worker result, samples, logs, and checksum manifest. The provider verifies identities/checksums, retries retrieval without rerunning the kernel, and only treats execution as successful when accounting and a valid worker result agree.

On Waterloo, remote data lives under `~/.cache/kcompute/runs/RUN_ID/`. Worker stdout/stderr are capped at 1 MiB each with truncation flags. Temporary build/cache directories are per run. Retention is explicit: no automatic cleanup or retention command is implemented yet, and cluster storage quotas remain unverified. Do not put local state/logs into Git. No GPU process is retained between runs.

## Extending and using the service

The reusable [ExecutionService](src/kernel_compute/service.py) takes a [RunStore](src/kernel_compute/store.py) and provider factory. It owns request snapshots, run IDs, lifecycle reconciliation, retrieval, and result validation. The [provider protocol](src/kernel_compute/providers/base.py) owns scheduling/transport; [worker code](src/kernel_compute/worker/main.py) owns execution/measurement and imports no scheduler API. A future Python server can call the service directly; another server can consume the CLI's JSON contract through an argument-array subprocess.

The Waterloo adapter uses owner-scoped Slurm job identity checks, persistent submission intents and receipts, and conservative recovery. If acceptance cannot be established after a connection loss, state remains ambiguous rather than resubmitting. This avoids accidental duplicate work but does not claim exactly-once execution. A terminal job without a finalized result becomes a structured failure after a 30-second storage/accounting visibility grace period.

Add another provider by implementing the protocol and registering its configuration/environment mapping. Waterloo uses Slurm and offers no FIFO/start-time promise. The default remains the tested full GPU on `gpu-8-vm`.

This is a personal/team runner under the submitting user's identity, not an adversarial benchmark service. Per-run processes/directories and scheduler allocations do not isolate malicious code from the user's files or writable environment. Multi-user server deployment requires a separate isolation/authorization design.

## Tests and evidence

```bash
uv run pytest -q
uv run ruff check src tests examples

# Explicit opt-in: submits short jobs on gpu-8-vm under your account.
KCOMPUTE_LIVE=1 KCOMPUTE_SSH_USER=YOUR_WATERLOO_USERNAME uv run pytest -m live -q -s
```

The ordinary suite uses a fake provider, archive/parser fixtures, and protocol/error cases; it does not need CUDA or SSH. Live tests cover a real compiled CUDA kernel, a self-contained ModelNew workload, batch failures, wrong outputs, compiler errors, timeouts, resource mismatch, pending/running cancellation, unvalidated profiling, fresh-process retrieval, and dropped-acknowledgement recovery. They cancel unfinished jobs during cleanup and write an evidence JSON file to `/tmp/kcompute-live-results.json` (override with `KCOMPUTE_LIVE_REPORT`).

The GPU environment's canonical lock is [providers/worker.lock](src/kernel_compute/providers/worker.lock); [worker.in](environments/worker.in) lists the direct requirements. Regenerate deliberately with `uv pip compile environments/worker.in --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes --index-strategy unsafe-best-match --output-file src/kernel_compute/providers/worker.lock`, then run setup and the live compatibility tests again. This creates a new environment hash instead of mutating the previous profile.

See the [implementation and live-test report](../wiki/artifacts/compute-cli-validation.md), [original plan](../wiki/artifacts/compute-cli-plan.md), and [cluster investigation](../wiki/artifacts/datasci-gpu-investigation.md). Slurm acceptance/file-staging semantics are documented in [`sbatch`, Description](https://slurm.schedmd.com/sbatch.html); CUDA event behavior is documented in [PyTorch Event](https://docs.pytorch.org/docs/2.15/generated/torch.cuda.Event.html). The checked-in code and live evidence define this release's behavior.
