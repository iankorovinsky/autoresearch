# Compute CLI implementation and live-test report

**Implemented and tested: 2026-11-21. Final cluster audit: 22:10:31 UTC.**

This is implementation evidence, separate from the [original proposal](compute-cli-plan.md). See the [CLI README](../../compute/README.md) for installation, configuration, task adapters, commands, metrics, and the reusable service interface. Frozen observations, source SHA-256 hashes, full result objects, final raw samples, environment inventory, and Slurm accounting output are in [the evidence JSON](compute-cli-validation-results.json).

## Delivered

The new `compute/` Python package provides `kcompute providers list`, `doctor`, `setup`, `run`, `submit`, `status`, `logs`, `result`, `cancel`, `list`, and `resume`. Multiple manifests submit independent jobs. A shared execution service and GPU worker keep kernel execution/measurement separate from the Waterloo provider's SSH/Slurm implementation.

Waterloo defaults to `gpu-8-vm`, partition `ALL`, one full GPU, four CPU cores, and 8 GiB host RAM. User-local configuration uses the previously verified `ikorovin` account; credentials were not copied into the repo or task bundles. Local state is under `~/.local/state/kcompute/`, and remote runs/environments under `~/.cache/kcompute/`.

Durable run IDs, local SQLite records, frozen source bundles, remote intents/receipts, identity checks, and checked artifact transfers support later retrieval and conservative reconnect recovery. `run` waits for submission and retrieval; interrupting the wait detaches. Cancellation is explicit. Failed runs retain structured diagnostics instead of fabricated measurements.

## Environment actually validated

Setup job **8115** completed on `gpu-8-vm` in **96 seconds**, with four CPUs, 8 GiB host RAM, one GPU, and a 20-minute maximum. Setup installed hash-checked wheels into a user-owned environment, executed a PyTorch GPU smoke test, and compiled/executed a CUDA program with the cluster toolkit.

- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition; compute capability **12.0**; MIG **disabled**.
- PyTorch-visible device memory: **101,975,851,008 bytes**. This runtime figure differs from the earlier physical-board `nvidia-smi` inventory figure; neither represents free memory throughout a workload.
- Driver: **595.58.03**; driver-reported CUDA support: **13.2**.
- Compiler: `/usr/local/cuda-13.2/bin/nvcc`, **13.2.86**.
- Framework: **PyTorch 2.9.1+cu130**, framework CUDA **13.0**, Triton **3.5.1**; Python **3.12.3**.
- Environment lock SHA-256: `127fdfce142035609925c7623b2a13d863be68961e53a622b951535e7c2a2b35`.
- Final workload observed **four CPUs in its CPU affinity mask**. Requested host RAM was 8 GiB; this report does not claim a separately measured memory cgroup ceiling.

The toolkit/framework version difference was exercised successfully here; this does not establish compatibility for every CUDA extension or framework. The worker checks the installed package inventory against the setup record before loading task code.

## Verification

**33 ordinary tests passed**, with the seven live tests skipped by default. **Seven opt-in live tests passed in 77.34 seconds**, using nine workload jobs. The final worker revision was then exercised by another successful vector-add run. The package built into a source distribution and wheel, and the wheel's CLI was checked in an isolated environment. Ruff and whitespace checks passed.

Ordinary tests cover bundle snapshots/path traversal/symlinks/credential paths, explicit validation policy, statistics and units, GPU-free coordinator imports, invalid CLI arguments before submission, fake-provider recovery after lost acknowledgement and delayed accounting, missing/malformed results, interrupted downloads, cancellation/completion races, detach timeouts, partial batches, Slurm owner/name matching, remote submission-intent handling, archive collisions, and artifact checksums/resumed retrieval.

Live outcomes:

| Slurm job | Case | CLI execution outcome | Validation | Failure reason |
| --- | --- | --- | --- | --- |
| 8117 | CUDA vector addition + fresh-process retrieval | succeeded | passed | — |
| 8118 | Incorrect output | failed | failed | validation_failed |
| 8119 | Invalid CUDA source | failed | not_run | compile_error |
| 8120 | Worker timeout | timed_out | not_run | worker_timeout |
| 8121 | Insufficient requested VRAM | failed | not_run | environment_mismatch |
| 8122 | Running cancellation | cancelled | not_run | scheduler_cancelled |
| 8123 | Pending cancellation | cancelled | not_run | scheduler_cancelled |
| 8124 | Dropped acknowledgement + ModelNew | succeeded | passed | — |
| 8125 | Unvalidated profiling | succeeded | not_requested | — |

The lost-acknowledgement test deliberately discarded the response **after real Slurm acceptance**. A fresh service recovered the same job, resumed the batch, and retrieved the result without resubmitting. This tests that concrete failure window, not exactly-once execution under every network/filesystem failure.

The pending cancellation had **zero allocated CPUs**, no assigned node, and zero elapsed runtime. The running cancellation stopped the short test job. Incorrect-output and compile-error runs returned no valid performance measurements. Unvalidated profiling returned `not_requested`, never a correctness pass.

Slurm rounded requested wall limits to whole minutes: the 15-second timeout fixture had a one-minute scheduler limit, and 90-second requests appeared as two minutes. The worker supervisor's own timeout stopped the timeout fixture first (Slurm exit `124:0`, 11 seconds allocated), yielding `worker_timeout`; it was not a test of scheduler wall-limit enforcement. The supervisor reserves five seconds from the requested limit for teardown, with a one-second minimum child budget.

## Final measured smoke test

Final job **8126**, run **`a9ad1e3c255044789a7bc817c1af38e6`**, completed on `gpu-8-vm` with Slurm exit **0:0**. This run included the final package-inventory verification and metadata changes after the seven-test live suite. Initial bring-up job 8116 and acceptance job 8117 also successfully compiled/executed the CUDA example.

Workload: **1,048,576 FP32 vector elements**, custom CUDA addition compiled by `nvcc -O3 -arch=sm_120`; compare against PyTorch addition on seeds **42, 43, 44**, `rtol=1e-5`, `atol=1e-6`, `equal_nan=false`. All three checks passed. Timing used **10 warm-up calls and 100 samples**, current-stream CUDA events, warm cache, and protocol `cuda-event-warm-v1`.

| Metric | Observed value |
| --- | --- |
| Median callable latency | 13.728 µs |
| Mean / p95 | 14.090 / 15.104 µs |
| Minimum / maximum | 13.120 / 31.104 µs |
| Population standard deviation | 1.942 µs |
| Empty-event-pair overhead | 8.512 µs |
| Allocator baseline / peak allocated | 8,388,608 / 12,582,912 bytes |
| Allocator baseline / peak reserved | 46,137,344 / 46,137,344 bytes |
| Preparation wall time | 1,094.105 ms |
| Worker wall time | 3,530.728 ms |
| Queue / allocation duration | 0 / 5,000 ms (second-resolution accounting) |

**Interpretation:** these are smoke-test callable measurements, including host-dispatch gaps between GPU events, not isolated device-kernel duration or a performance improvement claim. The latency is close enough to event-pair overhead that the worker flagged it; overhead was not subtracted. Allocator peaks include resident inputs/model memory and do not capture external CUDA allocations or total VRAM usage. There was no baseline speedup experiment or Nsight collection.

Task source hash: `f8680a66fda02b65a2710bd23869fd5248012660a7078de58fd536ac58d7f852`.

Worker bundle hash: `784c9e12250b13abbe9ae59b6bc737b20e1eacf5eeb8af933bb3992fdf13db29`.

Request hash: `7fbd4e88d38461174fee53afc94a48834cbb3ef8191361cf58af9e75834ce751`.

Raw samples and the exact result are in the evidence JSON. Original bundled sources/results remain in the local/remote run directories for the named run. The evidence JSON records the final implementation's source hashes and each live run's own worker hash, distinguishing earlier test revisions from the final smoke run.

## Cleanup and remaining scope

The final owner-scoped `squeue` query returned **no rows**, and `gpu-8-vm` was **idle**. All jobs **8115–8126** were terminal: successful setup/smoke/valid workloads, expected failure cases, or deliberate cancellations. No benchmark process or GPU reservation was retained. The reusable Python environment, frozen run bundles, logs, and results remain for explicit retention/cleanup.

The following are outside this implementation:

- Baseten/Velocity adapters, multi-GPU jobs, automatic placement, persistent GPU workers, a web server, and multi-user isolation.
- Explicit MIG-profile selection; the tested default is the complete GPU on `gpu-8-vm`.
- A general KernelBench importer. The included ModelNew example is self-contained and does not import KernelBench/Modal.
- Nsight counters, baseline speedups, stateful/mutating inputs, and unmanaged CUDA streams.
- Automatic artifact deletion, confirmed personal storage quotas, and automated retry of failed/ambiguous environment setup.

Implementation details that refine the original proposal: `setup` bootstraps the verified environment; `resume` identifies recorded batches; the provider observes by persisted request/run identity to recover missing handles; transport uses a fixed Python helper over verified SSH with checksummed archives rather than a separate SFTP client; `list` is cached and single-run `status` polls the scheduler. The library's store is injectable, but PostgreSQL/object-storage adapters are not implemented.

A Slurm allocation and separate process are not a security boundary against malicious code. This release is for trusted personal/team workloads; a future upload/server service must establish its own authorization, isolation, quotas, and trusted evaluation model.

Return to the [artifacts index](README.md).
