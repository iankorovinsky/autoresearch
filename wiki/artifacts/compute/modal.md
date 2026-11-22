# Modal compute provider investigation

The Modal backend was implemented and tested on **2026-11-21 (America/Toronto)**. The live tests covered a Tesla T4, Modal's lowest-priced listed GPU at the time. The backend exposes all 14 documented GPU selectors. GPU execution and profiler results in this report are specific to the tested T4 environment. [Modal GPU guide](https://modal.com/docs/guide/gpu) · [Modal pricing](https://modal.com/pricing)

## Integration

`kcompute` accepts `--provider modal --gpu TYPE` for runs and submissions. The profile defaults to `T4` and can select a Modal environment, app, and artifact volume. The catalog includes `T4`, `L4`, `A10`, `L40S`, `A100`, `A100-40GB`, `A100-80GB`, `RTX-PRO-6000`, `H100`, `H100!`, `H200`, `B200`, `B200+`, and `B300`. Each workload requests one full GPU. The manifest controls CPU, host memory, minimum GPU memory, and wall time, with a maximum of 24 hours on Modal. [Provider implementation](../../../compute/src/kernel_compute/providers/modal_backend.py) · [Modal Sandbox guide](https://modal.com/docs/guide/sandboxes)

The adapter runs each frozen task bundle in a named Modal Sandbox and reuses the shared benchmark worker. A Modal Volume holds the bundle, submission receipt, result, samples, and worker logs. The CLI verifies artifact identity and checksums on download. A submitted run can be retrieved from a fresh CLI process. Modal's polling API groups provisioning and execution into a single pending-or-running state, and worker logs are available after publication. [Sandbox runner](../../../compute/src/kernel_compute/providers/modal_runner.py) · [Compute CLI setup and usage](../../../compute/README.md)

From `autoresearch/compute`:

```bash
uv run kcompute providers list --provider modal
uv run kcompute setup --provider modal
uv run kcompute probe --provider modal --gpu T4
uv run kcompute run examples/vector-add/task.toml --provider modal --gpu T4
```

## Live T4 results

The allocation exposed one Tesla T4 with compute capability **7.5** and **15,360 MiB** reported by `nvidia-smi`. The worker used NVIDIA driver **620.70.06** (driver-reported CUDA support **13.5**), CUDA Toolkit **13.0.2** (`nvcc` **13.0.88**), PyTorch **2.9.1+cu130**, and Triton **3.5.1**.

- **Vector addition:** run `09d9fe77835041ffa8e2ccdcc5207b56` finished successfully and was retrieved in a fresh process. The compiled CUDA adapter added 1,048,576 FP32 elements and passed reference validation on seeds 42, 43, and 44 with `rtol=1e-5` and `atol=1e-6`. After 10 warm-up calls, 100 current-stream CUDA event samples gave **67.49 µs median** and **70.05 µs p95** callable latency. Empty event-pair overhead was **13.31 µs** and was left in the reported timings. Callable latency includes host dispatch gaps between events. PyTorch allocator peak allocated memory was **12 MiB**. [Vector-add manifest](../../../compute/examples/vector-add/task.toml) · [Measurement protocol](../../../compute/README.md#metrics-and-artifacts)
- **Resource validation:** the first vector-add run, `6eb06979d97d4efb8e5a651a7d241e68`, reached the T4 and returned a structured insufficient-memory failure because the example inherited an 80 GiB minimum from the Waterloo setup. The example now requests 1 GiB of minimum GPU memory.
- **Capability probe:** `nvidia-smi`, CUDA events, CUDA compilation and execution with `nvcc`, and a Triton JIT kernel all passed. Nsight Systems captured the probe CUDA kernel in a trace. Nsight Compute returned the `sm__cycles_elapsed.avg` hardware counter after disabling GPU clock locking with `--clock-control none`. Its initial attempt failed when locking GPU clocks. The final PyTorch Profiler probe captured **50 of 50** matrix-multiplication CUDA kernel events. [Probe implementation](../../../compute/src/kernel_compute/providers/modal_probe.py)

## Why the first PyTorch trace was empty

In a fresh Python process, the first profiler session recorded CPU operations and zero CUDA kernel or runtime events. Increasing the workload from 50 to 5,000 iterations, warming up CUDA before starting, and using a scheduled warm-up within that first session each reproduced the failure. A second 50-iteration session in the same process captured all 50 kernels and 102 CUDA runtime events. Kineto reported an empty GPU trace and zero out-of-range records for the first session.

CUDA 13.0 enables per-thread CUPTI activity buffers by default. The Kineto revision bundled with PyTorch 2.9.1 clears that setting before the pending CUPTI buffers are flushed. Direct checks on the T4 measured the setting changing from enabled to disabled across the first session. Setting it to disabled before the first session recovered all 50 kernel events. Setting it to enabled before each session reproduced zero kernel events on both sessions. The upstream Kineto change moved the reset to after the flush. [CUDA 13.0 CUPTI guide](https://docs.nvidia.com/cupti/13.0.0/main/main.html) · [Kineto source pinned by PyTorch 2.9.1](https://github.com/pytorch/kineto/blob/5e7501833f1021ce6f618572d3baf657b6319658/libkineto/src/CuptiActivityApi.cpp#L385-L421) · [Upstream Kineto fix](https://github.com/pytorch/kineto/commit/19283706e915)

The shipped probe uses the public PyTorch API to complete a short, separate warm-up profiler session in the same process. It exports the second session and checks for actual CUDA kernel records. The revised probe passed on the T4 with the existing PyTorch 2.9.1 image. Workloads that use PyTorch Profiler directly can use the same two-session pattern. [Profiler probe](../../../compute/src/kernel_compute/providers/modal_probe.py#L62)

## Shutdown and storage

The post-test audit found zero active Modal containers. The `kcompute` app was deployed with zero tasks, and `kcompute-artifacts-v2` held 15 files totaling **73,711 bytes**. At the time of the audit, Modal listed T4 compute at **$0.000164 per second** (about **$0.59 per GPU-hour**) plus CPU and memory. Compute billing stops when the app has zero containers. Volume storage includes **1 TiB per month** and is listed at **$0.09 per GiB-month** above the allowance. The test volume was well within that allowance on its own. Actual storage charges depend on workspace-wide usage. [Modal pricing](https://modal.com/pricing) · [Modal billing guide](https://modal.com/docs/guide/billing)
