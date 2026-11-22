# Baseten compute provider investigation

On **2026-11-21 (America/Toronto)**, Baseten rejected a T4 Training Job in the tested workspace, even though its public pricing and the workspace instance catalog listed T4. An A10G Training Job completed and returned a GPU capability report. Six of seven checks passed. Nsight Compute launched but could not access hardware performance counters. [Baseten pricing](https://www.baseten.co/pricing/) · [Instance type reference](https://docs.baseten.co/deployment/resources#instance-type-reference)

## Provider integration

Baseten Training Jobs run a container to completion and provide job status, cancellation, and durable checkpoints. The adapter uploads the frozen task bundle using Baseten-issued temporary S3 credentials, launches a CUDA development image, and installs the hash-locked worker dependencies. The shared worker writes results, timing samples, and bounded logs. Baseten syncs those files as checkpoints, and `kcompute` checks artifact identity and checksums when it downloads them. The job name preserves the local run ID so an uncertain submission can be looked up before another attempt. [Provider implementation](../../../compute/src/kernel_compute/providers/baseten_backend.py) · [Job runner](../../../compute/src/kernel_compute/providers/baseten_runner.py) · [Training Job lifecycle](https://docs.baseten.co/training/lifecycle) · [Checkpoint file API](https://docs.baseten.co/reference/training-api/get-training-job-checkpoint-files)

The CLI takes a GPU family with `--gpu`. Kernel tasks request one GPU and set CPU, RAM, minimum GPU memory, and the MIG policy in their manifests. The public GPU families are T4, L4, A10G, A100, H100MIG, H100, H200, B200, and RTX-PRO-6000. The selector also accepts the additional NVIDIA types exposed by Truss and valid GPU selectors returned by the workspace catalog. The authenticated catalog includes instance sizes and prices. Training Jobs eligibility is determined when the job is submitted. [Instance price API](https://docs.baseten.co/reference/management-api/instance-types/gets-instance-type-prices) · [Truss accelerator definitions](https://github.com/basetenlabs/truss/blob/5d687d04b89e45e205b4cfca958dc236d26d7d11/truss/base/truss_config.py#L85-L102)

From `autoresearch/compute`, set `BASETEN_API_KEY` in the ignored `.env` file and run the following commands. The `baseten` extra installs `boto3` for source uploads.

```bash
uv run --locked --extra baseten --env-file .env kcompute providers list --provider baseten
uv run --locked --extra baseten --env-file .env kcompute probe --provider baseten --gpu A10G
```

## GPU choice

- **T4:** The least expensive published GPU was `T4x4x16` at **$0.01052/min**. The Training Jobs API returned HTTP 400 and suggested A10G or H100. The attempted T4 submission launched no GPU job.
- **A10G:** The cheaper of the two alternatives named by the API was `A10Gx4x16` at **$0.02012/min**. The probe requested one GPU, 4 vCPUs, and 16 GiB of host RAM. A10G is now the adapter's default.
- **Other options:** The authenticated catalog returned 25 GPU instance configurations, including the cheaper L4 at **$0.01414/min**. L4 and the other GPU types remain untested on this Training Jobs backend. Baseten's Training Jobs guide names H100 and H200, while the live API suggested H100 and A10G in response to the rejected T4 request. Workspace access, product support, and capacity can differ. [Public pricing](https://www.baseten.co/pricing/) · [Training Jobs compute guide](https://docs.baseten.co/training/concepts/basics#compute-resources)

## Live A10G results

The final probe, Baseten job `qrek55w`, completed and the CLI retrieved its report. The runtime reported **NVIDIA A10G**, **23,028 MiB** of device memory, NVIDIA driver **580.178.04**, `nvcc` **13.0.88**, PyTorch **2.9.1+cu130**, and Triton **3.5.1**.

- **`nvidia-smi`:** Reported the GPU, driver, and device memory.
- **CUDA events:** Timed a CUDA operation and verified its output. The single **74.78 µs** sample records probe activity. The workload benchmark protocol uses repeated samples and separate validation.
- **`nvcc`:** Compiled and ran a CUDA kernel that wrote a known value.
- **Triton:** JIT-compiled and ran a kernel with an output check. The first A10G job, `qv7njjq`, reached the GPU but Triton compilation failed because `Python.h` was missing. Adding `python3-dev` to the job environment resolved it.
- **PyTorch Profiler:** Captured 50 CUDA kernel events.
- **Nsight Systems:** Captured the probe CUDA kernel in an actual trace using Nsight Systems **2025.3.2**.
- **Nsight Compute:** Started with GPU clock locking disabled, but returned `ERR_NVGPUCTRPERM` when attempting to collect `sm__cycles_elapsed.avg`. The tested job lacked GPU hardware counter access.

These results cover the capability probe on A10G in one workspace. Local tests exercised task submission and checkpoint retrieval with simulated API responses. A live task-manifest benchmark, cancellation, multi-GPU execution, and profiler behavior on other Baseten GPU types remain unverified. [Probe code](../../../compute/src/kernel_compute/providers/modal_probe.py) · [Local Baseten tests](../../../compute/tests/test_baseten.py)

## What carried over from Modal

Baseten reused the capability probe originally built for Modal and the same worker dependency lock. Modal schedules a named Sandbox and stores artifacts on a Volume. Baseten runs a Training Job and retrieves artifacts after checkpoint sync. On the tested Modal T4, Nsight Compute returned a hardware counter after clock locking was disabled. The same clock setting on Baseten A10G reached the counter permission error. Each result reflects its own GPU and provider environment. [Modal investigation and T4 results](modal.md)
