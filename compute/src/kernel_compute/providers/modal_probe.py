"""Run small, independent GPU checks and report what each tool actually produced."""

import csv
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl

CUDA_SOURCE = """
#include <cuda_runtime.h>
__global__ void kcompute_probe(int *value) { *value = 42; }
int main() {
    int *device, host = 0;
    if (cudaMalloc(&device, sizeof(int)) != cudaSuccess) return 1;
    kcompute_probe<<<1, 32>>>(device);
    if (cudaMemcpy(&host, device, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 2;
    return host != 42;
}
"""
COUNTER = "sm__cycles_elapsed.avg"


def command(*args, timeout=30):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stderr + result.stdout)[-2000:].strip())
    return result.stdout


def check(action):
    try:
        return {"status": "passed", "detail": action()}
    except FileNotFoundError as exc:
        return {"status": "missing", "detail": str(exc)}
    except Exception as exc:
        return {"status": "failed", "detail": f"{type(exc).__name__}: {exc}"[:2200]}


def cuda_events():
    x = torch.arange(1024, device="cuda")
    x + x
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    y = x + x
    end.record()
    end.synchronize()
    if not torch.equal(y, 2 * x):
        raise RuntimeError("incorrect CUDA result")
    elapsed = start.elapsed_time(end)
    if elapsed <= 0:
        raise RuntimeError("CUDA events did not report positive elapsed time")
    return {"sample_us": elapsed * 1000, "torch_version": torch.__version__}


def torch_profiler():
    x = torch.ones((512, 512), device="cuda")
    x @ x
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    # On Modal T4, Kineto can receive zero CUDA records in the first session.
    # Complete a separate profiling session before collecting the trace we check.
    with torch.profiler.profile(activities=activities):
        x @ x
        torch.cuda.synchronize()
    with torch.profiler.profile(activities=activities) as profile:
        for _ in range(50):
            x @ x
        torch.cuda.synchronize()
    path = Path("torch-trace.json")
    profile.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [event["name"] for event in events if event.get("cat") == "kernel"]
    if not kernels:
        raise RuntimeError("PyTorch profiler collected no CUDA kernel events")
    return {"cuda_events": len(kernels), "example": kernels[0][:300]}


@triton.jit
def add_one(source, target, block_size: tl.constexpr):
    offset = tl.arange(0, block_size)
    tl.store(target + offset, tl.load(source + offset) + 1)


def triton_jit():
    x = torch.arange(128, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    add_one[(1,)](x, y, 128)
    if not torch.equal(y, x + 1):
        raise RuntimeError("incorrect Triton result")
    return {"version": triton.__version__}


def compile_cuda():
    Path("probe.cu").write_text(CUDA_SOURCE)
    major, minor = torch.cuda.get_device_capability()
    compiler = str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc")
    command(compiler, f"-arch=sm_{major}{minor}", "probe.cu", "-o", "probe")
    command("./probe")
    return {"version": command(compiler, "--version").strip()}


def nsight_systems():
    command(
        "nsys",
        "profile",
        "--trace=cuda",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=true",
        "--output=nsys-probe",
        "./probe",
    )
    trace = command(
        "nsys", "stats", "--report=cuda_gpu_kern_sum", "--format=csv", "nsys-probe.nsys-rep"
    )
    if "kcompute_probe" not in trace:
        raise RuntimeError(f"CUDA kernel missing from Nsight Systems trace: {trace[-1500:]}")
    return {"trace": "CUDA kernel captured", "version": command("nsys", "--version").strip()}


def nsight_compute():
    executable = shutil.which("ncu") or str(Path(os.environ["CUDA_HOME"]) / "bin/ncu")
    report = command(
        executable, "--clock-control", "none", "--csv", "--metrics", COUNTER, "./probe"
    )
    for row in csv.reader(report.splitlines()):
        if COUNTER not in row or not any("kcompute_probe" in field for field in row):
            continue
        try:
            value = float(row[-1].replace(",", ""))
        except ValueError:
            continue
        if value > 0:
            return {"metric": COUNTER, "value": value}
    raise RuntimeError(f"Nsight Compute collected no numeric GPU counter: {report[-1500:]}")


def nvidia_smi():
    return command(
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"
    ).strip()


def main():
    with tempfile.TemporaryDirectory(prefix="kcompute-probe-") as root:
        os.chdir(root)
        tools = {
            "nvidia-smi": check(nvidia_smi),
            "cuda-events": check(cuda_events),
            "nvcc": check(compile_cuda),
            "torch-profiler": check(torch_profiler),
            "triton": check(triton_jit),
        }
        for name, action in (("nsys", nsight_systems), ("ncu", nsight_compute)):
            tools[name] = (
                check(action)
                if tools["nvcc"]["status"] == "passed"
                else {"status": "blocked", "detail": "CUDA probe executable did not build and run"}
            )
        actual_gpu = check(torch.cuda.get_device_name)
        print(json.dumps({"actual_gpu": actual_gpu.get("detail"), "tools": tools}))


if __name__ == "__main__":
    main()
