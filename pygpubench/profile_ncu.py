"""Profile a KernelBench candidate with Nsight Compute on a Modal GPU.

Usage (from the repository root, with the Modal CLI logged in):

    modal run pygpubench/profile_ncu.py --task-id 1 --level 1 \
        --kernel-path pygpubench/candidates/correct_cuda.py --output profile.json

The candidate is loaded exactly as in `evaluate_pygpubench.py`: the reference
problem comes from KernelBench, the candidate's `ModelNew` is compiled, and the
same generated inputs are used. One warm-up call runs outside the profiled
range, then `cudaProfilerStart`/`cudaProfilerStop` brackets a single call so
only the candidate's kernels are collected. The raw CSV is reduced by
`ncu_parse.py` to throughput, occupancy, launch, and stall summaries.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SECTIONS = ("SpeedOfLight", "Occupancy", "LaunchStats", "WarpStateStats", "MemoryWorkloadAnalysis")

DRIVER = textwrap.dedent(
    """
    import json, os, sys, torch
    sys.path.insert(0, os.environ["AR_PYGPUBENCH_DIR"])
    import evaluate_pygpubench as ev
    args, _ = ev.generate_test_case(seed=int(os.environ.get("AR_SEED", "42")))
    for _ in range(int(os.environ.get("AR_WARMUP", "2"))):
        ev.kernel(*args)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    ev.kernel(*args)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print("AR_DRIVER_DONE", flush=True)
    """
)


def find_ncu() -> str | None:
    found = shutil.which("ncu")
    if found:
        return found
    for pattern in ("/usr/local/cuda*/bin/ncu", "/opt/nvidia/nsight-compute/*/ncu", "/usr/local/NVIDIA-Nsight-Compute*/ncu"):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    return None


def profile_candidate(task_id: int, level: int, kernel_path: str, *, warmup: int = 2, seed: int = 42,
                      init_seed: int = 42, precision: str = "fp32", timeout: int = 1800,
                      build_dir: str = "/tmp/kb_build") -> dict:
    """Run the candidate once under Nsight Compute and return a reduced summary."""
    import torch

    sys.path.insert(0, str(SCRIPT_DIR))
    import evaluate_pygpubench as ev
    import ncu_parse

    kb_eval, kb_dataset = ev._kernelbench()
    problem = kb_dataset.construct_kernelbench_dataset(level).get_problem_by_id(task_id)
    Path(build_dir).mkdir(parents=True, exist_ok=True)
    result = {
        "task_id": task_id,
        "level": level,
        "task_name": problem.name,
        "kernel_path": str(kernel_path),
        "gpu": torch.cuda.get_device_name(0),
        "precision": precision,
        "profiled": False,
        "kernels": [],
        "summary": None,
        "feedback": None,
        "ncu_version": None,
    }
    ncu = find_ncu()
    if not ncu:
        result["feedback"] = "Nsight Compute (ncu) is not installed in the profiling image."
        return result
    version = subprocess.run([ncu, "--version"], capture_output=True, text=True)
    result["ncu_version"] = (version.stdout or version.stderr).strip().splitlines()[-1:] or None

    config = {
        "ref_src": problem.code,
        "kernel_src": Path(kernel_path).read_text(),
        "init_seed": init_seed,
        "precision": precision,
        "build_dir": str(Path(build_dir).resolve()),
        "kb_trials": 1,
    }
    env = dict(
        os.environ,
        AR_PYGPUBENCH_DIR=str(SCRIPT_DIR),
        AR_WARMUP=str(warmup),
        AR_SEED=str(seed),
        PYTHONPATH=os.pathsep.join(p for p in ev._import_paths() + [os.environ.get("PYTHONPATH", "")] if p),
    )
    env[ev.CONFIG_ENV] = json.dumps(config)
    command = [ncu, "--csv", "--page", "raw", "--profile-from-start", "off", "--target-processes", "all",
               # Containers cannot lock GPU clocks; profile at whatever clocks the GPU runs.
               "--clock-control", "none"]
    for section in SECTIONS:
        command += ["--section", section]
    command += [sys.executable, "-c", DRIVER]
    try:
        run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        result["feedback"] = f"Nsight Compute timed out after {timeout} s."
        return result
    stderr_tail = run.stderr[-4000:]
    if "ERR_NVGPUCTRPERM" in run.stdout + run.stderr:
        result["feedback"] = ("The GPU denied performance-counter access to Nsight Compute "
                              "(ERR_NVGPUCTRPERM). Profiling is unavailable on this machine; "
                              "use evaluate_kernel timings instead.")
        return result
    if "AR_DRIVER_DONE" not in run.stdout:
        result["feedback"] = ("The candidate did not run to completion under Nsight Compute:\n"
                              + (run.stdout[-2000:] + "\n" + stderr_tail).strip())
        return result
    kernels = ncu_parse.parse_ncu_raw_csv(run.stdout)
    result["kernels"] = kernels
    result["profiled"] = bool(kernels)
    if not kernels:
        result["feedback"] = "Nsight Compute produced no kernel rows:\n" + stderr_tail
        return result
    result["summary"] = ncu_parse.summarize_kernels(kernels)
    result["feedback"] = f"profiled {len(kernels)} kernel launch(es) on {result['gpu']}"
    return result


def print_summary(result: dict) -> None:
    status = "OK" if result["profiled"] else "FAIL"
    print(f"[{status}] level {result['level']} task {result['task_id']} ({result['task_name']}) on {result['gpu']}")
    if result.get("summary"):
        print(textwrap.indent(result["summary"], "  "))
    print(f"  feedback: {result['feedback']}")


try:
    import modal
except ImportError:  # local tests without Modal
    modal = None

if modal is not None:
    import evaluate_pygpubench as _ev

    image = (
        _ev.base_image
        .apt_install("cuda-nsight-compute-13-0")
        .add_local_dir(str(_ev.LOCAL_KERNELBENCH), remote_path="/opt/KernelBench", ignore=_ev.KERNELBENCH_IGNORE)
        .add_local_python_source("evaluate_pygpubench", "ncu_parse", "profile_ncu")
    )
    app = modal.App("kernelbench-ncu-profile", image=image)

    @app.function(gpu="L4", timeout=3600, scaledown_window=300)
    def profile_remote(task_id: int, level: int, kernel_name: str, kernel_src: str, warmup: int) -> dict:
        kernel_dir = Path("/tmp/candidates")
        kernel_dir.mkdir(parents=True, exist_ok=True)
        kernel_file = kernel_dir / kernel_name
        kernel_file.write_text(kernel_src)
        return profile_candidate(task_id, level, str(kernel_file), warmup=warmup)

    @app.local_entrypoint()
    def main(task_id: int = 1, level: int = 1, kernel_path: str = "pygpubench/candidates/correct_cuda.py",
             warmup: int = 2, output: str = ""):
        path = Path(kernel_path)
        result = profile_remote.remote(task_id, level, path.name, path.read_text(), warmup)
        print_summary(result)
        if output:
            Path(output).write_text(json.dumps(result, indent=2))
            print(f"  saved:    {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile a KernelBench candidate with Nsight Compute on the local GPU")
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--kernel-path", default="pygpubench/candidates/correct_cuda.py")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", default="")
    ns = parser.parse_args()
    res = profile_candidate(ns.task_id, ns.level, ns.kernel_path, warmup=ns.warmup)
    print_summary(res)
    if ns.output:
        Path(ns.output).write_text(json.dumps(res, indent=2))
