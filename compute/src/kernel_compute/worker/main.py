"""A bounded supervisor and the GPU worker it launches; no scheduler dependencies."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

from kernel_compute.common import atomic_json, digest, failure_result, now
from kernel_compute.worker.measure import check_output, measure, memory

MAX_LOG_BYTES = 1024 * 1024


def environment(torch):
    prop = torch.cuda.get_device_properties(0)
    xml = subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=15)
    root = ET.fromstring(xml)
    uuid = str(getattr(prop, "uuid", ""))
    devices = root.findall("gpu")
    match = [
        gpu
        for gpu in devices
        if (gpu.findtext("uuid") or "").removeprefix("GPU-") == uuid.removeprefix("GPU-")
    ]
    # One visible torch device and a single physical board is unambiguous on gpu-8-vm.
    gpu = match[0] if len(match) == 1 else devices[0] if len(devices) == 1 else None
    if gpu is None:
        raise RuntimeError("cannot identify allocated GPU in driver metadata")
    mig_mode = gpu.findtext("mig_mode/current_mig")
    mig = mig_mode == "Enabled" or "MIG" in prop.name or uuid.startswith("MIG-")
    if mig_mode not in {"Enabled", "Disabled", "N/A"}:
        raise RuntimeError("cannot establish GPU MIG configuration")
    cuda_home = os.environ["CUDA_HOME"]
    nvcc = subprocess.check_output([f"{cuda_home}/bin/nvcc", "--version"], text=True, timeout=15)
    ready_path = Path(sys.executable).parent.parent / "ready.json"
    ready = json.loads(ready_path.read_text()) if ready_path.exists() else {}
    packages = dict(
        sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions())
    )
    if ready.get("packages") != packages:
        raise RuntimeError("worker environment packages changed since setup validation")
    return {
        "node": platform.node(),
        "gpu_name": prop.name,
        "gpu_uuid": uuid,
        "compute_capability": f"{prop.major}.{prop.minor}",
        "device_memory_bytes": prop.total_memory,
        "mig": mig,
        "mig_mode": mig_mode,
        "driver_version": root.findtext("driver_version"),
        "driver_cuda_support": root.findtext("cuda_version"),
        "cuda_home": cuda_home,
        "nvcc": nvcc.strip(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python_version": platform.python_version(),
        "environment_lock_hash": ready.get("lock_hash"),
        "installed_packages": packages,
        "power_limit_w": gpu.findtext("gpu_power_readings/current_power_limit")
        or gpu.findtext("power_readings/power_limit"),
        "visible_devices": torch.cuda.device_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
    }


def load_task(request):
    path, name = request["task"]["entrypoint"].split(":")
    task_path = Path("task") / path
    sys.path.insert(0, str(task_path.parent.resolve()))
    spec = importlib.util.spec_from_file_location("kcompute_candidate", task_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def execute(request):
    started = time.perf_counter()
    result = failure_result(request, "failed", "worker_error", "worker did not complete", "setup")
    result["execution"]["started_at"] = now()
    stage = "environment"
    try:
        import torch

        torch.set_num_threads(request["task"]["resources"]["cpu_cores"])
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one allocated CUDA device is required")
        torch.cuda.set_device(0)
        result["environment"] = environment(torch)
        actual = result["environment"]
        resources = request["task"]["resources"]
        if actual["environment_lock_hash"] != request["environment_lock_hash"]:
            raise RuntimeError("installed environment lock differs from request")
        if actual["device_memory_bytes"] < resources["min_gpu_memory_gib"] * 1024**3:
            raise RuntimeError("allocated GPU has insufficient memory")
        if resources["mig"] == "forbid" and actual["mig"]:
            raise RuntimeError("full GPU requested, but a MIG allocation was provided")
        if resources["mig"] == "require" and not actual["mig"]:
            raise RuntimeError("MIG requested, but a full GPU was provided")
        stage = "prepare"
        begin = time.perf_counter()
        task_spec = request["task"]
        torch.manual_seed(int(task_spec["case"].get("seed", 42)))
        task = load_task(request)(task_spec["case"], torch.device("cuda:0"))
        if getattr(task, "mutates_inputs", True) or getattr(task, "stateful", True):
            raise ValueError("task must declare mutates_inputs=False and stateful=False")
        if getattr(task, "timing_scope", None) != "callable_current_stream":
            raise ValueError("task must declare timing_scope='callable_current_stream'")
        torch.cuda.synchronize()
        result["execution"]["prepare_wall_ms"] = (time.perf_counter() - begin) * 1000
        stage = "validation"
        policy = task_spec["validation"]
        with torch.inference_mode():
            if policy["mode"] == "reference":
                result["validation"] = {
                    "status": "not_run",
                    "policy": policy,
                    "passed_seeds": [],
                    "case": task_spec["case"],
                }
                for seed in policy["seeds"]:
                    torch.manual_seed(seed)
                    inputs = task.make_inputs(seed)
                    if not isinstance(inputs, tuple) or not all(
                        isinstance(x, torch.Tensor) for x in inputs
                    ):
                        raise ValueError("make_inputs must return a tuple of tensors")
                    reference_inputs = tuple(x.clone() for x in inputs)
                    actual_output = task.run(*inputs)
                    expected_output = task.reference(*reference_inputs)
                    torch.cuda.synchronize()
                    check_output(torch, actual_output, expected_output, policy)
                    result["validation"]["passed_seeds"].append(seed)
                    del inputs, reference_inputs, actual_output, expected_output
                result["validation"]["status"] = "passed"
            else:
                result["validation"] = {"status": "not_requested"}
            stage = "measurement"
            seed = int(task_spec["case"].get("seed", 42))
            inputs = task.make_inputs(seed)
            latency, samples, warmup_ms = measure(torch, task, inputs, task_spec["measurement"])
            del inputs
            result["latency"] = latency
            result["execution"]["warmup_wall_ms"] = warmup_ms
            atomic_json(Path("samples.json"), {"units": "us", "samples": samples})
            inputs = task.make_inputs(seed)
            result["memory"] = memory(torch, task, inputs)
            del inputs
            if latency["median"] <= 5 * max(latency["event_pair_overhead_us"], 0.001):
                result["warnings"].append(
                    "latency is close to event-pair overhead; no subtraction applied"
                )
        result["execution"]["status"] = "succeeded"
        result["execution"]["error"] = None
        result["unavailable_reasons"] = {}
    except Exception as exc:
        reason = "worker_error"
        if stage == "environment":
            reason = "environment_mismatch"
        elif isinstance(exc, AssertionError) and stage == "validation":
            reason = "validation_failed"
            result["validation"]["status"] = "failed"
        elif "out of memory" in str(exc).lower():
            reason = "out_of_memory"
        elif stage == "prepare" and isinstance(exc, subprocess.CalledProcessError):
            reason = "compile_error"
        elif "CUDA" in str(exc):
            reason = "cuda_error"
        result["execution"]["error"] = {
            "stage": stage,
            "reason": reason,
            "message": str(exc)[:8192],
        }
        result["unavailable_reasons"] = {
            key: reason for key in ("latency", "memory") if result[key] is None
        }
        traceback.print_exc()
    result["execution"]["finished_at"] = now()
    result["execution"]["worker_wall_ms"] = (time.perf_counter() - started) * 1000
    atomic_json(Path("result.json"), result)
    return 0 if result["execution"]["status"] == "succeeded" else 1


def drain(stream, path, truncated):
    size = 0
    with path.open("wb") as output:
        while chunk := stream.read(65536):
            output.write(chunk[: max(0, MAX_LOG_BYTES - size)])
            output.flush()
            size += len(chunk)
    truncated[path.name] = size > MAX_LOG_BYTES


def supervise(request):
    process = subprocess.Popen(
        [sys.executable, "-m", "kernel_compute.worker.main", "--child"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    truncated = {}
    threads = [
        threading.Thread(target=drain, args=(stream, Path(name), truncated))
        for stream, name in ((process.stdout, "stdout.log"), (process.stderr, "stderr.log"))
    ]
    for thread in threads:
        thread.start()
    try:
        code = process.wait(timeout=max(1, request["task"]["resources"]["wall_time_s"] - 5))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        code = 124
    finally:
        # Also stop descendants left behind after a worker failure/success.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for thread in threads:
            thread.join()
    if not Path("result.json").exists() or code == 124:
        result = failure_result(
            request,
            "timed_out" if code == 124 else "failed",
            "worker_timeout" if code == 124 else "worker_crash",
            f"worker exited with code {code}",
            "supervisor",
        )
    else:
        result = json.loads(Path("result.json").read_text())
        if code and result["execution"]["status"] == "succeeded":
            result = failure_result(request, "failed", "worker_crash", f"worker exited {code}")
    result["execution"]["exit_code"] = code
    result["artifacts"] = {"truncated_logs": truncated}
    atomic_json(Path("result.json"), result)
    names = ["request.json", "result.json", "stdout.log", "stderr.log"]
    if Path("samples.json").exists():
        names.append("samples.json")
    atomic_json(
        Path("completion.json"),
        {
            "schema_version": 1,
            "run_id": request["run_id"],
            "request_hash": request["request_hash"],
            "files": {
                name: {"sha256": digest(Path(name).read_bytes()), "size": Path(name).stat().st_size}
                for name in names
            },
        },
    )
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    request = json.loads(Path("request.json").read_text())
    return execute(request) if args.child else supervise(request)


if __name__ == "__main__":
    raise SystemExit(main())
