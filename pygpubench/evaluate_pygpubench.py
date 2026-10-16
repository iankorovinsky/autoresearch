import argparse
import dataclasses
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PYGPUBENCH_VERSION = "0.0.3"
CONFIG_ENV = "KB_PYGPUBENCH_CONFIG"
PRECHECK_MARKER = "KB_PRECHECK_RESULT:"
SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_NAME = Path(__file__).stem
LOCAL_KERNELBENCH = SCRIPT_DIR.parent / "KernelBench"
SUPPORTED_DTYPES = ("torch.float32", "torch.float16", "torch.bfloat16")

_cache = {}
_reference_cache = {}


def _config():
    return json.loads(os.environ[CONFIG_ENV])


def _import_paths():
    paths = [str(SCRIPT_DIR)]
    if (LOCAL_KERNELBENCH / "src").is_dir():
        paths.append(str(LOCAL_KERNELBENCH / "src"))
    return paths


def _kernelbench():
    for p in _import_paths():
        if p not in sys.path:
            sys.path.insert(0, p)
    from kernelbench import eval as kb_eval
    from kernelbench import dataset as kb_dataset
    return kb_eval, kb_dataset


def _device():
    import torch
    return torch.device("cuda:0")


def _load_reference():
    if "reference" in _cache:
        return _cache["reference"]
    import torch
    kb_eval, _ = _kernelbench()
    cfg = _config()
    device = _device()
    precision = kb_eval.get_torch_dtype_from_string(cfg["precision"])
    context = {}
    Model, get_init_inputs, get_inputs = kb_eval.load_original_model_and_inputs(cfg["ref_src"], context)
    kb_eval.set_seed(cfg["init_seed"])
    init_inputs = [kb_eval._process_input_tensor(x, device, "cuda", precision) for x in get_init_inputs()]
    with torch.no_grad():
        kb_eval.set_seed(cfg["init_seed"])
        model = Model(*init_inputs).to(device=device, dtype=precision)
    torch.cuda.synchronize(device)
    _cache["reference"] = (model, get_inputs, init_inputs, precision, device)
    return _cache["reference"]


def _load_candidate():
    if "candidate" in _cache:
        return _cache["candidate"]
    import torch
    kb_eval, _ = _kernelbench()
    cfg = _config()
    _, _, init_inputs, precision, device = _load_reference()
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    context = {}
    ModelNew = kb_eval.load_custom_model(cfg["kernel_src"], context, cfg["build_dir"])
    if ModelNew is None:
        raise RuntimeError("ModelNew not found in candidate kernel file")
    with torch.no_grad():
        kb_eval.set_seed(cfg["init_seed"])
        model = ModelNew(*init_inputs).to(device=device, dtype=precision)
    torch.cuda.synchronize(device)
    torch.set_grad_enabled(False)
    _cache["candidate"] = model
    return model


def generate_test_case(*, seed, **kwargs):
    import torch
    kb_eval, _ = _kernelbench()
    model, get_inputs, _, precision, device = _load_reference()
    kb_eval.set_seed(seed)
    inputs = [kb_eval._process_input_tensor(x, device, "cuda", precision) for x in get_inputs()]
    with torch.no_grad():
        expected = model(*inputs)
    if not isinstance(expected, torch.Tensor):
        raise TypeError(f"reference Model must return a single tensor, got {type(expected).__name__}")
    if str(expected.dtype) not in SUPPORTED_DTYPES:
        raise TypeError(f"PyGPUBench cannot check outputs of dtype {expected.dtype}")
    expected = expected.contiguous()
    out = torch.empty_like(expected)
    tol = kb_eval.get_tolerance_for_precision(precision)
    args = (out,) + tuple(x.contiguous() if isinstance(x, torch.Tensor) else x for x in inputs)
    return args, (expected, tol, tol)


def kernel(out, *inputs):
    model = _load_candidate()
    out.copy_(model(*inputs))


def reference_kernel(out, *inputs):
    import torch
    model, _, _, _, _ = _load_reference()
    with torch.no_grad():
        out.copy_(model(*inputs))


def _precheck():
    kb_eval, _ = _kernelbench()
    cfg = _config()
    check = kb_eval.eval_kernel_against_ref(
        cfg["ref_src"], cfg["kernel_src"],
        seed_num=cfg["init_seed"], num_correct_trials=cfg["kb_trials"], measure_performance=False,
        build_dir=cfg["build_dir"], device=_device(), backend="cuda",
        precision=kb_eval.get_torch_dtype_from_string(cfg["precision"]),
    )
    payload = None if check is None else check.model_dump()
    print(PRECHECK_MARKER + json.dumps(payload, default=str), flush=True)


def _parse_precheck(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith(PRECHECK_MARKER):
            return json.loads(line[len(PRECHECK_MARKER):])
    return None


def _describe_kb_failure(metadata, tolerance):
    if "compilation_error" in metadata:
        return f"candidate failed to compile: {metadata['compilation_error']}"[:4000]
    if "runtime_error" in metadata:
        text = f"runtime error in ModelNew: {metadata['runtime_error']}"
        tb = metadata.get("runtime_error_traceback")
        if tb:
            text += "\n" + tb[-2500:]
        return text
    issue = metadata.get("correctness_issue")
    if issue:
        text = f"KernelBench correctness check failed: {issue}"
        if metadata.get("max_difference"):
            text += (f" (max abs diff {metadata['max_difference'][-1]}, "
                     f"mean abs diff {metadata['avg_difference'][-1]}, tolerance {tolerance})")
        if metadata.get("correctness_trials"):
            text += f"; trials passed {metadata['correctness_trials']}"
        return text
    return "KernelBench correctness check failed: " + json.dumps(metadata, default=str)[:2000]


def _stats(time_us):
    import pygpubench
    valid = [t for t in time_us if t > 0]
    if len(valid) < 2:
        return None
    return dataclasses.asdict(pygpubench.basic_stats(time_us))


def _bench(qualname, repeats, seed, timeout):
    import pygpubench
    return pygpubench.do_bench_isolated(
        qualname,
        generate_test_case,
        {},
        repeats,
        seed,
        discard=True,
        timeout=timeout,
        landlock=False,
        mseal=False,
        allow_root=True,
    )


def evaluate_pygpubench(task_id, level, kernel_path, repeats=10, seed=42, init_seed=42, precision="fp32",
                        kb_trials=1, measure_reference=True, timeout=900, build_dir="/tmp/kb_build"):
    import torch
    kb_eval, kb_dataset = _kernelbench()
    problem = kb_dataset.construct_kernelbench_dataset(level).get_problem_by_id(task_id)
    kernel_path = str(Path(kernel_path).resolve())
    build_dir = str(Path(build_dir).resolve())
    Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ[CONFIG_ENV] = json.dumps({
        "ref_src": problem.code,
        "kernel_src": Path(kernel_path).read_text(),
        "init_seed": init_seed,
        "precision": precision,
        "build_dir": build_dir,
        "kb_trials": kb_trials,
    })
    gpu = torch.cuda.get_device_name(0)
    tolerance = kb_eval.get_tolerance_for_precision(precision)
    result = {
        "task_id": task_id,
        "level": level,
        "task_name": problem.name,
        "kernel_path": kernel_path,
        "gpu": gpu,
        "precision": precision,
        "tolerance": tolerance,
        "repeats": repeats,
        "compiled": False,
        "correct": False,
        "feedback": None,
        "kernelbench_check": None,
        "errors": None,
        "latency_us": None,
        "time_us": None,
        "event_overhead_us": None,
        "reference_latency_us": None,
        "reference_cached": False,
        "speedup": None,
    }
    pythonpath = os.pathsep.join(p for p in _import_paths() + [os.environ.get("PYTHONPATH", "")] if p)
    env = dict(os.environ, PYTHONPATH=pythonpath)
    precheck_run = subprocess.run(
        [sys.executable, "-c", f"import {MODULE_NAME} as m; m._precheck()"],
        env=env, capture_output=True, text=True, timeout=timeout,
    )
    precheck = _parse_precheck(precheck_run.stdout)
    if precheck_run.returncode != 0:
        result["feedback"] = "KernelBench check crashed:\n" + precheck_run.stderr[-4000:]
        return result
    if precheck is None:
        result["feedback"] = ("KernelBench could not evaluate the candidate (build lock or missing file):\n"
                              + precheck_run.stderr[-4000:])
        return result
    result["compiled"] = precheck["compiled"]
    result["kernelbench_check"] = precheck
    if not precheck["compiled"] or not precheck["correctness"]:
        result["feedback"] = _describe_kb_failure(precheck["metadata"], tolerance)
        return result
    try:
        res = _bench(f"{MODULE_NAME}.kernel", repeats, seed, timeout)
    except Exception:
        result["feedback"] = "PyGPUBench benchmark failed:\n" + traceback.format_exc()[-4000:]
        return result
    result["correct"] = res.success
    result["errors"] = res.errors or 0
    result["time_us"] = res.time_us
    result["event_overhead_us"] = res.event_overhead_us
    result["latency_us"] = _stats(res.time_us)
    if res.errors:
        result["feedback"] = (f"KernelBench check passed but PyGPUBench flagged {res.errors} output elements outside "
                              f"tolerance {tolerance} across {len(res.time_us)} timed runs; "
                              "check for stream, synchronization or uninitialized-memory issues")
        return result
    if not res.full:
        result["feedback"] = "PyGPUBench did not complete all repeats; timings are informational only"
        return result
    if measure_reference:
        key = (level, task_id, gpu, repeats, seed, init_seed, precision)
        if key in _reference_cache:
            result["reference_latency_us"] = _reference_cache[key]
            result["reference_cached"] = True
        else:
            try:
                ref = _bench(f"{MODULE_NAME}.reference_kernel", repeats, seed, timeout)
                result["reference_latency_us"] = _stats(ref.time_us)
                if ref.success and result["reference_latency_us"]:
                    _reference_cache[key] = result["reference_latency_us"]
            except Exception:
                result["reference_error"] = traceback.format_exc()[-2000:]
    s = result["latency_us"]
    if s and result["reference_latency_us"]:
        result["speedup"] = result["reference_latency_us"]["mean"] / s["mean"]
        result["feedback"] = (f"correct; {s['mean']:.1f} us vs torch {result['reference_latency_us']['mean']:.1f} us "
                              f"({result['speedup']:.2f}x)")
    elif s:
        result["feedback"] = f"correct; {s['mean']:.1f} us"
    return result


def print_summary(result):
    status = "PASS" if result["correct"] else "FAIL"
    print(f"[{status}] level {result['level']} task {result['task_id']} ({result['task_name']}) on {result['gpu']}")
    print(f"  kernel:   {result['kernel_path']}")
    print(f"  compiled: {result['compiled']}  correct: {result['correct']}  errors: {result['errors']}")
    if result["latency_us"]:
        s = result["latency_us"]
        print(f"  latency:  {s['mean']:.1f} +/- {s['std']:.1f} us  [best {s['best']:.1f} / median {s['median']:.1f} / worst {s['worst']:.1f}]  ({s['runs']}/{s['expected']} runs)")
    if result["reference_latency_us"]:
        r = result["reference_latency_us"]
        cached = " (cached)" if result["reference_cached"] else ""
        print(f"  torch:    {r['mean']:.1f} +/- {r['std']:.1f} us  [best {r['best']:.1f} / median {r['median']:.1f} / worst {r['worst']:.1f}]{cached}")
    if result["speedup"]:
        print(f"  speedup:  {result['speedup']:.2f}x")
    print(f"  feedback: {result['feedback']}")


def _write_output(results, output):
    if output:
        data = results[0] if len(results) == 1 else results
        Path(output).write_text(json.dumps(data, indent=2))
        print(f"  saved:    {output}")


try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    image = (
        modal.Image.from_registry("nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04", add_python="3.12")
        .entrypoint([])
        .apt_install("libseccomp2")
        .uv_pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
        .uv_pip_install(
            f"pygpubench=={PYGPUBENCH_VERSION}",
            "numpy", "requests", "pydantic", "python-dotenv", "tqdm", "openai", "litellm", "ninja",
        )
        .env({
            "PYTHONPATH": "/opt/KernelBench/src",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TORCH_EXTENSIONS_DIR": "/tmp/kb_build",
        })
        .add_local_dir(
            str(LOCAL_KERNELBENCH),
            remote_path="/opt/KernelBench",
            ignore=["**/__pycache__", "notebooks", "assets", "results", "scripts", ".env*"],
        )
    )
    app = modal.App("kernelbench-pygpubench", image=image)

    @app.function(gpu="L4", timeout=3600, scaledown_window=300)
    def evaluate_remote(task_id: int, level: int, kernel_name: str, kernel_src: str, repeats: int, measure_reference: bool):
        kernel_dir = Path("/tmp/candidates")
        kernel_dir.mkdir(parents=True, exist_ok=True)
        kernel_file = kernel_dir / kernel_name
        kernel_file.write_text(kernel_src)
        return evaluate_pygpubench(task_id, level, str(kernel_file), repeats=repeats, measure_reference=measure_reference)

    @app.local_entrypoint()
    def main(task_id: int = 1, level: int = 1, kernel_path: str = "candidates/correct_cuda.py", repeats: int = 10,
             skip_reference: bool = False, output: str = ""):
        results = []
        for item in kernel_path.split(","):
            path = Path(item.strip())
            result = evaluate_remote.remote(task_id, level, path.name, path.read_text(), repeats, not skip_reference)
            print_summary(result)
            results.append(result)
        _write_output(results, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a KernelBench task through PyGPUBench on the local GPU")
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--kernel-path", default="candidates/correct_cuda.py")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    results = []
    for item in args.kernel_path.split(","):
        result = evaluate_pygpubench(args.task_id, args.level, item.strip(), repeats=args.repeats,
                                     measure_reference=not args.skip_reference)
        print_summary(result)
        results.append(result)
    _write_output(results, args.output)
    raise SystemExit(0 if all(r["correct"] for r in results) else 1)
