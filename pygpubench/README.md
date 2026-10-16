# KernelBench task evaluation through PyGPUBench

`evaluate_pygpubench.py` loads one KernelBench task, runs the PyTorch reference to get expected outputs, loads a candidate `ModelNew` from a file, and hands everything to PyGPUBench for correctness checking and latency measurement.

```python
evaluate_pygpubench(task_id=1, level=1, kernel_path="candidates/correct_cuda.py")
```

## Run on Modal

```bash
cd pygpubench
modal run evaluate_pygpubench.py --kernel-path candidates/correct_cuda.py
modal run evaluate_pygpubench.py --kernel-path candidates/incorrect_cuda.py
modal run evaluate_pygpubench.py --kernel-path candidates/torch_matmul.py --output result.json
```

Options: `--task-id` (default 1), `--level` (default 1), `--repeats` (default 10), `--skip-reference`, `--output <json path>`.

The first run builds the image (CUDA 13 base, PyTorch cu130, `pygpubench==0.0.3` wheel). Later runs reuse it. The repo's own `KernelBench/` folder (identical to upstream commit `423217d`) is mounted into the container at `/opt/KernelBench`, so task files and the `kernelbench` package come from this checkout.

The function runs on an L4. The published PyGPUBench wheel is compiled for compute capability 8.0 and 9.0, so Ampere or newer is required; a T4 will not work.

## Run on a Linux GPU box

Needs Python 3.12 and `pip install pygpubench`. The script adds `../KernelBench/src` to the import path itself. Run from inside this folder, otherwise the folder name shadows the real `pygpubench` package:

```bash
cd pygpubench
python evaluate_pygpubench.py --kernel-path candidates/correct_cuda.py
```

## What happens

1. `construct_kernelbench_dataset(level).get_problem_by_id(task_id)` fetches the task source.
2. A separate subprocess imports the candidate once so `load_inline` compiles before any timing starts. Compile errors are reported as `compiled: false`.
3. `pygpubench.do_bench_isolated("evaluate_pygpubench.kernel", generate_test_case, {}, repeats, seed, ...)` spawns the benchmark worker.
4. Inside the worker, `generate_test_case(seed=...)` uses KernelBench's `load_original_model_and_inputs`, `set_seed`, `_process_input_tensor`, `get_init_inputs()` and `get_inputs()`, runs the reference `Model`, and returns PyGPUBench's test-case format: `((out, *inputs), (expected, rtol, atol))`. `out` is an empty tensor matching `expected`, all tensors are contiguous CUDA tensors, and the tolerance is KernelBench's `get_tolerance_for_precision` (1e-4 for fp32).
5. PyGPUBench copies `expected` into memory it allocated with `cudaMalloc` and zeroes the original before importing the candidate, so the candidate cannot read expected outputs. Reference computation happens in the generator, outside the timed region.
6. `kernel(out, *inputs)` loads `ModelNew` through KernelBench's `load_custom_model`, instantiates it with the same init seed as the reference, and writes `model(*inputs)` into `out`. That copy is inside the timed region for both the candidate and the reference baseline, so the speedup is apples to apples.
7. Correctness comes from PyGPUBench's GPU checker: `correct` is `result.success`, `errors` is the number of sampled elements outside tolerance.

## Result format

```json
{
  "task_id": 1, "level": 1, "task_name": "1_Square_matrix_multiplication_.py",
  "gpu": "NVIDIA L4", "precision": "fp32", "tolerance": 0.0001, "repeats": 10,
  "compiled": true, "correct": true, "errors": 0, "error_message": null,
  "latency_us": {"runs": 10, "expected": 10, "best": ..., "worst": ..., "median": ..., "mean": ..., "std": ..., "err": ...},
  "time_us": [...], "event_overhead_us": ...,
  "reference_latency_us": {...}, "speedup": ...
}
```

## Candidates

- `candidates/correct_cuda.py`: 16x16 shared-memory tiled fp32 matmul via `load_inline`. Expected to pass.
- `candidates/incorrect_cuda.py`: same kernel with the last K tile skipped. Expected to fail with a large `errors` count.
- `candidates/torch_matmul.py`: plain `torch.matmul`, for checking the plumbing without compiling anything.

## Notes

- Landlock and mseal are disabled and `allow_root=True`, as in PyGPUBench's own Modal CI, because Modal containers run as root without those kernel features.
- After the summary prints, Modal shows a traceback ending in `RuntimeError: supervisor: connection closed unexpectedly`. That is PyGPUBench's seccomp supervisor helper process exiting because the sandbox never hands it a seccomp notification handle. It happens after results are returned and does not affect them.
- The PyPI `pygpubench==0.0.3` wheel is slightly older than the GitHub master source: it has no `writable_paths` option and does not make generator tensors contiguous itself, so the generator here does that explicitly.
- PyGPUBench runs under `torch.use_deterministic_algorithms(True)`, so `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set.
- PyGPUBench only checks fp32, fp16 and bf16 outputs and requires the reference to return a single tensor. Tasks that return tuples are rejected with a clear error.
- Each repeat holds its own inputs, expected output, output buffer and shadow copies on the GPU. Lower `--repeats` for large tasks.
