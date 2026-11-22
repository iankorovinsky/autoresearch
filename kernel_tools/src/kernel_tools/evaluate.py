"""Evaluate a KernelBench candidate for correctness and speedup via PyGPUBench on Modal."""
from __future__ import annotations

from typing import Any

from .runner import run_modal_script
from .summarize import summarize_evaluation

EVALUATOR = "pygpubench/evaluate_pygpubench.py"
MAX_SOURCE_BYTES = 512 * 1024


MIN_REPEATS = 4


def validate_candidate(level: int, task_id: int, kernel_source: str) -> str | None:
    """Return a problem description, or None when the request is worth a GPU run."""
    if not isinstance(level, int) or level < 1 or level > 4:
        return "level must be a KernelBench level between 1 and 4"
    if not isinstance(task_id, int) or task_id < 1:
        return "task_id must be a positive KernelBench problem id"
    if not kernel_source or not kernel_source.strip():
        return "kernel_source is empty; pass the full Python file that defines ModelNew"
    if len(kernel_source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return "kernel_source is larger than 512 KB"
    if "ModelNew" not in kernel_source:
        return "kernel_source must define a torch.nn.Module named ModelNew (KernelBench convention)"
    return None


def evaluate_kernel(level: int, task_id: int, kernel_source: str, *, repeats: int = 10,
                    measure_reference: bool = True, timeout_s: int = 1800) -> dict[str, Any]:
    """Run the repository's evaluator on Modal and return its result dict plus `ok` and `summary`."""
    problem = validate_candidate(level, task_id, kernel_source)
    if problem:
        result = {"ok": False, "error": problem, "level": level, "task_id": task_id}
        result["summary"] = summarize_evaluation(result)
        return result
    # PyGPUBench needs at least five timed test cases (repeats + 1), so floor at 4.
    repeats = max(MIN_REPEATS, min(int(repeats), 100))
    args = ["--task-id", str(task_id), "--level", str(level), "--repeats", str(repeats)]
    if not measure_reference:
        args.append("--skip-reference")
    run = run_modal_script(EVALUATOR, args, kernel_source=kernel_source, kernel_name="candidate.py",
                           timeout_s=timeout_s)
    if not run.ok or run.result is None:
        result = {"ok": False, "error": run.error, "level": level, "task_id": task_id,
                  "stderr": run.stderr[-2000:]}
    else:
        result = dict(run.result)
        result["ok"] = True
    result["summary"] = summarize_evaluation(result)
    return result
