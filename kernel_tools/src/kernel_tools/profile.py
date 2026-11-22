"""Profile a KernelBench candidate with Nsight Compute on Modal."""
from __future__ import annotations

from typing import Any

from .evaluate import validate_candidate
from .runner import run_modal_script
from .summarize import summarize_profile

PROFILER = "pygpubench/profile_ncu.py"


def profile_kernel(level: int, task_id: int, kernel_source: str, *, warmup: int = 2,
                   timeout_s: int = 2400) -> dict[str, Any]:
    """Run the repository's Nsight Compute profiler on Modal and return the reduced result."""
    problem = validate_candidate(level, task_id, kernel_source)
    if problem:
        result = {"ok": False, "error": problem, "level": level, "task_id": task_id}
        result["summary"] = summarize_profile(result)
        return result
    warmup = max(0, min(int(warmup), 20))
    args = ["--task-id", str(task_id), "--level", str(level), "--warmup", str(warmup)]
    run = run_modal_script(PROFILER, args, kernel_source=kernel_source, kernel_name="candidate.py",
                           timeout_s=timeout_s)
    if not run.ok or run.result is None:
        result = {"ok": False, "error": run.error, "level": level, "task_id": task_id,
                  "stderr": run.stderr[-2000:]}
    else:
        result = dict(run.result)
        result["ok"] = True
    result["summary"] = summarize_profile(result)
    return result
