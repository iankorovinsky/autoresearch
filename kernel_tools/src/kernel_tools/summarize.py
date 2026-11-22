"""Compact, model-readable summaries of evaluator and profiler results."""
from __future__ import annotations

import json
from typing import Any

# Keys that are large arrays or redundant for a chat reply.
_DROP_FROM_JSON = {"time_us", "kernelbench_check", "stdout", "stderr"}


def summarize_evaluation(result: dict[str, Any]) -> str:
    if not result.get("ok", True):
        return f"evaluation failed before producing a result: {result.get('error')}"
    lines = []
    status = "CORRECT" if result.get("correct") else ("COMPILED but incorrect" if result.get("compiled") else "FAILED")
    lines.append(f"{status}: level {result.get('level')} task {result.get('task_id')} "
                 f"({result.get('task_name')}) on {result.get('gpu')}")
    lat = result.get("latency_us")
    if isinstance(lat, dict) and lat.get("mean") is not None:
        lines.append(f"candidate latency: {lat['mean']:.1f} us mean, {lat.get('median', 0):.1f} median, "
                     f"{lat.get('std', 0):.1f} std over {lat.get('runs')} runs")
    ref = result.get("reference_latency_us")
    if isinstance(ref, dict) and ref.get("mean") is not None:
        lines.append(f"torch reference latency: {ref['mean']:.1f} us mean")
    if result.get("speedup"):
        lines.append(f"speedup over torch: {result['speedup']:.2f}x")
    if result.get("errors"):
        lines.append(f"output elements outside tolerance during timed runs: {result['errors']}")
    if result.get("feedback"):
        lines.append(f"feedback: {result['feedback']}")
    return "\n".join(lines)


def summarize_profile(result: dict[str, Any]) -> str:
    if not result.get("ok", True):
        return f"profiling failed before producing a result: {result.get('error')}"
    head = (f"{'PROFILED' if result.get('profiled') else 'NOT PROFILED'}: level {result.get('level')} "
            f"task {result.get('task_id')} ({result.get('task_name')}) on {result.get('gpu')}")
    body = result.get("summary") or ""
    tail = f"feedback: {result.get('feedback')}" if result.get("feedback") else ""
    return "\n".join(part for part in (head, body, tail) if part)


def compact_json(result: dict[str, Any]) -> str:
    slim = {k: v for k, v in result.items() if k not in _DROP_FROM_JSON}
    return json.dumps(slim, indent=None, default=str)


def tool_text(summary: str, result: dict[str, Any]) -> str:
    """What the model receives: a readable summary followed by machine-readable JSON."""
    return summary + "\n\nJSON:\n" + compact_json(result)
