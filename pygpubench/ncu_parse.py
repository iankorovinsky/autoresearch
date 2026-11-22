"""Reduce Nsight Compute raw CSV output to a small, model-readable summary.

`ncu --csv --page raw` prints one header row of column names, one row of
units, then one row per profiled kernel launch. Section metrics such as
`smsp__average_warps_issue_stalled_*` appear as ordinary columns. This module
has no GPU or Modal dependency so it can be unit-tested anywhere.
"""
from __future__ import annotations

import csv
import io
from typing import Any

STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"

# Raw metric name -> short field name in the summary.
FIELDS = {
    "gpu__time_duration.sum": "duration",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "memory_throughput_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
    "sm__maximum_warps_per_active_cycle_pct": "theoretical_occupancy_pct",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__shared_mem_per_block_static": "static_smem_bytes",
    "launch__shared_mem_per_block_dynamic": "dynamic_smem_bytes",
    "launch__grid_size": "grid_size",
    "launch__block_size": "block_size",
    "launch__waves_per_multism": "waves_per_sm",
    "launch__waves_per_multiprocessor": "waves_per_sm",
    "launch__occupancy_limit_registers": "occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem": "occupancy_limit_shared_mem",
    "launch__occupancy_limit_blocks": "occupancy_limit_blocks",
    "lts__t_sector_hit_rate.pct": "l2_hit_rate_pct",
    "l1tex__t_sector_hit_rate.pct": "l1_hit_rate_pct",
    "smsp__issue_active.avg.pct_of_peak_sustained_active": "issue_slot_utilization_pct",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active": "tensor_pipe_pct",
}

_IDENTITY_COLUMNS = {
    "ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context",
    "Stream", "Block Size", "Grid Size", "Device", "CC", "Section Name",
}


def _number(value: str) -> Any:
    text = value.strip().replace(",", "")
    if text in ("", "n/a", "N/A"):
        return None
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return value.strip()


def _to_us(value: Any, unit: str) -> Any:
    if not isinstance(value, (int, float)):
        return value
    unit = unit.strip().lower()
    scale = {"nsecond": 1e-3, "ns": 1e-3, "usecond": 1.0, "us": 1.0,
             "msecond": 1e3, "ms": 1e3, "second": 1e6, "s": 1e6}.get(unit)
    return round(value * scale, 3) if scale else value


def _to_bytes(value: Any, unit: str) -> Any:
    if not isinstance(value, (int, float)):
        return value
    unit = unit.strip().lower()
    scale = {"byte": 1, "bytes": 1, "b": 1, "kbyte": 1024, "kb": 1024,
             "mbyte": 1024 ** 2, "mb": 1024 ** 2}.get(unit)
    return int(round(value * scale)) if scale else value


def parse_ncu_raw_csv(text: str) -> list[dict[str, Any]]:
    """Return one summary dict per profiled kernel launch, in launch order."""
    lines = [line for line in text.splitlines() if line.strip()]
    # ncu may print warnings before the table; the header starts with "ID".
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID"')), None)
    if start is None:
        return []
    rows = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
    if len(rows) < 3:
        return []
    header, units = rows[0], rows[1]
    kernels: list[dict[str, Any]] = []
    for row in rows[2:]:
        if len(row) != len(header):
            continue
        record = dict(zip(header, row))
        unit_of = dict(zip(header, units))
        entry: dict[str, Any] = {
            "name": record.get("Kernel Name", ""),
            "grid": record.get("Grid Size", ""),
            "block": record.get("Block Size", ""),
        }
        stalls: dict[str, float] = {}
        for column, raw in record.items():
            if column in _IDENTITY_COLUMNS:
                continue
            value = _number(raw)
            if column in FIELDS:
                key = FIELDS[column]
                if key == "duration":
                    entry["duration_us"] = _to_us(value, unit_of.get(column, ""))
                elif key.endswith("_bytes"):
                    entry[key] = _to_bytes(value, unit_of.get(column, ""))
                else:
                    entry[key] = value
            elif column.startswith(STALL_PREFIX) and column.endswith(STALL_SUFFIX):
                reason = column[len(STALL_PREFIX):-len(STALL_SUFFIX)]
                if isinstance(value, (int, float)):
                    stalls[reason] = float(value)
        if stalls:
            ranked = sorted(stalls.items(), key=lambda item: item[1], reverse=True)
            entry["top_stalls"] = [{"reason": r, "cycles_per_issue": round(v, 2)} for r, v in ranked[:4]]
            total = sum(stalls.values())
            entry["stall_share_pct"] = {r: round(100 * v / total, 1) for r, v in ranked[:4]} if total else {}
        static = entry.get("static_smem_bytes") or 0
        dynamic = entry.get("dynamic_smem_bytes") or 0
        if isinstance(static, int) and isinstance(dynamic, int) and (static or dynamic):
            entry["shared_mem_bytes"] = static + dynamic
        entry["verdict"] = classify(entry)
        kernels.append(entry)
    return kernels


def classify(entry: dict[str, Any]) -> str:
    """A coarse bound classification from the speed-of-light percentages."""
    sm = entry.get("sm_throughput_pct")
    mem = entry.get("memory_throughput_pct")
    if mem is None:
        mem = entry.get("dram_throughput_pct")
    occ = entry.get("achieved_occupancy_pct")
    if isinstance(mem, (int, float)) and mem >= 60 and (not isinstance(sm, (int, float)) or mem >= sm):
        return "memory-bound"
    if isinstance(sm, (int, float)) and sm >= 60:
        return "compute-bound"
    if isinstance(occ, (int, float)) and occ < 25:
        return "latency-bound (low occupancy)"
    if isinstance(sm, (int, float)) or isinstance(mem, (int, float)):
        return "latency-bound (both pipes underused)"
    return "unknown"


def summarize_kernels(kernels: list[dict[str, Any]]) -> str:
    """One line per kernel for a chat reply."""
    if not kernels:
        return "no kernels were profiled"
    lines = []
    for k in kernels:
        parts = [k.get("name", "?")[:60]]
        if k.get("duration_us") is not None:
            parts.append(f"{k['duration_us']:.1f} us")
        for key, label in (("sm_throughput_pct", "SM"), ("memory_throughput_pct", "mem"),
                           ("achieved_occupancy_pct", "occupancy")):
            if isinstance(k.get(key), (int, float)):
                parts.append(f"{label} {k[key]:.0f}%")
        if k.get("registers_per_thread") is not None:
            parts.append(f"{k['registers_per_thread']} regs")
        if k.get("top_stalls"):
            parts.append("stalls: " + ", ".join(s["reason"] for s in k["top_stalls"][:3]))
        parts.append(k.get("verdict", ""))
        lines.append(" | ".join(str(p) for p in parts if p != ""))
    return "\n".join(lines)
