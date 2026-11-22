"""MCP server exposing `evaluate_kernel` and `profile_kernel` over stdio.

Start it with `python -m kernel_tools.mcp_server` (or the `kernel-tools-mcp`
script). Any MCP client can attach: the Claude Agent SDK and Claude Code via an
`mcpServers` stdio entry, Codex via `[mcp_servers.autoresearch]` in its config,
or the OpenAI Agents SDK via `MCPServerStdio`. Each call runs on the user's
Modal account and spends GPU time.
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from . import evaluate as _evaluate
from . import profile as _profile
from .summarize import tool_text

INSTRUCTIONS = (
    "Tools for automated GPU kernel research on KernelBench tasks. evaluate_kernel is the "
    "only source of truth for correctness and speedup: never report a speedup you did not "
    "measure with it. Each call runs on a remote Modal GPU, takes one to several minutes, "
    "and costs money, so compile-check locally first and batch your changes. Pass the full "
    "Python source of a file that defines ModelNew, as KernelBench expects."
)

server = MCPServer(
    "autoresearch",
    instructions=INSTRUCTIONS,
    version="0.1.0",
)


@server.tool(
    name="evaluate_kernel",
    description=(
        "Check a candidate kernel against the KernelBench reference for correctness and measure its "
        "speedup over PyTorch with PyGPUBench on a Modal L4 GPU. Returns a summary and JSON with "
        "compiled, correct, latency_us, reference_latency_us, speedup, and feedback. Use repeats=5 "
        "for a quick check and the default 10 for a reportable number (minimum 4)."
    ),
)
def evaluate_kernel(level: int, task_id: int, kernel_source: str, repeats: int = 10) -> str:
    result = _evaluate.evaluate_kernel(level, task_id, kernel_source, repeats=repeats)
    return tool_text(result["summary"], result)


@server.tool(
    name="profile_kernel",
    description=(
        "Profile a candidate kernel with Nsight Compute on a Modal L4 GPU. Returns, per kernel launch, "
        "duration, SM and memory throughput as percent of peak, achieved occupancy, registers, shared "
        "memory, L1/L2 hit rates, the top warp stall reasons, and a memory-bound/compute-bound/"
        "latency-bound verdict. Slower and costlier than evaluate_kernel; use it to decide what to "
        "optimize next, not to measure speedup."
    ),
)
def profile_kernel(level: int, task_id: int, kernel_source: str, warmup: int = 2) -> str:
    result = _profile.profile_kernel(level, task_id, kernel_source, warmup=warmup)
    return tool_text(result["summary"], result)


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
