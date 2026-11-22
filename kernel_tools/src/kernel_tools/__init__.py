"""Kernel evaluation and profiling tools that any agent harness can call.

Plain functions live in `evaluate` and `profile`; `mcp_server` exposes them
over the Model Context Protocol so Claude Code, the Claude Agent SDK, Codex,
or an OpenAI Agents SDK orchestrator can attach the same tools.
"""

from .evaluate import evaluate_kernel
from .profile import profile_kernel

__all__ = ["evaluate_kernel", "profile_kernel"]
