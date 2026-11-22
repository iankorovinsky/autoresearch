# Kernel tools

Harness-agnostic tools for the kernel research loop, exposed as a
[Model Context Protocol](https://modelcontextprotocol.io) server so any agent
harness can call them: the Claude Agent SDK, Claude Code, Codex, or an
OpenAI Agents SDK orchestrator. The logic is plain Python, so a harness that
prefers function tools can import it directly.

| Tool | What it does | Backed by |
| --- | --- | --- |
| `evaluate_kernel(level, task_id, kernel_source, repeats=10)` | KernelBench correctness check, then PyGPUBench timing against the PyTorch reference. Returns compiled, correct, latency, speedup, feedback. | `pygpubench/evaluate_pygpubench.py` on a Modal L4 |
| `profile_kernel(level, task_id, kernel_source, warmup=2)` | Nsight Compute run reduced to throughput, occupancy, launch stats, top stall reasons, and a bound verdict per kernel launch. | `pygpubench/profile_ncu.py` on a Modal L4 |

Both tools take the full Python source of a file that defines `ModelNew`,
write it to a temporary file, run the Modal script as a subprocess from the
repository root, and return a readable summary followed by JSON. Every call
uses the Modal account logged in on this machine and spends GPU time; expect
one to several minutes per call, more on the first run while the image builds.

## Setup

This component has its own environment, like `compute/` and `tooling/`.

```powershell
cd kernel_tools
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e .
```

With `uv`: `uv sync --python 3.12` inside `kernel_tools/`.

Modal must be logged in (`python -m modal setup`). Tests run without Modal:

```powershell
.venv\Scripts\python -m unittest discover -s tests
```

## Run the server

```powershell
kernel_tools\.venv\Scripts\python -m kernel_tools.mcp_server
```

The server speaks MCP over stdio. Environment variables:

- `AUTORESEARCH_REPO_ROOT`: repository root if the package is installed elsewhere.
- `AUTORESEARCH_MODAL_PYTHON`: interpreter with the Modal client, when it is not the one running the server.

## Attach it to a harness

Claude Agent SDK (what `frontend/` does when `AUTORESEARCH_CLAUDE_TOOLS=1`):

```ts
mcpServers: {
  autoresearch: {
    type: "stdio",
    command: "kernel_tools/.venv/Scripts/python",
    args: ["-m", "kernel_tools.mcp_server"],
    cwd: repoRoot,
  },
},
allowedTools: ["mcp__autoresearch__evaluate_kernel", "mcp__autoresearch__profile_kernel"],
```

Claude Code CLI:

```powershell
claude mcp add autoresearch -- kernel_tools\.venv\Scripts\python -m kernel_tools.mcp_server
```

Codex CLI (`~/.codex/config.toml`):

```toml
[mcp_servers.autoresearch]
command = "C:/path/to/autoresearch/kernel_tools/.venv/Scripts/python"
args = ["-m", "kernel_tools.mcp_server"]
```

OpenAI Agents SDK:

```python
from agents.mcp import MCPServerStdio
async with MCPServerStdio(params={"command": ".venv/Scripts/python", "args": ["-m", "kernel_tools.mcp_server"]}) as server:
    agent = Agent(name="optimizer", mcp_servers=[server], ...)
```

Direct import, no MCP:

```python
from kernel_tools import evaluate_kernel, profile_kernel
result = evaluate_kernel(1, 1, open("candidate.py").read(), repeats=5)
print(result["summary"])
```

## Result shape

`evaluate_kernel` returns the evaluator's dictionary (`compiled`, `correct`,
`latency_us` stats, `reference_latency_us`, `speedup`, `feedback`, and more)
plus `ok` and `summary`. `profile_kernel` returns `profiled`, `kernels` (one
entry per launch with `duration_us`, `sm_throughput_pct`,
`memory_throughput_pct`, `achieved_occupancy_pct`, `registers_per_thread`,
`shared_mem_bytes`, `l2_hit_rate_pct`, `top_stalls`, `verdict`), `summary`,
and `feedback`. When Modal or the script fails, `ok` is false and `error`
holds the tail of the output. Large arrays are dropped from the JSON that
reaches the model.

## Limits

- KernelBench tasks only, through the existing evaluator; other benchmarks
  need another script and a tool that calls it.
- Nsight Compute needs GPU performance-counter access; when the host denies it
  the tool reports `ERR_NVGPUCTRPERM` instead of numbers.
- No anti-cheating checks beyond the evaluator's own correctness trials yet.
