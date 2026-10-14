# Run your first KernelBench candidate

This guide evaluates an existing kernel against a PyTorch task using a remote
NVIDIA T4 on Modal. You can launch it from a Mac without a local NVIDIA GPU.

## 1. Get the repository and dependencies

Clone this repository or copy the complete checkout, including the updated
`scripts/check_candidate.py`. Open a terminal in the repository root (the folder
containing `pyproject.toml`). Keep `src/`, `scripts/`, and `KernelBench/` together.

You need `uv`, internet access, and a Modal account with access to GPU compute.
Remote runs use your Modal account's compute allowance or billing.

Install `uv` using the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
If you already have Python and pip, you can install it with:

```bash
python3 -m pip install uv
```

Then install the Python version and dependencies used by this project:

```bash
uv python install 3.10
uv sync --locked
```

`pyproject.toml` defines dependencies and `uv.lock` pins the resolved versions.
Use these files for setup; `requirements.txt` is archived. You do not need to
activate the virtual environment when using `uv run`.

For Modal evaluation, the base dependencies are sufficient on your computer.
The evaluator builds the remote CUDA image and installs the `gpu` extras there.
LLM provider API keys are not needed to evaluate an existing candidate.

## 2. Authenticate with Modal

```bash
uv run modal token new
```

Follow the browser login flow. You only need to do this when the machine does
not already have working Modal credentials. See the
[Modal token documentation](https://modal.com/docs/cli/latest/token).

## 3. Run the included example

```bash
uv run python scripts/check_candidate.py 1 1 \
  src/kernelbench/prompts/few_shot/model_new_ex_tiled_matmul.py \
  --eval-mode modal --gpu T4 \
  --setting num_correct_trials=1 \
  --setting num_perf_trials=5
```

This selects level 1, problem 1 (square matrix multiplication), and checks the
included tiled matrix multiplication candidate. The evaluator automatically
selects the Turing architecture for a T4. The first run may take longer while
Modal builds the image and starts a GPU container.

Look for `compiled=True` and `correctness=True` in the output. Timing results
compare the candidate against PyTorch eager and `torch.compile`. A speedup
greater than `1.0x` means the candidate is faster; less than `1.0x` means slower.
The process exit code alone does not establish that correctness passed.

One correctness trial and five timing trials are useful for a quick check.
Omit those two settings to use the evaluator defaults of five correctness
trials and 100 timing trials.

## 4. Run your own candidate

```bash
uv run python scripts/check_candidate.py LEVEL PROBLEM_ID /path/to/candidate.py \
  --eval-mode modal --gpu T4
```

Replace `LEVEL` and `PROBLEM_ID` with numbers. Tasks are stored under
`KernelBench/level<LEVEL>/`; the filename prefix identifies the problem ID.
Your candidate must define `ModelNew` with constructor and forward arguments
compatible with the task's `Model`. Paths containing spaces should be quoted.
Relative candidate paths are resolved from the directory where you launch the command.

| Option | Purpose |
| --- | --- |
| `--eval-mode modal` | Execute on Modal instead of your local machine. Local is the default. |
| `--gpu T4` | Choose the remote GPU. T4 is the wrapper default for Modal. |
| `--dry-run` | Print the command without starting evaluation. |
| `--setting backend=triton` | Evaluate a Triton candidate; the default backend is CUDA. |
| `--setting num_correct_trials=5` | Set the number of correctness trials. |
| `--setting num_perf_trials=100` | Set the number of timing trials. |
| `--setting precision=fp32` | Select tensor precision; FP32 is the default. |

Repeat `--setting KEY=VALUE` to pass additional evaluator settings. See
`ScriptConfig` in `scripts/run_and_check.py` for supported settings.

## Troubleshooting

- **`CUDA is not available` / `Torch not compiled with CUDA enabled`:** Check
  that your command includes `--eval-mode modal`. Local CUDA execution requires
  a compatible GPU machine; changing the architecture flag cannot enable CUDA
  on a Mac.
- **`No module named kernelbench`:** Use the wrapper from this checkout. It adds
  the repository's `src` directory to the child process's `PYTHONPATH`. Ensure
  the full repository is present and run `uv sync --locked`.
- **`uv` not found:** Install `uv` and ensure its executable is on your `PATH`.
- **Modal authentication failure:** Run `uv run modal token new` and authenticate
  with the account you intend to use for GPU execution.
- **Missing candidate or task:** Check the candidate path and the matching
  `KernelBench/level<LEVEL>/<PROBLEM_ID>_*.py` task file.

## Optional: run locally on a GPU machine

On a compatible NVIDIA GPU machine with the required drivers and CUDA build
toolchain installed, install GPU dependencies and select the local architecture.
For a T4:

```bash
uv sync --locked --extra gpu
uv run python scripts/check_candidate.py 1 1 \
  src/kernelbench/prompts/few_shot/model_new_ex_tiled_matmul.py \
  --eval-mode local --setting 'gpu_arch=["Turing"]'
```

`--gpu` selects a Modal GPU; it does not select local hardware.

## PyGPUBench evaluation

`pygpubench/evaluate_pygpubench.py` evaluates a candidate `ModelNew` against a
KernelBench task through [PyGPUBench](https://github.com/gpu-mode/pygpubench),
the adversarial benchmarking harness. It needs only the `modal` package locally;
PyTorch, PyGPUBench and this repository's `KernelBench/` folder are set up in
the Modal image (CUDA 13, Python 3.12, L4 GPU).

```bash
cd pygpubench
modal run evaluate_pygpubench.py --task-id 1 --level 1 --kernel-path candidates/correct_cuda.py
```

Options: `--task-id`, `--level`, `--kernel-path` (one file or several separated
by commas), `--repeats` (default 10), `--skip-reference`, `--output result.json`.
Several kernels in one call share one container and one cached PyTorch baseline.

Each candidate goes through these stages, stopping at the first failure:

1. Compile and load `ModelNew` with KernelBench's `load_custom_model` in a subprocess.
2. KernelBench's `run_and_check_correctness` on random inputs. A failure returns
   the reason (shape mismatch, max and mean absolute difference, or traceback).
3. PyGPUBench benchmark in an isolated worker: expected outputs are hidden from
   the candidate, every output element is checked on the GPU, and latency is
   measured with CUDA events after clearing L2.
4. PyTorch reference timed the same way, once per task and GPU, then cached.

The result is a dict with `compiled`, `correct`, `speedup`, a one-line `feedback`
string, and the full latency statistics. `candidates/correct_cuda.py` is a tiled
CUDA matmul that passes task 1 at about 0.10x of cuBLAS, `candidates/incorrect_cuda.py`
is the same kernel with one tile dropped and fails at stage 2, and
`candidates/torch_matmul.py` is a plain PyTorch candidate for checking the setup.

Notes: PyGPUBench's published wheel targets compute capability 8.0 and 9.0, so
a T4 will not run it. Landlock and mseal are disabled on Modal, as in PyGPUBench's
own CI. The `supervisor: connection closed unexpectedly` traceback printed after
each result is PyGPUBench's helper process exiting on Modal's sandbox and does not
affect the result. PyGPUBench checks only fp32, fp16 and bf16 single-tensor outputs.

### Harbor task verified by PyGPUBench

`harbor/tasks/matmul-opt-pygpubench/` is the same matrix multiplication task with
`evaluate_pygpubench.py` as the verifier. The agent iterates with that script and
the final candidate is scored by it; the reward is the PyGPUBench speedup capped at 1.0.

```bash
CODEX_FORCE_AUTH_JSON=1 harbor run \
  --path ./harbor/tasks/matmul-opt-pygpubench \
  --agent codex \
  --model openai/gpt-6-luna \
  --agent-kwarg reasoning_effort=none \
  --env modal \
  --n-attempts 1 \
  --n-concurrent 1 \
  --jobs-dir ./harbor/tasks/matmul-opt-pygpubench/runs \
  --job-name run1
```

`CODEX_FORCE_AUTH_JSON=1` makes Harbor use the Codex login in `~/.codex/auth.json`.
Results land in `harbor/tasks/matmul-opt-pygpubench/runs/run1/`, with the agent's
trajectory under `agent/`, the scored kernel under `artifacts/`, and the PyGPUBench
output under `verifier/`.

## Run kernel autoresearch with Harbor

A Harbor task for optimizing the level 1 square matrix multiplication kernel is in [harbor/README.md](harbor/README.md). It runs the agent and correctness/performance verifier in a Modal L4 environment. The starter kernel is based on the existing tiled CUDA example.

## Conversation database

The [`data/`](data/README.md) package manages projects, threads, and ordered messages with Prisma migrations for Supabase PostgreSQL. It is separate from the frontend and contains no seed data. See its README for local validation and deployment instructions.
