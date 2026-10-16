# Kernel autoresearch with Harbor

This folder contains a Harbor task where a coding agent creates a CUDA kernel from scratch, then iteratively improves it using KernelBench feedback.

## Workflow

1. Harbor starts an isolated task environment on a Modal NVIDIA L4.
2. The agent reads the KernelBench reference and creates /workspace/candidate.py.
3. The agent runs check_candidate.py to fix correctness issues and compare speed.
4. Harbor runs the verifier on the final candidate and records the reward, logs, trajectory, and candidate artifact.

## Start

Authenticate Modal and the selected model provider, then run:

~~~bash
RUNS_DIR="./harbor/tasks/matmul-opt/runs"
RUN_NUMBER=1
while [ -e "$RUNS_DIR/run$RUN_NUMBER" ]; do
  RUN_NUMBER=$((RUN_NUMBER + 1))
done

harbor run \
  --path ./harbor/tasks/matmul-opt \
  --agent codex \
  --model openai/gpt-6-luna \
  --agent-kwarg reasoning_effort=none \
  --env modal \
  --jobs-dir "$RUNS_DIR" \
  --job-name "run$RUN_NUMBER" \
  --n-attempts 1 \
  --n-concurrent 1
~~~

The task uses an L4; the first run also builds a CUDA image. Model usage and GPU time may incur charges. Each invocation picks the next unused name (`run1`, `run2`, ...) and stores its output in a separate folder under `harbor/tasks/matmul-opt/runs/`.

## Current scope

The first operation is KernelBench level 1, problem 1: square matrix multiplication. KernelBench supplies the reference operation, correctness checks, and performance measurements. Harbor runs the agent task and records the verifier score. PyGPUBench is not wired in yet.
