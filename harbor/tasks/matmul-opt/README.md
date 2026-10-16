# Harbor task: create and optimize CUDA matrix multiplication

This Harbor task starts without a candidate kernel. The coding agent reads the KernelBench level 1, problem 1 reference, creates /workspace/candidate.py with a custom CUDA ModelNew, and uses check_candidate.py to debug and optimize it.

Harbor provisions an L4 through Modal. The verifier independently checks correctness and measures speedup over PyTorch eager. Incorrect or missing candidates receive reward 0; correct candidates receive their speedup capped at 1.0.

## Run it

From the repository root:

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

Authenticate Modal and the selected model provider before launching. One trial uses an L4 and model usage; the first run also builds the CUDA image. Each invocation picks the next unused name (`run1`, `run2`, ...) and stores its output in a separate folder under `harbor/tasks/matmul-opt/runs/`.

## Files

- task.toml: Harbor task, resource, and timeout settings.
- instruction.md: create-from-scratch and optimization instructions.
- environment/Dockerfile: CUDA environment and pinned KernelBench checkout. It does not provide a candidate file.
- tests/test.sh: independent final correctness/performance evaluation and Harbor reward.
