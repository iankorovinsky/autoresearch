# Harbor task: create and optimize CUDA matrix multiplication

This Harbor task starts without a candidate kernel. The coding agent reads the KernelBench level 1, problem 1 reference, creates /workspace/candidate.py with a custom CUDA ModelNew, and uses check_candidate.py to debug and optimize it.

Harbor provisions an L4 through Modal. The verifier independently checks correctness and measures speedup over PyTorch eager. Incorrect or missing candidates receive reward 0; correct candidates receive their speedup capped at 1.0.

## Run it

From the repository root:

~~~bash
harbor run \
  --path ./harbor/tasks/matmul-opt \
  --agent codex \
  --model openai/gpt-6-luna \
  --agent-kwarg reasoning_effort=none \
  --env modal \
  --n-attempts 1 \
  --n-concurrent 1
~~~

Authenticate Modal and the selected model provider before launching. One trial uses an L4 and model usage; the first run also builds the CUDA image.

## Files

- task.toml: Harbor task, resource, and timeout settings.
- instruction.md: create-from-scratch and optimization instructions.
- environment/Dockerfile: CUDA environment and pinned KernelBench checkout. It does not provide a candidate file.
- tests/test.sh: independent final correctness/performance evaluation and Harbor reward.
