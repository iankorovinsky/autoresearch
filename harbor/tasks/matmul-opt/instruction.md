# Create and optimize square matrix multiplication

Create a custom CUDA implementation for KernelBench level 1, problem 1: square matrix multiplication.

Start without a candidate file. Read the reference task at /opt/autoresearch/KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py to understand the expected PyTorch behavior, inputs, and output. Inspect nearby KernelBench examples if useful.

Create /workspace/candidate.py with a PyTorch-compatible ModelNew class and a custom CUDA implementation. Do not call torch.matmul or use the reference implementation as the candidate.

## Experiment loop

1. Create the initial implementation in /workspace/candidate.py.
2. Run a quick evaluation from /opt/autoresearch/KernelBench:

   ~~~bash
   uv run python scripts/check_candidate.py 1 1 /workspace/candidate.py \
     --eval-mode local \
     --setting num_correct_trials=1 \
     --setting num_perf_trials=5 \
     --setting 'gpu_arch=["Turing"]'
   ~~~

3. If it fails to compile or pass correctness, inspect the error, fix the candidate, and rerun the evaluation.
4. Once it passes correctness, make focused optimizations. Keep a change only when it remains correct and improves the printed Speedup over eager. Save the best correct candidate before trying another change; restore it if the new candidate is incorrect or slower.
5. Repeat focused experiments while time remains. Leave the fastest correct implementation in /workspace/candidate.py.

Before finishing, report the best speedup over PyTorch eager and torch.compile, the number of experiments, and any failed approaches. The verifier will independently evaluate the final candidate on an L4.
