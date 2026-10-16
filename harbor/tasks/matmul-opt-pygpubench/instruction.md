# Create and optimize square matrix multiplication

Create a custom CUDA implementation for KernelBench level 1, problem 1: square matrix multiplication.

Start without a candidate file. Read the reference task at /opt/autoresearch/KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py to understand the expected PyTorch behavior, inputs, and output. Inspect nearby KernelBench examples if useful.

Create /workspace/candidate.py with a PyTorch-compatible ModelNew class and a custom CUDA implementation. Do not call torch.matmul or use the reference implementation as the candidate.

## Experiment loop

1. Create the initial implementation in /workspace/candidate.py.
2. Evaluate it:

   ~~~bash
   python /opt/autoresearch/pygpubench/evaluate_pygpubench.py --task-id 1 --level 1 --kernel-path /workspace/candidate.py
   ~~~

   The last line of the summary, `feedback:`, states either why the candidate failed (compile error, output mismatch with the measured difference, or runtime error) or its latency and speedup over PyTorch when it passed. The evaluation first runs KernelBench's correctness check and then times correct candidates with PyGPUBench, which checks every output element on the GPU. A run takes one to two minutes.
3. If it fails, read the feedback, fix the candidate, and rerun.
4. Once it passes, make focused optimizations. Keep a change only when it remains correct and improves the speedup. Save the best correct candidate before trying another change; restore it if the new candidate is incorrect or slower.
5. Repeat focused experiments while time remains. Leave the fastest correct implementation in /workspace/candidate.py.

Before finishing, report the best speedup, the number of experiments, and any failed approaches. The verifier will independently evaluate the final candidate on an L4 with the same tool.
