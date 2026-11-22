# Upstream research code

These checkouts are for reading and comparing upstream implementations. Project code runs through the existing [KernelBench](../../KernelBench/) and [PyGPUBench](../../pygpubench/) integrations. See the [paper analyses](../papers/analysis/README.md) for the experimental conditions and limits of each source.

## Benchmarking and verification

- **[KernelBench](./KernelBench/)** provides the original task definitions, candidate interface, timing code, and evaluator. Use it to compare our local integration with upstream. [Paper 01](../papers/analysis/01-kernelbench.md)
- **[KernelBenchX](./KernelBenchX/)** adds harder tasks, optimized baselines, and records of correctness and performance across optimization attempts. [Paper 02](../papers/analysis/02-kernelbench-x.md)
- **[KernelBench-Verified](./kernel_bench_verified/)** provides hidden test cases, stricter baseline comparisons, and checks for suspicious submissions. [Paper 03](../papers/analysis/03-kernelbench-verified.md)
- **[PyGPUBench](./pygpubench/)** is the upstream GPU benchmarking library used by our adapter. Start here for how it isolates execution, validates output, and measures runtime. [Upstream](https://github.com/gpu-mode/pygpubench)
- **[robust-kbench](./robust-kbench/)** provides an independent benchmark, verification methods, and an optimization loop with profiler feedback. [Paper 08](../papers/analysis/08-robust-agentic-cuda.md)
- **[MMA-Sim](./MMA-Sim/)** models the arithmetic behavior of GPU matrix instructions. Use it when investigating device-specific differences in matmul results. [Paper 09](../papers/analysis/09-bit-accurate-gpu-mma.md)

## Task sources and agent scaffolding

- **[KernelBenchFiltered](./KernelBenchFiltered/)** contains METR's filtered tasks, Level 5 workloads, and released solutions. [Article 21](../papers/analysis/21-measuring-automated-kernel-engineering.md)
- **[PTXBench](./PTXBench/)** provides a common harness for comparing kernels across GPU programming languages and GPUs. [Paper 04](../papers/analysis/04-ptxbench.md)
- **[Caesar](./caesar/)** is the multi-turn inference engine used for KernelBench's iterative refinement experiments. Use it to see how correctness, timing, and profiling are passed back between attempts. [Upstream](https://github.com/ScalingIntelligence/caesar)
- **[GPU MODE reference kernels](./reference-kernels/)** provide leaderboard task definitions and reference implementations for additional workloads. [Upstream](https://github.com/gpu-mode/reference-kernels)

Other useful sources include [TritonBench](https://github.com/thunlp/TritonBench) for Triton-specific task coverage, [MultiKernelBench](https://github.com/wzzll123/MultiKernelBench) for non-GPU accelerator coverage, and [ISO-Bench](https://github.com/Lossfunk/ISO-Bench) for optimization inside existing inference repositories. [AutoKernel](https://github.com/RightNow-AI/autokernel) and [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) illustrate lightweight agent loops. The [LARK paper](https://arxiv.org/abs/2611.02028v1) describes the Lean verification workflow. Its public code repository has yet to be identified.

### FlashInfer competition

**[flashinfer-competition](./flashinfer-competition/)** provides an agent loop and a Modal B200 evaluation setup for optimizing a CuTe DSL kernel for DeepSeek Sparse Attention. The benchmark runs the candidate kernel directly on the GPU. [Pinned setup](https://github.com/AndreSlavescu/flashinfer-competition/blob/5f827c364cdf3469a0af293e8a32ebe0ca79534f/README.md#L32-L59) · [Benchmark entry point](https://github.com/AndreSlavescu/flashinfer-competition/blob/5f827c364cdf3469a0af293e8a32ebe0ca79534f/scripts/bench.py#L110-L125)

- **Earlier Claude Code setup (`7110054`).** The kernel-optimizer agent definition describes a research, implementation, validation, and profiling loop. It uses Markdown files under `.claude/agent-memory/` to carry variant timings and bottleneck findings across sessions. The commit also adds Nsight Compute profiling and SASS disassembly tools. Its standalone harness checks output shape and dtype and compares both outputs with absolute, relative, and cosine-difference tolerances. It defaults to 10 warmups and 100 CUDA-event timings, with an L2 flush before each timed call, and reports median and p99 latency. [Claude workflow and memory](https://github.com/AndreSlavescu/flashinfer-competition/blob/71100543ff4a5a66978ab63ba13f1693e1bbb054/.claude/agents/kernel-optimizer.md#L79-L241) · [Memory instructions](https://github.com/AndreSlavescu/flashinfer-competition/blob/71100543ff4a5a66978ab63ba13f1693e1bbb054/.claude/agents/kernel-optimizer.md#L315-L386) · [Validation](https://github.com/AndreSlavescu/flashinfer-competition/blob/71100543ff4a5a66978ab63ba13f1693e1bbb054/tools/harness/validate.py#L129-L186) · [Timing](https://github.com/AndreSlavescu/flashinfer-competition/blob/71100543ff4a5a66978ab63ba13f1693e1bbb054/tools/harness/bench.py#L56-L182)
- **Pinned OpenAI loop (`5f827c3`).** The Python runner uses the OpenAI Agents SDK and defaults to `gpt-5.4` for planning, design, coding, and optimization. It feeds benchmark results back into later rounds and saves the round history, best latency, and optional Codex worker thread IDs in `loop_state.json` for resumption. [Agent setup](https://github.com/AndreSlavescu/flashinfer-competition/blob/5f827c364cdf3469a0af293e8a32ebe0ca79534f/README.md#L13-L37) · [Model defaults](https://github.com/AndreSlavescu/flashinfer-competition/blob/5f827c364cdf3469a0af293e8a32ebe0ca79534f/main.py#L2217-L2255) · [Checkpoint](https://github.com/AndreSlavescu/flashinfer-competition/blob/5f827c364cdf3469a0af293e8a32ebe0ca79534f/main.py#L136-L170) · [Optimization rounds](https://github.com/AndreSlavescu/flashinfer-competition/blob/5f827c364cdf3469a0af293e8a32ebe0ca79534f/main.py#L2066-L2195)
- **Fast correctness checks.** The synthetic sweep runs seeded cases with 1, 2, 6, 7, and 8 query tokens and compares results with cached reference outputs. A case passes if the maximum absolute error is below `0.01` for the attention output and below `0.001` for the softmax log-sum-exp. The sweep returns pass/fail and errors. [Case configuration](https://github.com/AndreSlavescu/flashinfer-competition/blob/5f827c364cdf3469a0af293e8a32ebe0ca79534f/scripts/bench_synthetic_common.py#L15-L20) · [Reference fixtures](https://github.com/AndreSlavescu/flashinfer-competition/blob/5f827c364cdf3469a0af293e8a32ebe0ca79534f/scripts/bench_synthetic_service.py#L211-L235) · [Comparison](https://github.com/AndreSlavescu/flashinfer-competition/blob/5f827c364cdf3469a0af293e8a32ebe0ca79534f/scripts/bench_synthetic_service.py#L434-L511)
- **Full correctness and performance evaluation.** The Modal runner delegates to FlashInfer Bench on the competition trace workloads. The timed configuration uses 3 warmups, 100 iterations, and 5 trials. It reports per-workload status, maximum absolute and relative error when available, candidate and reference latency, and speedup. The summary takes the arithmetic mean of reported speedups. [Benchmark configuration and results](https://github.com/AndreSlavescu/flashinfer-competition/blob/5f827c364cdf3469a0af293e8a32ebe0ca79534f/scripts/bench.py#L165-L230) · [Speedup summary](https://github.com/AndreSlavescu/flashinfer-competition/blob/5f827c364cdf3469a0af293e8a32ebe0ca79534f/scripts/bench.py#L536-L555)

## GPU kernel implementations

**[gn-kernels](./gn-kernels/)** packages hand-written Triton, CuTe DSL, and CUDA kernels with implementation tests and microbenchmarks. The checkout is a source of working kernel examples. [Upstream snapshot](https://github.com/gau-nernst/gn-kernels/tree/c065ac750cc9429bba67959f5ebb171841645137)

- **Correctness.** Tests compare outputs on sampled inputs against PyTorch operations or explicit references using `torch.testing.assert_close`. Most use PyTorch's dtype-specific tolerances. Triton attention tests compare fixed- and variable-length outputs to PyTorch scaled dot-product attention with `rtol=0.016` and `atol=0.001`. CuTe DSL matmul tests use references appropriate to each format, including PyTorch integer or scaled matmul and matmul on explicitly dequantized operands. The matmul benchmark checks outputs only for entries supplied with a reference. The attention benchmark checks each timed implementation, using dequantized Q and K for the INT8 and FP8 reference calculations. [Attention tests](https://github.com/gau-nernst/gn-kernels/blob/c065ac750cc9429bba67959f5ebb171841645137/tests/test_triton_attn.py#L10-L82) · [CuTe DSL tests](https://github.com/gau-nernst/gn-kernels/blob/c065ac750cc9429bba67959f5ebb171841645137/tests/test_cutedsl_sm120.py#L20-L129) · [Matmul checks](https://github.com/gau-nernst/gn-kernels/blob/c065ac750cc9429bba67959f5ebb171841645137/benchmark_mm.py#L63-L114) · [Attention checks](https://github.com/gau-nernst/gn-kernels/blob/c065ac750cc9429bba67959f5ebb171841645137/benchmark_attn.py#L94-L138)
- **Performance.** The main scripts time repeated calls with Triton's `do_bench` and use the median latency. Matmul counts `2 × M × N × K` operations and reports throughput for square sizes 1024, 2048, and 4096 by default. Attention reports latency and estimated throughput from the two attention matmuls, with inputs already quantized for the low-precision runs. The separate speed-of-light sweep reports the best measured matmul throughput by dtype across larger square sizes and the best device-copy bandwidth. [Matmul benchmark](https://github.com/gau-nernst/gn-kernels/blob/c065ac750cc9429bba67959f5ebb171841645137/benchmark_mm.py#L30-L72) · [Attention benchmark](https://github.com/gau-nernst/gn-kernels/blob/c065ac750cc9429bba67959f5ebb171841645137/benchmark_attn.py#L65-L109) · [Speed-of-light sweep](https://github.com/gau-nernst/gn-kernels/blob/c065ac750cc9429bba67959f5ebb171841645137/benchmark_sol.py#L15-L78)

## Check out and update

Initialize the pinned versions from the project root:

```sh
git submodule update --init -- wiki/codebases
```

To run upstream tasks that require robust-kbench's CUTLASS checkout, initialize that nested submodule with `git -C wiki/codebases/robust-kbench submodule update --init third_party/cutlass`.

To review a new upstream version, advance one checkout to the branch configured in [.gitmodules](../../.gitmodules), inspect the commits, and stage the new pin:

```sh
git submodule update --remote wiki/codebases/KernelBench
git diff --submodule=log -- wiki/codebases/KernelBench
git add wiki/codebases/KernelBench
```

Each project commit records the exact submodule revisions. For experiment reproduction, record the specific revision used in the run.
