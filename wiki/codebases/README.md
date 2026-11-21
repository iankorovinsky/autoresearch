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
