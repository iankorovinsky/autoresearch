---
title: "KernelFoundry: Hardware-aware evolutionary GPU kernel optimization"
source_url: "https://arxiv.org/abs/2603.12440"
full_text_url: "https://arxiv.org/html/2603.12440v2"
version: "2603.12440v2"
authors: ["Wiedemann, Nina", "Leboutet, Quentin", "Paulitsch, Michael", "Wofk, Diana", "Ummenhofer, Benjamin"]
retrieved_date: "2026-11-21"
conversion: "HTML to Markdown with TeX math"
pdf: "30-kernelfoundry.pdf"
original_html: "sources/30-kernelfoundry.html"
---

# KernelFoundry: Hardware-aware evolutionary GPU kernel optimization

Nina Wiedemann
Affiliation: Intel Corporation * Equal contribution
Correspondence to: [nina.wiedemann@intel.com](mailto:nina.wiedemann@intel.com)

Quentin Leboutet
Affiliation: Intel Corporation * Equal contribution

Michael Paulitsch
Affiliation: Intel Corporation * Equal contribution

Diana Wofk
Affiliation: Intel Corporation * Equal contribution

Benjamin Ummenhofer
Affiliation: Intel Corporation * Equal contribution

###### Abstract

GPU kernel optimization challenges LLMs beyond standard coding tasks, as it requires an understanding of hardware architecture, parallel computing optimization strategies, and profiling outputs. However, most existing approaches leveraging LLMs for kernel generation apply standard prompting and feedback loops, considering hardware only through profiling feedback. We introduce KernelFoundry, an evolutionary framework that efficiently explores the space of GPU kernels through (1) MAP-Elites quality-diversity search with kernel-specific behavioral dimensions to sustain exploration; (2) meta-prompt evolution that co-evolves prompts with kernels to uncover task-specific optimization strategies, and (3) a template-based parameter optimization approach to tune kernels to inputs and hardware. We evaluate this framework on KernelBench, robust-kbench and custom tasks, generating SYCL kernels as a cross-platform GPU programming paradigm, and CUDA kernels for comparison to prior work. Our approach consistently outperforms the baseline methods and achieves an average speedup of 2.3 on KernelBench for SYCL. Moreover, KernelFoundry is implemented as a distributed framework with remote access to diverse hardware, allowing quick benchmarking and featuring a flexible user input layer to support kernel generation for a wide range of real use cases beyond benchmarking.

###### Keywords:

GPU Kernel Optimization, Large Language Models, Evolutionary Algorithms, Quality-Diversity, SYCL, CUDA, Triton

## 1 Introduction

Writing high-performance GPU kernels requires deep expertise in hardware architectures, memory hierarchies, and parallel programming paradigms: knowledge that remains scarce even among experienced engineers ([Li et al., 2025c](https://arxiv.org/html/2603.12440v2#bib.bib28)). Yet, efficient GPU implementations are often decisive for a method’s success and adoption, as illustrated by FlashAttention ([Dao et al., 2022](https://arxiv.org/html/2603.12440v2#bib.bib56)), which was instrumental to the practical scalability of Transformers.
Following the release of KernelBench ([Ouyang et al., 2025a](https://arxiv.org/html/2603.12440v2#bib.bib2)), a growing body of work has explored automating GPU kernel development using LLMs. With few exceptions based on model fine-tuning ([Li et al., 2025c](https://arxiv.org/html/2603.12440v2#bib.bib28); [Baronio et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib27)), most approaches rely on an iterative generate$\to$verify$\to$measure loop: kernels proposed by an LLM are compiled, validated, benchmarked, and the resulting feedback is used to steer subsequent generations. Variants of this paradigm have been demonstrated across CUDA ([Chen et al., 2025b](https://arxiv.org/html/2603.12440v2#bib.bib10); [Zhang et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib11); [Lange et al., 2025b](https://arxiv.org/html/2603.12440v2#bib.bib7)), Triton ([Wang et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib20); [Li et al., 2025a](https://arxiv.org/html/2603.12440v2#bib.bib13)), and Metal ([Sereda et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib17)), often augmented with profiling tools ([Zhang et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib11); [Chen et al., 2025b](https://arxiv.org/html/2603.12440v2#bib.bib10); [Lange et al., 2025b](https://arxiv.org/html/2603.12440v2#bib.bib7); [Sereda et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib17)), multi-agent architectures ([Wei et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib9); [Zhang et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib11); [Lei et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib25)), or evolutionary search techniques ([Liao et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib14); [Lange et al., 2025a](https://arxiv.org/html/2603.12440v2#bib.bib8); [Yan et al., 2026](https://arxiv.org/html/2603.12440v2#bib.bib1)).
Despite known shortcomings in KernelBench that necessitate careful interpretation of reported speedups ([Lange et al., 2025b](https://arxiv.org/html/2603.12440v2#bib.bib7)),
these studies collectively demonstrate that LLMs can produce correct and performant kernels when guided by execution feedback.
However, existing approaches rely on general code-generation strategies that fail to address the unique challenges of kernel optimization. Unlike standard programming tasks, kernel generation demands (1) knowledge of memory access patterns and an understanding of how parallelism maps to specific architectures, and (2) a systematic exploration of optimization strategies. Furthermore, iterative prompting and kernel validation loops suffer from mode collapse – iterative refinement behaves as greedy local search, with LLMs repeatedly proposing variants close to recent successes, leading to premature convergence – and context degradation – as optimization histories grow, failed attempts dominate the prompt context.

To overcome these limitations, we propose KernelFoundry, an evolutionary framework that maintains a diverse archive of kernel implementations and systematically explores optimization techniques.
To counteract mode collapse, KernelFoundry adapts a MAP-Elites evolutionary strategy ([Mouret and Clune, 2015](https://arxiv.org/html/2603.12440v2#bib.bib47)). Rather than defining generic behavioral descriptors such as code length ([Lehman et al., 2023](https://arxiv.org/html/2603.12440v2#bib.bib22)), kernel implementations are indexed along well-defined, domain-specific dimensions addressing hardware system architecture, such as memory access patterns or parallelism strategies.
To mitigate context degradation, we leverage meta-prompt evolution, allowing prompts to co-evolve with kernels and uncover task-specific optimization strategies.
Finally, we propose to repurpose templated kernels to allow the LLM to tune hardware-dependent values like tile and block sizes.

We evaluate KernelFoundry on KernelBench, robust-kbench, and custom benchmark suites that span single-kernel operators, fusion patterns, and full model architectures.
Unlike prior work that targets NVIDIA’s proprietary ecosystem almost exclusively, we additionally use SYCL, an open-standard C++ abstraction that is vendor-agnostic and more expressive than other cross-platform GPU programming frameworks such as Triton.
Furthermore, our framework is not limited to standardized tasks or benchmarks. We introduce a custom input format that allows users to optimize kernels based on arbitrary instructions (not restricted to PyTorch code) and to employ complex test frameworks, provided they are compatible with pytest. As a case study, we showcase the optimization of a specific operation within a LLama3 model, illustrating the practical utility of our pipeline for real-world, model-level tasks.

#### Conflict of Interest Disclosure.

The authors declare no financial conflicts of interest related to this work.

## 2 Related Work

#### Traditional Kernel Optimization.

High-performance computing has long relied on separating algorithm specification from optimization schedules, as exemplified in Halide ([Ragan-Kelley et al., 2013](https://arxiv.org/html/2603.12440v2#bib.bib36)) or TVM ([Chen et al., 2018a](https://arxiv.org/html/2603.12440v2#bib.bib37)), whereas lower-level models like CUDA and SYCL require both to be handled together in code. Triton ([Tillet et al., 2019](https://arxiv.org/html/2603.12440v2#bib.bib35)) offers a middle ground with a Python-based DSL that simplifies custom GPU kernel development but still exposes scheduling control.
Vendor libraries (cuDNN ([Chetlur et al., 2014](https://arxiv.org/html/2603.12440v2#bib.bib43)), CUTLASS ([NVIDIA Corporation, 2017](https://arxiv.org/html/2603.12440v2#bib.bib44)), oneDNN ([Intel Corporation, 2016](https://arxiv.org/html/2603.12440v2#bib.bib45))) deliver peak performance but lack flexibility for novel operators. Classical autotuners like ATLAS ([Whaley et al., 2001](https://arxiv.org/html/2603.12440v2#bib.bib38)) and FFTW ([Frigo and Johnson, 1998](https://arxiv.org/html/2603.12440v2#bib.bib39)) demonstrate that systematic search can match hand-tuned code. To avoid exhaustively evaluating every schedule, systems like AutoTVM ([Chen et al., 2018b](https://arxiv.org/html/2603.12440v2#bib.bib40)), AnsorAnsor ([Zheng et al., 2020](https://arxiv.org/html/2603.12440v2#bib.bib41)), and Ithemal ([Mendis et al., 2019](https://arxiv.org/html/2603.12440v2#bib.bib42)) use learned cost models to predict performance, but these require extensive training data and struggle to generalize across hardware. Despite these tools, algorithmic kernel development remained a highly manual and expertise-driven process.

#### LLM-Based Kernel Generation.

Large language models have sparked intense interest in automated kernel synthesis, yet generating correct GPU code remains challenging, and generating fast code is harder still. Early studies ([Valero-Lara et al., 2023](https://arxiv.org/html/2603.12440v2#bib.bib32); [Godoy et al., 2023](https://arxiv.org/html/2603.12440v2#bib.bib33)) evaluated Llama-2 and Codex on generating HPC kernels. The introduction of KernelBench ([Ouyang et al., 2025a](https://arxiv.org/html/2603.12440v2#bib.bib2)) established the de facto benchmark, comprising 250 tasks spanning single operators, fusion patterns, and complete architectures. Results showed that even frontier models achieve correctness on merely ${\sim}20\%$ of problems with substantial gaps to cuBLAS/cuDNN baselines. Recent work has also highlighted validation challenges: spurious speedups of up to 50x can arise from incorrect kernels that pass test cases ([Lange et al., 2025b](https://arxiv.org/html/2603.12440v2#bib.bib7); [Zhu et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib31)), motivating more rigorous evaluation protocols.
Despite these challenges, KernelBench has inspired a plethora of follow-up approaches (see Table [5](https://arxiv.org/html/2603.12440v2#A1.T5) for a comprehensive overview).
The dominant paradigm is prompting with validation loop: correctness and profiling outputs guide successive refinements of
CUDA ([Chen et al., 2025b](https://arxiv.org/html/2603.12440v2#bib.bib10); [Zhang et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib11); [Lange et al., 2025b](https://arxiv.org/html/2603.12440v2#bib.bib7)), Triton ([Liao et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib14); [Li et al., 2025a](https://arxiv.org/html/2603.12440v2#bib.bib13)), and Metal ([Sereda et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib17)) kernels. PEAK ([Tariq et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib21)) expresses optimization strategies as natural language transformations, achieving up to 95% of cuBLAS performance on MatMul across CUDA, HIP, and HLSL. Multi-agent architectures decompose the task into specialized roles: Astra ([Wei et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib9)) separates planning, coding, testing, and profiling agents; STARK ([Dong et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib23)) uses plan-code-debug agents; QiMeng ([Zhu et al., 2026](https://arxiv.org/html/2603.12440v2#bib.bib24)) employs macro/micro coding agents; and PRAGMA ([Lei et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib25)) coordinates coder, verifier, and conductor roles. SwizzlePerf ([Tschand et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib19)) stands out by injecting detailed microarchitecture knowledge (bank conflicts, warp operations, coalescing) into prompts, demonstrating the value of hardware awareness. Despite their sophistication, these approaches share a common limitation: they explore the kernel design space implicitly through generic code generation, without mechanisms to preserve diversity across optimization trajectories.

An orthogonal direction is finetuning LLMs for kernel generation. CUDA-L1 ([Li et al., 2025c](https://arxiv.org/html/2603.12440v2#bib.bib28)) combines supervised finetuning (SFT) with “contrastive RL”; CUDA-L2 ([Su et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib29)) adds NCU profiler metrics and multi-stage RL, exceeding cuBLASLt by 11% on HGEMM, TritonRL ([Woo et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib30)) presents good results with SFT and GRPO for Triton, and Kevin ([Baronio et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib27)) uses multi-turn RL to refine kernels across dialogue turns. These methods achieve promising results but often cannot match the performance of the latest closed-source LLMs.

Existing LLM kernel work targets CUDA almost exclusively. MultiKernelBench ([Wen et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib3)) addresses this with benchmarks spanning NVIDIA GPUs, Huawei NPUs, and Google TPUs, and NPUEval ([Kalade and Schelle, 2025](https://arxiv.org/html/2603.12440v2#bib.bib5)) provides AMD NPU benchmarks. Translation systems like BabelTower ([Wen et al., 2022](https://arxiv.org/html/2603.12440v2#bib.bib57)) and CodeRosetta ([TehraniJamsaz et al., 2024](https://arxiv.org/html/2603.12440v2#bib.bib58)) convert between CUDA and HIP/OpenCL but preserve rather than optimize performance. Our work contributes to this direction by targeting SYCL ([Keryell et al., 2015](https://arxiv.org/html/2603.12440v2#bib.bib34)) and hence enabling portability across Intel, NVIDIA, and AMD accelerators.

#### LLM-Guided Evolutionary Search.

Evolutionary algorithms have a rich history in program synthesis ([Koza, 1992](https://arxiv.org/html/2603.12440v2#bib.bib46)). Quality-diversity (QD) methods ([Pugh et al., 2016](https://arxiv.org/html/2603.12440v2#bib.bib48)) shift from single-objective optimization to maintaining diverse collections of high-performing solutions: a property particularly relevant for kernel optimization, where multiple valid implementation strategies exist. MAP-Elites ([Mouret and Clune, 2015](https://arxiv.org/html/2603.12440v2#bib.bib47)) partitions the solution space by behavioral descriptors, CMA-ME ([Fontaine et al., 2020](https://arxiv.org/html/2603.12440v2#bib.bib49)) adds continuous optimization, and PGA-MAP-Elites ([Fontaine and Nikolaidis, 2021](https://arxiv.org/html/2603.12440v2#bib.bib50)) integrates policy gradients. Recent work combines LLMs with evolutionary search: FunSearch ([Romera-Paredes et al., 2024](https://arxiv.org/html/2603.12440v2#bib.bib51)) discovers mathematical programs beyond human baselines, AlphaEvolve ([Novikov et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib52)) adds multi-objective optimization, and CodeEvolve ([Assumpção et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib54)) evolves prompts rather than code directly.
PACEvolve ([Yan et al., 2026](https://arxiv.org/html/2603.12440v2#bib.bib1)) characterizes three failure modes in LLM-assisted evolution: context pollution from accumulated failures, mode collapse from poor exploration-exploitation balance, and weak collaboration in multi-island settings. While PACEvolve addresses these through hierarchical context management for general search, kernel optimization presents a structured space where domain-specific dimensions can be directly exploited. Our approach leverages this structure: kernel-specific behavioral coordinates (memory patterns, parallelism strategies) inherently partition the solution space, preventing mode collapse by construction, while meta-prompt evolution ([Agrawal et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib55)) counteracts context degradation by enabling prompts to specialize for different optimization regions.

## 3 Method

### 3.1 System Architecture

User Input

Prompt

Constructor

Inference:

Large Language Model

Evaluation

Engine

Meta-Prompt

Evolution

MAP-Elites Archive

Selection & Evolution

Figure 1: The KernelFoundry pipeline: evolutionary kernel optimization with multi-level feedback and meta-prompt co-evolution.

We present KernelFoundry, an evolutionary framework for LLM-driven GPU kernel optimization. Figure [1](https://arxiv.org/html/2603.12440v2#S3.F1) illustrates the architecture, which comprises five tightly-coupled components. The process begins with a task specification layer that accepts kernel generation problems in a flexible format, supporting PyTorch reference implementations, high-level natural language descriptions, or existing kernels requiring optimization. The system natively supports KernelBench ([Ouyang et al., 2025a](https://arxiv.org/html/2603.12440v2#bib.bib2)) tasks as well as custom workloads with user-defined test frameworks. A prompt construction engine then assembles context-aware prompts by combining the task specification with hardware-specific optimization guidance, sampled parent programs and inspirations from the archive, gradient-derived mutation hints, and dynamically evolved prompt components (§[3.5](https://arxiv.org/html/2603.12440v2#S3.SS5)). The resulting prompt is served to an LLM inference backend, which provides a unified interface to both API-based models (OpenAI, Anthropic) and locally-hosted models via vLLM, and generates multiple candidate kernels. Candidates are processed by a compilation & evaluation pipeline that compiles to the target backend (SYCL via Intel DPC++, CUDA via nvcc, or Triton), validates numerical correctness against reference implementations, measures execution time, and optionally collects profiling data via Intel *unitrace* or NVIDIA *Nsight*. Finally, an evolutionary archive implements MAP-Elites ([Mouret and Clune, 2015](https://arxiv.org/html/2603.12440v2#bib.bib47)) by organizing kernels according to behavioral coordinates in an optimization feature space (§[3.2](https://arxiv.org/html/2603.12440v2#S3.SS2)). It retains the best-performing kernel (elite) for each occupied cell, thereby preserving diversity across qualitatively different optimization strategies.
In a similar spirit, we evolve sections of the prompt with meta-prompt evolution, which maintains a separate archive with successful guidance examples.

The evolutionary loop proceeds as follows. At each iteration, parent programs are sampled from the archive using selection strategies informed by fitness, exploration potential, and gradient estimates. The prompt constructor assembles a generation prompt, incorporating parent code, feedback from prior evaluations, and optimization guidance.
The LLM generates candidate kernels, which are then compiled, correctness-tested, and performance-measured. Candidates that run correctly are assigned behavioral coordinates via static code analysis; those improving upon existing elites are inserted into the archive. Feedback from all outcomes (including failures) informs subsequent iterations through the gradient estimator and meta-prompter.

### 3.2 Quality-Diversity Search with MAP-Elites

Standard evolutionary approaches optimize a single objective (e.g., runtime), risking convergence to local optima and failing to explore alternative solution strategies. Quality-Diversity (QD) algorithms ([Pugh et al., 2016](https://arxiv.org/html/2603.12440v2#bib.bib48)) address this limitation by maintaining diverse collections of high-performing solutions in independent behavioral cells, enabling discovery of stepping-stone solutions that facilitate escape from local optima.

#### MAP-Elites Algorithm.

MAP-Elites ([Mouret and Clune, 2015](https://arxiv.org/html/2603.12440v2#bib.bib47)) partitions the solution space into a discrete grid based on behavioral descriptors, which are low-dimensional features characterizing solution behavior independent of fitness. Each cell maintains the highest-fitness solution discovered for that behavioral region. The algorithm iterates in four phases (selection, variation, evaluation, and insertion). During selection, the system samples parent kernel(s) from the archive using a configurable strategy (uniform, fitness-proportionate, curiosity-driven, or island-based). During variation, it produces offspring via LLM-based code modification guided by the parent kernel, optimization hints, and meta-evolved prompt content. During evaluation, the offspring is compiled, validated for correctness, benchmarked for performance, and assigned behavioral coordinates via static code analysis. During insertion, the offspring replaces the incumbent elite if it improves upon the existing elite (or if the target cell is empty); otherwise, it is discarded.
This mechanism maintains diversity by construction: the archive cannot collapse because each cell evolves independently.

![Refer to caption](assets/30-kernelfoundry/figure-001.png)

Figure 2: Gradient-informed MAP-Elites for kernel optimization. The archive partitions kernels by behavioral coordinates $(d_{\text{mem}},d_{\text{algo}},d_{\text{sync}})$. Elites are shown in yellow. A Transition Tracker records parent$\to$child transitions with behavioral coordinates and fitness deltas. The Gradient Estimator combines fitness gradients ($\nabla F$), improvement-rate gradients ($\nabla R$), and exploration gradients ($\nabla E$) to produce sampling weights and natural-language mutation hints that guide subsequent generations.

#### Kernel-Specific Behavioral Descriptors.

Adapting the MAP-Elites algorithm to kernel optimization requires defining behavioral descriptors that capture meaningful optimization characteristics. With KernelFoundry, we propose a three-dimensional optimization feature space with discrete levels reflecting increasingly sophisticated optimization patterns. These coordinates are computed deterministically from generated code via static pattern matching on SYCL and CUDA constructs, ensuring reproducibility and reducing execution-time variability.

1. 1.

   Memory Access Pattern ($d_{\text{mem}}\in\{0,1,2,3\}$):

   0:
   :   Scalar, strided, or uncoalesced access

   1:
   :   Coalesced/vectorized (vec4, aligned loads)

   2:
   :   Shared/local memory with explicit tiling

   3:
   :   Multi-lev. hierarchy (SLM + reg. blocking + prefetch)
2. 2.

   Algorithmic Structure ($d_{\text{algo}}\in\{0,1,2,3\}$):

   0:
   :   Direct PyTorch translation

   1:
   :   Fused operations (single-pass over data)

   2:
   :   Reformulated algorithm (online norm., flash pattern)

   3:
   :   Novel/asymptotically improved algorithm
3. 3.

   Parallelism Coordination ($d_{\text{sync}}\in\{0,1,2,3\}$):

   0:
   :   No synchronization (embarrassingly parallel)

   1:
   :   Work-group barriers (group::barrier)

   2:
   :   Sub-group primitives (shuffles, reductions, broadc.)

   3:
   :   Global coordination (atomics, multi-pass with sync)

This three-dimensional grid yields $4^{3}=64$ behavioral cells.
The classifier employs weighted pattern matching with category-specific patterns to avoid double-counting related constructs (e.g., a kernel using *group_barrier* for SLM synchronization receives credit in $d_{\text{mem}}$ for SLM usage, not additionally in $d_{\text{sync}}$ for the same barrier).

#### Fitness Function.

The fitness function prioritizes correctness while rewarding performance:

$$
f(k)=\begin{cases}0&\text{if compilation fails},\\
0.1&\text{if compiles but incorrect},\\
0.5+0.5\cdot s_{\text{norm}}&\text{if correct},\end{cases}\vskip-5.69054pt
$$

where $s_{\text{norm}}=\min(1,\text{speedup}/\text{target})$ is the normalized speedup relative to a configurable target (default: $2\times$ over PyTorch baseline). This design ensures that correctness is a prerequisite for high fitness while providing a continuous gradient for performance optimization.

#### Selection Strategies.

We implement four parent selection strategies, with configurable mixing ratios.
Uniform selection samples randomly from occupied cells to maximize behavioral diversity. Fitness-proportionate selection weights sampling by elite fitness to exploit high-performing regions. Curiosity-driven selection prioritizes cells with high estimated improvement potential as inferred from the gradient signal. Island-based selection maintains $K$ independent sub-populations with periodic migration every $M$ generations, balancing isolated exploration with cross-pollination.

### 3.3 Gradient-Informed Evolution

While the archive structure prevents global mode collapse, individual cells may stagnate if the LLM repeatedly produces similar variants. Inspired by CMA-ME ([Fontaine et al., 2020](https://arxiv.org/html/2603.12440v2#bib.bib49)) and PGA-MAP-Elites ([Fontaine and Nikolaidis, 2021](https://arxiv.org/html/2603.12440v2#bib.bib50)), we augment MAP-Elites with gradient-like signals derived from evolutionary transition history. These gradients provide a soft steering mechanism that complements the archive’s guarantees of structural diversity.

#### Transition Tracking and gradient estimation.

We maintain a circular buffer of recent parent$\to$child transitions. Each transition record stores the parent and child behavioral coordinates $(\mathbf{b}_{p},\mathbf{b}_{c})$, the fitness delta $\Delta f=f_{c}-f_{p}$, the transition outcome (improvement when the child becomes an elite or discovers a new cell, neutral when it is competitive but does not update the archive, or regression when fitness decreases), as well as a timestamp and iteration number for temporal weighting.

From the accumulated transitions, we compute three gradient components for each occupied cell $\mathbf{b}$. The Fitness gradient $\nabla_{\mathbf{b}}F$ estimates which directions in behavioral space improve fitness. For each dimension $d$, we aggregate transitions weighted by fitness delta and direction:

$$
\nabla_{d}F\approx\frac{1}{|\mathcal{T}|}\sum_{t\in\mathcal{T}}\Delta f_{t}\cdot\text{sign}(b_{c}^{(d)}-b_{p}^{(d)})\cdot w(t)
$$

(1)

where $\mathcal{T}$ is the set of transitions originating from cell $\mathbf{b}$, and $w(t)$ applies exponential time decay to prioritize recent experience.
Improvement-rate gradient $\nabla_{\mathbf{b}}R$ estimates which directions yield higher improvement probability, independent of improvement magnitude:

$$
\nabla_{d}R\approx P(\text{imp.}\mid\Delta b_{d}>0)-P(\text{imp.}\mid\Delta b_{d}<0)
$$

(2)

Exploration gradient $\nabla_{\mathbf{b}}E$ points to empty or low-quality cells, weighted by inverse distance and improvement potential:

$$
\nabla_{\mathbf{b}}E\propto\sum_{c\in\mathcal{E}}\frac{(f_{\max}-f_{c})}{\|\mathbf{c}-\mathbf{b}\|_{1}}\cdot\frac{\mathbf{c}-\mathbf{b}}{\|\mathbf{c}-\mathbf{b}\|_{1}}
$$

(3)

where $\mathcal{E}$ is the set of empty cells and low-quality occupied cells.
The combined gradient is a weighted average, $\nabla_{\mathbf{b}}=\alpha\nabla F+\beta\nabla R+\gamma\nabla E$, with $(\alpha,\beta,\gamma)=(0.4,0.4,0.2)$.

#### Gradient-to-Prompt Translation.

Gradients inform the optimization process at two levels. For parent selection, cells with strong positive gradient magnitudes receive higher sampling probability, directing computational effort toward productive regions. For prompt construction, gradient directions are translated into natural-language mutation hints. For example, a positive gradient in $d_{\text{mem}}$ yields hints such as “consider adding shared memory tiling” or “implement register blocking for data reuse.” These hints are injected into the generation prompt, providing actionable guidance grounded in empirical transition success (see example in Appendix [E.3](https://arxiv.org/html/2603.12440v2#A5.SS3)).

### 3.4 Templated Kernels for Parameter Tuning

Beyond algorithmic transformations, kernel performance depends critically on hardware-specific parameters such as work-group dimensions, tile sizes, and unroll factors. Rather than requiring the LLM to guess optimal values, we enable explicit parameter exploration through templated kernel generation.
The prompt instructs the LLM to optionally produce a templated kernel with configurable parameters alongside a dispatch function enumerating valid parameter combinations.
Our evaluation pipeline detects templated kernels, extracts parameter configurations,
and evaluates each instantiation independently. The best-performing configuration determines the kernel’s fitness, while all results are logged to enable the LLM to refine parameter choices in subsequent iterations. This approach separates algorithmic optimization from parameter tuning, allowing the evolutionary search to explore both dimensions efficiently.

### 3.5 Meta-Prompting Against Context Degradation

As evolutionary search progresses, accumulated experiment histories can degrade generation quality as failed attempts dominate the context: a phenomenon termed context pollution ([Yan et al., 2026](https://arxiv.org/html/2603.12440v2#bib.bib1)).
Inspired by CodeEvolve ([Assumpção et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib54)) and GEPA ([Agrawal et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib55)), we introduce meta-prompt evolution: prompts are mutable and co-evolve with kernels, enabling successful optimization strategies to propagate while unsuccessful guidance is pruned.

#### Evolvable Prompt Regions.

The kernel generation prompt contains four evolvable sections, delimited by special markers. Specifically, we evolve (1) optimization philosophy, which encodes high-level principles that shape priorities (e.g., “prioritize memory bandwidth utilization before compute optimization”), (2) optimization strategies, which enumerate concrete techniques organized by category (memory, compute, parallelism) together with canonical code patterns and short explanations, (3) common pitfalls, which list anti-patterns and frequent mistakes to avoid (e.g., “avoid bank conflicts by adding padding to shared memory arrays”), and (4) analysis guidance, which provides a pre-coding reasoning scaffold that prompts the LLM to identify likely bottlenecks before generating code.

#### Meta-Prompter LLM.

This dedicated LLM (distinct from the kernel generator) analyzes generation outcomes and proposes prompt modifications. Given the current evolvable prompt sections together with the generated kernel code and evaluation metrics (correctness, speedup, and error messages), the meta-prompter first diagnoses which guidance was missing, misleading, or insufficiently specific for the observed outcome. It then prescribes targeted updates as SEARCH/REPLACE diffs restricted to the evolvable regions.
This two-LLM architecture separates analysis from generation, allowing the meta-prompter to specialize in prompt refinement without conflating it with kernel synthesis.

#### Prompt Archive and Co-Evolution.

Evolved prompts are maintained in their own archive, with fitness defined by the best kernel performance achieved using each prompt variant.
Kernels and prompts co-evolve in an interleaved schedule: every $N$ kernel generations (default $N=10$), the meta-prompter proposes prompt updates based on recent outcomes.
This co-evolutionary dynamic enables the system to discover task-specific optimization strategies that would be difficult to engineer manually.

### 3.6 Distributed Execution Framework for Scalability

Scalability is essential in adoption of kernel generation in practice. To support efficient parallel evaluation across heterogeneous hardware, KernelFoundry employs a distributed architecture with four worker types (see Appendix [C](https://arxiv.org/html/2603.12440v2#A3) [Figure 4](https://arxiv.org/html/2603.12440v2#A3.F4) for a graphical overview): (1) LLM server, (2) compilation worker, (3) execution workers and (4) database server. While parallelization for (1) is common, separation and distribution of (2) and (3) makes KernelFoundry truly scale.
The evaluation loop scales with flexible hardware allocation as only execution workers require GPUs, and users can target different backends (Intel, NVIDIA) by routing jobs to appropriate workers. Additionally, multiple workers with identical hardware can be used to speed up the pipeline to compile and evaluate kernels in parallel.

## 4 Experimental setup

The field of kernel generation faces challenges in making fair comparisons, because results depend heavily on the hardware used, the programming approach, and the abilities of the LLM. To address this, we carefully control these factors and conduct experiments across several benchmarks, using different hardware types and programming languages.

#### Programming paradigm and hardware.

KernelFoundry supports CUDA, SYCL, and Triton. Our primary focus is on SYCL for Intel hardware. SYCL offers several advantages: (1) it is an open industry standard (Khronos Group) rather than a proprietary language; (2) it provides C++ abstractions that
are more amenable to LLM generation than low-level intrinsics; (3) it enables true cross-platform portability with a single codebase. The experiments are conducted on an integrated Intel Arc 140V GPU and a discrete Intel Battlemage B580 GPU, as well as an NVIDIA RTX A6000 GPU with Ampere architecture (for CUDA comparisons). We implement a rigorous kernel performance benchmarking with several improvements over prior work (see [Appendix B](https://arxiv.org/html/2603.12440v2#A2)).

#### Benchmarks.

We evaluate on KernelBench ([Ouyang et al., 2025a](https://arxiv.org/html/2603.12440v2#bib.bib2)) as the de-facto standard for LLM kernel generation, comprising 250 tasks across three levels (L1: single operators, L2: fusion patterns, L3: full architectures). Following [Lange et al. (2025b)](https://arxiv.org/html/2603.12440v2#bib.bib7) and [METR (2025)](https://arxiv.org/html/2603.12440v2#bib.bib6), we filter out flawed tasks prone to reward hacking or with inefficient baselines, resulting in a filtered set of 111 tasks (80 L1, 31 L2). For most experiments, we use a representative subset of 40 tasks (20 L1, 20 L2), that is selected based on related work (see Appendix [D](https://arxiv.org/html/2603.12440v2#A4)).
Secondly, we utilize 12 tasks from Robust-kbench ([Lange et al., 2025b](https://arxiv.org/html/2603.12440v2#bib.bib7)). They provide 19 tasks including forward-backward operations; however, they published their results (i.e., the best generated kernel, required for comparison) only for 12 of those.

#### Baselines.

Following KernelBench, we compare to Pytorch eager execution and torch.compile. Unfortunately, most related works only publish aggregated numbers such as the average speedup per level, usually across all KernelBench problems, and not the generated kernels themselves. Since some KernelBench tasks are flawed and allow reward hacking, and the performance is highly hardware dependent, comparability to these results is inhibited. We thus compare to prior work that publish their best-performing kernels: robust_kbench ([Lange et al., 2025b](https://arxiv.org/html/2603.12440v2#bib.bib7)), the AI CUDA Engineer ([Lange et al., 2025a](https://arxiv.org/html/2603.12440v2#bib.bib8)) and Kernelsseum ([Ouyang et al., 2025b](https://arxiv.org/html/2603.12440v2#bib.bib16); [Ouyang et al., 2025a](https://arxiv.org/html/2603.12440v2#bib.bib2)).
Due to model deprecation and access restrictions, we are limited to using only the available subset of the LLMs used in the baselines, which presents additional challenges for our method. For comparison with robust-kbench, we utilize GPT-{o3,o4-mini,4.1}, excluding Sonnet 3.7 used in their method. For comparing to the AI CUDA engineer we could only use o3-mini (excluding DeepSeek-{v3,R1}, GPT-{o1-preview,o1-high}, and Sonnet 3.5).
Further hyperparameter settings are given in Appendix [B](https://arxiv.org/html/2603.12440v2#A2).

#### Metrics.

Following prior work, we report the correctness rate (fraction of tasks with a kernel that compiles and yields numerically correct outputs) and the fastp metric, defined as the proportion of tasks with a speedup greater than p: $\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\text{speedup}_{i}>p]$ where $N$ is the number of tasks and the speedup is the baseline time (pytorch eager if not denoted otherwise) divided by the kernel execution time.
For correctness, we apply stricter criteria than prior work since we noticed that the precision used by KernelBench is very low, in particular the absolute precision of $10^{-2}$, allowing erroneous kernels to pass in cases of small output values which commonly appear with AI operations. We thus rely on the relative precision $\nu=\frac{|y-\hat{y}|}{|y|+\epsilon}$, where $y$ is the expected output, $\hat{y}$ is the kernel output, and $\epsilon$ is a small value preventing division by zero. However, due to hardware imprecision, errors should be allowed in a small fraction of cases. Thus, we define a kernel as correct if $\nu<0.01$ in 99% of the output values. As a second measure for kernel correctness, we propose the cosine similarity of the flattened output tensors as a measure of their angular divergence.

## 5 Results

We first evaluate KernelFoundry’s ability to generate CUDA kernels in comparison to prior work, and then demonstrate its versatility by generating SYCL kernels. Furthermore, we showcase its adaptability to custom tasks and analyze hardware-specific aspects.

### 5.1 Baseline comparison

[Table 1](https://arxiv.org/html/2603.12440v2#S5.T1)compares our method to the kernels published by “AI CUDA Engineer” and the robust-kbench method on the respective task set for which the baseline kernels were available. We reproduce the speedup of these kernels on our NVIDIA hardware for fair comparison; original results are listed for reference but are not directly comparable due to hardware-specific PyTorch execution times. We further include the results originally published within KernelBench (“Kernelsseum”). [Table 1](https://arxiv.org/html/2603.12440v2#S5.T1) shows that the kernels generated with our approach consistently outperform the baselines, despite using only a subset of the LLMs. Specifically, our approach leads to an average speedup of 1.24 on the KernelBench subset L1 and 2.1 on L2, compared to 1.01 and 1.61 for the kernels published by [Lange et al. (2025a)](https://arxiv.org/html/2603.12440v2#bib.bib8). On the 12 robust-kbench tasks which include backward operations, our approach also results in a 36% higher speedup compared to the best kernels published by [Lange et al. (2025b)](https://arxiv.org/html/2603.12440v2#bib.bib7).
While for many tasks the general evolutionary selection procedure already finds the suitable parameters (e.g. tile size), our parameter optimization, applied only for 2 iterations (best@8), can push the performance in some cases, leading to a notable improvement. For results on a per-task level, see [Appendix F](https://arxiv.org/html/2603.12440v2#A6). The cost per kernel is around $5.7 which is comparable to related work (see Appendix [I](https://arxiv.org/html/2603.12440v2#A9)).

<table>
<tr>
<td>Task set $n=\text{count}$</td>
<td>Method</td>
<td>LLMs</td>
<td>Correct rate</td>
<td>fast<sub>1</sub></td>
<td>fast<sub>2</sub></td>
<td>Avg. speedup</td>
<td>Geom. speedup</td>
</tr>
<tr>
<td rowspan="5">KernelBench repr. set L1 $n=20$</td>
<td>Kernelsseum*</td>
<td>GPT-{o1,4o}, DeepSeek-Coder, Sonnet-3.5, LLama-3.1-405B</td>
<td>0.65</td>
<td>15 %</td>
<td>0 %</td>
<td>0.765</td>
<td>0.600</td>
</tr>
<tr>
<td>AI CUDA Engineer original<sup>†</sup></td>
<td>DeepSeek-{v3,R1}, GPT-{o1-preview,o1-high,o3-mini}, Sonnet-3.5</td>
<td>1.0</td>
<td>70 %</td>
<td>20 %</td>
<td>1.422</td>
<td>1.222</td>
</tr>
<tr>
<td>AI CUDA Engineer re-eval</td>
<td>DeepSeek-{v3,R1}, GPT-{o1-preview,o1-high,o3-mini}, Sonnet-3.5</td>
<td>1.0</td>
<td>40 %</td>
<td>50 %</td>
<td>1.005</td>
<td>0.909</td>
</tr>
<tr>
<td>Ours</td>
<td>o3-mini</td>
<td>1.0</td>
<td>55 %</td>
<td>10 %</td>
<td>1.204</td>
<td>1.092</td>
</tr>
<tr>
<td>Ours + parameter optim.</td>
<td>o3-mini</td>
<td>1.0</td>
<td>55 %</td>
<td>15 %</td>
<td>1.241</td>
<td>1.124</td>
</tr>
<tr>
<td rowspan="5">KernelBench repr. set L2 $n=20$</td>
<td>Kernelsseum*</td>
<td>GPT-{o1,4o}, DeepSeek-Coder, Sonnet-3.5</td>
<td>0.65</td>
<td>10 %</td>
<td>0 %</td>
<td>0.874</td>
<td>0.854</td>
</tr>
<tr>
<td>AI CUDA Engineer original<sup>†</sup></td>
<td>DeepSeek-{v3,R1}, GPT-{o1-preview,o1-high,o3-mini}, Sonnet-3.5</td>
<td>1.0</td>
<td>100%</td>
<td>10 %</td>
<td>1.589</td>
<td>1.524</td>
</tr>
<tr>
<td>AI CUDA Engineer re-eval</td>
<td>DeepSeek-{v3,R1}, GPT-{o1-preview,o1-high,o3-mini}, Sonnet-3.5</td>
<td>1.0</td>
<td>85 %</td>
<td>25 %</td>
<td>1.606</td>
<td>1.515</td>
</tr>
<tr>
<td>Ours</td>
<td>o3-mini</td>
<td>1.0</td>
<td>90 %</td>
<td>40 %</td>
<td>2.051</td>
<td>1.680</td>
</tr>
<tr>
<td>Ours + parameter optim.</td>
<td>o3-mini</td>
<td>1.0</td>
<td>90 %</td>
<td>45 %</td>
<td>2.104</td>
<td>1.730</td>
</tr>
<tr>
<td rowspan="4">Robust-kbench $n=12$</td>
<td>Robust-kbench original<sup>†</sup></td>
<td>GPT-{o3,o4-mini,4.1}, Sonnet-3.7</td>
<td>1.0</td>
<td>92 %</td>
<td>50 %</td>
<td>15.622</td>
<td>2.591</td>
</tr>
<tr>
<td>Robust-kbench re-eval</td>
<td>GPT-{o3,o4-mini,4.1}, Sonnet-3.7</td>
<td>1.0</td>
<td>92 %</td>
<td>58 %</td>
<td>8.865</td>
<td>3.499</td>
</tr>
<tr>
<td>Ours</td>
<td>GPT-{o3,o4-mini,4.1}</td>
<td>1.0</td>
<td>100 %</td>
<td>67 %</td>
<td>12.107</td>
<td>5.295</td>
</tr>
<tr>
<td>Ours + parameter optim.</td>
<td>GPT-{o3,o4-mini,4.1}</td>
<td>1.0</td>
<td>100 %</td>
<td>67 %</td>
<td>12.081</td>
<td>5.225</td>
</tr>
</table>

Table 1: Comparison to baselines on CUDA kernel optimization. Best results per task set in bold. **Kernelsseum* evaluated on Nvidia L40S. †*AI CUDA Engineer original* and *Robust-kbench original* both evaluated on Nvidia H100. All others evaluated on Nvidia A6000.


<table>
<tr>
<td>Task set $n=\text{count}$</td>
<td>Method</td>
<td>Language</td>
<td>LLMs</td>
<td>Correct rate</td>
<td>fast<sub>1</sub></td>
<td>fast<sub>2</sub></td>
<td>Avg. speedup</td>
<td>Geom. speedup</td>
</tr>
<tr>
<td rowspan="2">KernelBench filtered $n=111$</td>
<td>Ours</td>
<td>SYCL</td>
<td>GPT-{4.1, 5-mini}, Sonnet-4.5</td>
<td>0.97</td>
<td>71 %</td>
<td>42 %</td>
<td>2.32</td>
<td>1.38</td>
</tr>
<tr>
<td>Robust-kbench</td>
<td>CUDA</td>
<td>GPT-{o3, o4-mini, 4.1}, Sonnet-3.7</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>1.49</td>
<td>-</td>
</tr>
<tr>
<td rowspan="7">KernelBench repr. set L2 $n=20$</td>
<td>OpenEvolve (40 iters)</td>
<td>SYCL</td>
<td>GPT-{4.1, 5-mini}, Sonnet-4.5</td>
<td>1.0</td>
<td>70 %</td>
<td>40 %</td>
<td>2.535</td>
<td>1.775</td>
</tr>
<tr>
<td>Ours (40 iters + param. optim.)</td>
<td>SYCL</td>
<td>GPT-{4.1, 5-mini}, Sonnet-4.5</td>
<td>1.0</td>
<td>80 %</td>
<td>45 %</td>
<td>2.732</td>
<td>2.119</td>
</tr>
<tr>
<td>OpenEvolve (10 iters)</td>
<td>SYCL</td>
<td>GPT-{4.1, 5-mini}, Sonnet-4.5</td>
<td>1.0</td>
<td>70 %</td>
<td>20 %</td>
<td>1.483</td>
<td>1.202</td>
</tr>
<tr>
<td>Ours (10 iters)</td>
<td>SYCL</td>
<td>GPT-{4.1, 5-mini}, Sonnet-4.5</td>
<td>1.0</td>
<td>80 %</td>
<td>35 %</td>
<td>2.059</td>
<td>1.661</td>
</tr>
<tr>
<td>Ours (20 iters)</td>
<td>SYCL</td>
<td>GPT-{4.1, 5-mini}, Sonnet-4.5</td>
<td>1.0</td>
<td>70 %</td>
<td>50 %</td>
<td>2.539</td>
<td>1.833</td>
</tr>
<tr>
<td>Ours (20 iters) w/o prompt evolution</td>
<td>SYCL</td>
<td>GPT-{4.1, 5-mini}, Sonnet-4.5</td>
<td>1.0</td>
<td>70 %</td>
<td>55 %</td>
<td>2.505</td>
<td>1.600</td>
</tr>
<tr>
<td>Ours (20 iters) w/o gradient guidance</td>
<td>SYCL</td>
<td>GPT-{4.1, 5-mini}, Sonnet-4.5</td>
<td>1.0</td>
<td>75 %</td>
<td>35 %</td>
<td>2.178</td>
<td>1.695</td>
</tr>
</table>

Table 2: Evaluating KernelFoundry on SYCL kernel generation.

### 5.2 Evaluation on SYCL kernel generation

[Table 2](https://arxiv.org/html/2603.12440v2#S5.T2)presents our results for SYCL kernel generation. For comparison, we include robust-kbench CUDA performance on the filtered KernelBench tasks, as reported in ([Lange et al., 2025b](https://arxiv.org/html/2603.12440v2#bib.bib7)). Although SYCL is less commonly used and less familiar to LLMs than CUDA, our method finds correct kernels in 97% of cases and achieves an average speedup of 2.32. We further compare to OpenEvolve, the open-source implementation of AlphaEvolve ([Novikov et al., 2025](https://arxiv.org/html/2603.12440v2#bib.bib52)), which uses an evolutionary algorithm but lacks kernel-specific optimization strategies, meta-prompting, and parameter optimization. While OpenEvolve achieves a comparable speedup to our method after 40 iterations, it requires substantially more iterations to reach this level, as indicated by the notable performance gap after 10 iterations (see [Table 2](https://arxiv.org/html/2603.12440v2#S5.T2)). Ablation studies with 20 iterations show that the performance is reduced without prompt evolution or without gradient guidance. [Figure 3](https://arxiv.org/html/2603.12440v2#S5.F3) illustrates how KernelFoundry progressively improves kernel performance. [Figure 6](https://arxiv.org/html/2603.12440v2#A8.F6) shows two cases where meta-prompting becomes increasingly valuable with prolonged training, yielding consistent improvements over 100 iterations. For reproducibility, we provide results obtained with GPT-OSS in [Appendix G](https://arxiv.org/html/2603.12440v2#A7).

![Refer to caption](assets/30-kernelfoundry/figure-002.png)

Figure 3: Improvement over iterations (cumulative best)

### 5.3 Validating hardware awareness

To analyze whether our method produces hardware-aware kernels, we propose a crossover experiment: We run KernelFoundry on two distinctly different GPUs, here Intel Arc B580 and an integrated Intel Arc 140V GPU (LNL), and then benchmark the performance of the best generated kernels on the respective other GPU. If KernelFoundry is successful in generating hardware-aware kernels, we expect scenarios where the top-performing kernel of the LNL-run outperforms the best kernel optimized on B580 when both are benchmarked on LNL,
and vice versa.
To quantify this, we introduce the hardware-speedup metric $hws$, defined as the speedup of a kernel $k^{A}$ optimized on GPU $A$ over a kernel $k_{B}$ optimized on GPU $B$: $hws(k^{A})=\frac{t_{A}(k^{B})}{t_{A}(k^{A})}$, where $t_{A}$ is the runtime of a kernel on GPU $A$. $hws_{p}$ is the percentage of kernels for which $hws>p$. We again use the representative subset of KernelBench L2 (20 tasks) and report average $hws$ for all kernels and $hws_{p}$ for $p$ equal to 1.0 and 1.5, i.e. the relative amount of kernels being quicker and 1.5 times quicker, respectively.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Kernels | $hws_{1}$ | $hws_{1.5}$ | avg. hws | geom. hws |
| LNL-optimized $k^{LNL}$ | 70 $\%$ | 20 $\%$ | 1.537 | 1.297 |
| BMG-optimized $k^{B580}$ | 70 $\%$ | 15 $\%$ | 1.109 | 1.038 |

Table 3: Evaluating hardware awareness. Kernels optimized for the target GPU outperform kernels optimized on another GPU.

As shown in [Table 3](https://arxiv.org/html/2603.12440v2#S5.T3), 70% of kernels are faster than their counterpart optimized on another GPU, with average speed-up of 1.537 for LNL-optimized kernels and 1.109 for BMG-optimized kernels. See Appendix [F.5](https://arxiv.org/html/2603.12440v2#A6.SS5) for per-task results.

### 5.4 Beyond Pytorch Eager - comparison to oneDNN

KernelBench and follow-ups usually just use Pytorch Eager and torch.compile as baselines, which is a good common ground but adds additional factors, like CPU overhead for routing to specific kernels of a backend library or the effectiveness of the compiled graph.
To eliminate these factors and demonstrate the flexibility of our framework beyond the KernelBench format, we directly compare to the high performance kernel library oneDNN.
We benchmark our framework on the operations shown in [Table 4](https://arxiv.org/html/2603.12440v2#S5.T4) implementing them with the C++ API of oneDNN and use operator fusion where possible.
For the *concat(x, layernorm(x))* operation we use an initial implementation as starting point using our framework for kernel optimization. For the softmax operation we provide additional high level user guidance to implement an optimization idea that reduces the load on special function units inspired by Flash Attention 4.

[Table 4](https://arxiv.org/html/2603.12440v2#S5.T4)shows that although the kernels in oneDNN are highly optimized, oftentimes implemented directly in assembly language, the generated SYCL kernels are competitive and provide a speedup in three cases.

|  |  |  |  |
| --- | --- | --- | --- |
| Operation | Initial impl. | User instructions | Speedup |
| concat(x, layer_norm(x)) | X |  | 1.79 |
| Matmul with relu post-op |  |  | 0.35 |
| MaxPool + Linear |  |  | 0.72 |
| Sum Reduction |  |  | 1.10 |
| Softmax |  | X | 1.70 |

Table 4: Speedup compared to the oneDNN C++ implementation

### 5.5 Case study: Accelerating an operation of Llama 3

A common use case for kernel optimization is accelerating LMs on given hardware, often by allowing reduced precision as long as model performance is unaffected. As a case study, we target an operation in the Llama 3.2 1B model, the rotary positional embedding function, which presents a bottleneck on Intel hardware. We define a custom task (see [Appendix C](https://arxiv.org/html/2603.12440v2#A3)) with the PyTorch implementation (*apply_rotary_pos_emb*, combining unsqueeze + rotate-half) as the reference. Correctness is tested by comparing our kernel’s output to the reference output, and by verifying that a full Llama3 model pass with our kernel yields identical results on a simple query. KernelFoundry discovers a correct kernel in just two iterations and achieves a 7.9× speedup within ten iterations. This accelerates the rotary embedding operation, resulting in an 8% reduction in total forward pass time (from 0.413 s to 0.38 s) on a B580 Intel GPU.

## 6 Conclusion

We presented KernelFoundry, an evolutionary framework for GPU kernel optimization with LLMs.
We combine kernel-specific quality-diversity search, meta-prompting and parameter optimization to generate high-performing CUDA and SYCL kernels, demonstrating cross-platform applicability and outperforming SOTA methods.
Our experiments highlight the need for open, reproducible benchmarks and kernel databases to advance the field.

Future work includes templated kernel generation for speedups across input ranges and tensor shapes, improved formal verification to eliminate reward hacking, and deeper hardware specialization. Despite the success of prompting-based frameworks, it remains a promising avenue to finetune models, in particular through RL with verifiable rewards. As LLMs and hardware evolve, frameworks like KernelFoundry will be essential for automated, robust kernel optimization – enabling rapid deployment of new language models and fast adaptation of kernels to new hardware platforms.

## Impact Statement

This work advances Machine Learning and High-Performance Computing through automated kernel optimization. By enabling efficient cross-platform GPU kernel generation, our work aims to democratize access to high-performance computing across diverse hardware ecosystems and improve sustainability through optimized energy efficiency.
While automated code generation raises questions about code quality and security, our framework includes rigorous correctness validation to mitigate these concerns. We do not foresee specific negative societal consequences that must be highlighted beyond those common to advances in AI-assisted programming.

## Data and code availability

Code is available at <https://github.com/isl-org/kernelfoundry>.

## References

- Agrawal et al. (2025)
  L. A. Agrawal, S. Tan, D. Soylu, N. Ziems, R. Khare, K. Opsahl-Ong, A. Singhvi, H. Shandilya, M. J. Ryan, M. Jiang, C. Potts, K. Sen, A. G. Dimakis, I. Stoica, D. Klein, M. Zaharia, and O. Khattab
  GEPA: reflective prompt evolution can outperform reinforcement learning.
  External Links: 2507.19457
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px3.p1.1),
  [§3.5](https://arxiv.org/html/2603.12440v2#S3.SS5.p1.1).
- Assumpção et al. (2025)
  H. Assumpção, D. Ferreira, L. Campos, and F. Murai
  CodeEvolve: an open source evolutionary coding agent for algorithm discovery and optimization.
  arXiv preprint arXiv:2510.14150.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px3.p1.1),
  [§3.5](https://arxiv.org/html/2603.12440v2#S3.SS5.p1.1).
- Baronio et al. (2025)
  C. Baronio, P. Marsella, B. Pan, S. Guo, and S. Alberti
  Kevin: multi-turn RL for generating CUDA kernels.
  The Exploration in AI Today Workshop at ICML.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.25.1),
  [§1](https://arxiv.org/html/2603.12440v2#S1.p1.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p2.1).
- Chen et al. (2025a)
  J. Chen, Q. Wu, B. Li, L. Ma, X. Si, Y. Hu, S. Yin, and J. Yang
  cuPilot: a strategy-coordinated multi-agent framework for CUDA kernel evolution.
  arXiv preprint arXiv:2512.16465.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.9.1).
- Chen et al. (2018a)
  T. Chen, T. Moreau, Z. Jiang, L. Zheng, E. Yan, H. Shen, M. Cowan, L. Wang, Y. Hu, L. Ceze, et al.
  $\{$tvm$\}$: An automated $\{$end-to-end$\}$ optimizing compiler for deep learning.
  In 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18),
  pp. 578–594.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px1.p1.1).
- Chen et al. (2018b)
  T. Chen, L. Zheng, E. Yan, Z. Jiang, T. Moreau, L. Ceze, C. Guestrin, and A. Krishnamurthy
  Learning to optimize tensor programs.
  Advances in Neural Information Processing Systems 31.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px1.p1.1).
- Chen et al. (2025b)
  W. Chen, J. Zhu, Q. Fan, Y. Ma, and A. Zou
  CUDA-LLM: LLMs can write efficient CUDA kernels.
  arXiv preprint arXiv:2506.09092.
  Cited by: [§1](https://arxiv.org/html/2603.12440v2#S1.p1.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p1.1).
- Chetlur et al. (2014)
  S. Chetlur, C. Woolley, P. Vandermersch, J. Cohen, J. Tran, B. Catanzaro, and E. Shelhamer
  Cudnn: efficient primitives for deep learning.
  arXiv preprint arXiv:1410.0759.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px1.p1.1).
- Dao et al. (2022)
  T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Ré
  FlashAttention: fast and memory-efficient exact attention with IO-awareness.
  Advances in Neural Information Processing Systems 35, pp. 16344–16359.
  Cited by: [§1](https://arxiv.org/html/2603.12440v2#S1.p1.1).
- Dong et al. (2025)
  J. Dong, Y. Yang, T. Liu, Y. Wang, F. Qi, V. Tarokh, K. Rangadurai, and S. Yang
  STARK: strategic team of agents for refining kernels.
  arXiv preprint arXiv:2510.16996.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.20.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p1.1).
- Du et al. (2025)
  J. Du, Q. Yuan, Z. Zhang, Y. Yi, J. Hu, W. Chen, Y. Zhu, Q. Zheng, W. Zou, X. Chang, et al.
  AKG kernel agent: a multi-agent framework for cross-platform kernel synthesis.
  arXiv preprint arXiv:2512.23424.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.23.1).
- Fontaine and Nikolaidis (2021)
  M. C. Fontaine and S. Nikolaidis
  Differentiable quality diversity.
  In Advances in Neural Information Processing Systems,
  Vol. 34.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px3.p1.1),
  [§3.3](https://arxiv.org/html/2603.12440v2#S3.SS3.p1.1).
- Fontaine et al. (2020)
  M. C. Fontaine, J. Togelius, S. Nikolaidis, and A. K. Hoover
  Covariance matrix adaptation for the rapid illumination of behavior space.
  In Proceedings of the Genetic and Evolutionary Computation Conference,
  pp. 94–102.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px3.p1.1),
  [§3.3](https://arxiv.org/html/2603.12440v2#S3.SS3.p1.1).
- Frigo and Johnson (1998)
  M. Frigo and S. G. Johnson
  FFTW: an adaptive software architecture for the fft.
  In Proceedings of the 1998 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP’98 (Cat. No. 98CH36181),
  Vol. 3, pp. 1381–1384.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px1.p1.1).
- Godoy et al. (2023)
  W. Godoy, P. Valero-Lara, K. Teranishi, P. Balaprakash, and J. Vetter
  Evaluation of OpenAI Codex for HPC parallel programming models kernel generation.
  In Proceedings of the 52nd International Conference on Parallel Processing Workshops,
  pp. 136–144.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p1.1).
- Intel Corporation (2016)
  Intel Corporation
  oneAPI deep neural network library (oneDNN).
  Note: <https://github.com/oneapi-src/oneDNN>
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px1.p1.1).
- Kalade and Schelle (2025)
  S. Kalade and G. Schelle
  NPUEval: optimizing NPU kernels with LLMs and open source compilers.
  arXiv preprint arXiv:2507.14403.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.6.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p3.1).
- Keryell et al. (2015)
  R. Keryell, R. Reyes, and L. Howes
  Khronos sycl for opencl: a tutorial.
  In Proceedings of the 3rd International Workshop on OpenCL,
  pp. 1–1.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p3.1).
- Koza (1992)
  J. R. Koza
  Genetic programming: on the programming of computers by means of natural selection.
   MIT Press.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px3.p1.1).
- Lange et al. (2025a)
  R. T. Lange, A. Prasad, Q. Sun, M. Faldor, Y. Tang, and D. Ha
  The ai cuda engineer: agentic cuda kernel discovery, optimization and composition.
  Technical report
   Technical report, Sakana AI, 02 2025.
  Cited by: [§B.4](https://arxiv.org/html/2603.12440v2#A2.SS4.p1.1),
  [item 3](https://arxiv.org/html/2603.12440v2#A4.I1.i3.p1.1),
  [Table 8](https://arxiv.org/html/2603.12440v2#A6.T8),
  [§1](https://arxiv.org/html/2603.12440v2#S1.p1.1),
  [§4](https://arxiv.org/html/2603.12440v2#S4.SS0.SSS0.Px3.p1.1),
  [§5.1](https://arxiv.org/html/2603.12440v2#S5.SS1.p1.1).
- Lange et al. (2025b)
  R. T. Lange, Q. Sun, A. Prasad, M. Faldor, Y. Tang, and D. Ha
  Towards robust agentic CUDA kernel benchmarking, verification, and optimization.
  arXiv preprint arXiv:2509.14279.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.8.1),
  [§B.4](https://arxiv.org/html/2603.12440v2#A2.SS4.p1.1),
  [item 2](https://arxiv.org/html/2603.12440v2#A4.I1.i2.p1.1),
  [Appendix D](https://arxiv.org/html/2603.12440v2#A4.p1.1),
  [Appendix D](https://arxiv.org/html/2603.12440v2#A4.p2.1),
  [§1](https://arxiv.org/html/2603.12440v2#S1.p1.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p1.1),
  [§4](https://arxiv.org/html/2603.12440v2#S4.SS0.SSS0.Px2.p1.1),
  [§4](https://arxiv.org/html/2603.12440v2#S4.SS0.SSS0.Px3.p1.1),
  [§5.1](https://arxiv.org/html/2603.12440v2#S5.SS1.p1.1),
  [§5.2](https://arxiv.org/html/2603.12440v2#S5.SS2.p1.1).
- Lehman et al. (2023)
  J. Lehman, J. Gordon, S. Jain, K. Ndousse, C. Yeh, and K. O. Stanley
  Evolution through large models.
  In Handbook of evolutionary machine learning,
  pp. 331–366.
  Cited by: [§1](https://arxiv.org/html/2603.12440v2#S1.p2.1).
- Lei et al. (2025)
  K. Lei, H. Yang, H. Zhang, X. You, K. Zhang, Z. Luan, Y. Liu, and D. Qian
  PRAGMA: a profiling-reasoned multi-agent framework for automatic kernel optimization.
  arXiv preprint arXiv:2511.06345.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.22.1),
  [§1](https://arxiv.org/html/2603.12440v2#S1.p1.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p1.1).
- Li et al. (2025a)
  H. Li, K. Man, P. Kanuparthy, H. Chen, W. Sun, S. Tallam, C. Zhu, K. Zhu, and Z. Qian
  TritonForge: profiling-guided framework for automated triton kernel optimization.
  arXiv preprint arXiv:2512.09196.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.13.1),
  [§1](https://arxiv.org/html/2603.12440v2#S1.p1.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p1.1).
- Li et al. (2025b)
  J. Li, S. Li, Z. Gao, Q. Shi, Y. Li, Z. Wang, J. Huang, H. Wang, J. Wang, X. Han, et al.
  TritonBench: benchmarking large language model capabilities for generating triton operators.
  In Findings of the Association for Computational Linguistics: ACL 2025,
  pp. 23053–23066.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.5.1).
- Li et al. (2025c)
  X. Li, X. Sun, A. Wang, J. Li, and C. Shum
  CUDA-L1: improving CUDA optimization via contrastive reinforcement learning.
  arXiv preprint arXiv:2507.14111.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.26.1),
  [§1](https://arxiv.org/html/2603.12440v2#S1.p1.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p2.1).
- Liao et al. (2025)
  G. Liao, H. Qin, Y. Wang, A. Golden, M. Kuchnik, Y. Yetim, J. J. Ang, C. Fu, Y. He, S. Hsia, et al.
  KernelEvolve: scaling agentic kernel coding for heterogeneous AI accelerators at meta.
  arXiv preprint arXiv:2512.23236.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.14.1),
  [§1](https://arxiv.org/html/2603.12440v2#S1.p1.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p1.1).
- Mendis et al. (2019)
  C. Mendis, A. Renda, S. Amarasinghe, and M. Carbin
  Ithemal: accurate, portable and fast basic block throughput estimation using deep neural networks.
  In International Conference on machine learning,
  pp. 4505–4515.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px1.p1.1).
- METR (2025)
  METR
  Measuring automated kernel engineering.
  External Links: [Link](https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering/)
  Cited by: [Appendix D](https://arxiv.org/html/2603.12440v2#A4.p1.1),
  [§4](https://arxiv.org/html/2603.12440v2#S4.SS0.SSS0.Px2.p1.1).
- Mouret and Clune (2015)
  J. Mouret and J. Clune
  Illuminating search spaces by mapping elites.
  arXiv preprint arXiv:1504.04909.
  Cited by: [§1](https://arxiv.org/html/2603.12440v2#S1.p2.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px3.p1.1),
  [§3.1](https://arxiv.org/html/2603.12440v2#S3.SS1.p1.1),
  [§3.2](https://arxiv.org/html/2603.12440v2#S3.SS2.SSS0.Px1.p1.1).
- Novikov et al. (2025)
  A. Novikov, N. Vũ, M. Eisenberger, E. Dupont, P. Huang, A. Z. Wagner, S. Shirobokov, B. Kozlovskii, F. J. Ruiz, A. Mehrabian, et al.
  AlphaEvolve: a coding agent for scientific and algorithmic discovery.
  arXiv preprint arXiv:2506.13131.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px3.p1.1),
  [§5.2](https://arxiv.org/html/2603.12440v2#S5.SS2.p1.1).
- NVIDIA Corporation (2017)
  NVIDIA Corporation
  CUTLASS: CUDA templates for linear algebra subroutines.
  Note: <https://github.com/NVIDIA/cutlass>
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px1.p1.1).
- Ouyang et al. (2025a)
  A. Ouyang, S. Guo, S. Arora, A. L. Zhang, W. Hu, C. Re, and A. Mirhoseini
  KernelBench: can llms write efficient gpu kernels?.
  In International Conference on Machine Learning,
  pp. 47356–47415.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.3.1),
  [§1](https://arxiv.org/html/2603.12440v2#S1.p1.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p1.1),
  [§3.1](https://arxiv.org/html/2603.12440v2#S3.SS1.p1.1),
  [§4](https://arxiv.org/html/2603.12440v2#S4.SS0.SSS0.Px2.p1.1),
  [§4](https://arxiv.org/html/2603.12440v2#S4.SS0.SSS0.Px3.p1.1).
- Ouyang et al. (2025b)
  A. Ouyang, S. Guo, S. Arora, A. L. Zhang, W. Hu, C. Ré, and A. Mirhoseini
  Kernelsseum.
  External Links: [Link](https://scalingintelligence.stanford.edu/KernelBenchLeaderboard/)
  Cited by: [§4](https://arxiv.org/html/2603.12440v2#S4.SS0.SSS0.Px3.p1.1).
- Pugh et al. (2016)
  J. K. Pugh, L. B. Soros, and K. O. Stanley
  Quality diversity: a new frontier for evolutionary computation.
  Frontiers in Robotics and AI 3, pp. 40.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px3.p1.1),
  [§3.2](https://arxiv.org/html/2603.12440v2#S3.SS2.p1.1).
- Ragan-Kelley et al. (2013)
  J. Ragan-Kelley, C. Barnes, A. Adams, S. Paris, F. Durand, and S. Amarasinghe
  Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines.
  Acm Sigplan Notices 48 (6), pp. 519–530.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px1.p1.1).
- Romera-Paredes et al. (2024)
  B. Romera-Paredes, M. Barekatain, A. Novikov, M. Balog, M. P. Kumar, E. Dupont, F. J. Ruiz, J. S. Ellenberg, P. Wang, O. Fawzi, et al.
  Mathematical discoveries from program search with large language models.
  Nature 625 (7995), pp. 468–475.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px3.p1.1).
- Sereda et al. (2025)
  T. Sereda, N. Serrino, and Z. Asgar
  Speeding up PyTorch inference on Apple devices with AI-generated Metal kernels.
  External Links: [Link](https://gimletlabs.ai/blog/ai-generated-metal-kernels)
  Cited by: [§1](https://arxiv.org/html/2603.12440v2#S1.p1.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p1.1).
- Sharma (2025)
  A. Sharma
  OpenEvolve: an open source implementation of Google DeepMind’s AlphaEvolve.
  Note: <https://github.com/codelion/openevolve>
  Cited by: [Table 9](https://arxiv.org/html/2603.12440v2#A6.T9).
- Su et al. (2025)
  S. Su, X. Sun, X. Li, A. Wang, J. Li, and C. Shum
  CUDA-l2: surpassing cublas performance for matrix multiplication through reinforcement learning.
  arXiv preprint arXiv:2512.02551.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.27.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p2.1).
- Tariq et al. (2025)
  M. U. Tariq, A. Jangda, A. Moreira, M. Musuvathi, and T. Sorensen
  PEAK: a performance engineering ai-assistant for gpu kernels powered by natural language transformations.
  arXiv preprint arXiv:2512.19018.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.16.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p1.1).
- TehraniJamsaz et al. (2024)
  A. TehraniJamsaz, A. Bhattacharjee, L. Chen, N. K. Ahmed, A. Yazdanbakhsh, and A. Jannesari
  CodeRosetta: pushing the boundaries of unsupervised code translation for parallel programming.
  Advances in Neural Information Processing Systems 37, pp. 100965–100999.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p3.1).
- Tillet et al. (2019)
  P. Tillet, H. Kung, and D. Cox
  Triton: an intermediate language and compiler for tiled neural network computations.
  In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages,
  pp. 10–19.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px1.p1.1).
- Tschand et al. (2025)
  A. Tschand, M. Awad, R. Swann, K. Ramakrishnan, J. Ma, K. Lowery, G. Dasika, and V. J. Reddi
  SwizzlePerf: hardware-aware LLMs for GPU kernel performance optimization.
  arXiv preprint arXiv:2508.20258.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.12.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p1.1).
- Valero-Lara et al. (2023)
  P. Valero-Lara, A. Huante, M. Al Lail, W. F. Godoy, K. Teranishi, P. Balaprakash, and J. S. Vetter
  Comparing Llama-2 and GPT-3 LLMs for HPC kernels generation.
  In International Workshop on Languages and Compilers for Parallel Computing,
  pp. 20–32.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p1.1).
- Wang et al. (2025)
  J. Wang, V. Joshi, S. Majumder, X. Chao, B. Ding, Z. Liu, P. P. Brahma, D. Li, Z. Liu, and E. Barsoum
  GEAK: introducing triton kernel AI agent & evaluation benchmarks.
  arXiv preprint arXiv:2507.23194.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.15.1),
  [§1](https://arxiv.org/html/2603.12440v2#S1.p1.1).
- Wang (2025)
  L. Wang
  KernelFalcon: autonomous GPU kernel generation via deep agents.
  External Links: [Link](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-%20generation-via-deep-agents/)
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.10.1).
- Wei et al. (2025)
  A. Wei, T. Sun, Y. Seenichamy, H. Song, A. Ouyang, A. Mirhoseini, K. Wang, and A. Aiken
  Astra: a multi-agent system for GPU kernel performance optimization.
  In NeurIPS 2025 Fourth Workshop on Deep Learning for Code,
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.18.1),
  [§1](https://arxiv.org/html/2603.12440v2#S1.p1.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p1.1).
- Wen et al. (2022)
  Y. Wen, Q. Guo, Q. Fu, X. Li, J. Xu, Y. Tang, Y. Zhao, X. Hu, Z. Du, L. Li, et al.
  BabelTower: learning to auto-parallelized program translation.
  In International Conference on Machine Learning,
  pp. 23685–23700.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p3.1).
- Wen et al. (2025)
  Z. Wen, Y. Zhang, Z. Li, Z. Liu, L. Xie, and T. Zhang
  MultiKernelBench: a multi-platform benchmark for kernel generation.
  arXiv preprint arXiv:2507.17773.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.4.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p3.1).
- Whaley et al. (2001)
  R. C. Whaley, A. Petitet, and J. J. Dongarra
  Automated empirical optimizations of software and the atlas project.
  Parallel computing 27 (1-2), pp. 3–35.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px1.p1.1).
- Woo et al. (2025)
  J. Woo, S. Zhu, A. Nie, Z. Jia, Y. Wang, and Y. Park
  TritonRL: training LLMs to think and code triton without cheating.
  arXiv preprint arXiv:2510.17891.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.28.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p2.1).
- Yan et al. (2026)
  M. Yan, B. Peng, B. Coleman, Z. Chen, Z. Xie, Z. He, N. Sachdeva, I. Ye, W. Wang, C. Wang, et al.
  PACEvolve: enabling long-horizon progress-aware consistent evolution.
  arXiv preprint arXiv:2601.10657.
  Cited by: [§1](https://arxiv.org/html/2603.12440v2#S1.p1.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px3.p1.1),
  [§3.5](https://arxiv.org/html/2603.12440v2#S3.SS5.p1.1).
- Zaeed et al. (2025)
  M. Zaeed, T. Z. Islam, and V. Indić
  Opal: a modular framework for optimizing performance using analytics and LLMs.
  arXiv preprint arXiv:2510.00932.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.11.1).
- Zhang et al. (2025)
  Z. Zhang, R. Wang, S. Li, Y. Luo, M. Hong, and C. Ding
  CudaForge: an agent framework with hardware feedback for CUDA kernel optimization.
  arXiv preprint arXiv:2511.01884.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.19.1),
  [§1](https://arxiv.org/html/2603.12440v2#S1.p1.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p1.1).
- Zheng et al. (2020)
  L. Zheng, C. Jia, M. Sun, Z. Wu, C. H. Yu, A. Haj-Ali, Y. Wang, J. Yang, D. Zhuo, K. Sen, et al.
  Ansor: generating $\{$high-performance$\}$ tensor programs for deep learning.
  In 14th USENIX symposium on operating systems design and implementation (OSDI 20),
  pp. 863–879.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px1.p1.1).
- Zhu et al. (2026)
  X. Zhu, S. Peng, J. Guo, Y. Chen, Q. Guo, Y. Wen, H. Qin, R. Chen, Q. Zhou, K. Gao, et al.
  Qimeng-kernel: macro-thinking micro-coding paradigm for llm-based high-performance gpu kernel generation.
  In Proceedings of the AAAI Conference on Artificial Intelligence,
  Vol. 40, pp. 29168–29176.
  Cited by: [Table 5](https://arxiv.org/html/2603.12440v2#A1.T5.2.1.21.1),
  [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p1.1).
- Zhu et al. (2025)
  Y. Zhu, T. Jin, Y. Pruksachatkun, A. Zhang, S. Liu, S. Cui, S. Kapoor, S. Longpre, K. Meng, R. Weiss, et al.
  Establishing best practices for building rigorous agentic benchmarks.
  arXiv preprint arXiv:2507.02825.
  Cited by: [§2](https://arxiv.org/html/2603.12440v2#S2.SS0.SSS0.Px2.p1.1).

## Appendix A Summary of the Related Works

Table 5: Overview of related work on LLM-based kernel generation and optimization. Methods are
grouped by approach: benchmarks, prompting-validation loops, multi-agent, and finetuning. Hardware is
GPU unless denoted otherwise.

<table>
<tr>
<td>Reference</td>
<td>Language</td>
<td>Hardware</td>
<td>Benchmark [Count]</td>
<td>Method</td>
<td>Key Features</td>
</tr>
<tr>
<td colspan="6">Benchmarks</td>
</tr>
<tr>
<td>KernelBench (Ouyang et al., 2025a)</td>
<td>CUDA / Triton</td>
<td>NVIDIA</td>
<td>KernelBench [250]</td>
<td>Repeated prompting</td>
<td>De-facto standard</td>
</tr>
<tr>
<td>MultiKernelBench (Wen et al., 2025)</td>
<td>CUDA / Pallas / AscendC</td>
<td>NVIDIA / TPU / NPU</td>
<td>KernelBench revised [285]</td>
<td>Prompting</td>
<td>Multi-platform</td>
</tr>
<tr>
<td>TritonBench (Li et al., 2025b)</td>
<td>Triton</td>
<td>NVIDIA</td>
<td>TritonBench [184]</td>
<td>Prompting</td>
<td>Triton-specific</td>
</tr>
<tr>
<td>NPUEval (Kalade and Schelle, 2025)</td>
<td>C++ AIE</td>
<td>AMD NPU</td>
<td>Custom [102]</td>
<td>Prompting + RAG</td>
<td>NPU focus</td>
</tr>
<tr>
<td colspan="6">Approaches based on inference-validation loops</td>
</tr>
<tr>
<td>Robust-kbench (Lange et al., 2025b)</td>
<td>CUDA</td>
<td>NVIDIA</td>
<td>KernelBench + Custom [269]</td>
<td>Evolve</td>
<td>Robustness focus</td>
</tr>
<tr>
<td>cuPilot (Chen et al., 2025a)</td>
<td>CUDA</td>
<td>NVIDIA</td>
<td>KernelBench L1 [100]</td>
<td>Evolve</td>
<td>Strategy coordination</td>
</tr>
<tr>
<td>KernelFalcon (Wang, 2025)</td>
<td>Triton</td>
<td>NVIDIA</td>
<td>KernelBench [250]</td>
<td>Repeated prompting</td>
<td>100% correctness</td>
</tr>
<tr>
<td>Opal (Zaeed et al., 2025)</td>
<td>CUDA / HIP</td>
<td>NVIDIA / AMD</td>
<td>Custom [8]</td>
<td>Prompting + 3 profilers</td>
<td>Modular profiling</td>
</tr>
<tr>
<td>SwizzlePerf (Tschand et al., 2025)</td>
<td>HIP</td>
<td>AMD</td>
<td>Custom [10]</td>
<td>HW-aware prompting</td>
<td>Microarch knowledge</td>
</tr>
<tr>
<td>TritonForge (Li et al., 2025a)</td>
<td>Triton</td>
<td>NVIDIA</td>
<td>TritonBench [185]</td>
<td>Profiling-guided</td>
<td>Automated profiling</td>
</tr>
<tr>
<td>KernelEvolve (Liao et al., 2025)</td>
<td>Triton / CuTE</td>
<td>NVIDIA / AMD</td>
<td>KernelBench [250]</td>
<td>Tree search</td>
<td>Meta scaling</td>
</tr>
<tr>
<td>Geak (Wang et al., 2025)</td>
<td>Triton</td>
<td>AMD</td>
<td>TritonBench-revised</td>
<td>Prompting + Reflection</td>
<td>AI agent</td>
</tr>
<tr>
<td>PEAK (Tariq et al., 2025)</td>
<td>CUDA / HIP / HLSL</td>
<td>NVIDIA / AMD / Qualcomm</td>
<td>MatMul</td>
<td>Natural transformations</td>
<td>Iterative + Modular</td>
</tr>
<tr>
<td colspan="6">Multi-Agent Approaches</td>
</tr>
<tr>
<td>Astra (Wei et al., 2025)</td>
<td>CUDA</td>
<td>NVIDIA</td>
<td>Custom [3]</td>
<td>Multi-agent</td>
<td>Plan/Code/Test/Profile</td>
</tr>
<tr>
<td>CudaForge (Zhang et al., 2025)</td>
<td>CUDA</td>
<td>NVIDIA</td>
<td>KernelBench [25]</td>
<td>Coder + Judge</td>
<td>Hardware feedback</td>
</tr>
<tr>
<td>STARK (Dong et al., 2025)</td>
<td>CUDA</td>
<td>NVIDIA</td>
<td>KernelBench [250]</td>
<td>Plan + Code + Debug</td>
<td>Strategic agents</td>
</tr>
<tr>
<td>QiMeng (Zhu et al., 2026)</td>
<td>Triton</td>
<td>NVIDIA</td>
<td>KernelBench + TritonBench</td>
<td>Macro/Micro coding</td>
<td>Two-stage thinking</td>
</tr>
<tr>
<td>PRAGMA (Lei et al., 2025)</td>
<td>Triton / C++</td>
<td>NVIDIA / Intel CPU</td>
<td>KernelBench</td>
<td>Coder + Verifier + Conductor</td>
<td>Profiling-reasoned</td>
</tr>
<tr>
<td>AKG Agent (Du et al., 2025)</td>
<td>Triton</td>
<td>NVIDIA / Ascend NPU</td>
<td>KernelBench L1 [100]</td>
<td>Coder + Designer</td>
<td>Cross-platform</td>
</tr>
<tr>
<td colspan="6">Finetuning Approaches</td>
</tr>
<tr>
<td>Kevin (Baronio et al., 2025)</td>
<td>CUDA</td>
<td>NVIDIA</td>
<td>KernelBench [100]</td>
<td>Multi-turn RL</td>
<td>Turn-level rewards</td>
</tr>
<tr>
<td>CUDA-L1 (Li et al., 2025c)</td>
<td>CUDA</td>
<td>NVIDIA</td>
<td>KernelBench [250]</td>
<td>Contrastive RL</td>
<td>3.12$\times$ speedup</td>
</tr>
<tr>
<td>CUDA-L2 (Su et al., 2025)</td>
<td>CUDA</td>
<td>NVIDIA (A100)</td>
<td>HGEMM [1000]</td>
<td>Multi-stage RL + NCU</td>
<td>Beats cuBLAS (+11%)</td>
</tr>
<tr>
<td>TritonRL (Woo et al., 2025)</td>
<td>Triton</td>
<td>NVIDIA</td>
<td>KernelBench [200]</td>
<td>SFT + GRPO</td>
<td>No reward hacking</td>
</tr>
<tr>
<td>KernelFoundry (Ours)</td>
<td>SYCL / CUDA / Triton</td>
<td>Intel / NVIDIA</td>
<td>KernelBench + Custom</td>
<td>MAP-Elites + Meta-prompt</td>
<td>QD + Parameter optim.</td>
</tr>
</table>

## Appendix B Implementation Details

### B.1 System Configuration

KernelFoundry is implemented in Python with the following core dependencies:

- •

  LLM inference: OpenAI API, Anthropic API, vLLM for local models
- •

  Compilation: Intel oneAPI DPC++ 2025.2, NVIDIA CUDA Toolkit 12.8, Triton 3.5
- •

  Profiling: Profiling Tools Interfaces for GPU (unitrace 2.3) for SYCL, NVIDIA Nsight Compute 2025.3.0.0 for CUDA
- •

  Testing: PyTorch 2.9

### B.2 Benchmarking kernel runtime

In contrast to prior work, we implement several improvements in the kernel benchmarking to reduce the variance of runtime measurements. Our implementation proceeds as follows: First, we run a fixed number of initial trials to determine the rough runtime of the kernel. This initial measurement informs the number of warmup trials and main trials, which are set based on a minimal total time rather than a fixed amount of trials (if the kernel is slow, less warmup trials and main trials are required). Furthermore, we noticed that for very fast kernels, the torch.cuda.synchronize or torch.xpu.synchronize operation has a significant overhead. We reduce this overhead by running an inner loop within the main trials, such that multiple trials are executed before each synchronize.

For all experiments, we set the minimum warmup time to 1 second, the mininum warmup iterations to 10, the inner-loop minimum time to 0.01 seconds, the minimum number of main iterations to 10 and the minimum runtime of the main performance measurement to 1s. This ensures that measured durations are comfortably larger than the timer precision and synchronization overhead while guaranteeing enough measurements to compute runtime statistics.

Furthermore, we noticed that the runtime measurement of backward operations in robust-kbench has a significant overhead due to using torch.autograd, rather than measuring the kernel execution time directly. We found a way to measure the isolated runtime of the kernel which is more stable. However, for the reference we still need to measure torch.autograd.grad. This leads to high speedups for backward operations as can be seen in [Table 7](https://arxiv.org/html/2603.12440v2#A6.T7); however, this appeared as a minor drawback to us compared to an incorrect kernel runtime measurement.

### B.3 Profiler Feedback

For correct kernels, optional profiling provides performance insights. We collect and provide the following information:

- •

  Execution time: Wall-clock runtime
- •

  Memory bandwidth: Achieved vs. theoretical peak (via unitrace or Nsight Compute)
- •

  Compute utilization: ALU occupancy, instruction throughput
- •

  Bottleneck identification: Memory-bound vs. compute-bound classification

Profiler feedback is structured into natural language summaries (e.g., “Kernel is memory-bound
at 45% of peak bandwidth. Consider shared memory tiling to improve data reuse.”).

### B.4 Hyperparameters

Table [6](https://arxiv.org/html/2603.12440v2#A2.T6) lists the default hyperparameters used in our experiments.
The number of iterations depends on the experiment: For a fair comparison to The AI CUDA Engineer ([Lange et al., 2025a](https://arxiv.org/html/2603.12440v2#bib.bib8)), we set the number of iterations to 40 and the population size to 4 to align with their experimental setup. Robust-kbench ([Lange et al., 2025b](https://arxiv.org/html/2603.12440v2#bib.bib7)) sample 8 suggestions from the LLM in each iteration and show the results over 40 iterations, so we use the same (40 x 8) as used for SYCL kernel generation ([Table 2](https://arxiv.org/html/2603.12440v2#S5.T2)). As explained in [section 4](https://arxiv.org/html/2603.12440v2#S4), we also align the LLMs to the ones in the baseline papers. For the SYCL experiment, we chose to prompt a powerful language model (Claude Sonnet 4.5) in the first iteration to avoid getting trapped in a local minimum at the beginning. After the first iteration, we use an ensemble of GPT 5 mini and GPT 4.1 (equal weights). After the 40 iterations, we run 2 iterations of parameter optimization.

<table>
<tr>
<td>Parameter</td>
<td>Value</td>
</tr>
<tr>
<td colspan="2">Evolution</td>
</tr>
<tr>
<td>Max generations</td>
<td>40 (*)</td>
</tr>
<tr>
<td>Population per generation</td>
<td>8 (*)</td>
</tr>
<tr>
<td>Selection strategy</td>
<td>Curiosity-driven</td>
</tr>
<tr>
<td>Archive dimensions</td>
<td>4</td>
</tr>
<tr>
<td>Bins per dimension</td>
<td>4</td>
</tr>
<tr>
<td colspan="2">Evaluation</td>
</tr>
<tr>
<td>Warmup iterations</td>
<td>10</td>
</tr>
<tr>
<td>Timing iterations</td>
<td>100</td>
</tr>
<tr>
<td>Target speedup</td>
<td>2.0$\times$</td>
</tr>
<tr>
<td colspan="2">Meta-prompting</td>
</tr>
<tr>
<td>Prompt update frequency</td>
<td>Every 10 generations</td>
</tr>
<tr>
<td>Max prompt mutations</td>
<td>3 per update</td>
</tr>
<tr>
<td>Prompt archive size</td>
<td>16</td>
</tr>
<tr>
<td colspan="2">LLM</td>
</tr>
<tr>
<td>Temperature</td>
<td>0.3</td>
</tr>
<tr>
<td>Max tokens</td>
<td>8000</td>
</tr>
<tr>
<td>Top-p</td>
<td>1</td>
</tr>
</table>

Table 6: Hyperparameters for KernelFoundry. (*) number of operations, population size, and LLM model vary dependent on the experiment.

## Appendix C Distributed System & Custom Task Input

Infrastructure  As shown in [Figure 4](https://arxiv.org/html/2603.12440v2#A3.F4), the KernelFoundry pipeline utilizes a lightweight main thread that interacts with four servers. The server processes can run on a single machine or on multiple remote machines in the same network.

1. 1.

   LLM Server: Hosts the generation model (Cloud services with REST-API or local vLLM instances).
2. 2.

   Compilation Workers: Executes compiler toolchains (DPC++, nvcc). Does not need a GPU.
3. 3.

   Execution Workers: Run correctness tests and benchmarks on GPU nodes. Connected via a task queue ensuring single-task-per-GPU isolation.
4. 4.

   Database Server: Stores all generated kernels, evaluation results, and evolutionary state for reproducibility and analysis.

The LLM server, compile worker, and execution worker are connected via a load balancer and queue system and allows to scale the number of workers and increase parallelism.

Custom task  Tasks are defined by a set of files with special markers, making them highly customizable and easy to integrate into existing projects.
A task is defined by a config file in YAML format containing hyperparameters; a python module with a build function and correctness and performance tests defined in the pytest framework; and a language specific file for the generated code.
Special markers are used to define sections for the reference code, optional user instructions, and optional initial kernel implementations passed to the model.

![Refer to caption](assets/30-kernelfoundry/figure-003.svg)

Figure 4: Overview of the KernelFoundry infrastructure and the custom task format

## Appendix D KernelBench task filtering

We follow [Lange et al. (2025b)](https://arxiv.org/html/2603.12440v2#bib.bib7) and [METR (2025)](https://arxiv.org/html/2603.12440v2#bib.bib6) and filter out KernelBench tasks that are flawed. [Lange et al. (2025b)](https://arxiv.org/html/2603.12440v2#bib.bib7) propose 5 criteria: (1) small output value range, (2) small output StD, (3) small output StD across any axis, (4) small impact of inputs on the output, (5) baseline inefficiencies. For our experiments, we select the same task set as ([Lange et al., 2025b](https://arxiv.org/html/2603.12440v2#bib.bib7)) to ensure full consistency.

However, it is worth noting that we generally believe that criterion (3) is too strict, as we did not observe issues with the kernels generated for such tasks, and e.g. for outputs with shape 2 along one axis the StD can naturally be low. Tasks with baseline inefficiencies (5) do not necessarily need to be removed either, as the comparison is still fair. [Lange et al. (2025b)](https://arxiv.org/html/2603.12440v2#bib.bib7) used an LLM to identify these tasks which sometimes gave wrong responses on closer inspection. Filtering according to criteria (1), (2) and (4) would retain 223 tasks across level 1, 2 and 3.

The set of representative tasks (20 L1, 20 L2, see Section [4](https://arxiv.org/html/2603.12440v2#S4.SS0.SSS0.Px2)) was selected based on the following criteria:

1. 1.

   Diversity: 20 tasks from level 1, 20 from level 2, covering all kind of operations (matmul, conv, conv transpose, activation functions, etc.)
2. 2.

   Uncompromised: We only select tasks that comply with criteria (1)-(5) by [Lange et al. (2025b)](https://arxiv.org/html/2603.12440v2#bib.bib7), i.e. tasks that are not listed as “compromised tasks” in ([Lange et al., 2025b](https://arxiv.org/html/2603.12440v2#bib.bib7)) Appendix A.2 and A.3.
3. 3.

   Availability of correct baseline kernels: For a fair comparison to ([Lange et al., 2025a](https://arxiv.org/html/2603.12440v2#bib.bib8)), we select only tasks for which they publish a kernel as part of their Huggingface dataset, and filtering only for the ones where their fastest kernel (per task) passes our correctness tests. Since our tests are more strict than the test used in ([Lange et al., 2025a](https://arxiv.org/html/2603.12440v2#bib.bib8)), 53 out of the 229 kernels failed our tests.

## Appendix E Prompts

### E.1 Main prompt

You are a $<$language$>$ programming expert specializing in GPU kernel optimization. Given a reference $<$reference-language$>$ implementation, your objective is to create a performant kernel with identical functionality. The code you generate will be pasted into an existing project. Make sure to follow the existing code structure and function signatures. The $<$language$>$ code you generate will be saved in $<$ kernel.cpp / kernel.cu $>$ and loaded using torch.utils.cpp_extension.load():

````

‘‘‘load(name=task_name, sources=[<kernel.cpp / kernel.cu>]‘‘‘
````

Later, the function will be called via load(name=task_name, ...).forward and thoroughly tested.
Examples:

Here is an example of a correct $<$language$>$ kernel for a given $<$reference-language$>$ reference:

$<$example-reference-code$>$

$<$example-kernel-code$>$
Reference code / Task:

This is the reference $<$reference-language$>$ implementation:

$<$reference-code$>$
Top performing kernel:

This is the best kernel tested so far (Runtime: $<$top-kernel-runtime$>$):

$<$top-kernel-code$>$
Last tested kernel:

Here is the last kernel we tested (Runtime: $<$last-kernel-runtime$>$):

$<$last-kernel-code$>$

Console output from running this kernel:

$<$last-kernel-log$>$
Hardware specification:

Your code will run on the following hardware:

$<$hardware-specs$>$

Please consider the hardware specifications when improving the code.
Main Instructions:

1.

Provide a functional kernel that matches the reference implementation.
2.

Use constructs to efficiently run the code on GPU.
3.

Provide the complete code in a code block.
Optimization strategies:

$<$evolvable-optimization-strategies$>$
Critical Requirements:

1.

The kernel must exactly match the reference’s functionality.
2.

The code must compile and run properly on the GPU.
3.

Do not cache or reuse previous results; ensure the code executes fully on each run.
Response Format:

Please structure your response as follows:

1.

Analysis: Summarize the issues found in the previous kernel and log. Explain your proposed changes and optimizations.
2.

Code: Provide the complete, improved $<$language$>$ code in code blocks:

````
‘‘‘
Your code here
‘‘‘
````

### E.2 Templated kernels for parameter optimization

To guide the model towards converting parameters to template parameters which can then be tuned with our pipeline, we use the following prompt and code example:

You are a $<$language$>$ programming expert specializing in GPU kernel optimization. Your task is to optimize a given $<$language$>$ kernel.
Given kernel:
Here is the $<$language$>$ kernel that we tested:
$<$Code of best generated kernel$>$
To optimize this kernel for specific hardware, please propose a templated kernel with some template parameters that can be tuned. To do so, you need to write an extension for PyTorch that implements a templated $<$language$>$ kernel and a forward function for dispatching. Select suitable parameter options by adding them as dispatch-options in the forward function.

Requirements:

•

The kernel should be templated on suitable parameters, e.g., block size, etc.
•

The forward_templated function should launch the kernel with the given parameter values.
•

The forward function should have standard arguments corresponding to the template parameters of forward_templated, and should select the correct instantiation based on the input values.
•

The code must match the given kernel in functionality.
•

The code should include a Pybind11 interface exposing forward.
Here is an example of a templated kernel in the correct format:

Listing 1: Example SYCL kernel with parameter dispatch

````C++


#include <sycl/sycl.hpp>


#include <torch/extension.h>


#include <c10/xpu/XPUStream.h>


// Kernel name struct at namespace scope


template <int BX, int BY>


struct ElementwiseMulKernel {};


template <int BLOCK_X, int BLOCK_Y>


void elementwise_mul_sycl_kernel(


 torch::Tensor A,


 torch::Tensor B,


 torch::Tensor C,


 int N,


 int M


) {


 auto a_data = A.data_ptr<float>();


 auto b_data = B.data_ptr<float>();


 auto c_data = C.data_ptr<float>();


 sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();


 sycl::range<2> global_range(


 ((N + BLOCK_Y - 1) / BLOCK_Y) * BLOCK_Y,


 ((M + BLOCK_X - 1) / BLOCK_X) * BLOCK_X


 );


 sycl::range<2> local_range(BLOCK_Y, BLOCK_X);


 q.submit([&](sycl::handler& cgh) {


 cgh.parallel_for<ElementwiseMulKernel<BLOCK_X, BLOCK_Y>>(


 sycl::nd_range<2>(global_range, local_range),


 [=](sycl::nd_item<2> item) {


 int row = item.get_global_id(0);


 int col = item.get_global_id(1);


 if (row < N && col < M) {


 int idx = row * M + col;


 c_data[idx] = a_data[idx] * b_data[idx];


 }


 }


 );


 }).wait();


}


// 2. Templated forward function


template <int BLOCK_X, int BLOCK_Y>


torch::Tensor forward_templated(torch::Tensor A, torch::Tensor B) {


 int N = A.size(0);


 int M = A.size(1);


 auto C = torch::empty({N, M}, A.options());


 elementwise_mul_sycl_kernel<BLOCK_X, BLOCK_Y>(A, B, C, N, M);


 return C;


}


// 3. Dispatcher - must have arguments corresponding to the template parameters of forward_templated


torch::Tensor forward(torch::Tensor A, torch::Tensor B, int block_x, int block_y) {


 TORCH_CHECK(A.scalar_type() == torch::kFloat, "Only float32 supported in this example");


 TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Only 2D tensors supported");


 TORCH_CHECK(A.sizes() == B.sizes(), "Input sizes must match");


 if (block_x == 16 && block_y == 16) {


 return forward_templated<16, 16>(A, B);


 } else if (block_x == 32 && block_y == 8) {


 return forward_templated<32, 8>(A, B);


 } else if (block_x == 8 && block_y == 32) {


 return forward_templated<8, 32>(A, B);


 } else {


 TORCH_CHECK(false, "Unsupported block size combination");


 }


}


// 4. Pybind11 interface


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {


 m.def("forward", &forward, "Elementwise multiplication with block size dispatch");


}
````

### E.3 Example of optimization-aware prompting

To make optimization-aware prompting more tangible, consider the following example: A parent kernel achieved a runtime of 0.021 ms. The framework identified a positive fitness gradient along the memory dimension and injected the following natural-language hint into the next generation prompt: “Use register tiling: each thread computes an M×N output block in registers. Apply #pragma unroll to keep intermediate values in registers and avoid spilling. Combine with shared memory tiling for multi-level memory hierarchy optimization.” The child kernel acted on this hint by replacing a flat accumulator array with an explicit 2×2 register tile, restructuring the inner loop to accumulate all four overlapping convolution outputs simultaneously, and adding #pragma unroll across all inner loops. This yielded a runtime of 0.0195 ms, a  7% improvement.

## Appendix F Per-task results

### F.1 Robust-kbench comparison

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Robust-kbench original | Robust-kbench re-evaluated | Ours | Ours + parameter optim. |
| Operation |  |  |  |  |
| layernorm_forward | 165.678 | 55.116 | 55.625 | 55.625 |
| llama_ffw | 1.011 | 1.010 | 1.014 | 1.012 |
| llama_rmsnorm_forward | 3.920 | 2.911 | 3.529 | 3.517 |
| mnist_conv_relu_pool_forward | 2.790 | 3.947 | 5.068 | 5.007 |
| mnist_cross_entropy_backward | 0.958 | 10.388 | 23.905 | 23.905 |
| mnist_cross_entropy_forward | 1.933 | 0.400 | 1.434 | 1.389 |
| mnist_linear_backward | 1.200 | 9.700 | 22.000 | 22.000 |
| mnist_linear_forward | 2.517 | 1.558 | 1.669 | 1.520 |
| mnist_linear_relu_backward | 1.226 | 15.446 | 22.233 | 22.233 |
| mnist_linear_relu_forward | 2.494 | 1.345 | 2.726 | 2.675 |
| mnist_pool_backward | 1.116 | 3.354 | 4.761 | 4.761 |
| resnet_block | 2.625 | 1.205 | 1.325 | 1.328 |
| Average | 15.622 | 8.865 | 12.107 | 12.081 |

Table 7: Comparison on robust-kbench by task

### F.2 Comparison to AI CUDA Engineer

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | Level | AI CUDA Engineer original | AI CUDA Engineer re-evaluated | Ours | Ours + parameter optim. |
| Operation |  |  |  |  |  |
| 20_LeakyReLU | 1 | 1.134 | 0.621 | 0.910 | 0.955 |
| 21_Sigmoid | 1 | 1.109 | 0.554 | 0.805 | 0.805 |
| 25_Swish | 1 | 1.562 | 1.016 | 1.349 | 1.402 |
| 30_Softsign | 1 | 2.474 | 1.724 | 2.895 | 2.895 |
| 33_BatchNorm | 1 | 0.619 | 0.975 | 1.017 | 1.017 |
| 44_Average_Pooling_1D | 1 | 1.238 | 0.643 | 0.769 | 0.824 |
| 48_Mean_reduction_over_a_dimension | 1 | 1.760 | 0.567 | 0.887 | 0.887 |
| 4_Matrix_vector_multiplication_ | 1 | 0.939 | 0.897 | 0.985 | 0.990 |
| 53_Min_reduction_over_a_dimension | 1 | 2.187 | 0.981 | 1.317 | 1.317 |
| 5_Matrix_scalar_multiplication | 1 | 0.999 | 0.999 | 1.003 | 1.008 |
| 64_conv_transposed_1D | 1 | 1.021 | 0.578 | 1.017 | 1.213 |
| 67_conv_standard_1D | 1 | 1.411 | 1.199 | 1.566 | 1.576 |
| 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool | 1 | 1.047 | 1.030 | 0.315 | 0.319 |
| 76_conv_standard_1D_dilated_strided__ | 1 | 1.883 | 1.209 | 1.325 | 1.325 |
| 7_Matmul_with_small_K_dimension_ | 1 | 0.670 | 0.612 | 1.095 | 1.135 |
| 82_conv_depthwise_2D_square_input_square_kernel | 1 | 1.239 | 1.530 | 0.794 | 0.794 |
| 86_conv_depthwise_separable_2D | 1 | 0.808 | 0.574 | 0.940 | 0.961 |
| 87_conv_pointwise_2D | 1 | 0.249 | 0.499 | 1.000 | 1.000 |
| 89_cumsum | 1 | 2.214 | 1.319 | 1.824 | 2.127 |
| 99_TripletMarginLoss | 1 | 3.873 | 2.570 | 2.278 | 2.278 |
| Average | 1 | 1.422 | 1.005 | 1.204 | 1.241 |
| 16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling | 2 | 1.689 | 1.684 | 0.466 | 0.469 |
| 17_Conv2d_InstanceNorm_Divide | 2 | 1.786 | 2.036 | 1.920 | 2.022 |
| 1_Conv2D_ReLU_BiasAdd | 2 | 1.170 | 1.626 | 3.322 | 3.444 |
| 21_Conv2d_Add_Scale_Sigmoid_GroupNorm | 2 | 1.786 | 1.982 | 2.891 | 2.834 |
| 24_Conv3d_Min_Softmax | 2 | 1.001 | 1.026 | 1.013 | 1.013 |
| 32_Conv2d_Scaling_Min | 2 | 1.767 | 1.537 | 2.398 | 2.655 |
| 35_Conv2d_Subtract_HardSwish_MaxPool_Mish | 2 | 1.932 | 2.331 | 1.624 | 1.669 |
| 37_Matmul_Swish_Sum_GroupNorm | 2 | 1.499 | 1.804 | 1.256 | 1.443 |
| 46_Conv2d_Subtract_Tanh_Subtract_AvgPool | 2 | 1.393 | 2.333 | 5.560 | 5.622 |
| 47_Conv3d_Mish_Tanh | 2 | 1.086 | 1.192 | 1.361 | 1.379 |
| 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling | 2 | 1.000 | 1.019 | 3.039 | 3.099 |
| 59_Matmul_Swish_Scaling | 2 | 1.911 | 0.954 | 1.412 | 1.408 |
| 5_ConvTranspose2d_Subtract_Tanh | 2 | 1.092 | 1.153 | 0.199 | 0.200 |
| 67_Conv2d_GELU_GlobalAvgPool | 2 | 1.634 | 2.000 | 1.464 | 1.526 |
| 70_Gemm_Sigmoid_Scaling_ResidualAdd | 2 | 1.713 | 0.781 | 1.592 | 1.881 |
| 73_Conv2d_BatchNorm_Scaling | 2 | 1.024 | 0.924 | 1.739 | 1.741 |
| 82_Conv2d_Tanh_Scaling_BiasAdd_Max | 2 | 1.999 | 2.712 | 3.883 | 3.744 |
| 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp | 2 | 1.241 | 1.322 | 2.037 | 2.077 |
| 97_Matmul_BatchNorm_BiasAdd_Divide_Swish | 2 | 2.277 | 1.688 | 1.662 | 1.726 |
| 99_Matmul_GELU_Softmax | 2 | 2.778 | 2.020 | 2.183 | 2.124 |
| Average | 2 | 1.589 | 1.606 | 2.051 | 2.104 |

Table 8: Comparison to the AI CUDA Engineer ([Lange et al., 2025a](https://arxiv.org/html/2603.12440v2#bib.bib8)) by KernelBench task

### F.3 Comparison to OpenEvolve

|  |  |  |
| --- | --- | --- |
|  | Ours | OpenEvolve |
| Operation |  |  |
| 16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling | 1.786 | 1.792 |
| 17_Conv2d_InstanceNorm_Divide | 3.550 | 1.851 |
| 1_Conv2D_ReLU_BiasAdd | 2.917 | 3.295 |
| 21_Conv2d_Add_Scale_Sigmoid_GroupNorm | 3.045 | 6.612 |
| 24_Conv3d_Min_Softmax | 1.207 | 1.211 |
| 32_Conv2d_Scaling_Min | 7.871 | 10.296 |
| 35_Conv2d_Subtract_HardSwish_MaxPool_Mish | 1.972 | 1.985 |
| 37_Matmul_Swish_Sum_GroupNorm | 1.928 | 0.520 |
| 46_Conv2d_Subtract_Tanh_Subtract_AvgPool | 5.005 | 3.446 |
| 47_Conv3d_Mish_Tanh | 0.834 | 0.747 |
| 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling | 1.667 | 1.684 |
| 59_Matmul_Swish_Scaling | 0.913 | 0.674 |
| 5_ConvTranspose2d_Subtract_Tanh | 1.262 | 1.237 |
| 67_Conv2d_GELU_GlobalAvgPool | 2.498 | 4.061 |
| 70_Gemm_Sigmoid_Scaling_ResidualAdd | 0.615 | 0.767 |
| 73_Conv2d_BatchNorm_Scaling | 0.904 | 0.748 |
| 82_Conv2d_Tanh_Scaling_BiasAdd_Max | 7.955 | 2.023 |
| 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp | 3.031 | 3.043 |
| 97_Matmul_BatchNorm_BiasAdd_Divide_Swish | 1.441 | 0.471 |
| 99_Matmul_GELU_Softmax | 4.234 | 4.239 |
| Average | 2.732 | 2.535 |

Table 9: Comparison to OpenEvolve ([Sharma, 2025](https://arxiv.org/html/2603.12440v2#bib.bib53)) by KernelBench task

### F.4 Ablation study with respect to prompt evolution and gradient guidance

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W/o prompt evolution | Prompt evolution (every 10) | Prompt evolution (every 2) | W/o gradient |
| Operation |  |  |  |  |
| 16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling | 0.294 | 1.638 | 1.654 | 1.633 |
| 17_Conv2d_InstanceNorm_Divide | 3.249 | 3.293 | 1.636 | 2.866 |
| 1_Conv2D_ReLU_BiasAdd | 2.500 | 2.015 | 3.296 | 1.968 |
| 21_Conv2d_Add_Scale_Sigmoid_GroupNorm | 3.169 | 2.807 | 2.810 | 2.144 |
| 24_Conv3d_Min_Softmax | 1.504 | 1.089 | 1.098 | 1.103 |
| 32_Conv2d_Scaling_Min | 9.541 | 9.196 | 6.446 | 8.455 |
| 35_Conv2d_Subtract_HardSwish_MaxPool_Mish | 2.759 | 6.831 | 8.717 | 1.723 |
| 37_Matmul_Swish_Sum_GroupNorm | 1.452 | 0.775 | 1.727 | 1.983 |
| 46_Conv2d_Subtract_Tanh_Subtract_AvgPool | 4.747 | 3.592 | 1.774 | 1.728 |
| 47_Conv3d_Mish_Tanh | 1.368 | 0.649 | 0.525 | 0.646 |
| 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling | 2.617 | 1.221 | 2.972 | 1.565 |
| 59_Matmul_Swish_Scaling | 0.653 | 0.656 | 0.831 | 0.647 |
| 5_ConvTranspose2d_Subtract_Tanh | 0.111 | 0.729 | 1.176 | 0.617 |
| 67_Conv2d_GELU_GlobalAvgPool | 3.247 | 3.868 | 2.591 | 2.971 |
| 70_Gemm_Sigmoid_Scaling_ResidualAdd | 0.328 | 0.976 | 0.692 | 0.660 |
| 73_Conv2d_BatchNorm_Scaling | 0.696 | 1.670 | 2.057 | 1.698 |
| 82_Conv2d_Tanh_Scaling_BiasAdd_Max | 3.661 | 3.844 | 4.760 | 4.175 |
| 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp | 2.159 | 2.809 | 2.674 | 3.142 |
| 97_Matmul_BatchNorm_BiasAdd_Divide_Swish | 0.723 | 0.462 | 0.627 | 0.623 |
| 99_Matmul_GELU_Softmax | 5.324 | 2.657 | 2.787 | 3.208 |
| Average | 2.505 | 2.539 | 2.542 | 2.178 |
| Geometric mean | 1.600 | 1.833 | 1.944 | 1.695 |

Table 10: Ablation studies with respect to prompt evolution and gradient guidance.

### F.5 Evaluation of hardware-awareness

<table>
<tr>
<td></td>
<td colspan="2">Runtime [ms] on LNL</td>
<td></td>
<td colspan="2">Runtime [ms] on B580</td>
<td></td>
</tr>
<tr>
<td>Operation</td>
<td>Optimized on LNL</td>
<td>Optimized on B580</td>
<td>hws</td>
<td>Optimized on LNL</td>
<td>Optimized on B580</td>
<td>hws</td>
</tr>
<tr>
<td>16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling</td>
<td>3.130</td>
<td>3.140</td>
<td>1.003</td>
<td>0.850</td>
<td>0.866</td>
<td>0.982</td>
</tr>
<tr>
<td>17_Conv2d_InstanceNorm_Divide</td>
<td>0.269</td>
<td>0.401</td>
<td>1.491</td>
<td>0.090</td>
<td>0.071</td>
<td>1.268</td>
</tr>
<tr>
<td>1_Conv2D_ReLU_BiasAdd</td>
<td>0.253</td>
<td>0.214</td>
<td>0.846</td>
<td>0.055</td>
<td>0.025</td>
<td>2.228</td>
</tr>
<tr>
<td>21_Conv2d_Add_Scale_Sigmoid_GroupNorm</td>
<td>0.532</td>
<td>0.525</td>
<td>0.987</td>
<td>0.073</td>
<td>0.050</td>
<td>1.474</td>
</tr>
<tr>
<td>24_Conv3d_Min_Softmax</td>
<td>6.130</td>
<td>6.150</td>
<td>1.003</td>
<td>1.820</td>
<td>1.840</td>
<td>0.989</td>
</tr>
<tr>
<td>32_Conv2d_Scaling_Min</td>
<td>0.088</td>
<td>0.454</td>
<td>5.142</td>
<td>0.041</td>
<td>0.040</td>
<td>1.010</td>
</tr>
<tr>
<td>35_Conv2d_Subtract_HardSwish_MaxPool_Mish</td>
<td>0.465</td>
<td>0.476</td>
<td>1.024</td>
<td>0.053</td>
<td>0.048</td>
<td>1.109</td>
</tr>
<tr>
<td>37_Matmul_Swish_Sum_GroupNorm</td>
<td>0.075</td>
<td>0.110</td>
<td>1.473</td>
<td>0.033</td>
<td>0.055</td>
<td>0.608</td>
</tr>
<tr>
<td>46_Conv2d_Subtract_Tanh_Subtract_AvgPool</td>
<td>0.199</td>
<td>0.527</td>
<td>2.648</td>
<td>0.076</td>
<td>0.071</td>
<td>1.075</td>
</tr>
<tr>
<td>47_Conv3d_Mish_Tanh</td>
<td>1.630</td>
<td>1.160</td>
<td>0.712</td>
<td>0.420</td>
<td>0.263</td>
<td>1.597</td>
</tr>
<tr>
<td>50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling</td>
<td>50.500</td>
<td>36.700</td>
<td>0.727</td>
<td>16.100</td>
<td>10.400</td>
<td>1.548</td>
</tr>
<tr>
<td>59_Matmul_Swish_Scaling</td>
<td>0.127</td>
<td>0.128</td>
<td>1.008</td>
<td>0.051</td>
<td>0.044</td>
<td>1.155</td>
</tr>
<tr>
<td>5_ConvTranspose2d_Subtract_Tanh</td>
<td>0.959</td>
<td>0.977</td>
<td>1.019</td>
<td>0.199</td>
<td>0.196</td>
<td>1.015</td>
</tr>
<tr>
<td>67_Conv2d_GELU_GlobalAvgPool</td>
<td>0.265</td>
<td>0.211</td>
<td>0.796</td>
<td>0.090</td>
<td>0.085</td>
<td>1.056</td>
</tr>
<tr>
<td>70_Gemm_Sigmoid_Scaling_ResidualAdd</td>
<td>0.182</td>
<td>0.180</td>
<td>0.989</td>
<td>0.056</td>
<td>0.055</td>
<td>1.013</td>
</tr>
<tr>
<td>73_Conv2d_BatchNorm_Scaling</td>
<td>0.960</td>
<td>1.200</td>
<td>1.250</td>
<td>0.537</td>
<td>0.473</td>
<td>1.135</td>
</tr>
<tr>
<td>82_Conv2d_Tanh_Scaling_BiasAdd_Max</td>
<td>0.132</td>
<td>0.530</td>
<td>4.015</td>
<td>0.048</td>
<td>0.094</td>
<td>0.510</td>
</tr>
<tr>
<td>85_Conv2d_GroupNorm_Scale_MaxPool_Clamp</td>
<td>0.485</td>
<td>0.652</td>
<td>1.344</td>
<td>0.055</td>
<td>0.141</td>
<td>0.391</td>
</tr>
<tr>
<td>97_Matmul_BatchNorm_BiasAdd_Divide_Swish</td>
<td>0.102</td>
<td>0.126</td>
<td>1.235</td>
<td>0.055</td>
<td>0.057</td>
<td>0.974</td>
</tr>
<tr>
<td>99_Matmul_GELU_Softmax</td>
<td>0.009</td>
<td>0.018</td>
<td>2.027</td>
<td>0.009</td>
<td>0.009</td>
<td>1.038</td>
</tr>
<tr>
<td>Average</td>
<td>3.325</td>
<td>2.694</td>
<td>1.537</td>
<td>1.036</td>
<td>0.744</td>
<td>1.109</td>
</tr>
</table>

Table 11: Crossover-experiment: Kernels are optimized in two separate runs on an Intel B580 and Intel LNl GPU, and then benchmarked on the respective other hardware as well. The hardware-speedup $hws$ is the ratio of both runtimes. In most cases, the kernel optimized for the target GPU outperforms the kernel that was optimized on the other GPU (lower runtime on the target GPU, $hws>1$)

## Appendix G Results for open-source model (GPT-OSS 20B)

For the sake of reproducibility, we provide results of running KernelFoundry with GPT-OSS 20B in [Table 12](https://arxiv.org/html/2603.12440v2#A7.T12). We use the representative subset of 20 kernels of KernelBench level 2 and prompt the model to generate SYCL kernels. Unfortunately, the model’s lower capabilities led to failure in generating correct kernels in 7 out of 20 cases, even after 40 iterations and using a population of 4 per generation. However, in the successful cases, the kernels were optimized, achieving, for instance, a speedup of 1.775 for operation 16 on an Intel LNL GPU.

|  |  |
| --- | --- |
| Operation | Speedup |
| 16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling.py | 1.775 |
| 17_Conv2d_InstanceNorm_Divide.py | - |
| 1_Conv2D_ReLU_BiasAdd.py | 1.708 |
| 21_Conv2d_Add_Scale_Sigmoid_GroupNorm.py | - |
| 24_Conv3d_Min_Softmax.py | 0.989 |
| 32_Conv2d_Scaling_Min.py | 2.106 |
| 35_Conv2d_Subtract_HardSwish_MaxPool_Mish.py | 0.865 |
| 37_Matmul_Swish_Sum_GroupNorm.py | 0.043 |
| 46_Conv2d_Subtract_Tanh_Subtract_AvgPool.py | 1.576 |
| 47_Conv3d_Mish_Tanh.py | 0.304 |
| 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling.py | - |
| 5_ConvTranspose2d_Subtract_Tanh.py | 1.223 |
| 59_Matmul_Swish_Scaling.py | - |
| 67_Conv2d_GELU_GlobalAvgPool.py | 0.714 |
| 70_Gemm_Sigmoid_Scaling_ResidualAdd.py | - |
| 73_Conv2d_BatchNorm_Scaling.py | 0.031 |
| 82_Conv2d_Tanh_Scaling_BiasAdd_Max.py | 1.085 |
| 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp.py | - |
| 97_Matmul_BatchNorm_BiasAdd_Divide_Swish.py | 0.069 |
| 99_Matmul_GELU_Softmax.py | - |

Table 12: Results for GPT-OSS for reproducibility.

## Appendix H Convergence curves

[Figure 5](https://arxiv.org/html/2603.12440v2#A8.F5)shows convergence over trials corresponding to the heatmap in [Figure 3](https://arxiv.org/html/2603.12440v2#S5.F3), but reports the best speedup per iteration rather than the cumulative best. This reveals that some iterations are more exploratory, yielding lower speedups as the system searches broader regions of the solution space. [Figure 6](https://arxiv.org/html/2603.12440v2#A8.F6) presents two examples of convergence behavior when scaling to 100 iterations. Even after 80 iterations, KernelFoundry continues to discover higher-performing kernels, particularly when using meta-prompting.

![Refer to caption](assets/30-kernelfoundry/figure-004.png)

Figure 5: Speedup per iteration


![Refer to caption](assets/30-kernelfoundry/figure-005.png)

Figure 6: Convergence of the cumulative speedup over 100 iterations, shown for two examples, with and without prompt evolution activated.

## Appendix I Cost estimation

We measured token usage for five KernelBench level 2 problems using the same experimental configuration as [Table 1](https://arxiv.org/html/2603.12440v2#S5.T1) (40 iterations, 4 branches, o3-mini). Average token consumption per kernel was 11,102 input tokens and 5,377 output tokens. Using OpenAI’s pricing model (o3-mini: $1.10 per 1M input tokens, $4.40 per 1M output tokens), the cost to generate 160 kernels (40 iterations × 4 branches) amounts to approximately $5.74 per kernel. With more recent models (gpt-5.4), costs increase to $17.34 per kernel. These costs are comparable to related work: robust-kbench reports $5 API cost per kernel, while METR demonstrates convergence with costs up to $20 per kernel for o3-mini. Higher costs are justified when tools substantially reduce manual engineering effort.

## Appendix J Qualitative analysis of prompt evolution

To better understand how prompts evolve during the training process, we analyzed the changes in prompts generated across training iterations. [Figure 7](https://arxiv.org/html/2603.12440v2#A10.F7) visualizes pairwise similarities between prompts generated at different training steps for randomly selected problems, computed using difflib’s SequenceMatcher.

The analysis reveals distinct patterns in prompt evolution. While prompts change substantially for some problems, others exhibit clustering behavior where prompts remain similar across multiple iterations. Notably, we observe recurring patterns, with similarities appearing between prompts generated early in training and those generated substantially later, suggesting convergence toward stable prompt structures.

Manual inspection indicates that the “optimization strategies” and “common pitfalls” sections undergo the most frequent modifications, while “optimization philosophy” and “analysis guidance” remain relatively stable (see Section 3.5). Observed additions to prompts include specific anti-patterns and their fixes, such as “Reinterpreting host pointers to vector types inside kernels is unsafe unless the pointer is device-USM and properly aligned. Anti-pattern seen in generated code reinterpret_castsycl::float4*(conv_out_ptr)[i]. Fix: …”. Furthermore, additions include context-aware recommendations such as “Use direct convolution with shared memory when kernel sizes are small (3×3, 5×5) and memory reuse within a tile is high”. These findings suggest that the training process effectively learns to refine domain-specific guidance based on problem characteristics.

![Refer to caption](assets/30-kernelfoundry/figure-006.png)

Figure 7: Prompt similarity (of evolvable prompt regions) between iterations. Prompts are modified but patterns reoccur.
