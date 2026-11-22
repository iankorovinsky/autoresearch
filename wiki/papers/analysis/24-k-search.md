---
paper_id: "24"
title: "K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model"
source_version: "2602.19128v2"
source_markdown: "../pdfs/24-k-search.md"
source_pdf: "../pdfs/24-k-search.pdf"
source_url: "https://arxiv.org/abs/2602.19128"
analysis_date: "2026-11-21"
analysis_status: "draft"
topics: ["gpu-kernels", "planning", "world-model", "tree-search", "evolutionary-search", "agent-scaffolding", "execution-feedback", "iterative-refinement", "cuda", "triton", "inference-serving"]
summary: "K-Search has an LLM maintain a search tree of optimization intents and instantiates each by sampling evaluator-scored kernels; evidence is four FlashInfer kernels (three runs per method) and one TriMul task, with trace-based correctness testing and no ablations."
---

# K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model

## 1. Paper identity and scope

- The source is arXiv `2602.19128v2` by Cao, Mao, Gonzalez and Stoica (UC Berkeley), with a code link to `github.com/caoshiyi/K-Search`; the PDF margin stamp reads `[cs.AI] 26 Feb 2026`. [PDF p. 1](../pdfs/24-k-search.pdf)
- The research question is whether an LLM used as a planner over a tree of optimization intents finds better GPU kernels, under a fixed number of kernel evaluations, than LLM-driven evolutionary search that mutates programs directly. [§1; §3.1](../pdfs/24-k-search.md)
- The contribution is a method called "Search via Co-Evolving World Model", the K-Search system built on it, and a comparison with OpenEvolve and ShinkaEvolve. [Abstract; §1](../pdfs/24-k-search.md)
- The main evaluation covers four FlashInfer kernels written in CUDA (MLA paged prefill, MLA paged decode, GQA paged decode, FP8 MoE); a secondary result is one Triton solution for the GPUMODE TriMul task. [Table 2; §4.1; §4.7](../pdfs/24-k-search.md)
- Scores are relative to the expert-written FlashInfer kernel for each task, so they express performance against an optimized library, not against an unoptimized PyTorch baseline. [§4.1](../pdfs/24-k-search.md)
- The "world model" is not a learned simulator: its state changes only through in-context use of past observations by an LLM. [§3.3, Evolution: World Model Update](../pdfs/24-k-search.md)
- The work introduces no benchmark, correctness method or model training, and it contains no component ablation, no token or cost accounting, and no limitations section. Correctness checking is delegated to the FlashInfer-Bench and GPUMODE evaluators. [§4.1, Implementation details; §4.7](../pdfs/24-k-search.md)

## 2. Concepts and terminology

### Objective, score and budget

- The evaluator maps a kernel program `x` to an observation `o = (s, p, m)`: `s ∈ {0,1}` is correctness, `p` is the performance metric (latency), and `m` is metadata such as compiler logs and profiler output. [§3.1](../pdfs/24-k-search.md)
- The objective is `J(x) = s · (p_ref / p) · 100`, described as "the speedup relative to a reference SoTA baseline". A score of 100 is latency parity with the reference, below 100 is slower than the reference, and an incorrect program scores 0. [§3.1](../pdfs/24-k-search.md)
- Correctness enters only as the multiplicative gate `s`: a kernel must pass the functional correctness tests before receiving a non-zero score. There is no partial credit and no accuracy term. [§3.1; §4.1, Implementation details](../pdfs/24-k-search.md)
- The reported quantity is the best-so-far score per iteration, where the score is `J(x)` "averaged over workloads" with the FlashInfer kernel as reference. The averaging formula, and whether a failure on one workload zeroes that workload or the whole kernel, are not stated. [§3.1; §4.1](../pdfs/24-k-search.md)
- The budget `B` counts evaluator invocations. One "iteration" (also "round") is one evaluation of a single candidate kernel on the benchmark workloads, i.e. one compiled and benchmarked kernel. [§3.1; §3.4; §4.1](../pdfs/24-k-search.md)
- The headline multipliers ("2.10×", "14.3×") are ratios of mean final scores between search methods. The source calls them "improvements"; they are not wall-clock speedups of one kernel over another, and not speedups over FlashInfer. [§4.4](../pdfs/24-k-search.md)
- `Fast_p` denotes the fraction of workloads on which a system's best kernel reaches at least a given speedup over the FlashInfer baseline; "speedup" there is the per-workload ratio to the reference. [§4.4; Figure 3](../pdfs/24-k-search.md)

### Program-space search versus the world-model formulation

- The baseline formulation selects a subset of past programs with evolutionary heuristics such as MAP-Elites and generates the next candidate from raw program text and serialized feedback. The authors argue this couples intent with implementation, so a sound strategy can be dropped after one transient syntax error; this is an argument, not a measured failure rate. [§3.2, Baseline: Heuristic Search in Program Space](../pdfs/24-k-search.md)
- In K-Search the LLM represents a transition distribution over search states, `P_model(S_{t+1} | S_t, a_t)`. A state holds the history of explored actions and their performance, the frontier of pending actions, and a priority score over the frontier. [§3.2, Ours: Search via Co-Evolving World Model](../pdfs/24-k-search.md)
- An action is a pair `(x_parent, δ)`: a natural-language intent `δ` (example: "Resolve bank conflicts via padding") applied to a specific parent program. [§3.2](../pdfs/24-k-search.md)
- `V(a | S_t) ∈ [0,1]` is a scalar priority predicted by the LLM when an action is created or updated. The source calls it the model's "intrinsic assessment"; it is a judgment, not a measurement. [§3.2; Table 1](../pdfs/24-k-search.md)
- The state is an explicit tree of Closed nodes (visited actions whose refinement has ended, with the best program found attached) and Open nodes (the frontier, each holding a proposed optimization and a priority). [§3.3, Search State; Figure 1](../pdfs/24-k-search.md)
- "Co-evolving" means the model's beliefs are revised as results accumulate; the source says that in the current version this happens only by in-context learning. [§3.2; §3.3, Evolution: World Model Update](../pdfs/24-k-search.md)

### The three phases and the stagnation limit

- Action selection takes the frontier action with the highest `V`. It is a plain argmax; no exploration bonus or visit count is described. [§3.2; Algorithm 1](../pdfs/24-k-search.md)
- Program instantiation uses the LLM as a stochastic policy `π_code` that repeatedly samples and evaluates implementations of the selected action until `K` consecutive attempts bring no improvement. `K` is the stagnation limit. [§3.3, Execution: Local Refinement](../pdfs/24-k-search.md)
- In Algorithm 1 the best program and the stagnation counter are reset for every selected action, every sample costs one unit of budget, and the counter returns to zero whenever a sample beats the best one for that action. [Algorithm 1](../pdfs/24-k-search.md)
- The world-model update receives the best program and observation of the finished action and applies three tree edits: Insert (new child actions), Update (re-score frontier nodes) and Prune (permanently remove branches judged infeasible or redundant). [Algorithm 1; §3.3, Evolution: World Model Update](../pdfs/24-k-search.md)
- Derived: each selected action consumes at least `K` evaluations unless the budget ends, so a run with `B = 120` and `K = 7` completes at most 17 actions. This is the analyst's arithmetic, not a source statement. [Algorithm 1; §4.2](../pdfs/24-k-search.md)

### What the model decides and what is measured

- Decided by the LLM: which intents exist, their priorities, which is tried next, how results are interpreted, and which branches are deleted. None of these decisions is checked by an executable test. [§3.2; §3.3](../pdfs/24-k-search.md)
- Executed and measured: each sampled program is compiled, tested for functional correctness against the reference code, and timed by the evaluator. The score `J` comes only from this path. [§3.1; §4.1, Implementation details](../pdfs/24-k-search.md)
- Rationales attached to intents are hypotheses. In the case study the model expects fused multi-head processing to cut global memory traffic by 16× relative to independent processing; memory traffic is not measured. [§3.4, r1](../pdfs/24-k-search.md)

## 3. Methodology and experimental setup

### Tasks, workloads and correctness checking

- A K-Search task has a specification (PyTorch reference implementation, objective, task-specific instructions) and an evaluator that compiles candidates, validates functional correctness against the reference code, and measures performance. For the FlashInfer kernels the evaluator is FlashInfer-Bench. [§4.1, Implementation details](../pdfs/24-k-search.md)
- All candidates must be CUDA, and the same compilation toolchain, correctness suite and benchmark harness are used for all three methods. [§4.1, Implementation details](../pdfs/24-k-search.md)
- Each kernel has a fixed set of test traces "captured in real traffic", used for both correctness and benchmarking and identical across methods. [§4.3](../pdfs/24-k-search.md)
- Correctness here therefore means passing the evaluator's tests on a fixed trace set. This is sampled testing; no tolerance, input-generation procedure or proof is given. [§4.1; §4.3](../pdfs/24-k-search.md)
- The kernels are MLA Paged Prefill (Hopper, DeepSeek-V3), MLA Paged Decode (Hopper, DeepSeek-V3), GQA Paged Decode (Hopper, model context printed as "Qwen3-A3B-30B") and FP8 MoE (Blackwell, DeepSeek-V3); detailed configurations are stated to be omitted. [Table 2](../pdfs/24-k-search.md)
- Some configuration appears in the kernel analyses: MoE uses top-8 routing over 256 candidate experts, 32 local experts and hidden size 7168; the MLA tasks use 16 heads, CKV dimension 512 and KPE dimension 64. [§4.5; §6.1](../pdfs/24-k-search.md)
- The per-workload analysis covers 152 traces over the four kernels. Derived by the analyst from dot counts in the per-workload figure: 48 for GQA decode, 47 for MLA decode, 38 for MLA prefill and 19 for MoE; the source does not print per-kernel counts. [§4.4; Figure 3](../pdfs/24-k-search.md)

### Systems, models, budgets and hardware

- Three methods are compared: OpenEvolve, ShinkaEvolve and K-Search. All use `gemini-3-pro-preview`; ShinkaEvolve also uses Qwen3-8B as its embedding model. [§4.2](../pdfs/24-k-search.md)
- Every system runs for 120 iterations per kernel, each method is repeated three times, and plots show the mean curve with a shaded min–max band. [§4.1](../pdfs/24-k-search.md)
- K-Search uses `B = 120` and `K = 7` on the FlashInfer CUDA tasks. No sensitivity study for `K` is reported. [§4.2](../pdfs/24-k-search.md)
- The baselines use their default configurations with a shared code-generation prompt. It asks for CUDA optimized for a `{GPU}` placeholder, requires three named files (`kernel.h`, `kernel.cu`, `main.cpp`), permits cuBLAS, cuDNN and CUTLASS, and ends with "Performance targets (lower is better)" and a `{workloads}` placeholder whose substituted content is not shown. [§4.2; §6.2](../pdfs/24-k-search.md)
- That template is described as the one used for the baselines; the prompts K-Search uses for the world model and for `π_code` are not reproduced. [§6.2](../pdfs/24-k-search.md)
- Initial programs: the source says the same initial program was used for each kernel, and separately that an initial CUDA program was provided for MLA decode because the baselines struggle to write working kernels without one. [§4.2; §4.3](../pdfs/24-k-search.md)
- In the search-process figure only the MLA decode panel is titled "MLA decode (seed)"; the others are "GQA decode", "MLA prefill" and "MoE". [PDF p. 8](../pdfs/24-k-search.pdf)
- Hardware and software: NVIDIA H100 and B200 GPUs, CUDA 12.8, FlashInfer 0.5.3, PyTorch 2.8.0. The kernel table lists architectures (Hopper, Blackwell), but the text does not say which GPU ran which kernel. [§4.1, Implementation details; Table 2](../pdfs/24-k-search.md)
- The timing protocol (warm-up, repetitions per workload, variance handling) is not described in this paper. [§4.1](../pdfs/24-k-search.md)

### GPUMODE TriMul configuration

- GPUMODE is described as a public competition platform where Triton or CUDA submissions must pass "strict correctness checks" before standardized benchmarking; the checks are not specified here. [§4.7, Task Overview](../pdfs/24-k-search.md)
- TriMul (Triangle Multiplicative Update) operates on a pair representation of shape `B×N×N×C` and involves LayerNorm, five gated linear projections with optional masking, an `O(N^3)` pairwise contraction and a gated output projection. [§4.7, Task Overview](../pdfs/24-k-search.md)
- K-Search runs with `K = 5` (reduced from 7 because, the authors say, Triton is simpler than CUDA) for 300 iterations: 150 with GPT-5.2, then 150 with Gemini-3-Pro continuing from the best GPT-5.2 solution, with no seed Triton program. [§4.7, Configuration](../pdfs/24-k-search.md)
- Competing numbers come from the official leaderboard. Because submissions had closed, the K-Search kernel was measured locally with the official evaluator on NVIDIA H100 80 GB HBM3 GPUs. The number of TriMul runs is not stated, and the two evolutionary baselines are not run on this task. [§4.7, Results](../pdfs/24-k-search.md)

## 4. Results and evidence

### Final scores on the FlashInfer kernels

- Averaged over the four kernels, the mean final score is 56.13 for K-Search, 26.68 for OpenEvolve and 25.37 for ShinkaEvolve, reported as 2.10× and 2.21× improvements (gemini-3-pro-preview, 120 evaluations, three runs, correctness-gated score relative to FlashInfer). [§4.4](../pdfs/24-k-search.md)
- FP8 MoE (Blackwell): K-Search 44.1, OpenEvolve 3.09, ShinkaEvolve 27.9, reported as 14.3× and 1.58×. [§4.4](../pdfs/24-k-search.md)
- MLA prefill: K-Search 57.4, OpenEvolve 19.5, ShinkaEvolve 11.3, reported as 2.95× and 5.10×. [§4.4](../pdfs/24-k-search.md)
- GQA decode: K-Search 76.0, OpenEvolve 44.2, ShinkaEvolve 27.7 in the prose, reported as 1.72× and 2.74×; see the inconsistency note on the ShinkaEvolve value. [§4.4](../pdfs/24-k-search.md)
- MLA decode (the seeded task): K-Search 47.1, OpenEvolve 39.9, ShinkaEvolve 34.7, reported as 18% and 36% improvements (relative percent of score, not percentage points). [§4.4](../pdfs/24-k-search.md)
- Every mean final score is below 100: under the paper's metric the average best kernel of every method, including K-Search, is slower than the FlashInfer reference on the test traces. [§4.4](../pdfs/24-k-search.md)
- Derived: 2.10× is the ratio of the four-kernel average scores (56.13/26.68). The per-kernel ratios against OpenEvolve are about 14.3, 2.95, 1.72 and 1.18, with arithmetic mean about 5.0 and geometric mean about 3.0, so the headline depends on the aggregation chosen. [§4.4](../pdfs/24-k-search.md)
- Analyst digitization of the search-process curves reproduces eleven of the twelve prose endpoints within 0.1 score points and gives these K-Search min–max ranges over three runs at iteration 120: GQA decode 65.0–86.3, MLA decode 42.6–52.4, MLA prefill 51.1–63.5, MoE 28.2–55.8. Values are read from path coordinates and are approximate. [Figure 3](../pdfs/24-k-search.md)
- From the same digitization, the K-Search minimum at iteration 120 is above both baselines' maxima for GQA decode, MLA decode (42.6 versus 41.6 for OpenEvolve) and MLA prefill. On MoE the K-Search minimum (28.2) is below the ShinkaEvolve maximum (33.4), so the ranges overlap. The K-Search mean curve is above both baseline means from iteration 11 onward in all four panels. [Figure 3](../pdfs/24-k-search.md)

### Per-workload behaviour of the best kernels

- The source states that K-Search's best kernel outperforms the baselines' best kernels on "the vast majority" of the 152 traces, without a count. [§4.4](../pdfs/24-k-search.md)
- Analyst digitization of the per-workload dots: K-Search is strictly highest on 28/48 GQA decode traces, 43/47 MLA decode, 37/38 MLA prefill and 16/19 MoE, i.e. 124/152. Plotted dot values are integers. [Figure 3](../pdfs/24-k-search.md)
- On GQA decode K-Search underperforms the baselines on small-batch workloads, 16 with `batch_size=1` and 4 with `batch_size=16`, and not at larger batch sizes. [§4.4](../pdfs/24-k-search.md)
- The authors attribute this to K-Search's split-K design, which spreads the key-value sequence over thread blocks and adds synchronization cost that small batches cannot amortize, while the baselines' single-block-per-batch design is cheaper at `batch_size=1` and worse at large batches. The explanation comes from code inspection, not an ablation. [§4.4; §4.6](../pdfs/24-k-search.md)
- The source says generated kernels "rarely exceed" the FlashInfer kernel. Analyst digitization: best kernels exceed a per-workload score of 100 on 16/152 traces for K-Search (11 GQA decode, 1 MLA decode, 4 MLA prefill, 0 MoE), 16/152 for OpenEvolve and 7/152 for ShinkaEvolve (GQA decode only for both). The highest GQA decode dot is OpenEvolve's (about 171) against about 113 for K-Search. [§4.4; Figure 3](../pdfs/24-k-search.md)
- GQA decode thresholds: K-Search reaches speedup ≥ 0.36 on 100% of workloads versus 50% for ShinkaEvolve; at ≥ 0.50 it reaches 87.5% versus 50.0% (OpenEvolve) and 39.6% (ShinkaEvolve), reported as 1.75× and 2.21×. These are ratios of coverage fractions. [§4.4](../pdfs/24-k-search.md)
- MLA prefill: K-Search reaches speedup ≥ 0.40 on 57.9% of workloads; no baseline kernel reaches 0.40 on any workload. [§4.4](../pdfs/24-k-search.md)

### Search behaviour

- In the GQA logs the "vast majority" of ShinkaEvolve generations score zero (incorrect or failed programs); no count is given. OpenEvolve is described as having high per-iteration variance, and its mean final MoE score stays near 3 against K-Search's 44. [§4.4, Key observations](../pdfs/24-k-search.md)
- Case-study trace (one MLA paged decode run): the frontier starts with `fused_multi_head`, `split_k_decoding` and `independent_heads`; the model gives the first the highest priority and instantiates it first. The figure labels show 0.9, 0.7 and 0.3. [§3.4, r1; Figure 2](../pdfs/24-k-search.md)
- After `fused_multi_head` closes with score 34, the model inserts `register_resident_rescaling` and `occupancy_tuned_chunk32`, downgrades `independent_heads`, and by round 34 prunes it permanently. [§3.4, r14-r34](../pdfs/24-k-search.md)
- At round 42 the model deletes the root-level split-K action and inserts a variant, `low_overhead_split_k`, inside the `register_resident` branch. [§3.4, r42-r102](../pdfs/24-k-search.md)
- In the figure labels `split_k_decoding` shows a priority of 0.7 at r14 and r34 and no score before it is marked deleted at r42 (open nodes show priorities, closed nodes show scores). As far as the labels show, the root-level split-K action was never instantiated or measured in this trace. [PDF p. 6](../pdfs/24-k-search.pdf)
- The nine closed-node scores visible at r102 are 34, 45, 33, 44, 49, 48, 50, 45 and 52. Nodes closed after the 45-point node include 44 and 33, so later closures are not always improvements. [PDF p. 6](../pdfs/24-k-search.pdf)
- `chunk32_prescale_vectorized`, proposed after `chunk32_vectorized` succeeded, gives the best solution of the trace at r102. The trace is one run; its best labelled score of 52 is above the three-run mean of 47.1 for the same kernel. [§3.4, r42-r102; §4.4](../pdfs/24-k-search.md)

### Qualitative analyses of the generated kernels

- MoE: K-Search's kernel uses one 256-thread block per token with warp shuffles (`__shfl_down_sync`) for top-8 selection, sorts tokens by expert, uses tensor cores through WMMA on 16×16 blocks with double buffering, and skips empty experts. OpenEvolve's persistent kernel fetches tile indices with `atomicAdd` in a loop; ShinkaEvolve's uses plain loops and no tensor cores. [§4.5](../pdfs/24-k-search.md)
- GQA decode: K-Search's kernel splits the sequence across blocks with a counter-triggered merge and double-buffered loads; the baselines use one block per (batch, KV head), and ShinkaEvolve's kernel serializes loading and compute. [§4.6](../pdfs/24-k-search.md)
- MLA prefill: K-Search's kernel resolves sequence boundaries inside 16-row tiles on the GPU with a prefix-sum array and keeps all threads busy during score computation; the baselines precompute a tile-to-batch map on the CPU, and OpenEvolve's kernel limits score-and-softmax to one warp. [§6.1, MLA Paged Prefill](../pdfs/24-k-search.md)
- MLA decode: K-Search's best kernel splits adaptively, keeps the query in register-resident fragments instead of shared memory, and prefetches two chunks ahead; the baselines use 64-token chunks with a fixed minimum number of splits and standard double buffering. [§6.1, MLA Paged Decode](../pdfs/24-k-search.md)
- These analyses describe code structure. Their "is faster because" statements come without profiling data or experiments that remove one design element. [§4.5; §4.6; §6.1](../pdfs/24-k-search.md)

### GPUMODE TriMul

- Reported geometric-mean latency over a fixed set of benchmark cases on H100: K-Search (Triton, GPT-5.2 + Gemini-3-Pro, 300 iterations) 1030 µs; `shiyegao` (CUDA) 1074 µs; `Zeyu Shen` (Triton) 1140 µs; TTT (Triton, GPT-OSS-20B with RL, 25,600 iterations) 1161 µs. [Table 3](../pdfs/24-k-search.md)
- Derived: 1030 µs is about 4.1% lower than 1074 µs and about 11.3% lower than 1161 µs. The K-Search number is a local run of the official evaluator; the others are leaderboard entries. [Table 3; §4.7, Results](../pdfs/24-k-search.md)

## 5. Authors' conclusions

- LLMs have latent planning ability that lets them act as effective intrinsic world models, not only as code generators. [§5](../pdfs/24-k-search.md)
- Replacing static search heuristics with a co-evolving world model lets the LLM reason over the search space and separate valid strategies from implementation noise. [§5](../pdfs/24-k-search.md)
- The 2.1× average improvement, the 14.3× MoE gain and the TriMul result are presented as validating this "paradigm shift". [§5](../pdfs/24-k-search.md)
- The authors hypothesize (their word) that the advantage comes from maintaining and co-evolving a persistent search state that conditions new proposals on past attempts. [§4.4, Key observations](../pdfs/24-k-search.md)
- From the case study they infer that split-K is ineffective as an isolated starting point but effective when composed on a strong fused kernel, and that starting from intents avoids enumerating a sparse program space. [§3.4, r42-r102; §3.4, Takeaway](../pdfs/24-k-search.md)
- They argue that LLMs can serve as the core planning engine for complex optimization problems and drive the search strategy autonomously. [§5](../pdfs/24-k-search.md)

## 6. Limitations and open questions

### Stated by the source

- Generated kernels rarely exceed the expert-optimized FlashInfer kernel. [§4.4](../pdfs/24-k-search.md)
- World-model evolution is "merely" in-context learning of past observations in the current version. [§3.3, Evolution: World Model Update](../pdfs/24-k-search.md)
- The best K-Search GQA decode kernel is worse than the baselines' kernels on 20 small-batch workloads. [§4.4](../pdfs/24-k-search.md)
- An initial CUDA program was supplied for MLA decode because the baselines could not reliably write a working kernel without one. [§4.3](../pdfs/24-k-search.md)

### Analyst caveats and missing evidence

- **No ablation:** nothing separates the effect of the intent tree, the priority scores, pruning, local refinement with a stagnation limit, or simply keeping a persistent history. The causal account is a hypothesis. [Basis: §4.4, Key observations](../pdfs/24-k-search.md)
- **Budget definition:** the budget equalizes evaluator calls only. World-model calls, tokens, cost and wall-clock time are not reported, so equal-evaluation comparisons may not be equal-cost comparisons. [Basis: §3.1; §4.1](../pdfs/24-k-search.md)
- **Small sample:** three runs per method per kernel with min–max bands and no statistical test; the MoE ranges of K-Search and ShinkaEvolve overlap in the analyst's digitization. [Basis: §4.1; Figure 3](../pdfs/24-k-search.md)
- **Baseline configuration:** the baselines run at default settings with one generic prompt, while K-Search's prompts are not shown. [Basis: §4.2; §6.2](../pdfs/24-k-search.md)
- **Correctness evidence is sampled testing:** the same fixed traces serve as correctness tests and benchmark, and a workloads placeholder appears in the generation prompt. Nothing tests behaviour off those traces, and no numerical tolerance is given. [Basis: §4.3; §6.2](../pdfs/24-k-search.md)
- **Priority scores are unvalidated:** no analysis compares `V` with the measured outcome of the action, and pruning is permanent on the model's judgment alone. [Basis: §3.2; §3.3](../pdfs/24-k-search.md)
- **An insight that was not executed:** the authors read the round-42 edit as showing split-K is ineffective in isolation, but the trace labels show no evaluation of the root-level split-K action. In this trace that is the model's judgment, not a measured outcome. [Basis: §3.4, r42-r102; Figure 2](../pdfs/24-k-search.md)
- **Aggregation hides workload trade-offs:** a mean score rewards a kernel that is faster on most traces even if it regresses on a subset, as the GQA small-batch case shows. [Basis: §4.4](../pdfs/24-k-search.md)
- **TriMul comparability:** the result uses two models in sequence, a different `K`, an unstated number of runs and a local measurement, with no evolutionary baseline under the same models and budget. [Basis: §4.7](../pdfs/24-k-search.md)
- **Headline wording:** 14.3× is against OpenEvolve only; against ShinkaEvolve on the same kernel the ratio is 1.58×. The conclusion's "over state-of-the-art evolutionary baselines" does not make this distinction. [Basis: §4.4; §5](../pdfs/24-k-search.md)
- **Refinement context is unspecified:** Algorithm 1 writes `x ∼ π_code(· | a_t)`; whether a retry sees the previous failed attempt and its error log is not stated. [Basis: Algorithm 1; §3.3](../pdfs/24-k-search.md)

### Source inconsistencies and ambiguities

- **Source inconsistency (figure versus prose, analyst digitization):** the prose gives ShinkaEvolve's GQA decode final score as 27.7 (and derives 2.74× and the 25.37 average from it), but the ShinkaEvolve mean curve in the GQA panel ends at about 35.3 at iteration 120, after steps from about 25.0 to 30.1 to 35.3 over iterations 117–119. The other eleven endpoints match. The prose value is text-specific; the 35.3 reading rests on path coordinates and was not confirmed by rendering. [§4.4; Figure 3](../pdfs/24-k-search.md)
- **Source ambiguity (initial programs):** one sentence says the same initial program was used for each kernel, another says an initial CUDA program was provided for MLA decode. What the other three kernels started from is not stated. [§4.2; §4.3](../pdfs/24-k-search.md)
- **Source ambiguity (seed accounting, analyst digitization):** in the seeded MLA decode panel the K-Search mean is about 29.4 at iteration 1, while both baselines are at 0 at iteration 1 and reach about 24–30 by iterations 3–6. How the seed program's own evaluation is counted for each system is not stated. [Figure 3; §4.3](../pdfs/24-k-search.md)
- **Source ambiguity (per-workload plot versus scores, analyst digitization):** arithmetic means of the per-workload dots do not coincide with the final scores; the OpenEvolve GQA decode dots average about 76 while the OpenEvolve band tops out near 57, and the K-Search GQA dots average about 80. Which run supplies each "best kernel", whether it was re-benchmarked, and how workloads are weighted in `J` are not stated. Do not recompute scores from the dots. [Figure 3; §4.1; §4.4](../pdfs/24-k-search.md)
- **Wording versus figure (analyst digitization):** "rarely exceed" coexists with roughly 23% (K-Search), 33% (OpenEvolve) and 15% (ShinkaEvolve) of GQA decode traces above parity for the best kernels; the statement fits the other three kernels. [§4.4; Figure 3](../pdfs/24-k-search.md)
- **Numbering difference:** the local Markdown numbers the appendix §6, §6.1 and §6.2, while the PDF labels it Appendix A, A.1 and A.2 and captions the prompt "Listing 1". Locators here follow the Markdown. [PDF p. 14](../pdfs/24-k-search.pdf)

## 7. Relevance to this project — interpretation

- **Proposal:** represent the Stage 3 planning output as explicit intent records (parent kernel from the pool, natural-language intent, planner priority), separate from the code written for them, and tag every kernel in a batch with the intent it instantiates. [Basis: §3.2; §3.3, Search State](../pdfs/24-k-search.md)
- **Proposal:** give each planned intent a bounded number of write attempts before dropping it, so a compile error or slow first draft does not retire the idea, and charge every attempt to the round's evaluation budget. [Basis: Algorithm 1; §3.3, Execution: Local Refinement](../pdfs/24-k-search.md)
- **Proposal:** decide deliberately how a batch of size b is spent. K-Search spends consecutive evaluations on one intent; a batch could hold b samples of one intent or one sample each of b intents. Compare these at equal evaluation counts. [Basis: Algorithm 1; §4.2](../pdfs/24-k-search.md)
- **Proposal:** treat planner priorities as scheduling metadata only; pool admission stays tied to the fuzzing check and the PyGPUBench measurement, mirroring the source's split between model-estimated `V` and evaluator-produced `J`. Log each predicted priority beside the measured outcome so planner calibration can be assessed, which the source does not do. [Basis: §3.1; §3.2](../pdfs/24-k-search.md)
- **Proposal:** make pruning a logged, reversible state change, and let the planner propose an optimization as a composition on a strong pool member as well as a standalone variant. The case study removes an idea at one tree position and reintroduces it elsewhere. [Basis: §3.3, Evolution: World Model Update; §3.4, r42-r102](../pdfs/24-k-search.md)
- **Proposal:** store the correctness flag, raw latency and reference latency as separate fields and compute any gated relative score from them; report ratios of scores between methods as score ratios, never as speedups. [Basis: §3.1; §4.4](../pdfs/24-k-search.md)
- **Proposal:** report per-workload results and threshold-coverage curves for the selected kernel in Stage 4, and check for subsets (such as small batch sizes) where it regresses against other pool members. [Basis: §4.4; Figure 3](../pdfs/24-k-search.md)
- **Proposal:** keep fuzzing inputs distinct from timing inputs, and keep Stage 4 inputs unseen by the Stage 3 loop; the source uses one fixed trace set for both purposes. [Basis: §4.3; §6.2](../pdfs/24-k-search.md)
- **Proposal:** record whether a run starts from a supplied kernel and how that kernel's evaluation counts against the round budget, which the source leaves unclear. [Basis: §4.2; §4.3](../pdfs/24-k-search.md)
- **Proposal:** if a planner is added, run the ablation the source lacks: the same loop with and without the intent layer, at equal evaluation budget, with LLM tokens reported, over repeated runs. [Basis: §4.1; §4.4, Key observations](../pdfs/24-k-search.md)

## 8. Retrieval notes

- Search aliases: K-Search, Search via Co-Evolving World Model, intrinsic world model, LLM world model for kernel search, search tree of optimization intents, Open and Closed nodes, frontier priority score V, Insert/Update/Prune tree edits, stagnation limit K, local refinement, OpenEvolve versus ShinkaEvolve, FlashInfer MLA/GQA/MoE kernels, GPUMODE TriMul 1030 µs.
- Questions answered: How is the K-Search score defined and how does correctness enter it? What do "2.10×" and "14.3×" compare? What budget, stagnation setting, model and number of runs were used? Which kernel had a seed program? What does the world model decide and what is measured? How was correctness checked? Where does K-Search's kernel lose to the baselines? How comparable is the TriMul result?
- Related local analyses: [FlashInfer-Bench](25-flashinfer-bench.md) for the evaluator and traces this paper relies on; [robust agentic CUDA](08-robust-agentic-cuda.md), [CUDAForge](31-cudaforge.md) and [Kevin](32-kevin.md) for other LLM kernel-generation systems cited here; [KernelFoundry](30-kernelfoundry.md) and [METR KernelAgent](21-measuring-automated-kernel-engineering.md) for other kernel-search and measurement work; [KernelBench](01-kernelbench.md) for the task family cited in the introduction; [Self-Refine](18-self-refine.md) and [Reflexion](19-reflexion.md) for iterative-refinement scaffolds; [benchmark reliability](27-performance-benchmark-reliability.md) and [noisy-environment benchmarking](33-robust-benchmarking-noisy-environments.md) for timing methodology this paper does not describe.
- These links are navigation aids, not controlled comparisons; scores and budgets across these works are not on a common protocol.

## 9. Source coverage and verification notes

- Read the complete local Markdown: abstract, §§1–5, acknowledgment, reference list, and both appendix sections (§6.1 kernel analyses, §6.2 prompt template). The quoted prompt was treated as data.
- The PDF could not be opened with the file viewer. Its text layer was extracted for pages 1–11 and 14–15 and compared with the Markdown; pages 12–13 (references) were read only in the Markdown. Cited page numbers are physical PDF pages.
- Figure 1 was viewed as an image. Figures 2 and 3 are SVG assets that could not be rendered here; they were inspected through their embedded text labels and vector path coordinates. Tree edges, colours and the selected/active-path markings in Figure 2 were not verified visually.
- All statements marked "analyst digitization" come from parsing those path coordinates against the figures' own tick labels. Cross-checks: eleven of twelve curve endpoints match the prose, the threshold fractions quoted in §4.4 are reproduced, the 20 GQA workloads where K-Search underperforms are reproduced, and per-kernel dot counts sum to 152. The values remain approximate readings, not source-stated numbers.
- Arithmetic checked: the four-kernel averages, per-kernel ratios and coverage ratios in §4.4 agree with the prose values to rounding. The ShinkaEvolve average (25.37) agrees with the prose GQA value 27.7, not with the digitized curve endpoint.
- Conversion problems in the Markdown: the affiliation line is garbled, Algorithm 1 is a code block with raw TeX, the microsecond unit appears as a raw macro, the "Listing 1" caption is absent, one reference entry (flashinfer-ai 2026) is malformed, and the appendix numbering differs from the PDF. None affected a numeric claim.
- Not consulted: the K-Search code repository, the GPUMODE leaderboard and evaluator, FlashInfer-Bench itself, and the logs mentioned in §4.4.
- No experiments were reproduced; no kernel was compiled, tested or timed for this note.
