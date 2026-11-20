---
paper_id: "21"
title: "Measuring Automated Kernel Engineering"
source_version: "web snapshot 2026-11-20"
source_markdown: "../pdfs/21-measuring-automated-kernel-engineering.md"
source_pdf: null
source_url: "https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering/"
analysis_date: "2026-11-20"
analysis_status: "reviewed"
topics: ["gpu-kernels", "benchmarking", "agent-scaffolding", "test-time-compute", "benchmark-validity", "cost-effectiveness", "training-data", "kernelbench"]
summary: "METR evaluates a tree-search kernel agent on a filtered, extended KernelBench suite, reporting speedups and cost tradeoffs while documenting major gaps from expert kernel engineering."
---

# Measuring Automated Kernel Engineering

## 1. Paper identity and scope

- This entry is a METR research article, dated February 14, 2025 in its source URL, rather than an arXiv paper. This analysis concerns the locally saved November 20, 2026 snapshot; there is no original PDF in this collection. [Source and introductory paragraphs](../pdfs/21-measuring-automated-kernel-engineering.md)
- The research question is how well existing language-model agents can optimize GPU kernels when given substantially more scaffolding and inference-time search than a minimal generation baseline. The article situates this within measuring automation of AI R&D. [Introduction](../pdfs/21-measuring-automated-kernel-engineering.md)
- The experimental scope is single-H100 inference workloads derived from KernelBench and a new Level 5. It does not evaluate distributed execution, backward passes, end-to-end training, or replacement of professional kernel engineers. [Introduction; Remaining limitations](../pdfs/21-measuring-automated-kernel-engineering.md)
- The released artifacts are best discovered solutions and a filtered task suite in `METR/KernelBenchFiltered`; the KernelAgent implementation is deliberately withheld. The published results therefore reveal the broad search design but not a fully reproducible agent implementation. [Our Kernel Agent and Results; Impact Statement](../pdfs/21-measuring-automated-kernel-engineering.md)

## 2. Concepts and terminology

### Kernel engineering and task realism

- A kernel is a low-level compute building block, ranging from an elementwise operation to matrix multiplication or fusion of multiple neural-network components. The optimization task is to preserve the PyTorch module's input-output behavior while reducing runtime. [Introduction](../pdfs/21-measuring-automated-kernel-engineering.md)
- The baseline's optimization level is central: these tasks start from relatively naive PyTorch, which leaves opportunities unlike those in already highly optimized production CUDA. [Remaining limitations](../pdfs/21-measuring-automated-kernel-engineering.md)
- Level 1 consists of individual PyTorch operations; Level 2 consists of sequences of three to six operations; Level 3 consists of complete older model architectures; Level 5 contains larger, more recent open-source workloads. Level 4 is excluded. [Introductory task table image](../pdfs/assets/21-measuring-automated-kernel-engineering/figure-001.png)
- The introductory table describes Level 1 and Level 3 as realistic but potentially memorizable, Level 2 as novel but unrealistic, and Level 5 as realistic. These are the article's qualitative characterizations, not measured realism scores. [Introductory task table image](../pdfs/assets/21-measuring-automated-kernel-engineering/figure-001.png)
- The table estimates expert effort at 15 minutes–4 hours for Level 1, 30 minutes–10 hours for Level 2, 8–100+ hours for Level 3, and 40–500+ hours for Level 5. These are estimates; the study does not conduct a matched human evaluation. [Introductory task table image; Remaining limitations](../pdfs/assets/21-measuring-automated-kernel-engineering/figure-001.png)

### Metrics and search

- Per-task speedup is reference runtime divided by candidate runtime. The chosen candidate is the best of the sampled attempts, and the original implementation is retained whenever all candidates are slower, so the reported task score is floored at 1. [Introduction, metric definition](../pdfs/21-measuring-automated-kernel-engineering.md)
- “Average speedup” means the geometric mean across tasks, not an arithmetic average and not a speedup conditional only on tasks with successful improvements. The fallback to the original code means failures do not create ratios below 1 in this metric. [Introduction; Takeaways](../pdfs/21-measuring-automated-kernel-engineering.md)
- Best-of-k combines generation, verification and selection. A per-problem best across several agents is an oracle-style portfolio result and should not be described as the output of a single model call. [Introduction; Speedup distributions](../pdfs/21-measuring-automated-kernel-engineering.md)
- Capability elicitation means investing in prompts, tools, search and evaluation resources to expose capabilities not visible under a weak scaffold. The article argues that benchmark scores should be interpreted in light of that investment. [Introduction; Takeaways](../pdfs/21-measuring-automated-kernel-engineering.md)
- The agent uses parallel tree search: completed candidates and the original implementation become possible parents for further attempts, sampled using measured speedup. This is iterative code search with external feedback, not evidence of autonomous model-weight self-improvement. [Our Kernel Agent and Results](../pdfs/21-measuring-automated-kernel-engineering.md)
- Economic break-even compares one-time optimization cost with saved future execution cost. The reported threshold depends on GPU runtime, pricing, achieved speedup and the representativeness of the tested workloads. [Our Kernel Agent and Results; footnote 5](../pdfs/21-measuring-automated-kernel-engineering.md)

### Benchmark failure modes

- Very small outputs, specifically values within `(-0.01, 0.01)`, are vulnerable to weak signal relative to numerical tolerances. A nominally passing result may exploit the checker rather than reproduce a useful computation. [Task Filtering](../pdfs/21-measuring-automated-kernel-engineering.md)
- Outputs that barely vary with input seeds can enable caching or constant-output solutions; uniformity across tensor axes can similarly make tasks unrealistically easy. [Task Filtering](../pdfs/21-measuring-automated-kernel-engineering.md)
- Algebraically constant expressions such as `mean(softmax(x))`, saturated or inactive activations, and high-dimensional reductions with concentrated outputs are examples of problematic task distributions. These mechanisms are distinct from illegal code execution or malicious harness tampering. [Task Filtering](../pdfs/21-measuring-automated-kernel-engineering.md)
- State retained by an RNN between calls changes the semantics of repeated benchmark evaluation; METR modifies such tasks to accept state explicitly. [Task Filtering](../pdfs/21-measuring-automated-kernel-engineering.md)
- Asynchronous CUDA streams can invalidate timing when synchronization is missing; memory reuse by an uninitialized allocation can leak a previously computed reference output. Both require attention beyond ordinary numerical comparison. [Model solution filtering](../pdfs/21-measuring-automated-kernel-engineering.md)

## 3. Methodology and experimental setup

### Benchmark construction and measurement

- The article describes the starting suite as 270 tasks over four levels, with 250 in Levels 1–3 and 20 in Level 4. It reports removing 45 of the 250 tasks before adding 14 Level-5 tasks. There is a count discrepancy with the introductory image, documented below rather than silently resolved. [Introduction; Task Filtering](../pdfs/21-measuring-automated-kernel-engineering.md)
- Filtering tests output scale, variability across seeds and uniformity across axes; some tasks are repaired instead of discarded by varying weights or input variance. [Task Filtering](../pdfs/21-measuring-automated-kernel-engineering.md)
- The retained architecture tests run on a single H100. Larger Level-5 architectures are reduced to fit a single GPU, and initial implementations are intentionally somewhat naive, often relying on `torch.scaled_dot_product_attention`. [Introduction; More details of KernelBench Level 5](../pdfs/21-measuring-automated-kernel-engineering.md)
- Timing switches from the default KernelBench procedure to `triton.testing.do_bench`. The article presents this as an improvement in measurement reliability; it does not establish that every timing confound is removed. [Improving the KernelBench benchmark](../pdfs/21-measuring-automated-kernel-engineering.md)
- Seventeen model solutions using CUDA streams are excluded because the evaluation cannot reliably time them without explicit synchronization. One additional extremely fast result is removed after inspection; the suspected cause is `torch.empty` reusing reference-output memory. [Model solution filtering](../pdfs/21-measuring-automated-kernel-engineering.md)
- The correctness check described in the limitations uses absolute tolerance 0.01. The article explicitly distinguishes this from production accuracy requirements and downstream model/training evaluation. [Remaining limitations](../pdfs/21-measuring-automated-kernel-engineering.md)
- Level 4 is not evaluated because its tasks load model implementations through Hugging Face rather than exposing their internals; fair evaluation would require a scaffold capable of traversing those codebases. Exclusion is thus a scaffold/scope decision, not evidence that the tasks are impossible. [Task Filtering](../pdfs/21-measuring-automated-kernel-engineering.md)

### Search, models and comparison protocols

- Eight attempts run concurrently. When an attempt finishes, a completed candidate or the original reference is selected based on speedup to seed another attempt. [Our Kernel Agent and Results](../pdfs/21-measuring-automated-kernel-engineering.md)
- Most models receive 300 total attempts per problem; o1 receives 40 because it is described as eight times more expensive. The comparison therefore is not equal in number of generation attempts. [Our Kernel Agent and Results](../pdfs/21-measuring-automated-kernel-engineering.md)
- Models can produce PyTorch, Triton or CUDA and receive language-specific Triton and CUDA documentation. External verification supplies the signal for retaining improved candidates. [Our Kernel Agent and Results](../pdfs/21-measuring-automated-kernel-engineering.md)
- Comparisons include GPT-4o, Claude 3.5 Sonnet and o1 from the original KernelBench experiments, plus o3-mini, which was not in the original baseline. Baseline solutions are reselected on METR's modified task set. [Introduction; footnote 2](../pdfs/21-measuring-automated-kernel-engineering.md)
- The article also uses best-of-k over PyTorch compilation variants: `torch.compile`, `torch.jit_script`, CUDA graphs, autocasting to FP16/BF16, and explicit BF16 conversions. Because these options change precision as well as execution strategy, their results belong to the stated tolerance regime. [Introduction; footnote 3](../pdfs/21-measuring-automated-kernel-engineering.md)
- DeepSeek-R1 is excluded for latency, self-hosting expense and provider-reliability reasons. Its absence is not a reported negative performance result. [Footnote 4](../pdfs/21-measuring-automated-kernel-engineering.md)
- Cost comparisons include both model inference and GPU execution of candidate tests. The article gives approximately $20 per task per model in the headline comparison and mentions up to $35 per agent elsewhere. These are contextual figures, not a single universal per-task charge. [Introduction; Our Kernel Agent and Results; Takeaways](../pdfs/21-measuring-automated-kernel-engineering.md)

### Level 5, fine-tuning and qualitative examples

- The detailed Level-5 list contains S4, RWKV and Mamba2; Llama2 and Llama2 decode; five DeepSeek-V3 workloads covering MoE, MLA and a single-GPU model; three Hunyuan Video components; and Stable Diffusion 3 MMDiT. These sum to 14. [More details of KernelBench Level 5](../pdfs/21-measuring-automated-kernel-engineering.md)
- The shorter earlier category list mentions two “Llama 3” tasks, whereas the detailed list names Llama2; it also omits Stable Diffusion from its four category counts. Keep the named task inventory and source discrepancy visible when reconstructing the suite. [Model solution filtering; More details of KernelBench Level 5](../pdfs/21-measuring-automated-kernel-engineering.md)
- GPT-4o is fine-tuned on 1,200 best solutions from Levels 1–3, then evaluated on Level 5. This is a separate weight-updating experiment, unlike the main inference-time tree-search procedure. [Fine-Tuning GPT-4o](../pdfs/21-measuring-automated-kernel-engineering.md)
- To characterize task composition, o1 labels whether problems are elementwise, a single PyTorch op, convolutional, matrix-multiplication-based, a loop over layers, attention-based, or composed of submodules. These classifications are model-produced annotations. [Components of KernelBench problems by level](../pdfs/21-measuring-automated-kernel-engineering.md)

## 4. Results and evidence

### Headline effectiveness

- On the modified task suite, the best portfolio of GPT-4o, Sonnet and o1 yields 1.81× geometric-mean speedup, compared with 1.05× for the original KernelBench best-of-k comparison described by the article. The original 1.05× comparator selects best-of-k across all seven original models, whereas the 1.81× portfolio uses those three overlapping models. Scaffolding, prompting, sampling budget and task filtering are all part of this setting. [Introduction, model-progression figure caption; footnote 2](../pdfs/21-measuring-automated-kernel-engineering.md)
- The article calls this improvement “15 times higher.” This is not the ratio `1.81/1.05`; it appears to concern improvement above 1, and even `(1.81−1)/(1.05−1)` is about 16.2 using rounded numbers. Preserve the actual 1.81× and 1.05× results rather than propagate an ambiguous multiplier. [Introduction, headline results paragraph](../pdfs/21-measuring-automated-kernel-engineering.md)
- o3-mini-high alone is reported at 1.81×; choosing the best candidate across all agents per task yields 2.01×. These are different aggregation procedures and should not be conflated. [Introduction](../pdfs/21-measuring-automated-kernel-engineering.md)
- All evaluated agents improve between 69% and 95% of problems by more than 2%. The 2% cutoff is chosen to avoid counting changes that could be timing noise. [Speedup distributions](../pdfs/21-measuring-automated-kernel-engineering.md)
- Best-of-k across all agents achieves more than 5% speedup on 93% of problems; the article reports 6.5× speedup on 5.4% and a maximum of 30× on an individual task. These describe the distribution, not a 30× average improvement. [Speedup distributions](../pdfs/21-measuring-automated-kernel-engineering.md)
- After best-of-k selection, 80% of retained solutions use Triton or CUDA. The article says the best low-level-language solutions outperform the best PyTorch-only solutions, but this observational comparison does not isolate language choice from which tasks or candidates were selected. [Language choices for model-written solutions](../pdfs/21-measuring-automated-kernel-engineering.md)
- Solutions get longer at higher task levels; the article reports an average of about 500 lines for Level 5. Length is a descriptive complexity proxy, not a direct correctness or maintainability metric. [Language choices for model-written solutions](../pdfs/21-measuring-automated-kernel-engineering.md)

### Search scaling and economics

- Performance continues improving at 200–300 attempts in the reported runs. The article characterizes the relationship as smooth power-law scaling; its footnote describes approximately linear log-speedup versus number of attempts, so the precise scaling law is not formally resolved in the text. [Our Kernel Agent and Results; footnote 5](../pdfs/21-measuring-automated-kernel-engineering.md)
- o3-mini has the best cost-performance among the tested models at the prices used. The article explicitly warns that model-price changes can alter the ranking. This is a historical experimental result, not a current purchasing recommendation. [Our Kernel Agent and Results; footnote 5](../pdfs/21-measuring-automated-kernel-engineering.md)
- GPU testing represents 27% of the GPT-4o agent's cost and 2.5% of the o1 agent's cost. Thus expensive model calls and GPU evaluation can dominate in different regimes. [Our Kernel Agent and Results](../pdfs/21-measuring-automated-kernel-engineering.md)
- The article gives an average o3-mini break-even of 27 H100 compute hours, conditional on tested results being representative. A nearby broader estimate says at least 20 hours; retain the model-specific 27-hour figure and its assumptions when answering detailed economic questions. [Our Kernel Agent and Results](../pdfs/21-measuring-automated-kernel-engineering.md)
- Approximately four engineer-weeks were spent constructing the project. The authors interpret this as modest relative to potential optimization value; the study does not experimentally estimate the marginal return to further scaffold engineering. [Introduction; Takeaways](../pdfs/21-measuring-automated-kernel-engineering.md)

### Transfer and code examples

- Fine-tuning GPT-4o on 1,200 Level-1–3 solutions is reported to close almost all the gap to o3-mini on Level 5. The article explicitly qualifies this with small sample size and the need for stronger measurement. [Fine-Tuning GPT-4o](../pdfs/21-measuring-automated-kernel-engineering.md)
- Performance on a subset described as four DeepSeek questions remains substantially worse than on the rest of Level 5. This wording differs from the five DeepSeek tasks in the full task inventory; do not infer which task is omitted without additional evidence. [Fine-Tuning GPT-4o; More details of KernelBench Level 5](../pdfs/21-measuring-automated-kernel-engineering.md)
- The Level-1 sample is a CUDA argmin reduction labeled 2.19×: it groups eight columns into a block and uses warp-level reductions. Its code specializes to the benchmark dimension and shape, so it is an illustration of task-specific optimization rather than a general argmin implementation. [Best solutions, Level 1 index 52 listing](../pdfs/21-measuring-automated-kernel-engineering.md)
- The Level-2 sample, labeled 1.36×, retains the PyTorch convolution and fuses subtraction plus Mish into a vectorized CUDA kernel with fast-math approximations. It illustrates partial replacement rather than rewriting every operation. [Best solutions, Level 2 index 87 listing](../pdfs/21-measuring-automated-kernel-engineering.md)
- The Level-3 sample retains `nn.MultiheadAttention`, implements layout transforms and residual-plus-normalization CUDA kernels, and is labeled 1.00×. Its existence shows that substantial generated low-level code need not improve aggregate runtime. [Best solutions, Level 3 index 31 listing](../pdfs/21-measuring-automated-kernel-engineering.md)
- The Level-5 sample is a long model implementation with MLA, rotary embeddings, KV caches, routing, experts and BF16-related configuration, labeled 1.09× from fine-tuned GPT-4o. The listing is illustrative; this analysis does not certify its correctness outside the evaluation. [Best solutions, Level 5 index 3 listing](../pdfs/21-measuring-automated-kernel-engineering.md)

## 5. Authors' conclusions

- Better scaffolding and more sampling can reveal considerably stronger kernel-optimization performance than minimal one-shot or low-budget evaluations suggest. [Takeaways: Current AI agents...](../pdfs/21-measuring-automated-kernel-engineering.md)
- Overall speedup is presented as a useful complement to correctness rates, especially when a deployment can retain the original implementation on failed optimization attempts. [Takeaways: Current AI agents...](../pdfs/21-measuring-automated-kernel-engineering.md)
- The authors believe small or independent ML projects may benefit economically because model-driven optimization can amortize over far less compute than human expert optimization. This is their deployment inference from the cost results. [Our Kernel Agent and Results](../pdfs/21-measuring-automated-kernel-engineering.md)
- The results explicitly do not establish automation of professional kernel engineering. The authors report being unable to adapt FlashAttention 2 to new constraints and say their kernels are less novel and sophisticated than expert work. [Takeaways: Our results do not imply...](../pdfs/21-measuring-automated-kernel-engineering.md)
- The authors argue that more realistic evaluations remain difficult and that available benchmark scores may underestimate elicitable capabilities, while simultaneously overrepresenting easy optimization opportunities relative to frontier-lab workloads. [Takeaways; Remaining limitations](../pdfs/21-measuring-automated-kernel-engineering.md)
- Their impact assessment favors publishing measurements and benchmark artifacts but withholds the agent implementation; they expect the main benefit to be acceleration of algorithmic work outside frontier labs. These are policy judgments, not experimentally established effects. [Impact Statement](../pdfs/21-measuring-automated-kernel-engineering.md)

## 6. Limitations and open questions

### Explicit source limitations

- There is no matched human baseline, so success cannot be translated into “equivalent to an expert” or a measured fraction of expert capability. [Remaining limitations](../pdfs/21-measuring-automated-kernel-engineering.md)
- Fixed shapes, single-GPU execution and inference-only tasks omit important deployment requirements, communication and training stability. [Remaining limitations](../pdfs/21-measuring-automated-kernel-engineering.md)
- Absolute-tolerance checking can permit numerically inappropriate kernels; the article calls for considering downstream model performance and training stability. [Remaining limitations](../pdfs/21-measuring-automated-kernel-engineering.md)
- Filtering is incomplete and some retained tasks may still be flawed; conversely, removed tasks may be repairable rather than intrinsically useless. [Remaining limitations](../pdfs/21-measuring-automated-kernel-engineering.md)
- Output-token limits and inadequate context factoring are proposed barriers to sophisticated expert-level kernel development. This is an author hypothesis based on observed limitations, not a demonstrated impossibility theorem. [Takeaways: Our results do not imply...](../pdfs/21-measuring-automated-kernel-engineering.md)
- The lack of released agent code limits exact reproduction of search, prompt tuning and selection details. [Our Kernel Agent and Results; Impact Statement](../pdfs/21-measuring-automated-kernel-engineering.md)

### Analyst caveats and unresolved source inconsistencies

- **Analyst caveat:** improved performance is not causally attributable to one scaffold component, because budget, prompts, candidate reuse, task selection and measurement differ together. A component ablation or matched-budget comparison would be required. [Basis: Introduction; Our Kernel Agent and Results](../pdfs/21-measuring-automated-kernel-engineering.md)
- **Analyst caveat:** the fallback-to-reference metric hides the frequency and magnitude of slow or incorrect generated candidates; retain candidate correctness and latency distributions alongside the final best-of-k score. [Basis: metric definition](../pdfs/21-measuring-automated-kernel-engineering.md)
- **Count discrepancy:** the introductory image lists 90, 80 and 37 problems in Levels 1–3, totaling 207; the prose's `250 − 45` implies 205. Adding Level 5 yields 221 versus 219 respectively. The article alone does not reconcile these denominators. [Task Filtering](../pdfs/21-measuring-automated-kernel-engineering.md); [task table image](../pdfs/assets/21-measuring-automated-kernel-engineering/figure-001.png)
- **Inventory discrepancy:** earlier Llama-3 wording differs from the detailed Llama2 names, and the abbreviated category counts omit the Stable Diffusion task. Use the exact released task inventory if reconstructing the experiment. [Model solution filtering; More details of KernelBench Level 5](../pdfs/21-measuring-automated-kernel-engineering.md)
- **Analyst caveat:** the headline “15 times” and the scaling-law description should not be treated as precise derived quantities because the nearby formulas/rounded data do not uniquely support those formulations. [Introduction; footnote 5](../pdfs/21-measuring-automated-kernel-engineering.md)

## 7. Relevance to this project — interpretation

- **Proposal:** store a full candidate history, not only the winning speedup, so failed correctness checks, slow valid kernels and changes in search budget remain visible. This follows from the selection and flooring behavior of METR's metric, not from a claim that our harness already does so. [Basis: Introduction metric; Our Kernel Agent and Results](../pdfs/21-measuring-automated-kernel-engineering.md)
- **Proposal:** include input-dependence tests, scale perturbations, tensor-axis variability checks and explicit model-state handling when screening tasks. [Basis: Task Filtering](../pdfs/21-measuring-automated-kernel-engineering.md)
- **Proposal:** include synchronization-aware timing and tests that prevent candidate access to reference-output memory; extremely large speedups merit a separate validity audit. [Basis: Model solution filtering](../pdfs/21-measuring-automated-kernel-engineering.md)
- **Proposal:** compare parallel restarts, parent-conditioned search and fixed-prompt baselines under both equal-dollar and equal-attempt budgets. METR motivates these comparisons but does not establish which search component is causally best. [Basis: Our Kernel Agent and Results](../pdfs/21-measuring-automated-kernel-engineering.md)
- **Proposal:** distinguish research-scale naive-reference optimization from production-library optimization in the task taxonomy and in any reported headline. [Basis: Remaining limitations; Takeaways](../pdfs/21-measuring-automated-kernel-engineering.md)
- **Proposal:** treat fine-tuning on successful search trajectories as a separate experiment with held-out architectures and adequate samples; do not call ordinary candidate search persistent model learning. [Basis: Fine-Tuning GPT-4o](../pdfs/21-measuring-automated-kernel-engineering.md)

## 8. Retrieval notes

- Search aliases: METR KernelAgent, KernelBenchFiltered, Level 5, best-of-k kernel engineering, parallel tree search, H100 break-even, low-variance outputs, CUDA-stream timing.
- Useful questions: What task pathologies did METR filter? How does its speedup metric handle failure? How many attempts did each model receive? What was released? Why does its result not establish expert automation? What assumptions underlie 27-hour break-even?
- Related analyses for navigation: [KernelBench](01-kernelbench.md) for the starting benchmark; [robust agentic CUDA](08-robust-agentic-cuda.md) for additional correctness-focused benchmark construction; [KernelBench-Verified](03-kernelbench-verified.md) for another baseline/correctness protocol. These links are navigation, not a controlled cross-study comparison.

## 9. Source coverage and verification notes

- Read the full local article, including the task construction, agent/results, takeaways, impact statement, additional-results sections, Level-5 inventory, component taxonomy, all four illustrative code listings and footnotes. Detailed implementation listings are summarized by mechanism, not reproduced or independently executed.
- Visually inspected the introductory task-table image to recover counts, qualitative realism labels and expert-time estimates absent from the Markdown text; this exposed the 207-versus-205 discrepancy. Other exact quantitative findings above come from article prose, not estimated chart coordinates.
- The source is a saved web snapshot, not a versioned archival manuscript. All numerical claims retain the article's model/hardware/budget context; no current pricing or capability claims are inferred.
- No benchmark reruns, source-repository audit, independent correctness proof, or human-baseline study were performed for this analysis. The paper's experimental conclusions remain reported evidence.
