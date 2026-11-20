---
paper_id: "06"
title: "MultiKernelBench: A Multi-Platform Benchmark for Kernel Generation"
source_version: "2507.17773v2"
source_markdown: "../pdfs/06-multikernelbench.md"
source_pdf: "../pdfs/06-multikernelbench.pdf"
source_url: "https://arxiv.org/abs/2507.17773"
analysis_date: "2026-11-20"
analysis_status: "reviewed"
topics: ["multi-platform", "gpu-kernels", "ascendc", "pallas", "benchmarking", "category-aware-prompting", "correctness", "hardware-portability"]
summary: "MultiKernelBench evaluates 285 PyTorch-defined tasks on CUDA, AscendC, and Pallas, exposing platform/category differences and testing category-specific exemplars."
---

# MultiKernelBench: A Multi-Platform Benchmark for Kernel Generation

## Paper identity and scope

- This note covers arXiv 2507.17773v2, a benchmark and prompting study, rather than a trained kernel-generation system. [§1; §7](../pdfs/06-multikernelbench.md)
- The suite extends KernelBench's 250 tasks to 285 tasks across 14 functional categories and three language/platform pairs: NVIDIA CUDA, Huawei AscendC, Google TPU Pallas. [§3.3; Table 1](../pdfs/06-multikernelbench.md)
- Seven model configurations are evaluated, including reasoning and non-reasoning modes; 285 tasks across three platforms form 855 task-platform instances in the greedy comparison. [§4.1; §4.2; Table 4](../pdfs/06-multikernelbench.md)
- The benchmark explicitly aims to separate platform-specific execution infrastructure from the task definition and core evaluation logic. [§3.6](../pdfs/06-multikernelbench.md)

## Concepts and terminology

- A task is a PyTorch reference module plus its input-tensor specification; the generated replacement must expose a compatible module that invokes a custom kernel. [§3.1](../pdfs/06-multikernelbench.md)
- Functional categories capture computational patterns or DL roles, unlike a single broad difficulty-level ordering. [§3.3](../pdfs/06-multikernelbench.md)
- CUDA exposes thread/block/grid hierarchy and memory-space management; AscendC exposes scalar/vector/cube computation and explicit transfer/queue management. [§2.1–2.2](../pdfs/06-multikernelbench.md)
- Pallas expresses kernels in Python through JAX, with pallas_call and compiler lowering; concise syntax shifts rather than eliminates hardware constraints. [§2.2; §4.2](../pdfs/06-multikernelbench.md)
- Compilation@k is the fraction of tasks with at least one compilable sample among k; Pass@k requires at least one sample passing all correctness tests. [§4.1](../pdfs/06-multikernelbench.md)
- SpeedUp_alpha@k is the fraction of tasks with at least one sample meeting an alpha-fold runtime improvement over the baseline; SU_1 measures speed-winning task coverage rather than average speedup magnitude. [§4.1; Table 4](../pdfs/06-multikernelbench.md)
- The experiment evaluates functional correctness before performance; compilation success alone is a lower-level validity signal, with different interpretation for compiled C-like languages and Python. [§3.5; §4.2](../pdfs/06-multikernelbench.md)
- Greedy k=1 and sampling k=5 are separate decoding settings, not an unbiased pass@1 estimate from the same sample pool. [§4.1](../pdfs/06-multikernelbench.md)
- Category-aware one-shot prompting supplies one relevant example from the target functional category; the default supplies a vector-add example across categories. [§3.2; §4.4](../pdfs/06-multikernelbench.md)
- Kernel fusion reduces intermediate storage/transfers and launch overhead; structured-input specialization can exploit mathematical sparsity instead of invoking general-purpose routines. [§4.3](../pdfs/06-multikernelbench.md)

## Methodology and experimental setup

- Taxonomy construction begins by classifying all 250 KernelBench tasks, reviewing PyTorch's native CUDA library for gaps, and refining definitions with three developers having three years of kernel experience each. [§3.3](../pdfs/06-multikernelbench.md)
- The authors prune sparse/low-impact categories by expert consensus, explicitly excluding comparison operations as too simple/low impact for their evaluation goals. [§3.3](../pdfs/06-multikernelbench.md)
- New categories are Broadcast, Resize, and Optimizer; Index expands from two tasks. Added tasks are manually refined and verified. [§3.3](../pdfs/06-multikernelbench.md)
- Task counts are Activation 15, Broadcast 10, Convolution 34, Full Architecture 50, Fusion 100, Loss 7, Math 6, Matrix Multiply 17, Normalization 8, Optimizer 5, Pooling 6, Index 12, Resize 10, Reduce 5. [Table 1](../pdfs/06-multikernelbench.md)
- The largest categories remain Fusion 100 and Full Architecture 50: finer categorization does not imply equal category weighting. [Table 1](../pdfs/06-multikernelbench.md)
- Prompts combine platform-specific instructions, one-shot output-format guidance, and the same PyTorch target task across platforms. [§3.2](../pdfs/06-multikernelbench.md)
- CUDA output embeds C++/CUDA in Python and builds through cpp_extension; Pallas output includes kernel, pallas_call, JAX, and PyTorch integration. [§3.2](../pdfs/06-multikernelbench.md)
- AscendC output has separate host, tiling, device, and PyTorch-module components; glue scripts construct files and run the platform build pipeline. [§3.2; §3.4](../pdfs/06-multikernelbench.md)
- CUDA/Pallas code is extracted and dynamically executed; CUDA device assertions are enabled for failure diagnosis. [§3.4](../pdfs/06-multikernelbench.md)
- Correctness uses five randomly generated inputs with elementwise absolute difference below atol+rtol×abs(reference), setting atol=rtol=1e-2. [§3.5](../pdfs/06-multikernelbench.md)
- Authors report five-input testing usually agrees with 100-input testing and reproduce that observation on AscendC/Pallas; this is empirical agreement under their sampling, not a proof of correctness. [§3.5](../pdfs/06-multikernelbench.md)
- The relatively permissive tolerance is intended to accommodate reduced-precision strategies such as BF16/FP8. [§3.5](../pdfs/06-multikernelbench.md)
- CUDA and AscendC timing use synchronization markers; Pallas uses debugging tools because equivalent markers are unavailable in the interface used. [§3.5](../pdfs/06-multikernelbench.md)
- Authors report cross-time stability checks on the same machines, but do not provide a universal repetition count, numerical variance bound, or exact Pallas profiler procedure in the text. [§3.5](../pdfs/06-multikernelbench.md)
- Backend subclasses encapsulate device identification/initialization, compilation/execution, correctness/timing, and cleanup/memory management; registered names select them dynamically. [§3.6](../pdfs/06-multikernelbench.md)
- Experiments use NVIDIA L20, Huawei Ascend 910B2, and Google TPU v2-8; platform results are measurements on these particular devices/software integrations. [§4.1; Table 2](../pdfs/06-multikernelbench.md)
- Model configurations are DeepSeek-V3-0324, Qwen3-235B non-thinking, Qwen2.5-Coder-32B, GPT-4o, Claude-Sonnet 4, DeepSeek-R1-0528, and Qwen3-235B thinking. [Table 3](../pdfs/06-multikernelbench.md)
- The reference baseline is PyTorch eager; Dynamo and other compilation modes are excluded because of platform-dependent behavior. [§4.1](../pdfs/06-multikernelbench.md)
- Greedy uses temperature 0; five-sample decoding uses temperature 0.2/top-p 0.95. Reasoning models do not receive five-sample experiments because of inference cost. [§4.1](../pdfs/06-multikernelbench.md)
- Category-aware experiments collect one official/open-source exemplar for each of Activation, Loss, Matrix Multiply, Normalization, and Reduce on AscendC/Pallas. [§4.4](../pdfs/06-multikernelbench.md)
- These five categories contain 52 tasks in total, calculated from Table 1. Relative improvements in Tables 8–9 are consistent with this subset, not a rerun of all 285 tasks. [Table 1; Tables 8–9; §4.4](../pdfs/06-multikernelbench.md)
- Category-specific examples are selected partly because clear, concise implementations are available; this is a selected-category intervention rather than random example selection. [§4.4](../pdfs/06-multikernelbench.md)

## Results and evidence

- Default greedy total solved task-platform instances are Claude 164/855, R1 162/855, Qwen-thinking 149/855, GPT-4o 89/855, V3 81/855, Qwen 77/855, QwenCoder 63/855. [Table 4(a)](../pdfs/06-multikernelbench.md)
- On CUDA greedy, R1 leads Pass@1 at 52.6% and SU_1@1 at 26.0%; Claude records 47.0% and 20.4%, Qwen-thinking 44.2% and 19.3%. [Table 4(a)](../pdfs/06-multikernelbench.md)
- GPT-4o has the highest CUDA greedy compilation 97.5%, but only 23.2% correctness and 5.3% speed-winning coverage: syntax success and useful-kernel success differ strongly. [Table 4(a)](../pdfs/06-multikernelbench.md)
- On CUDA five-sample evaluation, Claude reaches 55.8% Pass@5 and 26.0% SU_1@5; Qwen reaches 43.9% and 13.0%, V3 35.1% and 13.7%. [Table 4(b)](../pdfs/06-multikernelbench.md)
- Claude's five-sample total rises to 200/855, while other sampled models range 113–149; five samples leave most task-platform instances unsolved. [Table 4(b)](../pdfs/06-multikernelbench.md)
- AscendC greedy correctness is only 0.7–2.5% across models; V3 leads at 2.5%. Claude's five-sample correctness reaches 3.9% and SU_1@5 reaches 2.1%. [Table 4](../pdfs/06-multikernelbench.md)
- Pallas greedy compilation spans 70.9–99.6%, but correctness only 2.8–8.4%; Claude leads correctness 8.4% and SU_1@1 7.7%. [Table 4(a)](../pdfs/06-multikernelbench.md)
- All five sampled models attain 100% Pallas compilation except V3 at 97.9%, while the highest Pass@5 is Claude 10.5%. [Table 4(b)](../pdfs/06-multikernelbench.md)
- R1's strong CUDA result does not transfer to Pallas: its 2.8% greedy correctness is lowest of the seven configurations. [Table 4(a)](../pdfs/06-multikernelbench.md)
- For the three selected top CUDA models, category-average Pass@1 is Activation 88.9%, Reduce 86.7%, Broadcast 83.3%, versus Convolution 13.7% and Full Architecture 21.3%. [Table 5](../pdfs/06-multikernelbench.md)
- CUDA Matrix Multiply has 74.5% average correctness, but each selected model wins runtime on only 5.9% of the 17 tasks; functional competence is not competitive GEMM performance. [Table 5](../pdfs/06-multikernelbench.md)
- Claude's CUDA Convolution compilation is 100% but correctness 0%; its Activation correctness is 100%, demonstrating task-family failure hidden by a common broad level. [Table 5](../pdfs/06-multikernelbench.md)
- CUDA Fusion accuracy is R1 59%, Claude 50%, Qwen-thinking 49%, with speed-winning fractions 44%, 37%, 32%; fusion is a substantial source of useful speed improvements. [Table 5](../pdfs/06-multikernelbench.md)
- Under default AscendC prompting, the displayed models succeed only in Activation and Broadcast; their Activation mean correctness is 35.6%, Broadcast 6.7%, all other categories 0%. [Table 6](../pdfs/06-multikernelbench.md)
- Default Pallas success is concentrated in Activation, Broadcast, and a few Full Architecture/Fusion/Loss/Normalization cases; Reduce and Matrix Multiply have zero correctness for the selected models. [Table 7; §4.3](../pdfs/06-multikernelbench.md)
- Among classified CUDA compiled-kernel failures, output mismatch accounts for 53.0%, shape mismatch 16.1%, and CUDA runtime errors 15.9%; these are failure-subset shares, not all-task probabilities. [§4.2](../pdfs/06-multikernelbench.md)
- AscendC compilation failures contain 'no member named' in 74.8% of cases; the illustrated softmax attempt calls an unavailable AscendC: : Softmax API. [§4.2; Figure 3](../pdfs/06-multikernelbench.md)
- Pallas failure analysis reports Unexpected Keyword Argument 22.4% and rank-at-least-one lowering errors 5%, reflecting hallucinated API keywords and unsupported scalar intermediates/outputs. [§4.2](../pdfs/06-multikernelbench.md)
- For very short activation kernels around 2 microseconds, authors explicitly acknowledge that small speed wins may reflect timing noise or reduced framework overhead. [§4.3](../pdfs/06-multikernelbench.md)
- The diagonal-matmul example replaces a general matrix multiply with row-wise diagonal scaling, exploiting input structure; the Pallas example fuses max/subtract-mean/GELU after GEMM. [§4.3; Figures 4–5](../pdfs/06-multikernelbench.md)
- In AscendC category-aware prompting, GPT-4o's subset compilation improves 380% relatively and correctness 160%; Claude's correctness improves 133.3% and V3's 100%. [Table 8](../pdfs/06-multikernelbench.md)
- AscendC Matrix Multiply correctness rises from 0% to GPT-4o 35.3%, Claude 23.5%, V3 23.5%, but all three still have 0% SU_1 on that category. [Table 8](../pdfs/06-multikernelbench.md)
- AscendC Normalization and Reduce remain 0% correct with in-category examples; the authors trace some Reduce errors to applying a 2D example incorrectly to 3D inputs. [§4.4; Table 8](../pdfs/06-multikernelbench.md)
- Pallas category-aware subset correctness improves relatively 145.5% for Qwen-thinking, 155.6% for V3, 133.3% for Qwen. [Table 9](../pdfs/06-multikernelbench.md)
- Pallas Matrix Multiply reaches 41.2/35.3/35.3% correctness for Qwen-thinking/V3/Qwen but remains 0% speed-winning coverage; Reduce reaches 80/40/20% correctness. [Table 9](../pdfs/06-multikernelbench.md)
- Improved correctness need not improve every speed metric: Qwen-thinking Pallas Activation correctness rises 66.7 → 73.3%, while SU_1 falls 66.7 → 40%. [Tables 7,9](../pdfs/06-multikernelbench.md)
- Pallas Qwen-thinking's overall selected-subset SU_1 relative improvement is 0%, despite 145.5% correctness improvement; prompt quality and speed quality require separate tracking. [Table 9](../pdfs/06-multikernelbench.md)

## Authors’ conclusions

- Authors conclude current models are highly sensitive to both programming platform and functional category, motivating multi-platform, fine-grained benchmark coverage. [§7](../pdfs/06-multikernelbench.md)
- They attribute weaker AscendC/Pallas results mainly to low representation in training corpora and API unfamiliarity; this is explanatory interpretation, not directly measured pretraining exposure. [§4.2](../pdfs/06-multikernelbench.md)
- They interpret reasoning-model strength and the smaller QwenCoder's weakness as evidence that reasoning and scale matter, although model families/training differ simultaneously. [§4.2](../pdfs/06-multikernelbench.md)
- They conclude category-aware examples provide useful domain-specific patterns, especially where generic add demonstrations fail. [§4.4](../pdfs/06-multikernelbench.md)
- They propose shape-aware prompting or finer shape-based categories after failures to generalize 2D examples to 3D tasks. [§4.4](../pdfs/06-multikernelbench.md)
- They suggest concise high-level DSLs and compiler-managed optimization can make kernel coding easier for LLMs, while presenting the modular backend as extensible infrastructure. [§3.6; §4.4; §7](../pdfs/06-multikernelbench.md)

## Limitations and open questions

- Explicit construction threat: reference and exemplar correctness may affect results; authors manually review added kernels and run compilation/execution tests. [§5](../pdfs/06-multikernelbench.md)
- Explicit generalization threats include untested models/platforms/languages; AMD/HIP and broader Triton integration are future targets, with the claimed Triton backend needing fewer than 20 extra lines. [§5](../pdfs/06-multikernelbench.md)
- Analyst caveat: three platforms simultaneously change hardware, language, integration, maturity, and timing method; platform gaps do not isolate a language effect. [§3.5; §4.1](../pdfs/06-multikernelbench.md)
- Analyst caveat: five random tests with 1e-2 tolerances cannot establish edge-case robustness, numerical equivalence, or portability to unseen shapes/distributions. [§3.5](../pdfs/06-multikernelbench.md)
- Analyst caveat: only eager baselines are compared; SU_1 success does not imply beating optimized compiled production libraries or full inference pipelines. [§4.1](../pdfs/06-multikernelbench.md)
- Analyst caveat: functional categories remain imbalanced, and category result tables show selected top-three models, sometimes different models per platform/setting. [Tables 1,5–9; §4.3–4.4](../pdfs/06-multikernelbench.md)
- Analyst caveat: category-aware selection covers only five available-example categories; no matched random-example or token-length control isolates why examples help. [§4.4](../pdfs/06-multikernelbench.md)
- Analyst caveat: the study does not report multi-turn agent repair, fine-tuning, hidden adversarial tests, or statistical confidence intervals for its core comparisons. [§4.1–4.4](../pdfs/06-multikernelbench.md)
- Source inconsistency: §4.2 calls Qwen3-235B a top-performing Pallas model despite Table 4 showing Claude higher; preserve table-specific rankings. [§4.2; Table 4](../pdfs/06-multikernelbench.md)
- Source metadata/prose inconsistencies include model-size mentions 681 B versus Table 3's 685 B, and Table 2 labeling all memory 'HBM'; hardware descriptions should not replace manufacturer specifications. [§1; §4.1; §5; Tables 2–3](../pdfs/06-multikernelbench.md)

## Relevance to this project — interpretation

- Project interpretation: retain per-platform/per-category correctness and speed-winning coverage alongside any aggregate, since aggregate CUDA accuracy hides specific failure families.
- Project interpretation: store exemplar provenance, category, shape, and interface constraints to support controlled prompt retrieval rather than indiscriminate example reuse.
- Project interpretation: backend adapters should expose identical semantic contracts while recording different timing/error semantics explicitly.
- Project interpretation: investigate mathematically justified structured-input specialization separately from invalid dependence on a test distribution; the diagonal example is a useful case study.
- Project interpretation: these are proposed uses of the paper, not features implemented or causal effects established by this analysis.

## Retrieval notes

- Aliases: MultiKernelBench; MKB; category-aware one-shot; CUDA/AscendC/Pallas; NPU/TPU kernel generation.
- Answers questions about platform portability, category difficulty, five-sample evaluation, eager-baseline coverage, exemplar selection, and shape-generalization failure.
- Related: [KernelBench](01-kernelbench.md) supplies the original 250-task base and joint correctness/speed framing; [TritonBench](05-tritonbench.md) covers a different DSL-specific benchmark and retrieval design.
- Related navigation: [KernelBench-X](02-kernelbench-x.md) studies a different multi-device benchmark; its suite/metrics are not interchangeable with this paper's three-language setting.

## Source coverage and verification notes

- Coverage: full main text §§1–7, Tables 1–9, all figure discussions, and source references consulted for context; this version has no substantive appendix. [§§1–7](../pdfs/06-multikernelbench.md)
- Figures 1, 3, 4, 5 have missing code content in Markdown conversion; their PDF text was additionally inspected. Figure 4's diagonal scaling and Figure 5's fused reductions/GELU were verified against that PDF content.
- Table numbers here refer to the linked Markdown's numbered tables. The CUDA figure reference in Markdown §2.2 incorrectly points to 1(b), whereas PDF text labels CUDA 1(a); this is a source/conversion discrepancy.
- The source PDF contains conference-template placeholder metadata; this analysis identifies the frozen arXiv version rather than inferring a publication venue.
- Numerical extraction preserves source rounding. Relative improvements were checked against selected-category task counts; no kernels, backend integrations, or timing measurements were independently reproduced.
