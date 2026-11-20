---
paper_id: "07"
title: "ISO-Bench: Can Coding Agents Optimize Real-World Inference Workloads?"
source_version: "2602.19594v1"
source_markdown: "../pdfs/07-iso-bench.md"
source_pdf: "../pdfs/07-iso-bench.pdf"
source_url: "https://arxiv.org/abs/2602.19594"
analysis_date: "2026-11-20"
analysis_status: "reviewed"
topics: ["repository-optimization", "inference-serving", "vllm", "sglang", "coding-agents", "llm-as-judge", "correctness", "agent-scaffolding", "benchmarking"]
summary: "ISO-Bench evaluates agent patches on 54 inference-serving optimization tasks using human-relative performance, judge-rated bottleneck targeting, and selective functional validation."
---

# ISO-Bench: Can Coding Agents Optimize Real-World Inference Workloads?

## Paper identity and scope

- This analysis covers arXiv 2602.19594v1, dated February 2026: a repository-level inference-optimization benchmark and agent study. [Title block; Abstract](../pdfs/07-iso-bench.md)
- ISO-Bench contains 54 tasks from merged performance PRs: 39 vLLM and 15 SGLang; each task starts from a pre-optimization repository snapshot. [§3.1–3.2](../pdfs/07-iso-bench.md)
- The agent receives a bottleneck description and must produce an optimization patch; this is guided optimization, not unconstrained discovery of any possible repository improvement. [§3.1; Appendix C.4](../pdfs/07-iso-bench.md)
- Comparators are expert human patches from the original PRs; the central analysis combines relative execution performance with judge-assessed target alignment. [§3.3](../pdfs/07-iso-bench.md)
- The main study has four configurations across three agent products; additional open-model failures are qualitative examples with incomplete aggregate coverage reporting. [§4.1; §6.1; Appendix D.5–D.7](../pdfs/07-iso-bench.md)

## Concepts and terminology

- Time to First Token (TTFT) is the serving latency metric; throughput is used for standalone benchmarks when TTFT is unavailable. [§3.3.1](../pdfs/07-iso-bench.md)
- TTFT delta is 100×(human TTFT−agent TTFT)/human TTFT, while throughput delta is 100×(agentThroughput−humanThroughput)/humanThroughput; positive is better in both. [Equations 1–2](../pdfs/07-iso-bench.md)
- The human post-optimization implementation is the denominator in these deltas, not the unoptimized parent commit. Both parent and human are also benchmarked in the protocol. [§3.3.1; §4.3](../pdfs/07-iso-bench.md)
- Hard categories are Beats if improvement over human exceeds 5%, Similar within±5%, Worse if degradation exceeds 5%, and Failed for benchmarking errors. [Table 1](../pdfs/07-iso-bench.md)
- The 5% band is a pragmatic noise tolerance, not a reported confidence interval or hypothesis test. [§3.3.1](../pdfs/07-iso-bench.md)
- Soft bottleneck targeting labels are Same target (same locations), Related target (same module/subsystem), Different target, and No optimization. [Table 2](../pdfs/07-iso-bench.md)
- Soft implementation labels are Similar approach, Valid alternative, Partial solution, and Ineffective; these describe strategy relative to the human patch. [Table 2](../pdfs/07-iso-bench.md)
- Q1 True Success means Same/Related target plus Beats/Similar performance; Q2 Good Intent means Same/Related target plus Worse/Failed. [§3.3.3; Figure 2](../pdfs/07-iso-bench.md)
- Q3 Lucky Win means Different/No target plus Beats/Similar; Q4 Complete Failure means Different/No target plus Worse/Failed. [§3.3.3; Figure 2](../pdfs/07-iso-bench.md)
- Hard Success=Q1+Q3; True Success=Q1. Implementation-approach labels are analyzed separately and do not enter the quadrant definition. [§3.3.3; §5.4](../pdfs/07-iso-bench.md)
- Because Similar counts as success, Q1 need not exceed human performance and may be as much as 5% worse than human; preserve the operational rule instead of interpreting every Q1 as a new speed record. [Table 1; §3.3.3](../pdfs/07-iso-bench.md)
- The 'understanding–execution gap' is operationalized by Q2 and correct-target counts Q1+Q2; it is an inference from patch location/performance, not a direct measurement of internal understanding. [§5.3; Figures 3–4](../pdfs/07-iso-bench.md)
- Functional validation is separate from the hard/soft classification and checks downstream model accuracy against the unoptimized baseline. [§3.3.4](../pdfs/07-iso-bench.md)

## Methodology and experimental setup

- Collection scans complete histories using ten keywords: optim, perf, speed, latency, throughput, memory, cache, kernel, fusion, batch; the keywords follow manual inspection of 100 commits per repository. [Appendix B.1](../pdfs/07-iso-bench.md)
- The automated scope filter retains commits modifying fewer than 10 files, then GPT-5-mini classifies genuine GPU inference optimizations from message and up-to 20,000-character diff. [Appendix B.1](../pdfs/07-iso-bench.md)
- The classifier emits a binary decision and confidence; the paper does not report a quantitative classification-accuracy audit. [Appendix B.1](../pdfs/07-iso-bench.md)
- vLLM filtering counts are 15,234 total commits → 892 keyword matches → 341 scope-qualified → 186 LLM-classified → 39 manually curated. [Table 6](../pdfs/07-iso-bench.md)
- SGLang filtering counts are 8,421 → 547 → 312 → 201 → 15 through the same stages. [Table 6](../pdfs/07-iso-bench.md)
- Manual PR review verifies genuine/reproducible optimization, extracts model/batch/request-rate and benchmark commands, records performance claims, and writes a bottleneck description without the solution. [§3.2; Appendix B.2](../pdfs/07-iso-bench.md)
- Additional exclusions include multi-GPU requirements, very old commits, and insufficient available Docker images; reproducibility/accessibility affects which tasks survive. [Appendix B.2](../pdfs/07-iso-bench.md)
- The illustrative task schema includes repository URL/human commit, parent metadata, CUDA/Python requirements, target-file patterns, and an optimization contract such as no public-API breakage. [Appendix C.1; Figure 7](../pdfs/07-iso-bench.md)
- The template schema's CUDA 12.4/Python 3.12 values are illustrative; actual containers use commit-specific dependencies and should not be assumed to share those versions. [Appendix C.1–C.2; Figure 7](../pdfs/07-iso-bench.md)
- The task prompt requests non-test-file edits, functional equivalence, no special-input overoptimization, and a baseline/profile/edit/rebuild/rebenchmark workflow. [Appendix C.4; Figure 8](../pdfs/07-iso-bench.md)
- Agents use isolated git worktrees in commit-specific Docker containers, with changes saved as git commits and final worktree captured on termination. [§4.2; Appendix C.2](../pdfs/07-iso-bench.md)
- Each task allows 120 minutes, one H100 GPU with 80 GB VRAM, eight CPUs, and 64 GB RAM; network downloads are permitted and logged. [Appendix C.2](../pdfs/07-iso-bench.md)
- Main configurations are Claude Code + Sonnet 4.5, Codex CLI + GPT-5 high reasoning, modified TRAE + Sonnet 4.5, and modified TRAE + GPT-5 high reasoning. [Table 7](../pdfs/07-iso-bench.md)
- Agents can inspect code, edit, run tests, benchmark, rebuild, and refine within the budget; equal wall-clock allowance does not establish equal token/tool-call budgets. [§4.2; Appendix C.2–C.4](../pdfs/07-iso-bench.md)
- Hard evaluation reuses original developers' benchmarking commands and compares agent/human/unoptimized versions on H100. [§3.3.1; §4.3](../pdfs/07-iso-bench.md)
- Soft evaluation uses Gemini-3-Flash-Preview to compare agent and human patches; the reported labels have not been validated against human annotators. [§3.3.2; §6.3](../pdfs/07-iso-bench.md)
- Functional evaluation is run for all Hard Success cases using LM Evaluation Harness and the model/accuracy tasks specified by the original PR. [§3.3.4; §5.6](../pdfs/07-iso-bench.md)
- The code/harness/configuration/analysis artifacts are released on GitHub under Apache 2.0, and the dataset on HuggingFace under CC BY 4.0. [Appendix A](../pdfs/07-iso-bench.md)

## Results and evidence

- On 39 vLLM tasks, Q1 rates are Claude 46.2% (18 tasks), TRAE-Sonnet 28.2% (11), Codex 20.5% (8), TRAE-GPT-5 17.9% (7). [Tables 3,5](../pdfs/07-iso-bench.md)
- On 15 SGLang tasks, Q1 rates are TRAE-GPT-5 86.7% (13), Codex 80% (12), TRAE-Sonnet 80% (12), Claude 26.7% (4). [Tables 3,5](../pdfs/07-iso-bench.md)
- Thus the best vLLM configuration is worst on SGLang in this sample; combining repositories into one headline ranking obscures that reversal. [Table 3; §5.4](../pdfs/07-iso-bench.md)
- vLLM Hard Success versus Q1 is Claude 56.4 vs 46.2%, Codex 33.3 vs 20.5%, TRAE-Sonnet 33.3 vs 28.2%, TRAE-GPT-5 20.5 vs 17.9%. [Table 4](../pdfs/07-iso-bench.md)
- The corresponding vLLM Q3 counts are 4, 5, 2, 1; reported gaps 10.2, 12.8, 5.1, 2.6 are percentage-point differences, not relative percentages. [Tables 4–5](../pdfs/07-iso-bench.md)
- On SGLang Claude's Hard Success 46.7% exceeds Q1 26.7% by 20 percentage points (three Q3 tasks); other configurations have zero Q3. [Tables 4–5](../pdfs/07-iso-bench.md)
- Full vLLM quadrant counts (Q1, Q2, Q3, Q4) are Claude (18, 15, 4, 2), Codex (8, 20, 5, 6), TRAE-Sonnet (11, 20, 2, 6), TRAE-GPT-5 (7, 27, 1, 4). [Table 5](../pdfs/07-iso-bench.md)
- Full SGLang counts are Claude (4, 8, 3, 0), Codex (12, 3, 0, 0), TRAE-Sonnet (12, 3, 0, 0), TRAE-GPT-5 (13, 2, 0, 0). [Table 5](../pdfs/07-iso-bench.md)
- vLLM correct-target counts Q1+Q2 are 33 Claude, 28 Codex, 31 TRAE-Sonnet, 34 TRAE-GPT-5, compared with 18, 8, 11, 7 Q1 successes. [Figure 3; Table 5](../pdfs/07-iso-bench.md)
- TRAE-GPT-5 targets correctly on 34/39 vLLM tasks but succeeds on 7/39; its 27 Q2 cases are the largest observed gap. [Figure 3; Table 5](../pdfs/07-iso-bench.md)
- Three of four vLLM configurations have more Q2 than any other quadrant; Claude instead has 18 Q1 versus 15 Q2. [Table 5; §5.3](../pdfs/07-iso-bench.md)
- On SGLang all configurations except Claude target correctly on 15/15; Claude targets correctly on 12/15 but eight of those are Q2. [Figure 4; Table 5](../pdfs/07-iso-bench.md)
- Same-model scaffold contrast: Sonnet 4.5 gets 46.2% Q1 in Claude Code versus 28.2% in TRAE on vLLM, but 26.7% versus 80% on SGLang. [§5.5; Table 3](../pdfs/07-iso-bench.md)
- GPT-5 scaffold contrast: Codex gets 20.5% vLLM/80% SGLang Q1 versus TRAE 17.9%/86.7%; the effect's direction/magnitude differs from the Sonnet comparison. [Table 3; Table 7](../pdfs/07-iso-bench.md)
- In the visually inspected vLLM approach chart, Claude has 3 Similar, 14 Partial, 14 Alternative, 8 Ineffective; Codex has 3, 9, 17, 10 respectively. [Figure 5](../pdfs/07-iso-bench.md)
- TRAE's vLLM displayed approach bars are Sonnet 11 Similar/10 Partial/7 Alternative/10 Ineffective and GPT-5 17/8/10/3; each sums to 38 rather than the 39-task caption. [Figure 5](../pdfs/07-iso-bench.md)
- The authors report that SGLang Claude never uses a Similar approach while other agents frequently do; its alternative/partial preference correlates with lower success in this repository. [§5.4; Figure 6](../pdfs/07-iso-bench.md)
- All Q1 cases reportedly retain model accuracy within the specified task tolerance; detailed per-task tolerances, accuracy values, and uncertainty are not enumerated. [§5.6](../pdfs/07-iso-bench.md)
- Bamba-9B commit fe66b347 yields a Q3 hard success for TRAE-Sonnet but downstream exact-match accuracy falls 32% → 0%, demonstrating that a fast patch can be behaviorally wrong. [§5.6; Appendix D.3](../pdfs/07-iso-bench.md)
- The paper attributes that regression to replacing dynamic expand dimensions (-1) with num_heads_per_rank in Mamba tensor preparation; the displayed patch also reorganizes unsqueeze/view operations. [Appendix D.3; Figure 11](../pdfs/07-iso-bench.md)
- Q1 case 98f47f2a moves query/key/value reshapes outside the FlashAttention custom op, reducing CPU overhead around CUDA-graph execution; reported throughput improves 21.09%. [Appendix D.1; Figure 9](../pdfs/07-iso-bench.md)
- The 21.09% case result is presented as a throughput gain, but the case caption does not independently restate the normalization denominator; retain its reported label rather than silently treating it as parent-relative. [Appendix D.1; Figure 9; Equation 2](../pdfs/07-iso-bench.md)
- Q2 case 015069b0 targets Qwen3 reasoning parsing with str.find/partition-style string operations replacing regex; benchmarking fails despite correct file/strategy selection. [Appendix D.2; Figure 10](../pdfs/07-iso-bench.md)
- The Q2 case commentary attributes failure to Python casing/name errors, but the displayed excerpt is incomplete and does not itself show the alleged none/None error. [Figure 10](../pdfs/07-iso-bench.md)
- Q4 case 2deb029d misses the prefix-cache warmup fix: human tracking records touched blocks and marks them computed after scheduling, while the agent edits tests/allocator logic without the needed scheduling linkage. [Appendix D.4; Figure 12](../pdfs/07-iso-bench.md)
- MiniMax-M2.1's illustrated scheduler attempt produces 81,782 output tokens over 477 seconds and 75 steps, with zero tool calls; the log repeatedly announces intended actions. [Appendix D.5; Figure 13](../pdfs/07-iso-bench.md)
- That MiniMax example reports 1,599,945 input tokens and 2,412 repetitions of a tool-use statement; these are trajectory totals, not an estimate of useful progress. [Figure 13](../pdfs/07-iso-bench.md)
- GPT-OSS-120B's illustrated failure makes about 84 tool calls attempting to create mock torch/triton/transformers packages in the task directory instead of optimizing the scheduler. [Appendix D.6; Figure 14](../pdfs/07-iso-bench.md)
- GLM-4.7's illustrated run makes 386 tool calls (327 bash, 59 str_replace), commits edits, then cycles through git operations and hits 400 steps without invoking finish. [Appendix D.7; Figure 15](../pdfs/07-iso-bench.md)
- The GLM excerpt reports a patch-reapplication error after edits were already present; authors interpret this as workflow/termination confusion. It is not quantitative evidence that those edits improved runtime. [Figure 15](../pdfs/07-iso-bench.md)
- Authors state none of the tested open models produced a working optimization under their TRAE setup; the paper provides no complete per-model task-count/results table supporting a 54-task-wide zero rate. [§6.1; Appendix D.5–D.7](../pdfs/07-iso-bench.md)

## Authors’ conclusions

- Authors argue performance-only evaluation can misclassify off-target or behavior-changing patches as success, and combine target judgments with performance and downstream correctness. [§3.3; §5.2; §5.6](../pdfs/07-iso-bench.md)
- They identify failed implementation after apparently correct targeting as a major observed failure mode, particularly for vLLM. [§5.3; §6.2](../pdfs/07-iso-bench.md)
- They conclude model capability alone does not determine repository optimization performance: exploration, decomposition, and stopping behavior in the scaffold matter. [§5.5; §6.2](../pdfs/07-iso-bench.md)
- They argue evaluation on one repository can overstate transfer, given observed rank reversals and strategy differences. [§5.4](../pdfs/07-iso-bench.md)
- They propose using the benchmark for model/RL improvements and scaffold development that generalizes across codebases. [§6.2](../pdfs/07-iso-bench.md)
- They identify tool invocation, environment understanding, and reliable task finalization as improvement targets from open-model examples, while acknowledging only one scaffold was tested. [§6.1](../pdfs/07-iso-bench.md)

## Limitations and open questions

- Explicit scope limitation: 54 tasks/two repositories omit broader inference ecosystems and large coordinated scheduling/memory/cache redesigns. [§6.3](../pdfs/07-iso-bench.md)
- Explicit hardware limitation: single H100/NVIDIA only, excluding tensor/pipeline-parallel and other multi-GPU optimizations. [§6.3](../pdfs/07-iso-bench.md)
- Explicit contamination limitation: public PRs may occur in frontier pretraining; authors propose temporal filtering, paraphrasing, or held-out repositories. [§6.3](../pdfs/07-iso-bench.md)
- Explicit judge limitation: one LLM judge without human-validation studies or judge ensembles limits reliability of the soft labels. [§6.3](../pdfs/07-iso-bench.md)
- Analyst caveat: Related target means the same subsystem, so target alignment is a coarse proxy and does not prove correct diagnosis or a sound algorithm. [Table 2; §5.3](../pdfs/07-iso-bench.md)
- Analyst caveat: a sound optimization of an alternative bottleneck can be useful despite Q3 labeling; 'Lucky Win' is a task-alignment category, not by itself evidence of cheating or incorrectness. [§3.3.3; §5.6](../pdfs/07-iso-bench.md)
- Analyst caveat: Q1's name should not be treated as a universal validity guarantee; it combines benchmark-competitive performance, judge alignment, and finite downstream tests. [§3.3.3–3.3.4; §5.6](../pdfs/07-iso-bench.md)
- Analyst caveat: downstream accuracy remaining similar does not establish identical per-example outputs, numerical equivalence, or correctness on every workload, although the task prompt requests identical outputs. [§3.3.4; Appendix C.4](../pdfs/07-iso-bench.md)
- Analyst caveat: testing only Hard Success cases leaves correctness of Q2/Q4 uncharacterized; runtime failure and numerical/semantic failure remain distinct mechanisms. [§3.3.4; Table 5](../pdfs/07-iso-bench.md)
- Analyst caveat: hard metrics lack detailed repetition/variance estimates in the paper, and TTFT versus throughput are substituted rather than jointly optimized for each task. [§3.3.1; §4.3](../pdfs/07-iso-bench.md)
- Analyst caveat: shared base models and wall-clock limits help scaffold comparisons, but no controlled ablation isolates individual tool, prompt, search, or stopping-policy effects. [§4.1–4.2; §5.5](../pdfs/07-iso-bench.md)
- Analyst caveat: 'scaffolding is as important as the model' is qualitative interpretation; no common scale estimates relative causal contributions of model and scaffold. [Abstract; §5.5; §6.2](../pdfs/07-iso-bench.md)
- Analyst caveat: project comparisons may reflect task distributions, historical commits, dependencies, and selection as well as repository generalization; only 15 SGLang tasks underpin its high rates. [§3.2; Appendix B; Table 3](../pdfs/07-iso-bench.md)
- Analyst caveat: agent access to repository history/network raises potential reference-patch leakage; the paper describes logged downloads but does not detail a blinded-history audit. [Appendix C.1–C.2](../pdfs/07-iso-bench.md)
- Analyst caveat: the Bamba excerpt does not establish the exact causal chain without full tensor shapes and executable comparison; num_heads_per_rank is derived, not a literal hardcoded constant, despite the source's wording. [Appendix D.3; Figure 11](../pdfs/07-iso-bench.md)
- Analyst caveat: GLM's sample change from+= to explicit addition is merely an edit; calling it a successful optimization would exceed the displayed performance evidence. [Figure 15](../pdfs/07-iso-bench.md)
- Source inconsistency: Appendix B retains fewer than 10 files, while §6.3 describes excluding more than 10; the treatment of exactly 10 files is ambiguous in prose. [Appendix B.1; §6.3](../pdfs/07-iso-bench.md)
- Source ambiguity: the time-limited protocol captures final worktree state, yet GLM is called a failure partly for not invoking finish; the operational difference between saved patch validity and agent completion needs harness inspection. [Appendix C.2; Appendix D.7](../pdfs/07-iso-bench.md)
- Source inconsistency: §6.2 summarizes overestimation as 10–20 percent, but Table 4 spans 0–20 percentage points and most nonzero rows are below 10; use row-specific denominators. [Table 4; §6.2](../pdfs/07-iso-bench.md)

## Relevance to this project — interpretation

- Project interpretation: preserve performance, task alignment, semantic correctness, execution errors, and completion status as separate fields instead of one success flag.
- Project interpretation: if using judges, retain prompts, model versions, rationales, and human-audit disagreements; target-location judgments should not replace executable evidence.
- Project interpretation: capture baseline/human/agent metrics and exact benchmark/model configurations so human-relative and parent-relative gains remain distinguishable.
- Project interpretation: compare scaffolds across repository families and log tool-use/verification/finalization behavior, while avoiding causal claims from a single aggregate ranking.
- Project interpretation: treat reported reward-hacking cases as hypotheses for targeted regression tests, verifying the full patch and tensor behavior before adopting a root-cause story.
- Project interpretation: these recommendations are possible knowledge-store uses, not implemented benchmark changes or independently validated results.

## Retrieval notes

- Aliases: ISO-Bench; inference serving optimization; Q1/Q2/Q3/Q4; True Success; Hard Success; Good Intent Bad Execution; Lucky Win.
- Answers questions about repository optimization task construction, human-relative TTFT/throughput, target-alignment judges, understanding/execution gaps, scaffold sensitivity, and correctness regressions.
- Related: [KernelBench](01-kernelbench.md) and [TritonBench](05-tritonbench.md) evaluate more isolated kernel generation with different tasks/baselines; their rates should not be compared directly.
- Related navigation only: SWE-agent for agent-computer interfaces; MT-Bench and LLM-judge papers for judge reliability; Measuring Automated Kernel Engineering for another optimization evaluation setting.

## Source coverage and verification notes

- Coverage includes full §§1–6, Appendices A–D.7, all seven tables, task schema/prompt, and every detailed case-study patch/log discussion. [§§1–6; Appendices A–D](../pdfs/07-iso-bench.md)
- Saved figure assets 3–6 were visually inspected: targeting counts agree with Table 5; Figure 5's two TRAE approach rows total 38 despite the 39-task scope. The missing case is not explained. [Figures 3–6; Table 5](../pdfs/07-iso-bench.md)
- Some main-text cross-references are wrong: §5.3 cites Figure 4 for vLLM although the vLLM chart isFigure 3; §5.4–5.5 repeatedly citeFigure 6 where vLLM isFigure 5. This note follows verified captions/assets. [§5.3–5.5; Figures 3–6](../pdfs/07-iso-bench.md)
- The broad prose claim that Codex consistently matches the human approach conflicts with its vLLM chart showing 17 Alternative versus 3 Similar. Numerical chart evidence is retained separately from that conclusion. [§5.4; Figure 5](../pdfs/07-iso-bench.md)
- Figure 6's saved image lacks agent labels/legend, so exact unlabeled row-to-agent assignments are not inferred beyond the supported prose. Figure 5 and PDF p. 9 establish the paired chart context.
- Source-derived figures/tables are cited using Markdown caption numbering; PDF p. 9 was inspected to confirm the approach-chart captions. No source patch, agent run, GPU benchmark, or judge label was independently reproduced.
- Open-model examples are treated as qualitative trajectories because the full evaluation denominator is not documented. All numerical claims retain their task counts, historical configuration, and source-reported status.
