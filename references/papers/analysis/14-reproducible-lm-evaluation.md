---
paper_id: "14"
title: "Lessons from the Trenches on Reproducible Evaluation of Language Models"
source_version: "2405.14782v3"
source_markdown: "../pdfs/14-reproducible-lm-evaluation.md"
source_pdf: "../pdfs/14-reproducible-lm-evaluation.pdf"
source_url: "https://arxiv.org/abs/2405.14782"
analysis_date: "2026-11-20"
analysis_status: "reviewed"
topics: ["evaluation", "reproducibility", "benchmarking", "statistics", "prompt-sensitivity", "tokenization", "lm-eval", "measurement-validity"]
summary: "Documents evaluation failure modes and reproducibility practices, with prompt-format case studies and formal descriptions of likelihood, perplexity, and generative scoring."
---

# Lessons from the Trenches on Reproducible Evaluation of Language Models

## Paper identity and scope

- A methods/practice paper based on experience building the Language Model Evaluation Harness (`lm-eval`), combining recommendations, implementation design, and empirical case studies. It is not primarily an LLM-as-a-judge algorithm paper. [§1](../pdfs/14-reproducible-lm-evaluation.md)
- Frozen version is **2405.14782v3**. Its descriptions of library defaults, API access, community norms, and available models are source-version statements, not guaranteed current documentation. [Source metadata; §§1–3](../pdfs/14-reproducible-lm-evaluation.md)
- The authors concentrate on multiple-choice evaluation because of their experience and its prevalence, while explaining how concerns extend to generation, perplexity, and model-based grading. [§2.1; Appendix B](../pdfs/14-reproducible-lm-evaluation.md)
- The organizing distinction is between **reproducibly computing a score** and **validly interpreting what that score says about a capability**. Both can fail independently. [§§3–4](../pdfs/14-reproducible-lm-evaluation.md)
- `lm-eval` is framed as orchestration infrastructure: non-prescriptive about which benchmarks matter, with documented defaults for how a task is evaluated and configurable alternatives. [§1](../pdfs/14-reproducible-lm-evaluation.md)

## Concepts and terminology

### Measurement validity and evaluation design

- **The Key Problem:** semantically equivalent answers can differ syntactically, but the best automated tools for detecting equivalence are often the very models being evaluated. [§2.1](../pdfs/14-reproducible-lm-evaluation.md)
- **Closed-domain conversion** restricts answer possibilities, such as multiple choice or constrained output, to make checking easier; it changes the task being measured. [§2.1](../pdfs/14-reproducible-lm-evaluation.md)
- **Model-based grading** approximates semantic or preference judgments but inherits the grader's limitations and configuration choices. Heuristic overlap metrics avoid some costs while missing semantic variation. [§2.1](../pdfs/14-reproducible-lm-evaluation.md)
- **Human evaluation** can assess nuanced equivalence but is costly, difficult to scale, and itself imperfect on complex correctness judgments. [§2.1](../pdfs/14-reproducible-lm-evaluation.md)
- A practical verifier, when available, offers a different checking mechanism; the paper identifies programmatic verification as a special opportunity rather than assuming it exists for all language tasks. [§2.1](../pdfs/14-reproducible-lm-evaluation.md)
- **Benchmark validity** asks whether the measured behavior supports the intended inference about deployment capability. Correct computation alone does not establish validity. [§4.1](../pdfs/14-reproducible-lm-evaluation.md)
- **Benchmark Lottery** describes how benchmark selection and arbitrary implementation choices can shape what the field treats as progress. [§4.1](../pdfs/14-reproducible-lm-evaluation.md)
- **Format compliance versus capability:** failing to emit an expected letter or answer format can depress scores even when another formulation reveals task knowledge. [§4.1; Appendix C.1](../pdfs/14-reproducible-lm-evaluation.md)
- **Retrofitting benchmarks:** tasks designed before in-context learning or chat usage may require new prompting/scoring decisions with no single original-author standard for the new setting. [§2.3](../pdfs/14-reproducible-lm-evaluation.md)
- **Micro average** weights individual examples equally; **macro average** weights task groups/subjects equally. For MMLU, the paper identifies original micro averaging over documents rather than an unweighted mean of 57 subject scores. [§3.1; Appendix C.1](../pdfs/14-reproducible-lm-evaluation.md)

### Sources of uncertainty

- Variation can come from model sampling, random seed, example selection, prompt wording, few-shot demonstration choice/order, normalization, output parsing, and implementation details. These are not one interchangeable uncertainty source. [§§3.1, 4.1; Appendices B, D](../pdfs/14-reproducible-lm-evaluation.md)
- **Repeated-run variation** measures stochastic model behavior when the benchmark is rerun. [§4.1; Figure 2](../pdfs/14-reproducible-lm-evaluation.md)
- **Bootstrap uncertainty** over examples estimates variability under resampling questions from a hypothetical population, conditional on the collected outputs. It is different from resampling model generations. [§4.3](../pdfs/14-reproducible-lm-evaluation.md)
- **Prompt variation** can change both absolute scores and relative model ordering; a single prompt is only one measurement configuration. [§3.1; Table 1](../pdfs/14-reproducible-lm-evaluation.md)
- A confidence interval answers a particular statistical question and depends on its resampling assumptions. Reporting an interval without explaining its source can obscure other unmeasured variation. [§4.3](../pdfs/14-reproducible-lm-evaluation.md)

### Likelihood-based measurement

- Conditional sequence log-likelihood sums token log-probabilities of a candidate continuation given the prompt and preceding continuation tokens. Teacher forcing permits scoring a known continuation without generating it. [Appendix B.1–B.2; Eq. 1](../pdfs/14-reproducible-lm-evaluation.md)
- Multiple-choice likelihood evaluation chooses the candidate with highest conditional log-likelihood, which can be evaluated without requiring a freely generated answer to follow the benchmark format. [Appendix B.2](../pdfs/14-reproducible-lm-evaluation.md)
- For single-token choices, the next-token distribution can score all alternatives in one model evaluation rather than one separate call per answer. [Appendix B.2](../pdfs/14-reproducible-lm-evaluation.md)
- Raw summed log-likelihood can favor short answer strings because additional token log-probabilities are nonpositive. [Appendix B.2, “Normalization”](../pdfs/14-reproducible-lm-evaluation.md)
- **Token-length normalization** divides by answer token count; **byte-length normalization** divides by byte count. The latter removes tokenizer-dependent length units, though the underlying model probabilities still differ. [Appendix B.2](../pdfs/14-reproducible-lm-evaluation.md)
- **Mutual-information-style normalization** subtracts an unconditional or null-context answer log-likelihood, measuring how much the question increases support for that answer. The chosen null context is itself a methodological choice. [Appendix B.2](../pdfs/14-reproducible-lm-evaluation.md)
- **Greedy exact match** can be checked by testing whether every target token is the argmax continuation along its target prefix. This is stricter than selecting the correct option among a finite candidate set. [Appendix B.2, “Computing Exact Match”](../pdfs/14-reproducible-lm-evaluation.md)
- Tokenizing prompt and answer separately need not equal tokenizing their concatenation. Boundary whitespace can change the actual scored tokens. [Appendix B.2, “Tokenization”](../pdfs/14-reproducible-lm-evaluation.md)
- The paper describes moving trailing prompt whitespace into target strings as a practical mitigation and mentions token healing and tokenizer research as other avenues. [Appendix B.2](../pdfs/14-reproducible-lm-evaluation.md)

### Perplexity and generation

- **Perplexity** is the exponential of mean negative token log-likelihood over the evaluation corpus. Corpus aggregation must weight token counts appropriately rather than blindly average document perplexities. [Appendix B.3; Eq. 2](../pdfs/14-reproducible-lm-evaluation.md)
- Token perplexity is not directly comparable across different tokenizers because segmentation changes both token count and prediction difficulty. [Appendix B.3](../pdfs/14-reproducible-lm-evaluation.md)
- **Bits per byte**, word-normalized perplexity, and byte-normalized perplexity change the normalization unit to enable more comparable text-distribution measurements. [Appendix B.3; Eqs. 3–4](../pdfs/14-reproducible-lm-evaluation.md)
- The consistent BPB relation supplied by Eq. 4 is `(total tokens / total bytes) * log2(PPL)`. Eq. 3 has a sign discrepancy documented below; do not copy it uncritically into code. [Appendix B.3; Eq. 4](../pdfs/14-reproducible-lm-evaluation.md)
- **Early Token Curse:** predictions near the start of a context window have less conditioning information and are harder than later predictions. Frequent window resets can penalize shorter-context models. [Appendix B.3, “Sliding Window”](../pdfs/14-reproducible-lm-evaluation.md)
- Non-overlapping perplexity windows reset context every `L` tokens; strided windows of length `L` advancing by `s` score new tokens with more preceding context. [Appendix B.3](../pdfs/14-reproducible-lm-evaluation.md)
- A naive strided implementation can require roughly `L/s` times the model calls/compute; cache reuse can reduce repeated work for some architectures. [Appendix B.3](../pdfs/14-reproducible-lm-evaluation.md)
- The source says `lm-eval` uses non-overlapping windows as a cost/standardization choice. That is a stated tradeoff, not proof that window-reset bias disappears. [Appendix B.3](../pdfs/14-reproducible-lm-evaluation.md)
- Generative evaluation must specify decoding, stopping, answer extraction, normalization, and scoring; each is part of the measured system. [Appendix B.4](../pdfs/14-reproducible-lm-evaluation.md)
- Regex/heuristic extraction can systematically reward models that happen to produce expected formats and penalize correct answers in alternative forms. [Appendix B.4](../pdfs/14-reproducible-lm-evaluation.md)
- Likelihood evaluation can be informative for weak/base models that struggle to follow output instructions; generative evaluation can be closer to actual interactive usage. Neither dominates for every intended measurement. [Appendix B.5](../pdfs/14-reproducible-lm-evaluation.md)

## Methodology and experimental setup

- The paper combines developer experience, cited examples from the literature, and an original prompt-format comparison. A cited external case is not automatically a new experiment performed by these authors. [§§1, 3–4](../pdfs/14-reproducible-lm-evaluation.md)
- The prompt study evaluates **GPT-NeoX-20B, Llama-2-7B, Falcon-7B, Mistral-7B, and Mixtral-8x7B** on ARC-Challenge and MMLU in **zero-shot** conditions. [Table 1](../pdfs/14-reproducible-lm-evaluation.md)
- **ARC cloze** presents a question and answer prefix, scoring each full candidate answer as a continuation. [Appendix C.1](../pdfs/14-reproducible-lm-evaluation.md)
- **ARC MMLU-style** includes the choices with letter labels and scores the answer letters. It changes both visible prompt format and continuation strings. [Appendix C.1](../pdfs/14-reproducible-lm-evaluation.md)
- **MMLU standard style** scores choice letters; **MMLU hybrid** keeps the listed choices but scores the full answer strings. [Appendix C.1](../pdfs/14-reproducible-lm-evaluation.md)
- Experiments use **lm-eval 0.4.2**, with explicit YAML configurations supplied for ARC and MMLU variants. They do not describe the current library release. [Appendix C.1](../pdfs/14-reproducible-lm-evaluation.md)
- Hardware is **8 NVIDIA A40 48GB GPUs**, with total usage reported below one day. [Appendix E](../pdfs/14-reproducible-lm-evaluation.md)
- Reported case-study intervals are **95% bootstrap confidence intervals**, as specified in the paper's reporting checklist, not variation over ten independent model runs. [Checklist item 3(c)](../pdfs/14-reproducible-lm-evaluation.md)
- The ten-run GPQA visualization is attributed to **Lukošiūtė (2024)** and contrasts point estimates with uncertainty; it is illustrative external evidence. [§4.1; Figure 2](../pdfs/14-reproducible-lm-evaluation.md)
- A task configuration includes dataset source/splits, rendered prompts, target mapping, candidate choices, metrics, aggregation, and version metadata. [Appendix A; Appendix C.1](../pdfs/14-reproducible-lm-evaluation.md)
- The LM abstraction treats **model plus tokenizer** as the evaluated system and dispatches string-oriented requests to it. [Appendix A](../pdfs/14-reproducible-lm-evaluation.md)
- Three request families are conditional log-likelihood, rolling log-likelihood/perplexity, and text generation with stopping conditions. [Appendix A](../pdfs/14-reproducible-lm-evaluation.md)
- Task-bound filters can implement chained output extraction and postprocessing, which must be reported alongside scores. [Appendix B.4](../pdfs/14-reproducible-lm-evaluation.md)

## Results and evidence

### Controlled prompt/scoring changes

- ARC-Challenge GPT-NeoX-20B accuracy changes from **38.0% ±2.78** cloze to **26.6% ±2.53** MMLU-style; Falcon-7B changes from **40.2% ±2.81** to **25.9% ±2.51**. [Table 1](../pdfs/14-reproducible-lm-evaluation.md)
- The direction reverses for Mistral-7B, **50.1% ±2.86 to 72.4% ±2.56**, and Mixtral-8x7B, **56.7% ±2.84 to 81.3% ±2.23**. These are different measurement formats on the same model/task, not training improvements. [Table 1](../pdfs/14-reproducible-lm-evaluation.md)
- The displayed ARC changes for Mistral and Mixtral are **22.3 and 24.6 percentage points**, respectively. The paper's broad “>20%” phrasing should not obscure the units. [Table 1; §3.1](../pdfs/14-reproducible-lm-evaluation.md)
- Llama-2-7B's ARC results are much closer, **43.5% ±2.84 versus 42.8% ±2.83**, demonstrating model-dependent sensitivity. [Table 1](../pdfs/14-reproducible-lm-evaluation.md)
- MMLU hybrid versus standard results are **27.6/24.5 GPT-NeoX, 39.8/41.3 Llama-2, 29.1/25.4 Falcon, 48.3/58.6 Mistral, and 59.7/67.1 Mixtral** in percent accuracy. No single style uniformly helps all models. [Table 1](../pdfs/14-reproducible-lm-evaluation.md)
- ARC-Easy GPT-NeoX accuracy is reported as **72.4% ±1.80 cloze versus 26.5% ±1.78 MMLU-style**, an even larger format-associated difference. [§4.1](../pdfs/14-reproducible-lm-evaluation.md)
- Qualitative inspection links GPT-NeoX's near-random letter-format performance to task/output-format failures; the authors caution against inferring a general lack of knowledge or a bad training corpus from that one score. [§4.1](../pdfs/14-reproducible-lm-evaluation.md)
- The authors infer that certain model reports likely used MMLU-style prompting on ARC because the numbers align, but explicitly lack precise reported setups. This is an inference, not verified knowledge of those developers' code. [§3.1](../pdfs/14-reproducible-lm-evaluation.md)

### Implementation details and interpretation

- The paper highlights an ARC question with five choices amid mostly four-choice items, MMLU micro/macro averaging, and HumanEval examples lacking otherwise-common example tests as cases where small implementation assumptions matter. [§3.1](../pdfs/14-reproducible-lm-evaluation.md)
- A comparison of three MMLU implementations is cited as evidence that scores and rankings differ despite sharing a benchmark name; it is attributed to Fourrier et al., not reproduced as the paper's own new table. [§3.1](../pdfs/14-reproducible-lm-evaluation.md)
- Benchmark scores can be optimized by matching separator/format conventions rather than improving the intended capability; the authors discuss leaderboard incentives as a concrete threat to interpretation. [§4.1](../pdfs/14-reproducible-lm-evaluation.md)
- Smaller evaluation sets make apparent gains more susceptible to uncertainty; HumanEval's **164 examples** is given as a motivating example. [§4.1](../pdfs/14-reproducible-lm-evaluation.md)
- Bootstrapped uncertainty, repeated-run uncertainty, and prompt sensitivity each answer different questions, so one cannot substitute one interval for all forms of reliability. [§4.3](../pdfs/14-reproducible-lm-evaluation.md)

## Authors' conclusions

- Publish exact evaluation code whenever possible; full prompts alone are necessary but insufficient to recover all methodological choices. [§3.2; Appendix D](../pdfs/14-reproducible-lm-evaluation.md)
- Disclose prompt engineering, few-shot selection, likelihood normalization, generation settings, extraction heuristics, and enough implementation detail to avoid a chain of underdetermined references. [§3.2; Appendix D](../pdfs/14-reproducible-lm-evaluation.md)
- Avoid importing baseline numbers from unrelated implementations; when unavoidable, mark them and qualify comparability. [Appendix D](../pdfs/14-reproducible-lm-evaluation.md)
- Save and share model outputs so others can inspect errors, rescore results, test alternate metrics, and study deprecated/API models without rerunning costly generation. [Appendix D](../pdfs/14-reproducible-lm-evaluation.md)
- Inspect a small batch qualitatively before running at scale, then analyze failure modes rather than treat a final score as the explanation. [§4.2; Appendix D](../pdfs/14-reproducible-lm-evaluation.md)
- Report uncertainty/statistical analyses and, where feasible, variation over seeds and few-shot selections. [§4.2; Appendix D](../pdfs/14-reproducible-lm-evaluation.md)
- Evaluate in realistic settings aligned with intended usage, while being explicit about what any benchmark can support. [§4.2](../pdfs/14-reproducible-lm-evaluation.md)
- Shared configurable infrastructure can reduce duplicated work and make rigor easier by default, but should permit research into alternative prompt/scoring choices. [§3.3; §5](../pdfs/14-reproducible-lm-evaluation.md)
- The authors' default-selection hierarchy prioritizes widespread trainer agreement, then a clear official implementation, then evaluator consensus, then a chosen common implementation. Their rationale includes social adoption and restricted model access, not a claim that trainer conventions are scientifically infallible. [§3.3](../pdfs/14-reproducible-lm-evaluation.md)
- `--limit`, `--log_samples`, task configurations, and library-version reporting operationalize inspection and provenance in the source's library workflow. [§3.3](../pdfs/14-reproducible-lm-evaluation.md)

## Limitations and open questions

- **Source-explicit:** the paper focuses on areas of author expertise, especially MCQA; broader validity issues and other settings are not treated exhaustively. [§2.1; Appendix F](../pdfs/14-reproducible-lm-evaluation.md)
- **Source-explicit:** human grading, model grading, and constrained-answer conversion all trade off different shortcomings; the Key Problem has no single context-free solution. [§2.1](../pdfs/14-reproducible-lm-evaluation.md)
- **Source-explicit:** social incentives, lack of model access, expense, and missing API capabilities can prevent independent reproductions even when desired. [§2.2](../pdfs/14-reproducible-lm-evaluation.md)
- **Analyst caveat:** code release and protocol matching improve reproducibility but do not prove deployment validity or immunity to benchmark contamination/gaming. [Evidence: §§3–4](../pdfs/14-reproducible-lm-evaluation.md)
- **Source discrepancy:** the BPB Eq. 3 has two negative signs, producing the opposite sign from its nonnegative description and Eq. 4. This also appears in the PDF; Eq. 4 provides the consistent relation. [PDF p. 26, Eqs. 3–4](../pdfs/14-reproducible-lm-evaluation.pdf#page=26)
- **Source notation caveat:** Appendix B.2's displayed `n+i` indexing should be reconciled with its zero-based token notation and the described next-token shift before implementation. The analysis records the semantic scoring procedure rather than presenting the expression as tested code. [Appendix B.1–B.2](../pdfs/14-reproducible-lm-evaluation.md)
- **Source incompleteness:** Appendix F's limitations paragraph ends mid-sentence in both saved Markdown and PDF; its missing continuation is not reconstructed. [Appendix F](../pdfs/14-reproducible-lm-evaluation.md); [PDF p. 31](../pdfs/14-reproducible-lm-evaluation.pdf#page=31)
- **Analyst caveat:** the case studies show sensitivity for specified models/tasks; they do not establish a universal preferred prompt or quantify all modern agentic evaluations. [Evidence: Table 1; Appendix C.1](../pdfs/14-reproducible-lm-evaluation.md)
- **Open question:** how to standardize reporting while allowing task-appropriate adaptation remains a governance and methodology issue, not something solved solely by a framework default. [§§3.2–3.3](../pdfs/14-reproducible-lm-evaluation.md)

## Relevance to this project — interpretation

- **Proposal:** treat a kernel experiment as a complete configuration: task source, shapes, dtype/tolerances, GPU/software versions, reference implementation, timing procedure, agent prompts, model version, budget, and selection rule. This extends the paper's provenance principle to our domain; these exact kernel fields are an analyst proposal. [Motivation: §§3.1–3.2](../pdfs/14-reproducible-lm-evaluation.md)
- **Proposal:** store raw candidate code, compiler/runtime feedback, correctness outputs, latency samples, and rejected attempts so later analyses can be recomputed. [Motivation: Appendix D, output release](../pdfs/14-reproducible-lm-evaluation.md)
- **Proposal:** distinguish runtime timing variance, generation-seed variance, task-population uncertainty, and prompt variation rather than report one ambiguous error bar. [Motivation: §4.3](../pdfs/14-reproducible-lm-evaluation.md)
- **Proposal:** before interpreting a low pass rate, inspect whether failures arise from model capability, interface/formatting, compilation, test design, or measurement. [Motivation: §4.1; Appendix D](../pdfs/14-reproducible-lm-evaluation.md)
- **Proposal:** preserve paired comparisons under the same harness and baseline; annotate imported paper numbers as external evidence rather than mixing them into our own run results. [Motivation: Appendix D](../pdfs/14-reproducible-lm-evaluation.md)
- **Proposal:** maintain both representative workloads and diagnostic subgroups so one aggregate does not silently define project success. This is a project-specific response to the Benchmark Lottery argument. [Motivation: §4.1](../pdfs/14-reproducible-lm-evaluation.md)

## Retrieval notes

- Aliases: lm-eval, LM Evaluation Harness, Key Problem, Benchmark Lottery, cloze versus MMLU-style, acc/acc_norm/acc_mutual_info, micro/macro average, early token curse, BPB, prompt variance.
- Questions answered: Why do the same benchmark names yield different numbers? What must be saved to reproduce an evaluation? What does a bootstrap interval measure? Why can tokenization alter answer scoring? Why is an LLM's score not a complete capability judgment?
- Related reading: [KernelBench](01-kernelbench.md), [KernelBench-Verified](03-kernelbench-verified.md), [MT-Bench/Chatbot Arena](16-mt-bench-chatbot-arena.md), and [Generation to Judgment](13-generation-to-judgment.md). These are navigation links; matched comparisons require checking each protocol.

## Source coverage and verification notes

- Read §§1–5, the reporting checklist, and Appendices A–G, including all likelihood/perplexity/generation descriptions, all five example task configurations, case-study details, hardware, and best-practice checklist.
- Verified Table 1 values and interval provenance, separated the cited ten-run GPQA example from original experiments, and checked PDF pages 26 and 31 for the BPB sign problem and incomplete limitations text.
- Equations and implementation descriptions are analyzed as written, not executed or endorsed as bug-free. Source-version library statements have not been checked against a current installation.
- Bibliographic works were not independently reviewed; external claims remain attributed through this paper.
