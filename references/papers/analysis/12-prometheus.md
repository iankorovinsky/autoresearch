---
paper_id: "12"
title: "Prometheus: Inducing Fine-grained Evaluation Capability in Language Models"
source_version: "2310.08491v2"
source_markdown: "../pdfs/12-prometheus.md"
source_pdf: "../pdfs/12-prometheus.pdf"
source_url: "https://arxiv.org/abs/2310.08491"
analysis_date: "2026-11-20"
analysis_status: "reviewed"
topics: ["llm-as-a-judge", "rubrics", "reference-guidance", "judge-training", "synthetic-data", "feedback", "human-alignment", "reward-models"]
summary: "Trains 7B/13B rubric-conditioned evaluators on synthetic feedback and studies reference materials, human agreement, distribution shift, and preference ranking."
---

# Prometheus: Inducing Fine-grained Evaluation Capability in Language Models

## Paper identity and scope

- Introduces **Prometheus**, an evaluator fine-tuned from Llama-2-Chat, and **Feedback Collection**, a synthetic rubric-conditioned evaluation dataset. This analysis refers to the original Prometheus paper, arXiv 2310.08491v2, not later similarly named models. [§§1, 3; source metadata](../pdfs/12-prometheus.md)
- The intended capability is scoring long-form responses under **user-specified criteria**, such as child-readability or formality, rather than assuming one universal helpfulness preference. [§1; Figure 1](../pdfs/12-prometheus.md)
- Motivations include control over model versions, inspectability, and avoiding per-call proprietary evaluation costs. The paper's price examples are historical motivations, not current cost estimates. [§1](../pdfs/12-prometheus.md)
- Its strongest human-correlation result is conditioned on a small, selected evaluation set and appropriate reference materials. “On par with GPT-4” is not a blanket claim across all tasks or all reported tables. [§§5.1–5.2, 6.2; Appendix J](../pdfs/12-prometheus.md)

## Concepts and terminology

### Evaluator interface

- Each example supplies **instruction, candidate response, customized score rubric, and reference answer**; output is **feedback followed by an integer score from 1 to 5**. [§3; Figure 2](../pdfs/12-prometheus.md)
- A score rubric includes a criterion description and separate descriptions of what earns each score. The score labels are semantic anchors, not merely an unexplained numeric range. [§3](../pdfs/12-prometheus.md)
- The reference answer is intended to earn a **5 under that rubric**, making it criterion-specific rather than necessarily an all-purpose ideal answer. [§3](../pdfs/12-prometheus.md)
- Reference guidance separates solving the instruction from judging a candidate: the evaluator can compare with a provided solution instead of generating the ideal answer internally. [§6.1](../pdfs/12-prometheus.md)
- Feedback is a rationale for the rating and is generated before the final score. The paper treats it as analogous to chain-of-thought supervision and as useful explanatory output. [§3.2](../pdfs/12-prometheus.md)
- The `[RESULT]` separator makes the final score easier to parse and helps prevent generation degeneration; structured output is explicitly learned during fine-tuning. [§3.2; Appendix C; Appendix H](../pdfs/12-prometheus.md)
- **Absolute grading** assesses one candidate on a fixed scale without an opponent; **ranking grading** compares independently scored candidates with human preferences. [§4.1](../pdfs/12-prometheus.md)
- An absolute evaluator's tied integer scores create a problem for binary preference tests; the paper's ranking procedure sometimes resamples until a winner emerges. [§4.1](../pdfs/12-prometheus.md)
- A **verbalizer** maps alternative score formats such as “Score: 4 out of 5” into a structured numeric value. It is used for untuned Llama-2-Chat baselines, not Prometheus. [Appendix C; Table 9](../pdfs/12-prometheus.md)

### Synthetic data and reference distribution

- **Feedback Collection** is training data; **Feedback Bench** is a distinct evaluation construction using the same generation procedure. Similar names should not be treated as identical splits. [§§3, 4.1](../pdfs/12-prometheus.md)
- The training dataset has **1,000 score rubrics, 20,000 instructions/reference answers, and 100,000 candidate responses with feedback**. There are 20 instructions per rubric and five scored responses per instruction. [Table 1](../pdfs/12-prometheus.md)
- Scores 1–5 each have **20,000** training responses to avoid teaching a skewed rating prior. Balanced synthetic labels do not establish calibration on a deployment distribution. [§3; Table 1](../pdfs/12-prometheus.md)
- Responses across score levels are generated to have similar lengths, intended to reduce a spurious association between length and quality. [§3; Appendix A](../pdfs/12-prometheus.md)
- Two distinct score-5 responses are generated: one serves as the reference input and another as the score-5 candidate. This avoids literal identity being the sole positive example. [§3.1](../pdfs/12-prometheus.md)
- **Seen-rubric evaluation** uses the 1,000 training rubrics with new instructions; **unseen-rubric evaluation** uses 50 new rubrics and 1,000 new instructions. [§4.1; Appendix B](../pdfs/12-prometheus.md)
- Lexical overlap checks using ROUGE-L are evidence about wording overlap, not a formal demonstration that train/test criteria are semantically independent. The stronger independence interpretation is an analyst caveat. [Appendices A–B](../pdfs/12-prometheus.md)

## Methodology and experimental setup

### Dataset construction and training

- Authors first curate **50 seed rubrics**; GPT-4 expands them by conditioning on four randomly sampled examples. [§3.1](../pdfs/12-prometheus.md)
- Rubric brainstorming and paraphrasing are iterated for **ten rounds**, aiming for both new criteria and robustness to differently worded criteria. [§3.1](../pdfs/12-prometheus.md)
- GPT-4 generates instructions relevant to each rubric, then reference answers, candidate responses at requested scores, and explanatory feedback. Score labels are deliberately requested during generation. [§3.1; Appendix G](../pdfs/12-prometheus.md)
- The generation prompt ties candidate sentence count to the reference's sentence count, attempting to keep quality differences from being reducible to length. [Appendix G](../pdfs/12-prometheus.md)
- Generating all fields simultaneously produced poor separation between scores in early experiments; the authors group instruction/reference generation separately from response/feedback generation. [Appendix G](../pdfs/12-prometheus.md)
- Prometheus-7B and -13B are fine-tuned from corresponding Llama-2-Chat models to generate feedback and then scores; training loss is on those output tokens, not on input instructions. [§3.2; Appendix C](../pdfs/12-prometheus.md)
- Training uses **8 A100 80GB GPUs**, PyTorch FSDP, AdamW, learning rate **1e-5**, StepLR, and maximum input-plus-output length **4,096**. Batch sizes are **28 for 7B** and **20 for 13B**. [Appendix C; Table 8](../pdfs/12-prometheus.md)
- Inference uses temperature **1.0**, top-p **0.9**, maximum output **256**, and repetition penalty **1.03** for open models. These settings matter when interpreting score variance and reproducing feedback quality. [Appendix C; Table 9](../pdfs/12-prometheus.md)
- The authors report poor feedback at temperature 0 in their tests; that observation is not a universal rule for all evaluators. [Appendix C](../pdfs/12-prometheus.md)
- Prometheus uses its evaluation prompt inside the Llama-2 conversation wrapper, maintaining training-format compatibility. [Appendix H](../pdfs/12-prometheus.md)

### Test sets, humans, and controls

- Human scoring uses **45 instances with distinct rubrics** from unseen Feedback Bench, Vicuna Bench, and MT Bench; **math and coding are excluded**. [§5.1; §6.2; Appendix J](../pdfs/12-prometheus.md)
- Nine crowd workers are split into three feedback-comparison groups: Prometheus/GPT-4, Prometheus/GPT-3.5, and GPT-4/GPT-3.5. They rate responses, choose better feedback, and explain rejection reasons. [§4.1; Appendix J](../pdfs/12-prometheus.md)
- Feedback display order is randomized; annotation questions are sequential to obtain response scores before feedback preference. Each annotator is asked to spend at least three hours on 45 instances and receives $50. [Appendix J](../pdfs/12-prometheus.md)
- Larger absolute-grading evaluation spans **1,222 rubrics and 2,360 instances** across Feedback Bench, Vicuna Bench, MT Bench, and FLASK Eval. Preserve those source units instead of conflating them with total sampled model judgments. [§5.2](../pdfs/12-prometheus.md)
- Feedback Bench targets are scores assigned during GPT-4-0613 data generation; other benchmark targets are scores obtained by running GPT-4-0613 as an evaluator. These are different notions of “agreement with GPT-4.” [§5.2](../pdfs/12-prometheus.md)
- Vicuna Bench and MT Bench use 80 prompts each with handcrafted rubrics and GPT-4-generated reference answers. MT Bench judges the last turn with preceding dialogue as context. [§4.1](../pdfs/12-prometheus.md)
- FLASK Eval has 200 prompts and 12 evaluation skills, including logical robustness/correctness/efficiency, factuality, commonsense, harmlessness, readability, comprehension, insightfulness, completeness, metacognition, and conciseness. [§4.1; Appendix D](../pdfs/12-prometheus.md)
- Baselines include Llama-2-Chat 7B/13B/70B, a coarse-rubric-trained 13B model, GPT-3.5-turbo-0613, several GPT-4 versions, and two reward models for ranking tests. [§4.2](../pdfs/12-prometheus.md)
- The coarse model trains on a FLASK split, reduced to **30K** balanced examples from **67.5K responses / 1.5K training instructions**; thus it differs in data source and size as well as rubric granularity. [Appendix D](../pdfs/12-prometheus.md)
- Correlation tables use three inference samples; GPT-4-0613 uses six in the external-benchmark comparison to assess self-consistency between score sets. [Tables 3–4 captions](../pdfs/12-prometheus.md)
- Ranking tests use **HHH Alignment (221 pairs)** and MT Bench Human Judgments. Reference answers are omitted in this setting to test reward-model-like use. [§4.1](../pdfs/12-prometheus.md)
- HHH comparisons resample independent scores at temperature 1 until a winner; MT Bench permits ties and does not need forced tie breaking. The authors explicitly say comparison with ranking models is **not exactly fair** and not designed to establish state of the art. [§4.1](../pdfs/12-prometheus.md)

## Results and evidence

### Human scoring and feedback preference

- Human Pearson correlation on the selected 45-instance setup is **0.897 Prometheus, 0.882 GPT-4, and 0.392 GPT-3.5**. This is correlation on the selected rubric-conditioned examples, not accuracy across all tasks. [§5.1; Figure 4](../pdfs/12-prometheus.md)
- Humans prefer Prometheus feedback over GPT-4 **58.62%** and over GPT-3.5 **79.57%** in the reported pairwise comparison. The introduction says **58.67%** for the former; the main result, figure caption, and conclusion say **58.62%**. [§1; §5.1; Figure 5; §7](../pdfs/12-prometheus.md)
- Rejection analysis finds GPT-4 feedback more often too general/abstract, while Prometheus feedback is more often overly critical. Criticalness can be useful or undesirable depending on the objective. [§5.1; Figure 6; Appendix E](../pdfs/12-prometheus.md)
- The authors hypothesize that specialization on evaluation produces a stronger tendency to criticize; this is an explanatory conjecture, not an isolated causal ablation. [§5.1; Appendix E](../pdfs/12-prometheus.md)

### Synthetic distribution versus external benchmarks

- Prometheus-13B Pearson correlation on seen/unseen Feedback Bench is **0.861/0.860**, versus base Llama-2-Chat-13B **0.441/0.450** and Llama-2-Chat-70B **0.572/0.558**. [Table 3](../pdfs/12-prometheus.md)
- Prometheus-7B is also strong on this distribution, **0.860/0.847**; the extra model size does not create a large gap there. [Table 3](../pdfs/12-prometheus.md)
- GPT-4-0613's agreement with the synthetic assigned scores is **0.742/0.743**. This does not mean Prometheus is generally more accurate than its teacher; the targets come from controlled score-conditioned generation on a familiar distribution. [Table 3; §6.2](../pdfs/12-prometheus.md)
- **Source arithmetic discrepancy:** §5.2 claims an unseen-rubric 13B gain of **+0.397**, while Table 3's **0.860 − 0.450 = 0.410**. The table cells are preserved; no silent reconciliation is attempted. [§5.2; Table 3](../pdfs/12-prometheus.md)
- On external benchmarks, Prometheus-13B Pearson is **0.466 Vicuna, 0.473 MT Bench, 0.467 FLASK**, versus GPT-3.5 **0.270/0.275/0.422**. [Table 4](../pdfs/12-prometheus.md)
- GPT-4-0613 self-consistency is much higher on those datasets, **0.925/0.952/0.835**. The paper explicitly acknowledges Prometheus still lags GPT-4 in this setting. [Table 4; §5.2](../pdfs/12-prometheus.md)
- Prometheus-13B improves over base 13B by **0.255, 0.493, and 0.202** Pearson on those three datasets; these are correlation differences, not percentage-point accuracies. [Table 4; §5.2](../pdfs/12-prometheus.md)
- The coarse FLASK-trained model scores **0.517 Pearson on FLASK**, above Prometheus's **0.467**, supporting task-distribution specialization within this experiment. It is trained on a FLASK training split, not necessarily the held-out evaluation instances. [Table 4; Appendix D](../pdfs/12-prometheus.md)
- The authors suggest long, detailed synthetic instructions versus shorter external instructions as one possible explanation for distribution-shift losses. This remains a hypothesis. [§5.2](../pdfs/12-prometheus.md)

### Ranking transfer and ablations

- Prometheus-13B achieves **79.19% HHH total** and **57.72% MT Bench Human Judgment**, versus base 13B **73.76%/52.34%** and GPT-4 **88.69%/63.87%**. [Table 5](../pdfs/12-prometheus.md)
- The improvement over base 13B is **5.43 and 5.38 percentage points**, respectively. It is a transfer result from absolute-grading training to a ranking evaluation. [Table 5; §5.3](../pdfs/12-prometheus.md)
- Prometheus-7B achieves **80.09% HHH** and **55.14% MT Bench**; the 13B model is not uniformly better across these ranking totals. [Table 5](../pdfs/12-prometheus.md)
- The evaluated reward models score **58.82% / 44.79%** (StanfordNLP) and **76.02% / 49.90%** (ALMOST). Comparisons must retain the resampling and reference-omission caveats. [Table 5; §4.1](../pdfs/12-prometheus.md)
- For 7B, removing reference answers lowers seen/unseen/Vicuna Pearson from **0.860/0.847/0.457** to **0.642/0.626/0.349**. [Table 6](../pdfs/12-prometheus.md)
- Removing feedback distillation gives **0.668/0.673/0.413**; removing score rubrics gives **0.837/0.745/0.355**. Each component contributes, with effects depending on the test distribution. [Table 6](../pdfs/12-prometheus.md)
- Using Code-Llama as the 7B base gives **0.470 Vicuna Pearson** versus **0.457** for the main 7B evaluator, while synthetic-set correlations are lower. The code-domain argument also draws on only seven code instances in Vicuna Bench. [Table 6; §6.5](../pdfs/12-prometheus.md)
- Training-data analysis finds response lengths around 200 tokens across score levels and reports bigram/trigram diversity. These checks support dataset construction goals but do not establish adversarial robustness. [Appendix A; Table 7](../pdfs/12-prometheus.md)
- Length-versus-score plots show no strong length preference in the tested datasets, but those datasets omit adversarial verbosity examples; the authors explicitly leave that bias question open. [Appendix F](../pdfs/12-prometheus.md)

## Authors' conclusions

- Reference materials plus feedback fine-tuning can produce a capable open evaluator that follows diverse user-defined rubrics. [§7](../pdfs/12-prometheus.md)
- Reference answers reduce the burden of independently solving the task, while rubrics identify the intended quality dimension; feedback supervision helps train the evaluation process. [§6.1](../pdfs/12-prometheus.md)
- The authors view the human scoring and feedback-preference results as evidence that a specialized smaller evaluator can approach GPT-4 on selected non-code/non-math settings. [§6.2](../pdfs/12-prometheus.md)
- Training on a target domain's feedback data may improve specialization, at the cost of collecting new data; base-model expertise may also matter. [§6.5](../pdfs/12-prometheus.md)
- Ranking transfer suggests possible reward-model use, but actual RLHF with Prometheus as the reward is future work, not an experiment completed here. [§5.3](../pdfs/12-prometheus.md)
- Practitioner guidance is to choose evaluation instructions, specify rubrics, prepare suitable references, generate candidate outputs, then score and inspect feedback. [§6.4](../pdfs/12-prometheus.md)

## Limitations and open questions

- **Source-explicit:** the human study excludes math and coding and is small; the strongest headline should not be generalized to those domains. [§6.2; Appendix J](../pdfs/12-prometheus.md)
- **Source-explicit:** train/test generation similarity may explain particularly high Feedback Bench results; external-benchmark performance is lower. [§§5.2, 6.2](../pdfs/12-prometheus.md)
- **Source-explicit:** ranking comparisons use iterative tie-breaking and are not directly fair to all reward-model baselines. [§4.1](../pdfs/12-prometheus.md)
- **Source-explicit:** adversarial length bias, domain-specialized judging, additional reference materials, and actual RLHF usage remain untested or future work. [§§5.3, 6.5; Appendix F](../pdfs/12-prometheus.md)
- **Analyst caveat:** GPT-4-derived data, reference answers, and GPT-4-based evaluation create shared teacher dependence; high teacher correlation is not independent truth validation. [Evidence: §§3.1, 4.1, 5.2](../pdfs/12-prometheus.md)
- **Analyst caveat:** the coarse-versus-fine comparison changes dataset source and number of training examples, so it does not isolate rubric count as the only causal variable. [Evidence: Appendix D; Table 1](../pdfs/12-prometheus.md)
- **Analyst caveat:** absence of a second candidate avoids the specific pair-order comparison bias, but does not prove that absolute scoring is free from all positional or contextual biases. [Source claim and scope: Appendix F](../pdfs/12-prometheus.md)
- **Analyst caveat:** a bad reference can anchor a judge incorrectly; the reference-answer ablation establishes usefulness under the supplied references, not safety under corrupted references. [Evidence: §6.1](../pdfs/12-prometheus.md)
- **Open question:** how much dataset/rubric/teacher shift can occur before the observed correlations fail, and how should references be independently audited? The paper does not settle these deployment questions. [Motivation: §§5.2, 6.4](../pdfs/12-prometheus.md)

## Relevance to this project — interpretation

- **Proposal:** for subjective kernel-agent qualities, define explicit score anchors and distinguish criterion-specific references from executable correctness or performance baselines. [Motivation: §§3, 6.1](../pdfs/12-prometheus.md)
- **Proposal:** version rubric, reference, judge weights, output parser, and sampling settings together so stored scores remain interpretable. [Motivation: §1; Appendix C](../pdfs/12-prometheus.md)
- **Proposal:** independently test judge transfer on our actual code and trajectory data; the 45-instance human result is not sufficient evidence for kernel verification. [Motivation: §6.2](../pdfs/12-prometheus.md)
- **Proposal:** use feedback as inspectable advice or an auxiliary signal, while letting executed tests and measured performance decide objective kernel success. This is an analyst boundary, not an implementation prescribed by the paper. [Scope: §§1, 4–5](../pdfs/12-prometheus.md)
- **Proposal:** if training a critic, vary response quality without automatically varying length, balance labels deliberately, and test held-out rubrics plus distribution shift. [Motivation: §3; Appendices A–B, G](../pdfs/12-prometheus.md)

## Retrieval notes

- Aliases: original Prometheus, Feedback Collection, Feedback Bench, fine-grained rubric, reference answer, feedback distillation, absolute grading, ranking grading.
- Questions answered: What inputs does Prometheus require? How are synthetic low-quality responses generated? Why do references help? Does 0.897 apply to coding? Is it proven as an RLHF reward? Why are synthetic and external correlations so different?
- Related reading: [G-Eval](15-g-eval.md), [MT-Bench/Chatbot Arena](16-mt-bench-chatbot-arena.md), [Fair Evaluators](17-fair-evaluators.md), and [Generation to Judgment](13-generation-to-judgment.md). Links are for topic navigation, not performance ranking.

## Source coverage and verification notes

- Read §§1–7 and Appendices A–J, including all nine tabulated datasets/results/configurations, data-generation and evaluator prompts, human-study design, length/diversity checks, and qualitative-example descriptions.
- Individual illustrated feedback examples in Appendix I are represented here through the authors' accompanying analysis and captions; their entire screenshot text is not independently transcribed. No visual numerical values were guessed.
- Recorded the 58.67/58.62 feedback-preference mismatch and +0.397 versus table-derived +0.410 discrepancy. Numerical bullets follow the explicitly named tables/sections.
- No new training or evaluator runs were performed. Source conclusions, empirical observations, analyst caveats, and project proposals are intentionally separated.
