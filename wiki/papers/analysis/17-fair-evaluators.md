---
paper_id: "17"
title: "Large Language Models are not Fair Evaluators"
source_version: "2305.17926v2"
source_markdown: "../pdfs/17-fair-evaluators.md"
source_pdf: "../pdfs/17-fair-evaluators.pdf"
source_url: "https://arxiv.org/abs/2305.17926"
analysis_date: "2026-11-20"
analysis_status: "reviewed"
topics: ["llm-as-a-judge", "position-bias", "calibration", "human-in-the-loop", "uncertainty", "pairwise-evaluation"]
summary: "A study of response-order bias in pairwise LLM judging, with evidence-first sampling, balanced positions, and entropy-based human review."
---

# Large Language Models are not Fair Evaluators

## Paper identity and scope

- Studies **positional bias** when an LLM evaluates two assistant responses; “fair” in the title concerns sensitivity to presentation position, not a comprehensive demographic-fairness claim. [§§1–2](../pdfs/17-fair-evaluators.md)
- The core perturbation swaps the same two answers without changing their contents or the question. An order-induced change therefore challenges the stability of the evaluation procedure. [§2.1](../pdfs/17-fair-evaluators.md)
- Contributions are an empirical bias demonstration, three calibration methods, and human win/tie/lose annotations for 80 Vicuna Benchmark questions. [§1; §4.1](../pdfs/17-fair-evaluators.md)
- This analysis concerns arXiv 2305.17926v2 and the historical models/settings in that version. It does not assert the same bias magnitude for current models. [Source metadata; §4.2](../pdfs/17-fair-evaluators.md)

## Concepts and terminology

### Judge setup and failure mode

- A template `T(q,r1,r2)` contains a user question and two responses, conventionally labeled Assistant 1 and Assistant 2. [§2.1; Table 1](../pdfs/17-fair-evaluators.md)
- The baseline scoring prompt requests helpfulness, relevance, accuracy, and level of detail, then two scores on a 1–10 scale. [Table 1](../pdfs/17-fair-evaluators.md)
- Its original output order puts scores before the explanation. The proposed evidence-first template reverses that sequence. [Tables 1, 3; §3.1](../pdfs/17-fair-evaluators.md)
- The baseline explicitly asks the judge to avoid order effects, yet the measured bias remains; an instruction to be unbiased is not sufficient in these experiments. [Table 1; Figure 1; Table 2](../pdfs/17-fair-evaluators.md)
- **Conflict Rate** is the fraction of examples whose identity-normalized win/tie/lose verdict changes when answer order is swapped. Identity normalization is necessary so a label change caused solely by relabeling assistants is not mistaken for a changed winner. [§2.2; Eq. 1](../pdfs/17-fair-evaluators.md)
- **Position preference** is directional: a judge may favor the first or second slot. **Conflict rate** measures inconsistency and does not by itself specify which position is favored. [§2.2; Table 2](../pdfs/17-fair-evaluators.md)
- **Score gap** is the difference in the judge's ratings; the paper examines whether near-tied responses have greater order sensitivity. It is not an independently measured true quality distance. [§2.2; Figure 2](../pdfs/17-fair-evaluators.md)

### Calibration methods

- **Evidence Calibration (EC):** ask for evaluative evidence before scoring, so the autoregressive score generation can condition on that evidence. [§3.1; Table 3](../pdfs/17-fair-evaluators.md)
- The authors' rationale is that an explanation generated after a score cannot have influenced that already-emitted score. This motivates output-order changes; it does not guarantee the preceding rationale is correct. [§3.1](../pdfs/17-fair-evaluators.md)
- **Multiple Evidence Calibration (MEC):** sample `k` evidence-first judgments rather than rely on a single rationale/score pair. [§3.1](../pdfs/17-fair-evaluators.md)
- **Balanced Position Calibration (BPC):** also evaluate `T(q,r2,r1)`, giving every candidate both presentation positions. [§3.2](../pdfs/17-fair-evaluators.md)
- Combined MEC+BPC obtains `2k` scores per response, averages that response's scores across both positions, and selects the higher average as better. [§3.2; Eq. 2](../pdfs/17-fair-evaluators.md)
- Equal allocation to positions is distinct from merely drawing more judgments in one fixed ordering; the ablation compares these at the same total sample count. [§4.3; Table 4](../pdfs/17-fair-evaluators.md)
- **Human-in-the-Loop Calibration (HITLC):** route selected examples to people after automatic calibration, rather than require full human annotation of all future comparisons. [§3.3](../pdfs/17-fair-evaluators.md)
- **Balanced Position Diversity Entropy (BPDE):** compute the frequency of win/tie/lose outcomes across the `2k` individual judgments, then Shannon entropy `-sum p(outcome) log p(outcome)`. [§3.3; Eqs. 3–5](../pdfs/17-fair-evaluators.md)
- BPDE is based on categorical verdict diversity, not the variance of raw scores or a calibrated probability of error. [Eqs. 3–5](../pdfs/17-fair-evaluators.md)
- High entropy flags comparisons whose judgments disagree; select the top `beta` fraction for manual correction. [§3.3](../pdfs/17-fair-evaluators.md)
- The human labels on selected items are integrated by majority opinion. Sampling and aggregation retain response identity across order swaps. [§§3.2–3.3, 4.1](../pdfs/17-fair-evaluators.md)
- **Vanilla Diversity Entropy** uses same-position repeated judgments; BPDE adds the information exposed by order reversal. The comparison fixes six total judgments for both. [§5.2](../pdfs/17-fair-evaluators.md)
- **Scoring** predicts numeric ratings for both candidates; **Comparing** directly returns Assistant 1, Assistant 2, or Same. The paper tests calibration beyond the original numeric format. [§5.3](../pdfs/17-fair-evaluators.md)

## Methodology and experimental setup

- Evaluation covers **80 questions in nine categories** from the Vicuna Benchmark. Main human-alignment tests compare ChatGPT and Vicuna-13B answers. [§1; §4.1](../pdfs/17-fair-evaluators.md)
- The positional-bias demonstration additionally compares Vicuna-13B with Alpaca-13B, allowing inspection of an easier-to-distinguish pair. [§2.2; Table 2](../pdfs/17-fair-evaluators.md)
- Three authors independently annotate win/tie/lose, using the same four quality dimensions and randomized response order. They are AI researchers. [§4.1](../pdfs/17-fair-evaluators.md)
- Annotation takes approximately three minutes per example, and the final comparison target is the majority opinion of the three annotators. [§4.1](../pdfs/17-fair-evaluators.md)
- ChatGPT's API identifier is `gpt-3.5-turbo-0301`; GPT-4 uses `gpt-4`. [§4.2](../pdfs/17-fair-evaluators.md)
- Single-generation methods use temperature 0; multiple-evidence settings default to temperature 1 and `k=3`. [§4.2](../pdfs/17-fair-evaluators.md)
- For methods without BPC, the authors randomize answer order and average results over **100 runs** to stabilize reported evaluation values. Fractional accuracy values should not be interpreted as a single integer correct-count out of 80. [§4.2](../pdfs/17-fair-evaluators.md)
- Metrics are accuracy against the final majority labels and a kappa agreement coefficient. Human rows also compare individual annotators with those majority labels. [§4.2; Table 4](../pdfs/17-fair-evaluators.md)
- Tested ablations include evidence count, temperature, human-review selection, scoring versus direct-comparison templates, and performance by category. [§5](../pdfs/17-fair-evaluators.md)
- Cost values use contemporaneous API costs and the paper's labor assumption of approximately $7.50/hour and three minutes per annotation. They are study-specific estimates, not current prices. [§4.3, footnote; Table 4](../pdfs/17-fair-evaluators.md)

## Results and evidence

### Response-order effects

- With GPT-4 judging Vicuna versus ChatGPT, Vicuna's win rate changes from **51.3% in position 1 to 23.8% in position 2**; verdicts conflict on **37/80 (46.3%)** comparisons. [Table 2](../pdfs/17-fair-evaluators.md)
- With ChatGPT judging the same pair, Vicuna changes from **2.5% in position 1 to 82.5% in position 2**, with **66/80 (82.5%)** conflicting verdicts. [Table 2](../pdfs/17-fair-evaluators.md)
- The abstract's statement that Vicuna can beat ChatGPT on 66/80 queries describes a manipulated evaluation configuration. It is not evidence that Vicuna is actually the stronger assistant. [Abstract; Table 2](../pdfs/17-fair-evaluators.md)
- For Vicuna versus Alpaca, GPT-4 gives **92.5%** Vicuna wins in both orders, yet still has **4/80 (5.0%)** individual conflicts. Equal aggregate win rates therefore need not imply per-example stability. [Table 2](../pdfs/17-fair-evaluators.md)
- ChatGPT's Vicuna-versus-Alpaca win rates are **37.5% and 90.0%**, with **42/80 (52.5%)** conflicts. [Table 2](../pdfs/17-fair-evaluators.md)
- Under this template GPT-4 tends to prefer the first position and ChatGPT the second. That direction is a property of these settings, not a general law about model families. [§2.2](../pdfs/17-fair-evaluators.md)
- Near-tied scores, such as gaps at most 1, have more order instability than large gaps, such as at least 3; the authors connect this to greater difficulty separating similar responses. [§2.2; Figure 2](../pdfs/17-fair-evaluators.md)

### Calibration and agreement

- Individual human accuracies against the final vote are **68.8%, 76.3%, and 70.0%**; their mean is **71.7%**, with mean kappa **0.54**. This is agreement with the study's majority labels, not universal human accuracy. [Table 4](../pdfs/17-fair-evaluators.md)
- GPT-4 improves from **52.7% / 0.24 kappa** (vanilla) to **56.5% / 0.29** (EC), **58.7% / 0.30** (MEC `k=3`), and **62.5% / 0.37** (MEC+BPC, three samples per position). [Table 4](../pdfs/17-fair-evaluators.md)
- GPT-4's vanilla-to-MEC+BPC improvement is **9.8 percentage points**. The paper sometimes calls this “9.8%”; the difference of displayed percentages is a percentage-point change. [Table 4; §1](../pdfs/17-fair-evaluators.md)
- ChatGPT improves from **44.4% / 0.06** to **52.6% / 0.23** (EC), **53.2% / 0.24** (MEC `k=3`), and **58.7% / 0.31** (MEC+BPC). [Table 4](../pdfs/17-fair-evaluators.md)
- ChatGPT's vanilla-to-MEC+BPC change is **14.3 percentage points**, not a 14.3% relative increase. [Table 4; §4.3](../pdfs/17-fair-evaluators.md)
- At six total judgments, GPT-4 MEC-only reaches **60.9%** versus **62.5%** for balanced positions; ChatGPT reaches **55.6%** versus **58.7%**. This comparison supports a benefit from position balance beyond sample count. [Table 4; §4.3](../pdfs/17-fair-evaluators.md)
- Adding HITLC on the top **20%** of examples yields **73.8% / 0.56 kappa** for GPT-4 and **71.3% / 0.52** for ChatGPT. These are hybrid human/model results, not autonomous judge accuracy. [Table 4](../pdfs/17-fair-evaluators.md)
- The cost table lists **$30** for one annotator's full set, **$23.10** for GPT-4 with HITLC, and **$18.30** for ChatGPT with HITLC. The claimed **39%** saving compares $18.30 with $30. [Table 4; §4.3](../pdfs/17-fair-evaluators.md)
- Automatic balanced evaluation costs **$6.38** for GPT-4 and **$0.34** for ChatGPT in the reported setup, versus vanilla **$2.00** and **$0.10**. Calibration has an inference-cost tradeoff before adding human labor. [Table 4](../pdfs/17-fair-evaluators.md)

### Ablations and generalization

- Evidence counts **1, 3, 5, and 7** are explored for ChatGPT. Gains level off or slightly decline beyond a moderate sample count; `k=3` is selected as a useful quality/cost balance. [§5.1; Figure 4](../pdfs/17-fair-evaluators.md)
- Temperatures **0.2** and **1.4** underperform intermediate settings in the study. The authors attribute this to insufficient diversity at low temperature and poorer generation quality at high temperature. [§5.1](../pdfs/17-fair-evaluators.md)
- Both diversity-based human-review selectors outperform random selection in the reported comparison, and BPDE outperforms one-position entropy at equal judgment count. [§5.2; Figure 5](../pdfs/17-fair-evaluators.md)
- For ChatGPT's numeric-scoring template, MEC reduces conflict from **82.5% to 35.0%** while improving accuracy from **44.4% to 53.2%**. [Table 5](../pdfs/17-fair-evaluators.md)
- For direct comparison, vanilla accuracy/kappa/conflict is **50.2% / 0.18 / 50.0%**; MEC gives **54.8% / 0.27 / 42.5%**; MEC+BPC gives **60.3% / 0.35**. [Table 5](../pdfs/17-fair-evaluators.md)
- Conflict is listed **N/A** for balanced methods, not zero: the reported single-order conflict metric is not applicable to the already-aggregated BPC output. [Table 5 caption](../pdfs/17-fair-evaluators.md)
- The category analysis reports stronger GPT-4 judging on challenging common-sense, coding, and math tasks and improvements from calibration in nearly all categories. Exact per-category values are not inferred from chart geometry here. [§5.4; Figure 6](../pdfs/17-fair-evaluators.md)

## Authors' conclusions

- LLM judges can exhibit severe systematic order dependence, allowing response rankings to be manipulated without changing content. [§7](../pdfs/17-fair-evaluators.md)
- Generating evidence before scores, sampling multiple evidence paths, and balancing positions improve human alignment in the tested settings. [§7](../pdfs/17-fair-evaluators.md)
- BPDE provides a useful signal for choosing which comparisons should receive human attention. [§§3.3, 5.2, 7](../pdfs/17-fair-evaluators.md)
- The authors present the calibration framework as a practical, relatively inexpensive improvement and release human annotations and code for further work. [§7](../pdfs/17-fair-evaluators.md)

## Limitations and open questions

- **Source-supported scope:** results come from 80 questions and a small set of historical model pairs; neither all task domains nor all evaluator designs are covered. [§§2.2, 4.1–4.2](../pdfs/17-fair-evaluators.md)
- **Analyst caveat:** the reference labels are a majority of three author-annotators. They are useful supervision but not objective correctness certificates or a representative survey of human preferences. [Evidence: §4.1](../pdfs/17-fair-evaluators.md)
- **Analyst caveat:** individual-human agreement with a majority containing that annotator is not independent human-versus-ground-truth validation. Interpret the “Human Average” comparison accordingly. [Evidence: §4.1; Table 4](../pdfs/17-fair-evaluators.md)
- **Analyst caveat:** consistency does not imply correctness. A model can consistently give the same mistaken answer and consequently have low BPDE. [Mechanism: Eqs. 3–5](../pdfs/17-fair-evaluators.md)
- **Analyst caveat:** BPC controls a known presentation variable but does not establish the absence of verbosity, style, factual, or domain-knowledge biases. [Study scope: §§2–3](../pdfs/17-fair-evaluators.md)
- **Analyst caveat:** repeated rationales can share systematic errors; agreement among samples is not independent corroboration. [Mechanism: §3.1](../pdfs/17-fair-evaluators.md)
- **Analyst caveat:** the cost comparison is tied to the paper's labor and API assumptions, and HITLC results include human contribution. It should not be presented as a current deployment cost estimate. [§4.3; Table 4](../pdfs/17-fair-evaluators.md)
- **Open question:** the paper establishes useful correlations between disagreement and difficulty, but does not prove that entropy thresholds transfer unchanged to other tasks, models, or response-quality distributions. [§§3.3, 5.2](../pdfs/17-fair-evaluators.md)

## Relevance to this project — interpretation

- **Proposal:** if comparing agent explanations, plans, or patch quality with a judge, keep candidate identity separate from display order, evaluate both orders, and retain raw ratings before aggregation. [Motivation: §§2–3](../pdfs/17-fair-evaluators.md)
- **Proposal:** calibrate any subjective evaluator against task-specific human-reviewed examples; store conflict rate and categorical disagreement alongside mean score. [Motivation: §§2.2, 4](../pdfs/17-fair-evaluators.md)
- **Proposal:** use order/sample disagreement to prioritize manual review, while also sampling low-disagreement cases to detect confidently repeated mistakes. The latter is an analyst extension, not a tested method in this paper. [Motivation: §3.3](../pdfs/17-fair-evaluators.md)
- **Proposal:** use executable correctness checks and measured speed for kernels; presentation-order controls are complementary for subjective judgments and do not validate numerical code. [Boundary: §§1, 4](../pdfs/17-fair-evaluators.md)
- **Proposal:** when comparing judge methods, match total calls/tokens and human effort, and distinguish autonomous from human-assisted results. [Motivation: Table 4; §5.2](../pdfs/17-fair-evaluators.md)

## Retrieval notes

- Aliases: FairEval, positional bias, EC, MEC, BPC, HITLC, BPDE, answer-order swapping, balanced position diversity entropy.
- Questions answered: Can rankings flip when answers swap places? Does rationale-first help? Is balanced sampling better than more same-order samples? How is BPDE calculated? What does the 66/80 claim actually mean?
- Navigate to [MT-Bench/Chatbot Arena](16-mt-bench-chatbot-arena.md) for a wider judge-bias discussion, [G-Eval](15-g-eval.md) for criterion-specific direct scoring, and [Generation to Judgment](13-generation-to-judgment.md) for the broader survey. These links identify related reading, not matched comparisons.

## Source coverage and verification notes

- Read §§1–7, all five tables, all five equations, and the discussions accompanying Figures 1–6. This saved version has no substantive appendix after its references.
- Checked quantitative claims against displayed tables, preserving fractional accuracies from repeated randomized-order runs and the BPC conflict-rate N/A convention.
- Chart conclusions are reported only where stated by the authors in the text; no exact unprinted values were estimated from figures. Bibliography entries were not independently analyzed.
- This analysis does not reproduce the experiments or prove the judge's fairness; it records evidence, methods, and their boundaries.
