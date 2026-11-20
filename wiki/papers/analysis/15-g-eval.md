---
paper_id: "15"
title: "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"
source_version: "2303.16634v3"
source_markdown: "../pdfs/15-g-eval.md"
source_pdf: "../pdfs/15-g-eval.pdf"
source_url: "https://arxiv.org/abs/2303.16634"
analysis_date: "2026-11-20"
analysis_status: "reviewed"
topics: ["llm-as-a-judge", "evaluation", "rubrics", "chain-of-thought", "human-alignment", "bias", "summarization"]
summary: "Prompted NLG evaluation using generated evaluation steps and probability-weighted scores, tested against human judgments on summarization and dialogue."
---

# G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment

## Paper identity and scope

- An inference-time evaluator for natural-language generation, rather than a new trained model or a method for optimizing GPU kernels. Its objects of evaluation are summaries and dialogue responses. [§§1–3](../pdfs/15-g-eval.md)
- The research question is whether a task-specific LLM scoring procedure can align with human quality judgments better than lexical-overlap, embedding, and existing learned metrics. [§1](../pdfs/15-g-eval.md)
- The frozen version is arXiv 2303.16634v3; all model names and performance claims below refer to that paper's experiments, not current services. [Source metadata](../pdfs/15-g-eval.md)
- The contribution has three components: task/criterion prompts, automatically generated evaluation steps, and continuous scores derived from discrete-score probabilities. [§2; Figure 1](../pdfs/15-g-eval.md)
- The paper also studies a possible preference for LLM-generated text, relevant when an evaluator becomes a training reward. [§4, “Will G-Eval prefer LLM-based outputs?”](../pdfs/15-g-eval.md)

## Concepts and terminology

### Evaluation objects and criteria

- **Meta-evaluation** evaluates the evaluator itself: compare metric scores with human ratings on existing benchmark outputs. It is not the same as measuring a generator's task success. [§3](../pdfs/15-g-eval.md)
- **Reference-free** means no human-written target answer is required for scoring; source documents, dialogue history, and grounding facts can still be necessary inputs. [§§1–2; Appendix A](../pdfs/15-g-eval.md)
- **Coherence** concerns organization and how sentences form a connected account, rather than merely sharing a topic. [§2; Appendix A, summarization prompt](../pdfs/15-g-eval.md)
- **Consistency** concerns agreement with the source document; **fluency** concerns linguistic quality; **relevance** concerns pertinent content. These are separate SummEval rating dimensions. [§3.2; Table 1](../pdfs/15-g-eval.md)
- Dialogue evaluation uses **naturalness, coherence, engagingness, and groundedness**. A response can be engaging without being grounded, so the paper retains separate dimensions. [§3.2; Table 2](../pdfs/15-g-eval.md)
- The example engagingness rubric is explicitly three-point, while the coherence example uses five points; G-Eval does not require a universal 1–5 scale. [Appendix A](../pdfs/15-g-eval.md)
- Hallucination evaluation asks whether a summary contains unsupported or misleading facts relative to its source, rather than judging general plausibility. [§3.6; Appendix A, hallucination prompt](../pdfs/15-g-eval.md)

### Prompt and scoring mechanism

- The prompt begins with a task introduction and a definition of the requested criterion. Criterion wording is part of the evaluator specification, not incidental context. [§2, “Prompt for NLG Evaluation”](../pdfs/15-g-eval.md)
- **Auto-CoT** is a generated sequence of evaluation instructions: the LLM expands a criterion into steps such as reading the article, comparing the summary, and assigning a rating. [§2, “Auto Chain-of-Thoughts for NLG Evaluation”](../pdfs/15-g-eval.md)
- These evaluation steps are generated from the task and criterion and then included when scoring examples. This differs from assuming that every score is accompanied by an independently validated factual proof. [§2; Figure 1](../pdfs/15-g-eval.md)
- **Form-filling** asks the model to emit a score in a designated field. It evaluates quality directly rather than treating the likelihood of the entire candidate text as the quality score. [§2, “Scoring Function”](../pdfs/15-g-eval.md)
- GPTScore is the key conceptual contrast: it scores conditional generation likelihood; G-Eval asks the model to perform the evaluative task explicitly. [§§2, 3.3](../pdfs/15-g-eval.md)
- Direct integer ratings can collapse around a common value and create many ties, limiting resolution among candidates. [§2, “Scoring Function”](../pdfs/15-g-eval.md)
- The probability-weighted score is `sum_i p(s_i) * s_i` over the permitted scores. It is an expected rating, not a calibrated probability that the answer is correct. [§2, Eq. 1](../pdfs/15-g-eval.md)
- Probability weighting provides a continuous score even when the model only emits discrete integers. It is intended to retain information in the distribution rather than only its most likely value. [§2, Eq. 1](../pdfs/15-g-eval.md)
- Pearson correlation measures linear score association, Spearman measures rank association, and Kendall evaluates pair ordering; the paper uses different subsets of these metrics across tasks. [Tables 1–3; §§3.4–3.6](../pdfs/15-g-eval.md)
- Tie handling matters: an improvement in one reported correlation need not mean uniformly improved judgments under another measure. [§4, “The Effect of Probability Normalization”](../pdfs/15-g-eval.md)

## Methodology and experimental setup

- Backbones are `text-davinci-003`, labeled G-Eval-3.5 by the paper, and GPT-4, labeled G-Eval-4. Preserve the actual backbone identifier rather than silently substituting a later chat model. [§3.1](../pdfs/15-g-eval.md)
- GPT-3.5 decoding uses temperature 0. This is a historical experimental setting, not a general guarantee of deterministic API behavior. [§3.1](../pdfs/15-g-eval.md)
- Because the GPT-4 API used in the study did not expose token probabilities, the authors sample 20 outputs with `temperature=1` and `top_p=1` to estimate the score distribution. [§3.1](../pdfs/15-g-eval.md)
- Consequently, GPT-4's probability-weighted evaluation involves multiple samples; it is not a one-call direct log-probability method in this experiment. [§3.1](../pdfs/15-g-eval.md)
- SummEval supplies summary-level human ratings for four criteria and is based on CNN/DailyMail. [§3.2](../pdfs/15-g-eval.md)
- Topical-Chat supplies turn-level human ratings for knowledge-grounded dialogue responses. G-Eval follows the four dimensions used by UniEval. [§3.2](../pdfs/15-g-eval.md)
- QAGS tests summary factual consistency on CNN/DailyMail and XSum; the latter is more abstractive in the paper's comparison. [§§3.2, 3.6](../pdfs/15-g-eval.md)
- Baselines span ROUGE/BLEU/METEOR, BERTScore/MoverScore, BARTScore, task-specific factuality evaluators, USR, UniEval, and GPTScore. Different tables contain different applicable subsets. [§3.3; Tables 1–3](../pdfs/15-g-eval.md)
- FactCC is a consistency classifier; QAGS uses question generation and answer checking; UniEval casts assessment into a QA formulation. These are distinct mechanisms, not interchangeable “LLM judges.” [§3.3](../pdfs/15-g-eval.md)
- The preference-bias experiment uses human-written and `text-davinci-003` summaries from an existing comparison dataset, separated by whether humans preferred the human summary, the model summary, or rated them equally. [§4, bias experiment](../pdfs/15-g-eval.md)
- G-Eval-4 then scores the summaries in those three groups. The experiment assesses agreement with those labels and possible stylistic preference; it does not directly identify a causal reason for the observed preference. [§4; Figure 2](../pdfs/15-g-eval.md)
- Appendix A gives concrete task prompts, context slots, criteria, evaluation steps, and output fields for coherence, dialogue engagingness, and hallucination detection. [Appendix A](../pdfs/15-g-eval.md)

## Results and evidence

### Summarization

- G-Eval-4 has author-reported AVG Spearman **0.514** and Kendall **0.418** across the four SummEval dimensions. These are correlations, not 51.4% correctness or win rate. [Table 1](../pdfs/15-g-eval.md)
- Its Spearman correlations are **0.582 coherence, 0.507 consistency, 0.455 fluency, and 0.547 relevance**. Thus the headline average conceals criterion-specific differences. [Table 1](../pdfs/15-g-eval.md)
- UniEval's corresponding average is **0.474 Spearman / 0.377 Kendall**; G-Eval-3.5 reports **0.401 / 0.320**. The weaker G-Eval backbone does not beat UniEval on this average. [Table 1](../pdfs/15-g-eval.md)
- GPTScore's reported average Spearman is **0.417**; no average Kendall result is supplied for that row. Missing entries should not be treated as zero. [Table 1](../pdfs/15-g-eval.md)
- Removing Auto-CoT from G-Eval-4 lowers average Spearman from **0.514 to 0.500** and Kendall from **0.418 to 0.407**. [Table 1; §4, CoT ablation](../pdfs/15-g-eval.md)
- The largest displayed Spearman drop in that ablation is fluency, **0.455 to 0.403**. The paper interprets generated steps as useful evaluation guidance. [Table 1; §4](../pdfs/15-g-eval.md)
- Removing probability weighting from G-Eval-4 lowers average Spearman to **0.502**, but raises reported Kendall to **0.446**. The paper attributes the latter to ties and cautions against treating it as a fair improvement comparison. [Table 1 caption; §4, probability normalization](../pdfs/15-g-eval.md)
- For G-Eval-3.5, removing probability weighting lowers average Spearman from **0.401 to 0.346**. [Table 1](../pdfs/15-g-eval.md)

### Dialogue and factual consistency

- On Topical-Chat, average **Pearson/Spearman** is **0.575/0.588** for G-Eval-4, **0.574/0.585** for G-Eval-3.5, and **0.552/0.417** for UniEval. These metric names follow the table column labels and §3.5. [Table 2; §3.5](../pdfs/15-g-eval.md)
- **Source inconsistency:** Table 2's caption says Spearman/Kendall, but its columns are `r` and `rho`, and §3.5 explicitly says Pearson and Spearman. Preserve this discrepancy rather than relabeling the numbers as Kendall. [Table 2 caption and header; §3.5](../pdfs/15-g-eval.md)
- GPT-4 is not best on every dialogue aspect: GPT-3.5 has higher engagingness values (**0.660/0.691** versus **0.627/0.631**) and groundedness (**0.586/0.567** versus **0.531/0.551**). [Table 2](../pdfs/15-g-eval.md)
- G-Eval-4's QAGS author-reported AVG Pearson/Spearman/Kendall is **0.599/0.611/0.525**; UniEval reports **0.571/0.575/0.465**. [Table 3](../pdfs/15-g-eval.md)
- On QAGS-CNN, G-Eval-4 Pearson is **0.631**, below BARTScore's **0.735**. The headline average advantage is not a claim of winning every dataset/metric cell. [Table 3](../pdfs/15-g-eval.md)
- On QAGS-XSum, G-Eval-4 Pearson is **0.558**, compared with **0.461** for UniEval and **0.184** for BARTScore. [Table 3](../pdfs/15-g-eval.md)
- G-Eval-3.5's QAGS average Pearson is **0.344**, substantially below G-Eval-4's **0.599**; capacity sensitivity is more visible here than in the dialogue aggregate. [Table 3; §3.6](../pdfs/15-g-eval.md)

### Preference-bias probe

- G-Eval-4 rates human-written summaries differently across human-preference groups, yet still assigns GPT-3.5 summaries higher scores than human summaries even in the human-preferred group. The within-group model-versus-human comparison is the relevant bias observation. [§4; Figure 2](../pdfs/15-g-eval.md)
- The source study's human preference labels themselves have low agreement, reported as **Krippendorff's alpha 0.07**; the authors offer this as one possible explanation for the discrepancy. [§4, bias experiment](../pdfs/15-g-eval.md)
- Another proposed explanation is shared generation/evaluation preferences among the LLMs. The paper presents this as a hypothesis, not a proven mechanism. [§4](../pdfs/15-g-eval.md)
- The bias experiment uses GPT-4 to judge GPT-3.5 text; it should not be described as direct proof that a model recognizes and favors its own exact outputs. [§4, experimental description and footnote](../pdfs/15-g-eval.md)

## Authors' conclusions

- The authors conclude that task-specific prompting, generated evaluation steps, and form-filling can produce stronger human correspondence than the compared metrics on the evaluated NLG tasks. [§6](../pdfs/15-g-eval.md)
- They interpret the CoT ablation as evidence that explicit evaluation guidance helps the judge. [§4, CoT ablation](../pdfs/15-g-eval.md)
- They argue that probability weighting better captures subtle differences than integer-only scores, while highlighting misleading tie-sensitive comparisons. [§4, probability normalization](../pdfs/15-g-eval.md)
- They interpret GPT-4's stronger results on difficult criteria as evidence favoring more capable evaluators. Their “model size” explanation is an interpretation of these backbone comparisons, not a controlled parameter-count experiment. [§4, model-size discussion](../pdfs/15-g-eval.md)
- They explicitly characterize the model-generated-text preference finding as preliminary and call for further investigation before using judges as self-training rewards. [§§4, 6](../pdfs/15-g-eval.md)

## Limitations and open questions

- **Source-explicit:** instruction and prompt sensitivity remain important; the evaluation criterion and its generated steps can affect the results. [§1, contributions; §4](../pdfs/15-g-eval.md)
- **Source-explicit:** potential preference for LLM-generated language could reinforce the evaluator's own criteria when its score is used for further tuning. [§4](../pdfs/15-g-eval.md)
- **Analyst caveat:** correlation with subjective human ratings is not proof of factual correctness, executable correctness, or robustness on adversarial inputs. The paper's evidence is benchmark-specific meta-evaluation. [Evidence scope: §§3–4](../pdfs/15-g-eval.md)
- **Analyst caveat:** a 20-sample estimate introduces sampling cost and variability; the paper's equations should not be implemented as though exact GPT-4 probabilities were observed. [Evidence: §3.1](../pdfs/15-g-eval.md)
- **Analyst caveat:** the experiments do not establish that modern models, code evaluation, or kernel numerical validation share these performance values. [Evidence scope: §§3.1–3.2](../pdfs/15-g-eval.md)
- **Analyst caveat:** stronger-backbone comparisons change more than an independently controlled size variable, so the causal explanation should remain qualified. [Evidence: §4, model-size discussion](../pdfs/15-g-eval.md)
- **Open question:** how much of the preference discrepancy comes from stylistic bias versus unreliable human labels is unresolved by the reported study. [§4](../pdfs/15-g-eval.md)
- **Open question:** the appendix's engagingness steps ask for an explanation while its output form requests scores only; a reproduction should preserve and explicitly resolve this prompt ambiguity rather than assume a unique explanation policy. [Appendix A, engagingness prompt](../pdfs/15-g-eval.md)

- **Unresolved aggregation discrepancy:** published AVG cells do not consistently equal the arithmetic mean of displayed component scores. In Table 1, G-Eval-4 Spearman components average 0.52275 versus printed 0.514, and Kendall components average 0.42325 versus printed 0.418. In Table 3, its Pearson components average 0.5945 versus printed 0.599 and Kendall components average 0.5315 versus printed 0.525. Retain the author-reported AVG values without assuming a simple unweighted aggregation or silently replacing them. [Tables 1, 3](../pdfs/15-g-eval.md)

## Relevance to this project — interpretation

- **Proposal:** use explicit rubrics and separately stored evaluation-step prompts when judging subjective aspects of agent trajectories; version both alongside the judge model and sampling configuration. This is a project design suggestion, not a reported kernel experiment. [Motivation: §2](../pdfs/15-g-eval.md)
- **Proposal:** use execution tests, numerical comparisons, and measured latency for kernel correctness/performance; reserve this judge method for criteria for which human-rated quality is the intended target. [Scope motivating this distinction: §3](../pdfs/15-g-eval.md)
- **Proposal:** retain the score distribution and repeated-sampling configuration rather than only an average score, so apparent changes can be inspected for ties and variance. [Motivation: Eq. 1; §3.1; §4](../pdfs/15-g-eval.md)
- **Proposal:** independently audit model-generated versus human-generated evaluations before using a judge as an optimization reward. [Motivation: §4, bias probe](../pdfs/15-g-eval.md)

## Retrieval notes

- Aliases: G-Eval, GEval, Auto-CoT evaluation, form-filling judge, probability-weighted scoring.
- Questions answered: What distinguishes G-Eval from GPTScore? How were GPT-4 probabilities estimated? Why can removing probability weighting raise Kendall? What evidence suggests preference for model-generated summaries?
- Navigate to [Prometheus](12-prometheus.md) for a trained rubric-conditioned evaluator, [MT-Bench/Chatbot Arena](16-mt-bench-chatbot-arena.md) for pairwise/preference judge evaluation, and [Fair Evaluators](17-fair-evaluators.md) for order calibration. These are navigation links, not a claim of matched experimental comparisons.

## Source coverage and verification notes

- Read §§1–6 and Appendix A, including all three result tables, Eq. 1, prompt examples, ablations, bias-experiment discussion, and figure captions. Bibliographic entries were not independently reviewed as separate studies.
- Checked numerical claims against table cells and surrounding method text; preserved the Table 2 caption/header discrepancy. Figure 2 supports the qualitative bias observation via the accompanying text; no unprinted bar values were inferred.
- This is a source-grounded analysis, not an experimental reproduction. No new model calls or benchmark runs were performed.
