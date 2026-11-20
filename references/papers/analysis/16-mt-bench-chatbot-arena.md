---
paper_id: "16"
title: "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
source_version: "2306.05685v4"
source_markdown: "../pdfs/16-mt-bench-chatbot-arena.md"
source_pdf: "../pdfs/16-mt-bench-chatbot-arena.pdf"
source_url: "https://arxiv.org/abs/2306.05685"
analysis_date: "2026-11-20"
analysis_status: "reviewed"
topics: ["llm-as-a-judge", "human-preference", "benchmarking", "multi-turn", "bias", "pairwise-evaluation", "judge-training"]
summary: "Introduces MT-Bench and Chatbot Arena and measures how LLM judges align with human preferences under explicit tie-handling and bias controls."
---

# Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

## Paper identity and scope

- Examines LLM judges as scalable approximations to **human preferences about chat-assistant responses**, not as universal correctness oracles. [§§1, 3](../pdfs/16-mt-bench-chatbot-arena.md)
- Introduces a fixed, manually designed multi-turn benchmark and an open-ended crowdsourced battle platform, then validates automated judges against human votes. [§§1–2](../pdfs/16-mt-bench-chatbot-arena.md)
- Frozen source is arXiv 2306.05685v4. Model rankings, dataset sizes, costs, and API behaviors are historical results in this version. [Source metadata; §4.1; Appendix B](../pdfs/16-mt-bench-chatbot-arena.md)
- The main argument is that knowledge/capability benchmarks and preference benchmarks assess complementary properties; conversational alignment can improve without equivalent movement on conventional tests. [§§1, 5](../pdfs/16-mt-bench-chatbot-arena.md)

## Concepts and terminology

### Benchmarks and judging modes

- **MT-Bench:** 80 questions, each with two turns, in eight categories with ten questions per category. The categories are writing, roleplay, extraction, reasoning, math, coding, STEM knowledge, and humanities/social-science knowledge. [§2.2; §3.5](../pdfs/16-mt-bench-chatbot-arena.md)
- The follow-up turn tests contextual instruction following, such as rewriting with a constraint or referring to an earlier answer, rather than simply adding unrelated independent questions. [Table 1; §3.5](../pdfs/16-mt-bench-chatbot-arena.md)
- **Chatbot Arena:** users ask the same question of two anonymous models and vote on preferred responses; identities are revealed after voting. This reduces direct brand effects at vote time. [§2.3](../pdfs/16-mt-bench-chatbot-arena.md)
- Arena has unrestricted user prompts, making it more diverse but less experimentally controlled than MT-Bench. The sampled population reflects platform usage. [§§2.3, 4; Appendix C.2](../pdfs/16-mt-bench-chatbot-arena.md)
- **Pairwise judging:** select A, B, or tie for the same question; all-pairs comparisons grow quadratically with the number of models. [§3.1](../pdfs/16-mt-bench-chatbot-arena.md)
- **Single-answer grading:** rate one answer independently, making evaluation more scalable; absolute scales can drift and miss subtle pairwise differences. [§3.1](../pdfs/16-mt-bench-chatbot-arena.md)
- **Reference-guided grading:** supply a reference solution when available, especially for math and reasoning. It can be combined with either pairwise or single-answer scoring. [§3.1; §3.4](../pdfs/16-mt-bench-chatbot-arena.md)
- A reference generated separately prevents the candidate responses from entering the reference-generation context. It does not automatically make that reference correct. [§3.4](../pdfs/16-mt-bench-chatbot-arena.md)
- **Average win rate** averages wins against other players; it depends on the comparison pool and tie policy. It is not interchangeable with head-to-head win rate or a 1–10 score. [§4.1; §4.3](../pdfs/16-mt-bench-chatbot-arena.md)
- The reported MT-Bench scalar score is the average of single-answer GPT-4 ratings across **160 turns**, each rated on a 10-point scale. [§5](../pdfs/16-mt-bench-chatbot-arena.md)

### Biases and mitigation

- **Position bias:** the preferred answer changes when the same responses swap display order. **Name bias:** labels such as Assistant A versus B influence choice even when positions change. [§3.3; Table 2](../pdfs/16-mt-bench-chatbot-arena.md)
- The bias probe deliberately uses two similar GPT-3.5 answers generated at temperature 0.7 for each first-turn question; it is harder than comparing clearly different models. [§3.3](../pdfs/16-mt-bench-chatbot-arena.md)
- **Verbosity bias:** longer responses can be preferred despite adding no useful information. The paper's repetitive-list attack duplicates information through paraphrase. [§3.3; Table 3](../pdfs/16-mt-bench-chatbot-arena.md)
- **Self-enhancement bias:** possible preference for a judge's own model outputs. The authors investigate it but explicitly cannot establish it conclusively with their data. [§3.3](../pdfs/16-mt-bench-chatbot-arena.md)
- **Limited reasoning during grading:** a judge can be misled by candidate answers even when it solves the same problem correctly in isolation. [§3.3; Appendix B](../pdfs/16-mt-bench-chatbot-arena.md)
- Conservative order mitigation requires the same candidate to win in both orders; otherwise record a tie. This reduces order-induced wins by increasing abstention/ties. [§3.4](../pdfs/16-mt-bench-chatbot-arena.md)
- Randomizing positions is another proposed aggregate control, but the reported subsequent experiments use conservative double judging. [§3.4](../pdfs/16-mt-bench-chatbot-arena.md)
- Few-shot judge examples cover A-wins, B-wins, and tie. Greater consistency is not automatically greater agreement with human labels. [§3.4; Appendix D.2](../pdfs/16-mt-bench-chatbot-arena.md)
- CoT judging asks the evaluator to solve the problem first in the same context; reference-guided judging solves independently before candidates are shown. This distinction matters because same-context reasoning can copy the candidate's mistake. [§3.4; Appendix B](../pdfs/16-mt-bench-chatbot-arena.md)
- For second-turn judging, presenting complete conversations separately for each assistant helps preserve which earlier response belongs to whom. [§3.5](../pdfs/16-mt-bench-chatbot-arena.md)

### Agreement and selection

- Agreement is defined between random distinct individuals of two judge types on a random shared question, rather than always comparing with a single gold consensus. [§4.1; Appendix D.3](../pdfs/16-mt-bench-chatbot-arena.md)
- **S1** includes wins, ties, and inconsistent results, mapping order inconsistency to tie; the nominal random baseline is 33%. [Table 5 caption](../pdfs/16-mt-bench-chatbot-arena.md)
- **S2** restricts the comparison to non-tie votes; its nominal random baseline is 50%. The retained vote count changes with judge pair. [Table 5 caption](../pdfs/16-mt-bench-chatbot-arena.md)
- Human-majority agreement is a separate quantity from agreement with a randomly selected human. Appendix D.3 explains how these can differ even on the same questions. [Appendix D.3; Table 14](../pdfs/16-mt-bench-chatbot-arena.md)
- The two-human comparison excludes choosing the same person twice. A majority-following model can therefore agree with individuals more often than individuals agree with each other without exceeding a literal human-correctness ceiling. [Appendix D.3](../pdfs/16-mt-bench-chatbot-arena.md)

## Methodology and experimental setup

- MT-Bench answers are generated by GPT-4, GPT-3.5, Claude-v1, Vicuna-13B, Alpaca-13B, and LLaMA-13B. LLM judges assess all pairs. [§4.1](../pdfs/16-mt-bench-chatbot-arena.md)
- **58 expert-level human labelers**, mostly graduate students, each evaluate at least 20 random multi-turn questions, yielding approximately **3,000 votes**. [§4.1](../pdfs/16-mt-bench-chatbot-arena.md)
- Human judgments are collected turn by turn; respondents can skip up to five uncertain questions and may see reference solutions for selected math/reasoning items. [Appendix C.1; Figure 16 in Markdown](../pdfs/16-mt-bench-chatbot-arena.md)
- The study pays $20 for judging 20 questions, corresponding to the authors' estimate of about $35/hour; these are historical annotation conditions. [Appendix C.1](../pdfs/16-mt-bench-chatbot-arena.md)
- When a human initially disagrees with GPT-4, the interface shows GPT-4's explanation and asks whether it is reasonable. This follow-up measures receptiveness, not independent validation of the explanation's correctness. [§4.2; Appendix C.1](../pdfs/16-mt-bench-chatbot-arena.md)
- Arena evaluation samples roughly **3,000 single-turn votes** from about **30,000 collected votes**, covering a wider model set and **2,114 unique IP addresses**. IP count is not a guaranteed count of distinct people. [§4.1](../pdfs/16-mt-bench-chatbot-arena.md)
- Arena users are unpaid, can participate without registering, and consent to terms governing data release; the authors describe PII cleaning and toxicity tagging for release. [Appendix C.2–C.3](../pdfs/16-mt-bench-chatbot-arena.md)
- Default pairwise prompts request an explanation followed by machine-readable `[[A]]`, `[[B]]`, or `[[C]]`, and explicitly discourage position, length, and name biases. [PDF p. 14, Figure 5](../pdfs/16-mt-bench-chatbot-arena.pdf#page=14)
- Default single-answer prompts request an explanation and a 1–10 rating enclosed in double brackets. [PDF p. 14, Figure 6](../pdfs/16-mt-bench-chatbot-arena.pdf#page=14)
- Reference-guided prompts include question, reference answer, and both candidate answers; multi-turn prompts group each assistant's full conversation rather than interleave identities ambiguously. [PDF pp. 15–16, Figures 8–9](../pdfs/16-mt-bench-chatbot-arena.pdf#page=15)
- Case-study examples use `gpt-4-0314`, and the authors warn that future GPT-4 versions may not reproduce them. [Appendix B](../pdfs/16-mt-bench-chatbot-arena.md)

## Results and evidence

### Bias, robustness, and grading capability

- Under the similar-answer default probe, consistency is **23.8% Claude-v1, 46.2% GPT-3.5, and 65.0% GPT-4**. These values describe this probe, not general human agreement. [Table 2](../pdfs/16-mt-bench-chatbot-arena.md)
- GPT-4's remaining cases are **30% first-position bias and 5% second-position bias**; it has no formatting errors in this configuration. [Table 2](../pdfs/16-mt-bench-chatbot-arena.md)
- Renaming labels raises Claude-v1 consistency to **56.2%** and changes its bias distribution, supporting sensitivity to assistant names as well as position. [Table 2](../pdfs/16-mt-bench-chatbot-arena.md)
- On **23 repetitive-list attacks**, failure is **91.3%** for Claude-v1 and GPT-3.5 and **8.7%** for GPT-4. Success for the attacker means the redundant response is preferred. [§3.3; Table 3](../pdfs/16-mt-bench-chatbot-arena.md)
- GPT-4's math grading failures on ten questions with two answer orders fall from **14/20 default** to **6/20 CoT** and **3/20 reference-guided**. Failure means an incorrect answer is judged correct. [Table 4](../pdfs/16-mt-bench-chatbot-arena.md)
- Few-shot prompting raises GPT-4 position consistency from **65.0% to 77.5%**, but the authors report similar human agreement to zero-shot and approximately **4×** prompt/API cost. [§3.4; Appendix D.2; Table 13](../pdfs/16-mt-bench-chatbot-arena.md)
- Changing to numeric scoring raises GPT-3.5 consistency to **55.0%** but lowers GPT-4 to **51.2%** and Claude-v1 to **20.0%**; prompt changes are model-dependent. [Appendix D.1; Table 10](../pdfs/16-mt-bench-chatbot-arena.md)
- Category consistency is **86% for math and coding**, versus **42% writing, 44% STEM, and 36% humanities** in the appendix probe. High math consistency does not rule out consistently wrong grading. [Appendix D.1; Table 11](../pdfs/16-mt-bench-chatbot-arena.md)
- GPT-4 consistency is **67.5%** for GPT-3.5 versus Claude, **73.8%** versus Vicuna, and **98.8%** versus LLaMA. The authors attribute lower instability to wider performance separation. [Appendix D.1; Table 12](../pdfs/16-mt-bench-chatbot-arena.md)
- The self-enhancement probe observes shifts favoring some judges' own outputs, but GPT-3.5 does not show the same pattern; limited data and style/quality confounding prevent a firm conclusion. [§3.3](../pdfs/16-mt-bench-chatbot-arena.md)

### Human agreement: preserve tie handling and counts

- First-turn MT-Bench GPT-4 pairwise/human agreement is **66% over 1,343 votes in S1**, versus **85% over 859 votes in S2**. The “over 80%” headline is conditional on non-tie inclusion. [Table 5(a)](../pdfs/16-mt-bench-chatbot-arena.md)
- First-turn human/human agreement is **63% over 721 votes in S1** and **81% over 479 in S2**. Different retained populations and the agreement definition matter when comparing these values. [Table 5(a); Appendix D.3](../pdfs/16-mt-bench-chatbot-arena.md)
- Second-turn GPT-4 pairwise/human agreement is **66% over 1,325 votes in S1** and **85% over 864 in S2**; human/human S2 agreement is **82%**. [Table 5(b)](../pdfs/16-mt-bench-chatbot-arena.md)
- Single-answer GPT-4/human agreement is **85% first-turn and 84% second-turn in S2**, supporting its use as a less expensive-to-scale proxy in this setting. [Table 5](../pdfs/16-mt-bench-chatbot-arena.md)
- GPT-4 pairwise versus single-answer agreement is **97% first-turn and 95% second-turn in S2**, but **70% in S1** for both turns. Agreement depends strongly on whether ties are counted. [Table 5](../pdfs/16-mt-bench-chatbot-arena.md)
- Arena GPT-4 pairwise/human agreement is **64% over 3,066 votes in S1** and **87% over 1,944 in S2**. GPT-4 single-answer/human S2 agreement is **85%**. [Table 6](../pdfs/16-mt-bench-chatbot-arena.md)
- GPT-3.5 and Claude pairwise judges have Arena S2 human agreement of **83% and 84%**, on **1,567 and 1,475** votes respectively; they retain fewer affirmative comparisons than GPT-4. [Table 6; §4.2](../pdfs/16-mt-bench-chatbot-arena.md)
- Humans found GPT-4's disagreeing judgments reasonable in **75%** of follow-ups and were willing to change choices in **34%**. This is evidence of persuasiveness/reconsideration under the interface, not an independently adjudicated correction rate. [§4.2](../pdfs/16-mt-bench-chatbot-arena.md)
- Agreement increases with model-pair performance disparity, from roughly 70% toward 100% among non-tied votes; this is the authors' aggregate trend rather than a universal threshold. [§4.2; plot caption labeled Table 7 in Markdown](../pdfs/16-mt-bench-chatbot-arena.md)

### Complementary benchmarks and judge training

- LLaMA-13B scores **47.0 MMLU, 0.26 TruthfulQA, 2.61 MT-Bench**, while Vicuna-13B (all) scores **52.1, 0.35, 6.39**. These different scales should not be combined into a single arithmetic quality claim. [Table 9](../pdfs/16-mt-bench-chatbot-arena.md)
- Vicuna-7B trained on the selected **4.8M-token / 3K-conversation** subset scores **5.95 MT-Bench** but **37.3 MMLU**; the all-data model scores **6.00 / 47.1**. Preference-style gains and knowledge gains need not scale together. [§5; Table 9](../pdfs/16-mt-bench-chatbot-arena.md)
- MT-Bench scores for GPT-3.5 and GPT-4 are **7.94 and 8.99** in this evaluation. These are historical averages over the paper's 160 turns and specified judge procedure. [§5; Table 9](../pdfs/16-mt-bench-chatbot-arena.md)
- Vicuna training starts from cleaned ShareGPT conversations: about **125K conversations**, with long ones split into context-sized sequences. The selected subset picks cluster representatives from multi-turn GPT-4 conversations. [Appendix E](../pdfs/16-mt-bench-chatbot-arena.md)
- Training uses batch size **128**, learning rate **2e-5**, context **2,048**, and **three epochs**, except **five** for the selected subset, on **8 A100s**. [Appendix E](../pdfs/16-mt-bench-chatbot-arena.md)
- Zero-shot Vicuna-13B judging suffers **11.2–16.2%** position consistency and **22.5–78.8%** format-error rates across tested templates. Instruction-following/parser failure is separate from judging quality. [Appendix F.1; Table 16](../pdfs/16-mt-bench-chatbot-arena.md)
- The trained judge uses **20K Arena votes for training and 2K validation**, formulates judging as three-way sequence classification, and tests on the main body's approximately 3K vote set. [Appendix F.2](../pdfs/16-mt-bench-chatbot-arena.md)
- “Both bad” and “tie” Arena votes are merged into the tie class; this loses their distinction in the trained judge. [Appendix F.2](../pdfs/16-mt-bench-chatbot-arena.md)
- Fine-tuning raises position consistency to **65.0%** and eliminates output-format errors by construction of the classifier. Reported human agreement is **56.8% with all labels** and **85.5% after excluding tie predictions and labels**. [Appendix F.2; Table 16](../pdfs/16-mt-bench-chatbot-arena.md)
- The appendix compares those numbers to GPT-4 **66%/87%**, while main Table 6 gives GPT-4/human **64%/87%** for Arena. Treat the first figure as a source inconsistency rather than silently harmonizing it. [Appendix F.2; Table 6](../pdfs/16-mt-bench-chatbot-arena.md)

## Authors' conclusions

- Strong LLM judges can approximate human chat preferences at useful scale, particularly under the non-tie agreement conditions reported. [§7; §4.2](../pdfs/16-mt-bench-chatbot-arena.md)
- Pairwise and single-answer judging provide complementary quality/cost tradeoffs, and single-answer grading can be practical for larger model sets. [§3.1; §4.2; §5](../pdfs/16-mt-bench-chatbot-arena.md)
- Bias and reasoning weaknesses require controls; reference guidance, order swapping, and suitable multi-turn prompt organization mitigate selected failures. [§3.4–3.5](../pdfs/16-mt-bench-chatbot-arena.md)
- Capability benchmarks should be used together with preference benchmarks; no single benchmark determines overall model quality. [§5](../pdfs/16-mt-bench-chatbot-arena.md)
- Training open judges on preference data is promising, but the Vicuna experiment is explicitly preliminary. [Appendix F](../pdfs/16-mt-bench-chatbot-arena.md)

## Limitations and open questions

- **Source-explicit:** the study emphasizes helpfulness and largely neglects safety; modifying prompts for harmlessness is suggested, not demonstrated as an established result. [§6](../pdfs/16-mt-bench-chatbot-arena.md)
- **Source-explicit:** accuracy, relevance, creativity, and other helpfulness aspects are merged into a single metric, limiting diagnosis. [§6](../pdfs/16-mt-bench-chatbot-arena.md)
- **Source-explicit:** self-enhancement cannot be established conclusively, and few-shot consistency improvements may introduce new biases. [§§3.3–3.4](../pdfs/16-mt-bench-chatbot-arena.md)
- **Analyst caveat:** S2 agreement is selected on non-ties. It should always be accompanied by S1 or coverage information when used to argue reliability. [Evidence: Tables 5–6](../pdfs/16-mt-bench-chatbot-arena.md)
- **Analyst caveat:** crowd preferences, expert preferences, and executable correctness are different targets. Similar agreement percentages do not make their labels interchangeable. [Evidence: §§4.1, 6; Appendix C](../pdfs/16-mt-bench-chatbot-arena.md)
- **Analyst caveat:** model-generated reference solutions and confident repeated judgments can remain wrong; neither format consistency nor position consistency certifies factual accuracy. [Evidence: §§3.3–3.4](../pdfs/16-mt-bench-chatbot-arena.md)
- **Source-version caveat:** labels in the HTML-derived Markdown differ from the PDF in several places; for example, the Markdown's prompt Figure 4 is PDF Figure 5, and Markdown Table 8 is PDF Table 7. Follow the linked artifact's labels. [Markdown Appendix A, Table 8](../pdfs/16-mt-bench-chatbot-arena.md); [PDF p. 14](../pdfs/16-mt-bench-chatbot-arena.pdf#page=14)
- **Open questions:** broader task coverage, more capable open judges, stronger math/reasoning evaluation, safety evaluation, and resistance to deliberate judge manipulation remain future directions. [§6](../pdfs/16-mt-bench-chatbot-arena.md)

## Relevance to this project — interpretation

- **Proposal:** keep subjective trajectory-quality judging separate from kernel correctness and measured runtime. A preference-aligned judge is not the authority for floating-point equivalence or speed. [Motivation: scope in §§1, 3–4](../pdfs/16-mt-bench-chatbot-arena.md)
- **Proposal:** store both answer orders, raw outcomes, tie policy, retained denominator, judge version, and reference-generation procedure in evaluation artifacts. [Motivation: §§3.4, 4.1; Tables 5–6](../pdfs/16-mt-bench-chatbot-arena.md)
- **Proposal:** preserve each agent's complete multi-turn trajectory with explicit ownership when comparing experiments; avoid mixing one agent's earlier output with another's follow-up. [Motivation: §3.5](../pdfs/16-mt-bench-chatbot-arena.md)
- **Proposal:** use single-answer grading for broad screening and human or pairwise review for close calls, but validate the resulting pipeline on project-specific examples before relying on it. This workflow is an analyst proposal, not an evaluated kernel workflow. [Motivation: §3.1; §4.2](../pdfs/16-mt-bench-chatbot-arena.md)

## Retrieval notes

- Aliases: MT-Bench, MT-bench, Chatbot Arena, LLM-as-a-judge, S1/S2 agreement, pairwise versus single-answer grading, reference-guided judge, repetitive-list attack.
- Questions answered: What does “85% agreement” include/exclude? Why can judge/human agreement exceed human/human agreement? How are multi-turn prompts structured? Does order consistency prove correctness? How was Vicuna trained as a judge?
- Related analyses: [Fair Evaluators](17-fair-evaluators.md) for MEC/BPC/HITLC, [G-Eval](15-g-eval.md) for criterion-specific scoring, [Prometheus](12-prometheus.md) for trained rubric-conditioned evaluation, and [Reproducible Evaluation](14-reproducible-lm-evaluation.md) for protocol reporting. Navigation does not imply controlled cross-paper comparisons.

## Source coverage and verification notes

- Read §§1–7 and Appendices A–F, all displayed result tables, prompt-template descriptions, case-study descriptions, collection methodology, Vicuna training configuration, and judge-training details.
- Inspected PDF pages 14–16 to recover prompt text missing from Markdown figure bodies, and page 8 to confirm table layout and format-dependent numbering. Exact plot coordinates were not inferred.
- Recorded the full-label judge-training comparison discrepancy and HTML/PDF label differences. Table labels in Markdown citations refer to that linked Markdown, not automatically to PDF numbering.
- No experiments were reproduced; the analysis separates recorded evidence from deployment suggestions.
