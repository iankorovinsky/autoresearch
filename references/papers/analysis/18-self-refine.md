---
paper_id: "18"
title: "Self-Refine: Iterative Refinement with Self-Feedback"
source_version: "2303.17651v2"
source_markdown: "../pdfs/18-self-refine.md"
source_pdf: "../pdfs/18-self-refine.pdf"
source_url: "https://arxiv.org/abs/2303.17651"
analysis_date: "2026-11-20"
analysis_status: "reviewed"
topics: ["self-feedback", "iterative-refinement", "test-time-compute", "code-optimization", "multi-aspect-evaluation", "llm-as-a-judge", "error-analysis"]
summary: "Tests a same-model generate–critique–revise loop across seven tasks, with strong task-dependent gains and a major bottleneck in detecting incorrect feedback."
---

# Self-Refine: Iterative Refinement with Self-Feedback

## 1. Paper identity and scope

- **Research question:** can an existing LLM improve its own output by generating explicit feedback and revising against it, without training a separate critic or refiner? [§1](../pdfs/18-self-refine.md)
- **Contribution:** an inference-time generate–feedback–refine algorithm and evaluation across seven natural-language/code tasks, with ablations of feedback, iteration, model capability, and external correctness signals. [§§1–4](../pdfs/18-self-refine.md)
- **Same-model default:** the generator, feedback provider, and refiner use the same underlying model with different prompts; this is distinct from updating weights or learning a persistent policy. [§2, Equations 1–4](../pdfs/18-self-refine.md)
- **Important scope:** main experiments include Python algorithm optimization, not CUDA/Triton kernel optimization or hardware profiling. [Appendix N](../pdfs/18-self-refine.md)
- **Version boundary:** this analysis preserves the v2 results and explicitly records inconsistencies between its main and appendix protocols rather than collapsing them into one experimental setting. [Local source metadata; §3; Appendices F, H, L–P](../pdfs/18-self-refine.md)

## 2. Concepts and terminology

### Algorithm and feedback contract

- **Initialization:** generate an initial candidate (y_0=M(p_{gen}\Vert x)), where (x) is the input and (p_{gen}) is a task prompt or few-shot examples. [§2, Equation 1](../pdfs/18-self-refine.md)
- **Feedback:** generate (fb_t=M(p_{fb}\Vert x\Vert y_t)); feedback is language describing specific weaknesses and possible improvements, not necessarily a scalar reward. [§2, Equation 2](../pdfs/18-self-refine.md)
- **Refinement:** condition the next output on the task, existing answer, and feedback. The implementation retains prior candidates and feedback, not only the most recent pair. [§2, Equations 3–4](../pdfs/18-self-refine.md)
- **In-context history:** retaining past attempts is intended to help avoid repeated mistakes within the same refinement episode. The algorithm does not define cross-task persistent memory. [§2](../pdfs/18-self-refine.md)
- **Actionable feedback:** specify an operation that could improve the output, such as replacing repeated computation with a closed-form formula. [§2, feedback](../pdfs/18-self-refine.md)
- **Specific feedback:** identify concrete phrases, code locations, or missing requirements rather than only stating that quality should improve. [§2; §4, feedback-quality ablation](../pdfs/18-self-refine.md)
- **Multi-aspect feedback:** assess several potentially competing dimensions, such as readability, efficiency, engagement, and relevance, with explanations and sometimes scores. [§2; Appendices M and Q](../pdfs/18-self-refine.md)
- **Prompt schemas:** generation demonstrations use input/output pairs; feedback uses input/output/feedback triples; revision uses input/output/feedback/improved-output quadruples. [§2](../pdfs/18-self-refine.md)
- **Stopping rule:** stop after a maximum iteration count or when feedback supplies a task-specific stopping indicator. Stopping based on the model's own assessment is only as reliable as that assessment. [§2](../pdfs/18-self-refine.md)
- **Default final output:** the general formulation returns the last candidate; some appendix task implementations instead select the highest self-rated candidate, so output selection is task-specific. [§2; Appendix H, acronym analysis; Appendix M.2](../pdfs/18-self-refine.md)
- **No added training:** the loop uses pretrained/instruction-tuned models and in-context examples; “no training” does not mean no prior supervision in the base model or no human-designed demonstrations. [§2; §6](../pdfs/18-self-refine.md)
- **External evaluation versus self-feedback:** benchmark tests or human judgments can score outputs without being supplied to the loop. Oracle correctness feedback is a separately identified variant. [§3.2; Appendix H.1](../pdfs/18-self-refine.md)
- **Refinement versus resampling:** refinement conditions on prior candidates and their critiques; independent samples spend more generation attempts without using those intermediate diagnoses. [§4](../pdfs/18-self-refine.md)
- **Refinement versus RL:** this method changes generated artifacts/context at inference time; it does not optimize model weights against a reward function. [§5; Appendix B](../pdfs/18-self-refine.md)

### Task-specific concepts

- **Code optimization / PIE:** start from a functionally correct Python program and seek algorithmic edits that improve runtime; the reported main metric is the percentage of programs optimized. [§3.2; Appendix N](../pdfs/18-self-refine.md)
- **Optimization diagnosis:** examples critique costly nested loops and propose dynamic programming or formulas; the intended improvement can change algorithmic complexity rather than only syntax. [Figure 5; Appendix N](../pdfs/18-self-refine.md)
- **Code readability:** evaluate meaningful variable names, comments per line, and function decomposition; these are proxies for readability and maintainability, not direct functional-correctness metrics. [Appendix L.2](../pdfs/18-self-refine.md)
- **Readability initialization exception:** the appendix starts with existing code and treats initialization as a no-op before critique and revision. [Appendix L.1](../pdfs/18-self-refine.md)
- **Program-aided math:** GSM8K answers are represented as Python solutions; self-feedback attempts to identify semantic misunderstandings or incorrect operations in the solution. [Appendix O](../pdfs/18-self-refine.md)
- **Sentiment reversal:** rewrite whole multi-sentence reviews toward the opposite sentiment, with evaluation of strength/alignment and expressive language rather than merely binary polarity. [Appendix P](../pdfs/18-self-refine.md)
- **Dialogue quality dimensions:** relevance, informativeness, interestingness, consistency, helpfulness, engagement, specificity, safety, user understanding, and fluency. [Appendix M.1](../pdfs/18-self-refine.md)
- **Acronym dimensions:** pronunciation, spelling, relation to title, positive connotation, and familiarity; improvement in one can harm another. [Appendix Q](../pdfs/18-self-refine.md)
- **CommonGen-Hard:** generate coherent text covering 20–30 concepts, compared with the original CommonGen's 3–5; missing concepts create explicit revision opportunities. [Appendices K and R](../pdfs/18-self-refine.md)
- **Oracle feedback:** external knowledge of whether an answer is correct determines whether to keep refining; it is a privileged signal and must not be reported as unaided self-correction. [Appendix H.1](../pdfs/18-self-refine.md)

## 3. Methodology and experimental setup

- **Main models:** GPT-3.5 `text-davinci-003`, ChatGPT `gpt-3.5-turbo`, GPT-4, and additional Codex `code-davinci-002` experiments for code tasks. [§3.1](../pdfs/18-self-refine.md)
- **Primary comparison:** each base model without feedback/refinement versus the same model with the loop; prompts are inherited from prior work where available and designed otherwise. [§3.1](../pdfs/18-self-refine.md)
- **Main iteration limit:** §3.1 says up to four iterations; appendices describe task variants with three dialogue iterations and five readability iterations, so there is no single universal run length across every reported result. [§3.1; Appendices L.2 and M.2](../pdfs/18-self-refine.md)
- **Dataset sizes in Table 4:** 1,000 sentiment reviews, 372 dialogue conversations, 1,000 optimization programs, 300 readability programs, 1,319 math questions, 250 acronym items, and 200 constrained-generation examples. Task-specific appendices sometimes report smaller evaluation subsets. [Appendix A; Table 4](../pdfs/18-self-refine.md)
- **Task metrics:** math solve rate, program-optimization rate, and concept coverage; preference-based outcomes are used for subjective generation tasks. These quantities are not interchangeable units for an overall capability score. [§3.2](../pdfs/18-self-refine.md)
- **Human preference protocol:** blinded A/B judgments by the authors, using input/task instructions and candidate outputs; Appendix C reports 150 examples per listed task and allows ties. [Appendix C; Table 6](../pdfs/18-self-refine.md)
- **LLM judge:** GPT-4 compares candidate outputs with task-specific criteria and can choose either, both, or neither; prompts request a brief explanation before the decision. [Appendix D; Figures 6–8](../pdfs/18-self-refine.md)
- **Judge/human relation:** §3.2 reports 82% for sentiment, 68% for acronyms, and 71% for dialogue under the word “correlation”; the statistic is not specified clearly enough to treat these percentages as Pearson/Spearman coefficients. [§3.2](../pdfs/18-self-refine.md)
- **Feedback ablation:** compare specific actionable feedback, generic feedback, and repeated refinement without explicit feedback; models are ChatGPT for optimization/sentiment and GPT-3.5 for acronyms. [§4; Table 2](../pdfs/18-self-refine.md)
- **Iteration study:** inspect scores at initial output and three successive revisions, averaging across GPT-3.5, ChatGPT, and GPT-4 for the plotted tasks. [Figure 4](../pdfs/18-self-refine.md)
- **Independent-sample comparator:** compare one refined output against four independently generated ChatGPT outputs in a one-versus-many preference setup. This compares candidate generation strategies, not necessarily equal token cost. [§4; Appendix H, Figure 9](../pdfs/18-self-refine.md)
- **Qualitative diagnostic sample:** manually analyze 70 examples, divided into 35 successes and 35 failures, from code optimization/math reasoning. The reported error percentages condition on this selected sample. [§4, Qualitative Analysis](../pdfs/18-self-refine.md)
- **Weak-model probe:** Vicuna-13B uses the same prompts as the stronger models, exposing failures in feedback formatting and following revision instructions. [§4; Appendix G](../pdfs/18-self-refine.md)
- **Mixed-model variant:** initialize with Vicuna-13B but use ChatGPT for critique and revision; this intentionally departs from the same-model/no-external-help default. [Appendix G](../pdfs/18-self-refine.md)
- **Readability appendix protocol:** 300 CodeNet examples, 60 human-rewrite comparisons, `text-davinci-003` critics at temperature 0 or 0.7, greedy code editing, and five iterations. [Appendix L.2](../pdfs/18-self-refine.md)
- **Dialogue appendix protocol:** three initialization demonstrations and task-aspect feedback examples; choose the highest feedback-scored revised response, excluding the original. Appendix M reports 342 automated-test cases and 100 human-evaluation cases for that setting. [Appendix M.1–M.2](../pdfs/18-self-refine.md)

## 4. Results and evidence

### Main table, with task metrics kept separate

- **Code optimization:** GPT-3.5 14.8→23.0% (+8.2 points), ChatGPT 23.9→27.5% (+3.6), GPT-4 27.3→36.0% (+8.7). These are fractions of programs optimized, not average runtime speedup factors. [Table 1](../pdfs/18-self-refine.md)
- **Math reasoning:** GPT-3.5 64.1→64.1%, ChatGPT 74.8→75.0%, GPT-4 92.9→93.1%. The same-model gains are zero or 0.2 points in the main table. [Table 1](../pdfs/18-self-refine.md)
- **Constrained generation:** reported coverage scores move 28→37 for GPT-3.5, 44→67 for ChatGPT, and 15→45 for GPT-4. The metric concerns concept coverage, not an independent guarantee of grammaticality or semantic coherence. [§3.2; Table 1](../pdfs/18-self-refine.md)
- **Dialogue preference:** baseline/refined rates are 36.4/63.6 for GPT-3.5, 40.1/59.9 for ChatGPT, and 25.4/74.6 for GPT-4. The +49.2-point GPT-4 contrast is a preference-rate difference, not a 49.2-point factual-correctness improvement. [Table 1; §3.3](../pdfs/18-self-refine.md)
- **Sentiment preference:** GPT-3.5 8.8/30.4, ChatGPT 11.4/43.2, GPT-4 3.8/36.2; remaining preference mass can be ties, so paired rates need not total 100. [Table 1; Table 13 caption](../pdfs/18-self-refine.md)
- **Acronym preference:** GPT-3.5 41.6/56.4, ChatGPT 27.2/37.2, GPT-4 30.4/56.0. Multi-aspect preference is not a single objectively verifiable task-success criterion. [Table 1; Appendix Q](../pdfs/18-self-refine.md)
- **Readability main metric:** GPT-3.5 37.4→51.3, ChatGPT 27.7→63.1, GPT-4 27.4→56.2; §3.2 describes a GPT-4-estimated fraction of appropriately named variables. [§3.2; Table 1](../pdfs/18-self-refine.md)
- **Small math gains explained:** the model often fails to recognize an incorrect solution; ChatGPT reportedly says everything looks good for 94% of instances in the discussed setting. [§3.3](../pdfs/18-self-refine.md)
- **Confidence reporting:** Table 13 marks many gains significant but does not mark the main math gains; the confidence-level discrepancy described below prevents uncritical reuse of the statistical labels. [Appendix J; Table 13](../pdfs/18-self-refine.md)

### Ablations, iterations, and feedback reliability

- **Optimization feedback ablation:** actionable 27.5, generic 26.0, no-feedback refinement 24.8; explicit diagnosis helps but some improvement remains without detailed critique. [Table 2](../pdfs/18-self-refine.md)
- **Sentiment feedback ablation:** actionable 43.2, generic 31.2, no-feedback 0 in this particular evaluation; do not generalize zero to all sentiment-generation settings. [Table 2](../pdfs/18-self-refine.md)
- **Acronym feedback ablation:** actionable 56.4, generic 54.0, no-feedback 48.0. [Table 2](../pdfs/18-self-refine.md)
- **Iteration trajectories:** optimization scores 22.0→27.0→27.9→28.8; sentiment 33.9→34.9→36.1→36.8; constrained generation 29.0→40.3→46.7→49.7, averaged across the three main models. [Figure 4](../pdfs/18-self-refine.md)
- **Diminishing returns:** the largest gain is usually early in these aggregate curves, but continued iterations yield smaller improvements. [§4; Figure 4](../pdfs/18-self-refine.md)
- **Non-monotonic individual quality:** the acronym example's self-rated totals are 11→17→12→17, showing that another revision can regress even when aggregate curves improve. [Appendix H; Table 10](../pdfs/18-self-refine.md)
- **Candidate selection:** the acronym discussion proposes picking the maximum self-score across iterations; this adds a selector whose reliability is distinct from the refiner's ability. [Appendix H](../pdfs/18-self-refine.md)
- **Independent samples:** the authors report human preference for refined outputs even against four initial samples; Figure 9's graphical label extraction is ambiguous, so exact bar percentages are not inferred here. [§4; Appendix H, Figure 9](../pdfs/18-self-refine.md)
- **Failure causes in the 35-case diagnostic failure subset:** 33% wrong error localization, 61% inappropriate suggested fix, and 6% failure to implement otherwise useful feedback. Feedback quality is the dominant measured bottleneck. [§4, Qualitative Analysis](../pdfs/18-self-refine.md)
- **Successful cases:** 61% use accurate useful feedback for a precise fix; 33% succeed despite partially incorrect feedback. These categories are not a calibrated robustness guarantee. [§4, Qualitative Analysis](../pdfs/18-self-refine.md)
- **Dialogue error analysis:** feedback can be wrong, generic, or mis-scored; revisions can ignore useful feedback, follow harmful feedback, or introduce new problems. The tables give overlapping-looking occurrence categories rather than an exhaustive disjoint probability model. [Appendix H; Tables 11–12](../pdfs/18-self-refine.md)
- **Vicuna-13B:** weaker formatting/instruction-following leads to repeated outputs, invented conversational turns, and poor critiques; identical prompts do not yield reliable refinement on this model. [Appendix G](../pdfs/18-self-refine.md)
- **Mixed refinement:** Vicuna initialization plus ChatGPT feedback/revision improves the reported math result from 24.18% to 40.5%; this is not evidence that Vicuna alone self-corrects. [Appendix G](../pdfs/18-self-refine.md)

### External signals and task appendices

- **Oracle math results:** Table 9 gives GPT-3.5 64.06→68.9%, ChatGPT 74.8→76.2%, and GPT-4 92.9→93.8% when correctness information gates refinement. [Appendix H.1; Table 9](../pdfs/18-self-refine.md)
- **Oracle interpretation:** these results isolate failure recognition as a bottleneck, but real deployments may lack the privileged correctness signal used by the experiment. [Appendix H.1](../pdfs/18-self-refine.md)
- **Human preferences:** Appendix C's 150-example samples report Self-Refine/direct/tie rates of 75.00/21.43/3.57 for sentiment, 44.59/12.16/43.24 for acronyms, and 47.58/19.66/32.76 for response generation. [Table 6](../pdfs/18-self-refine.md)
- **Readability comparison:** on 60 examples, human rewrites have meaningful-variable ratio 0.653, comments/line 0.24, function units 0.70; Self-Refine with critique temperature 0.7 has 0.700, 0.25, 1.33. Larger heuristic counts are not independent human proof of superior maintainability. [Table 14](../pdfs/18-self-refine.md)
- **Readability decoding effect:** higher-temperature critique encourages naming/comment edits; greedy critique produces more modularization suggestions. [Appendix L.2; Figure 14](../pdfs/18-self-refine.md)
- **Sentiment classification saturation:** both compared methods get 100% on positive polarity; negative polarity is 92% baseline versus 93.6% refined. Preference/style metrics expose changes that a binary classifier largely misses. [Appendix P.1](../pdfs/18-self-refine.md)
- **Pointed sentiment feedback:** a generic something-is-wrong ablation lowers the reported sentiment preference from 85% to 73% and dramatic-language preference from 80.09% to 58.92% in the appendix setting. These are separate from Table 2's evaluation. [Appendix P.1](../pdfs/18-self-refine.md)
- **Code algorithm example:** feedback transforms a multi-loop enumeration into dynamic programming with stated complexity (O(amount\times coins)); it illustrates a mechanism, not a formal proof for all programs. [Figure 5](../pdfs/18-self-refine.md)
- **Additional optimization appendix:** the displayed direct baseline is 9.7% optimized and two-step Self-Refine reaches 15.6%, while no-feedback reaches 10.4%. Its speedup columns lack enough local protocol detail for a clean comparison with Table 1, so they are not promoted to a general speedup claim. [Appendix N; Tables 16–17](../pdfs/18-self-refine.md)
- **Website examples:** ice-cream and photosynthesis pages illustrate iterative changes to layout, text, and styles; the paper does not provide a controlled usability benchmark for these demonstrations. [Appendix I](../pdfs/18-self-refine.md)

## 5. Authors' conclusions

- A single capable LLM can serve as generator, critic, and refiner, improving many outputs at test time without additional model training. [§7](../pdfs/18-self-refine.md)
- Specific, actionable feedback is more useful than generic exhortation in the studied ablations. [§4; Table 2](../pdfs/18-self-refine.md)
- Multiple revisions often help, with diminishing returns and possible regressions in multi-objective tasks. [§4; Appendix H](../pdfs/18-self-refine.md)
- Generating the correct critique can be harder than applying it; failure recognition and feedback validity are central improvement opportunities. [§3.3; §4; Appendix H.1](../pdfs/18-self-refine.md)
- The authors frame the method as broadly applicable, but the actual measured gains are task-dependent, including negligible main-table math improvements. [§7; Table 1](../pdfs/18-self-refine.md)

## 6. Limitations and open questions

### Explicit source limitations

- The base model needs sufficient few-shot learning/instruction-following ability; Vicuna experiments show the loop can fail even when initialization works. [§6; Appendix G](../pdfs/18-self-refine.md)
- Main models are proprietary, with undisclosed pretraining/model details and paid API access; released code/outputs mitigate but do not remove reproducibility limitations. [§6](../pdfs/18-self-refine.md)
- Experiments are English-only; transfer to other languages is not established. [§6](../pdfs/18-self-refine.md)
- The method does not explicitly prevent iterative optimization toward harmful/toxic outputs. [§6](../pdfs/18-self-refine.md)
- Incorrect critiques and non-monotonic revision can degrade an otherwise good artifact. [§4; Appendix H](../pdfs/18-self-refine.md)

### Analyst caveats and unresolved source details

- **Analysis:** a self-assigned quality score is not independent validation. When the same family generates, critiques, selects, and judges outputs, correlated errors and stylistic preferences remain plausible. [Basis: §2; §3.2; Appendix D](../pdfs/18-self-refine.md)
- **Analysis:** the headline average improvement combines preference rates, coverage, readability proxies, optimization fractions, and solve rates. It should not be reused as a universal percentage improvement in accuracy. [Basis: §3.2; Table 1](../pdfs/18-self-refine.md)
- **Analysis:** the paper supplies no comprehensive matched-token/cost analysis demonstrating that this loop is best under every fixed inference budget. [Basis: §4, multiple-sample comparison](../pdfs/18-self-refine.md)
- **Decoding ambiguity:** §3.1 says “greedy decoding with a temperature of 0.7,” while Appendix L explicitly distinguishes greedy 0.0 from sampling 0.7. Reproduction requires inspecting the released configuration rather than treating that sentence as unambiguous. [§3.1; Appendix L.2](../pdfs/18-self-refine.md)
- **Stopping/selection ambiguity:** the generic algorithm returns the last candidate; Appendix H says best-scored selection and attributes it to the algorithm; Appendix M excludes the initial response from best-score selection. Keep these settings separate. [§2; Appendix H; Appendix M.2](../pdfs/18-self-refine.md)
- **Oracle distinction:** Appendix O uses correct labels to control progression, whereas the main math table is presented without oracle feedback and Table 9 separately labels oracle results. Appendix O cannot be used to claim all math results are supervision-free. [Table 1; Appendix H.1; Appendix O](../pdfs/18-self-refine.md)
- **Cross-table inconsistency:** Appendix F gives GPT-4 math 94.5%, versus main Table 1 93.1% and oracle Table 9 93.8%; ChatGPT optimization is 26.7% in Table 8 versus 27.5% in Table 1. No single reconciled setting is established in the text. [Tables 1, 7–9](../pdfs/18-self-refine.md)
- **Confidence ambiguity:** Appendix J prose says 99% intervals, while Table 13's caption says 95%; report the ambiguity rather than asserting a confidence level. [Appendix J; Table 13](../pdfs/18-self-refine.md)
- **Dialogue inconsistencies:** Table 4 lists 372 conversations, Appendix M says 342 automated cases; Appendix C and M describe 150- and 100-example human samples; Appendix M refers to different automated judges than §3.2. These may be different experiments but are not consistently reconciled. [Table 4; §3.2; Appendices C and M](../pdfs/18-self-refine.md)
- **Malformed preference column:** Table 15's ChatGPT human-preference entries 48/18/50 sum to 116%; that column is not safe to normalize or interpret without external clarification. [Table 15](../pdfs/18-self-refine.md)
- **Analysis:** numerical code-readability proxies and sentiment preference do not by themselves establish semantic preservation of code or factual preservation of rewritten reviews. [Basis: Appendix L.2; Appendix P](../pdfs/18-self-refine.md)

## 7. Relevance to this project — interpretation

- **Proposal:** keep code generation, critique, and revision as separate logged steps so failed optimization attempts can be attributed to diagnosis versus implementation. [Basis: §2; §4 diagnostic study](../pdfs/18-self-refine.md)
- **Proposal:** require critiques to name a concrete bottleneck and a proposed transformation; generic requests to make a kernel faster may be less useful than localized feedback. [Basis: Table 2](../pdfs/18-self-refine.md)
- **Proposal:** combine self-feedback with trusted compilation, correctness, and timing observations; the oracle study motivates external feedback when the model cannot identify its own errors. This is a proposed hybrid, not the paper's unaided default. [Basis: Appendix H.1](../pdfs/18-self-refine.md)
- **Proposal:** retain previous validated candidates and roll back regressions rather than trusting the last revision or maximum self-score. Kernel selection should use executable evidence under fixed test conditions. [Basis: Appendix H](../pdfs/18-self-refine.md)
- **Proposal:** compare refinement against independent sampling at equal model/tool budgets and report both success coverage and speed among correct candidates. The paper's Python optimization results do not determine the corresponding GPU outcome. [Basis: §4; Appendix N](../pdfs/18-self-refine.md)
- **Scope warning:** this is inference-time artifact refinement; repeated self-feedback alone does not demonstrate persistent agent learning, recursive self-improvement, or a correct evaluator. [Basis: §2; §5](../pdfs/18-self-refine.md)

## 8. Retrieval notes

- **Aliases:** Self-Refine; self-feedback; feedback-refine; actionable critique; iterative revision; multi-aspect feedback; oracle feedback; CommonGen-Hard; PIE; mixed-refine.
- **Questions answered:** what the loop conditions on; whether weights change; what makes feedback actionable; whether revisions always improve; how oracle correctness changes math results; why the headline average cannot be read as accuracy; what the code-optimization metric measures.
- **Related analyses:** [Reflexion](19-reflexion.md) for reflection stored across trials; [ReAct](11-react.md) for interaction during a trial; [SWE-agent](10-swe-agent.md) for execution-grounded interface feedback; [Toolformer](20-toolformer.md) for actual parameter updates to learn tool use.
- **Navigation only:** [G-Eval](15-g-eval.md), [MT-Bench judging](16-mt-bench-chatbot-arena.md), and [Fair Evaluators](17-fair-evaluators.md) provide judge-evaluation context; no new cross-paper performance ranking is asserted here.

## 9. Source coverage and verification notes

- Covered main §§1–7 and all substantive appendix families: A task inventory; B related-method distinctions; C–D human/LLM evaluation; E model naming; F baseline comparisons; G weaker/mixed models; H diagnostics/oracle/selection; I website examples; J intervals; K new tasks; L–R individual task methods; S prompt structures. [Local full text](../pdfs/18-self-refine.md)
- Main results and conditions were checked against Tables 1–2, 4, 6–15 and Appendix N's displayed optimization table; prompt examples were used to understand initialization, critique, and revision, not treated as measurements.
- Repeated demonstration code and screenshot content are not reproduced wholesale. Figure 9's extracted plot labels are ambiguous, so this analysis records its qualitative conclusion without inventing exact percentages.
- The source contains material inconsistencies in decoding, subset sizes, judge identity, stopping/selection, oracle usage, and table values. These are retained explicitly; no experiments or original code were executed to resolve them.
