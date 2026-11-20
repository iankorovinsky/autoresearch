---
paper_id: "19"
title: "Reflexion: Language Agents with Verbal Reinforcement Learning"
source_version: "2303.11366v4"
source_markdown: "../pdfs/19-reflexion.md"
source_pdf: "../pdfs/19-reflexion.pdf"
source_url: "https://arxiv.org/abs/2303.11366"
analysis_date: "2026-11-20"
analysis_status: "reviewed"
topics: ["agents", "reflection", "episodic-memory", "self-generated-tests", "feedback", "program-synthesis", "reasoning", "evaluation"]
summary: "Reflexion converts trial feedback into bounded textual memories for subsequent attempts, with gains and failures across ALFWorld, HotpotQA, and code generation that depend strongly on evaluation quality."
---

# Reflexion: Language Agents with Verbal Reinforcement Learning

## 1. Paper identity and scope

- Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao propose Reflexion: adapting an agent through linguistic feedback stored in context rather than model-weight updates. This analysis describes the frozen v4 source. [Title; Abstract; §1](../pdfs/19-reflexion.md)
- The experiments cover sequential household decision-making, Wikipedia-based question answering, and function generation in interpreted and compiled languages. WebShop supplies an important negative result in the appendix. [§4; Appendix B.1](../pdfs/19-reflexion.md)
- The principal research question is whether an LLM can transform sparse success/failure feedback into actionable lessons that improve subsequent attempts on the same task. This is a trial-and-error inference procedure, not a demonstration of gradient-based reinforcement learning. [§§1, 3](../pdfs/19-reflexion.md)
- The paper also introduces LeetcodeHardGym: 40 hard-rated Leetcode questions, with infrastructure described as supporting 19 languages; the reported benchmark result is for Python. [§1 contributions; §4.3; Table 1](../pdfs/19-reflexion.md)
- Its headline HumanEval result is 91.0% pass@1 for the final product of a self-testing/reflection process, compared with a single-generation GPT-4 baseline of 80.1%. It must not be described as 91% correctness on the first internal code draft. [§4.3; Table 1](../pdfs/19-reflexion.md)

## 2. Concepts and terminology

### Agent components and memory

- **Actor:** a prompted LLM generates text or executable environment actions from observations, trajectory history, and stored experiences; the framework accommodates both CoT and ReAct actors. [§3, Actor](../pdfs/19-reflexion.md)
- **Evaluator:** a task-dependent component assesses a trajectory/output. It may use an exact-match answer checker, a heuristic, an LLM classifier, or execution of generated unit tests; it is not necessarily a separately trained neural reward model. [§3, Evaluator; §4](../pdfs/19-reflexion.md)
- **Self-reflection model:** an LLM receives the attempted trajectory, evaluation signal, and memory, then produces a natural-language explanation and proposed improvement. The three roles are modular functions; the paper does not require three different underlying model families. [§3, Self-reflection](../pdfs/19-reflexion.md)
- **Short-term memory:** the current trajectory of actions and observations, retaining details of the ongoing attempt. [§3, Memory](../pdfs/19-reflexion.md)
- **Long-term memory:** textual reflections from previous trials, supplied to future actor calls. “Long-term” here means persistence across attempts within the experiment, not an established cross-project or lifelong knowledge system. [§3, Memory and Reflexion process](../pdfs/19-reflexion.md)
- **Bounded experience buffer:** memory capacity Ω is normally one to three reflections because of model context limits; old experiences leave the sliding window. [§3, Reflexion process; §5](../pdfs/19-reflexion.md)
- **Verbal reinforcement / semantic gradient:** the authors' analogy for translating a scalar failure signal into directional textual feedback. No differentiable gradient through the environment or weight update is calculated. [§1; §3](../pdfs/19-reflexion.md)
- **Credit assignment:** the reflection attempts to identify which earlier choice caused downstream failure and propose a different choice or plan. This diagnosis is inferred by the model, not guaranteed by the scalar reward. [§1; §3, Self-reflection](../pdfs/19-reflexion.md)
- **Parameterized policy:** the algorithm writes θ as the actor together with its memory; changing memory changes the effective policy even when the LLM remains fixed. [§3; Figure 2 / Algorithm 1](../pdfs/19-reflexion.md)
- **Episodic-memory baseline (EPM):** passing the previous trajectory back to the actor without the full reflection step, used to distinguish raw experience access from distilled explanation. [§4.2, Analysis](../pdfs/19-reflexion.md)

### Feedback reliability and task metrics

- An environment's completion signal and an agent's decision to abandon a failing attempt serve different purposes. ALFWorld supplies success, while heuristics/LLM classification trigger resets and reflection before success. [§4.1](../pdfs/19-reflexion.md)
- Programming uses **internal generated tests** to guide refinement and **held-out benchmark tests** to measure the final submission; the authors argue the absence of hidden-test feedback during refinement preserves pass@1 eligibility. [§4.3](../pdfs/19-reflexion.md)
- A generated-test **false positive** is an incorrect program that passes internal tests; this can terminate refinement prematurely. A **false negative** is a correct program that fails faulty internal tests; reflection might recognize the tests are wrong. [§4.3, Analysis; Table 2](../pdfs/19-reflexion.md)
- The displayed Table 2 cells are not four joint probabilities summing to one: TP+FP and FN+TN each sum to one in its rows. The prose explicitly interprets FP as probability of final incorrectness conditional on tests passing. [§4.3, Analysis; Table 2](../pdfs/19-reflexion.md)
- **Cumulative solved proportion** over repeated ALFWorld/HotpotQA trials differs from single-attempt accuracy; memory, retry budget, and early stopping are part of the result. [Figures 3–4; §4.1–4.2](../pdfs/19-reflexion.md)

## 3. Methodology and experimental setup

### General loop

- Start with an actor-generated trajectory; evaluate it; use trajectory and reward to generate a reflection; append that reflection to the memory provided for subsequent trials. [§3; Figure 2](../pdfs/19-reflexion.md)
- Subsequent attempts reset the environment while retaining textual experience, so the actor can exploit knowledge from failures without resuming the exact failed environment state. [§4.1](../pdfs/19-reflexion.md)
- The prose says to continue until evaluation deems the trajectory correct, subject to a trial limit. The printed algorithm instead uses `not pass or t < max trials`; that logical OR is inconsistent with stopping on either success or exhaustion and should not be implemented literally as the intended policy. [§3; PDF p. 4, Algorithm 1](../pdfs/19-reflexion.pdf#page=4)

### ALFWorld

- Evaluate 134 text-based household environments spanning six task types, including finding, moving, and transforming objects. [§4.1](../pdfs/19-reflexion.md)
- Use a ReAct actor with GPT-3 and two domain-specific demonstration trajectories reused from the ReAct setup to reduce action-format errors. [§4.1](../pdfs/19-reflexion.md)
- Compare two self-evaluation routes: LLM natural-language classification and a hand-written heuristic. [§4.1; Figure 3](../pdfs/19-reflexion.md)
- The heuristic triggers when the same action produces the same response for more than three cycles, or when an attempt exceeds 30 actions. These detect looping/inefficient planning rather than prove a task impossible. [§4.1](../pdfs/19-reflexion.md)
- Baseline agents also reset and retry when the trigger fires; Reflexion agents additionally create and retain a lesson. This makes the comparison more informative than comparing multiple Reflexion trials with a strictly one-shot baseline alone. [§4.1](../pdfs/19-reflexion.md)
- Retain the latest three reflections and run 12 consecutive trials; reported success is cumulative across environments. [§4.1; Figure 3](../pdfs/19-reflexion.md)

### HotpotQA

- Main reasoning experiments use 100 questions from a dataset described as containing 113,000 question-answer pairs. The 100-question evaluation must not be confused with a full-dataset result. [§4.2; Figure 4](../pdfs/19-reflexion.md)
- Three actor conditions isolate different information access: CoT with question alone; CoT(GT) with supplied ground-truth supporting context; ReAct with Wikipedia API retrieval. [§4.2](../pdfs/19-reflexion.md)
- CoT uses six-shot prompting, ReAct two-shot prompting, and reflection two-shot prompting. Different actor strategies therefore also use different demonstration layouts. [§4.2](../pdfs/19-reflexion.md)
- A ground-truth exact-match grader supplies success/failure between trials without revealing the answer text; this is external correctness supervision, even though the model does not see the answer itself. [§4.2](../pdfs/19-reflexion.md)
- Retain three experiences; failed questions may be retried until three consecutive failed attempts. The reported baseline retry sampling uses temperature 0.7. [§4.2, Results](../pdfs/19-reflexion.md)
- CoT(GT)+EPM adds the most recent trajectory; the full comparison adds a self-reflective explanation, intended to measure the value of distilled reasoning beyond merely seeing a failed trace. [§4.2, Analysis](../pdfs/19-reflexion.md)

### Programming and additional-model tests

- HumanEval and MBPP ask for a function body from a natural-language specification; Rust versions use MultiPL-E translations. Python and Rust task subsets are not automatically identical. [§4.3](../pdfs/19-reflexion.md)
- HumanEval Rust uses the 50 hardest HumanEval Python problems translated to Rust; both the language comparison and ablations are affected by this selected subset. [Tables 2–3](../pdfs/19-reflexion.md)
- LeetcodeHardGym's 40 questions were released after October 8, 2022, described by the authors as GPT-4's pretraining cutoff. This is the paper's contamination-control rationale, not an independently audited training-data guarantee. [§4.3](../pdfs/19-reflexion.md)
- Generate a diverse unit-test pool with CoT and natural-language descriptions; reject tests that cannot form valid ASTs, then sample at most six unit tests. Syntactic validity is weaker than semantic correctness. [§4.3](../pdfs/19-reflexion.md)
- Code-generation memory is limited to one experience. Table 1 describes instruction-model code generation as zero-shot, while appendix templates mention user examples; the exact run configuration should be checked before replication. [§4.3; Table 1; Appendix C.2](../pdfs/19-reflexion.md)
- The Rust ablation compares baseline, reflection without test generation, testing without reflection, and the full procedure on the same 50-problem hard subset using GPT-4. [§4.3; Table 3](../pdfs/19-reflexion.md)
- Appendix A evaluates a weaker open model, starchat-beta, on HumanEval Python and evaluates text-davinci-003, gpt-3.5-turbo, and GPT-4 on 100 HotpotQA questions. [Appendix A; Tables 4–5](../pdfs/19-reflexion.md)

## 4. Results and evidence

### Decision-making and reasoning

- Heuristic-triggered ReAct+Reflexion solves 130/134 ALFWorld tasks over 12 trials, approximately 97.0%; this is cumulative success, not initial-trial performance. [§4.1, Results; Figure 3](../pdfs/19-reflexion.md)
- ReAct-only improvement plateaus around trials six to seven, whereas Reflexion continues improving over the trial sequence. [§4.1, Results](../pdfs/19-reflexion.md)
- The authors report a 22-percentage-point ALFWorld improvement and identify hallucinated possession of objects as a persistent baseline failure; ReAct-only hallucination reaches about 22% in the diagnostic curve. These are distinct quantities. [§1; §4.1, Analysis; Figure 3](../pdfs/19-reflexion.md)
- Reflection is proposed to help both identify an early mistake in a long trajectory and remember which locations have already been searched. The latter is environment-specific accumulated information, not necessarily general skill acquisition. [§4.1, Analysis](../pdfs/19-reflexion.md)
- On the reported HotpotQA experiment, repeated baseline sampling at temperature 0.7 did not solve additional initially failed questions; the authors report improvement with reflective memory. This is a result for that setup, not proof that resampling never helps. [§4.2, Results](../pdfs/19-reflexion.md)
- Main-text CoT(GT) fails on 39% of questions and improves by 14 percentage points with Reflexion, corresponding to 61%→75% from the stated values. [§4.2, Results](../pdfs/19-reflexion.md)
- Reflection adds an eight-percentage-point gain over the episodic-memory advantage in the CoT(GT) ablation, supporting a role for the verbal explanation beyond retaining a previous trajectory in this setting. [§4.2, Analysis; Figure 4c](../pdfs/19-reflexion.md)
- Appendix A's CoT(GT) results are text-davinci-003 60%→77%, gpt-3.5-turbo 57%→71%, and GPT-4 68%→80% on 100 questions. Keep these separate from main-text 61%→75%; the paper does not reconcile the discrepancy. [Appendix A; Table 5; §4.2](../pdfs/19-reflexion.md)
- Appendix A's ReAct results are text-davinci-003 30%→55%, gpt-3.5-turbo 26%→38%, and GPT-4 39%→51%, each on the stated 100-question setup. [Appendix A; Table 5](../pdfs/19-reflexion.md)
- The abstract/introduction's HotpotQA “20%” improvement should not be treated as a universal multiplier: gains vary with actor, model, context access, and which experiment is cited. [§1; §4.2; Appendix A, Table 5](../pdfs/19-reflexion.md)
- WebShop evaluates a two-shot ReAct+Reflexion agent on 100 shopping environments and stops after four trials because improvement is absent; it does not significantly beat ReAct. [Appendix B.1; Figure 6](../pdfs/19-reflexion.md)
- The WebShop diagnosis is that unhelpful reflections keep the agent in local choices when success requires substantially different, precise search queries; the authors contrast this with visible ALFWorld action options and less restrictive Wikipedia search. [Appendix B.1](../pdfs/19-reflexion.md)

### Code generation and evaluator diagnostics

- HumanEval Python improves from GPT-4 80.1% to Reflexion 91.0%, a 10.9-percentage-point increase in Table 1. The earlier CodeT+GPT-3.5 comparison is 65.8% and uses a different model/strategy. [Table 1](../pdfs/19-reflexion.md)
- HumanEval Rust improves 60.0%→68.0% on the 50 hardest translated problems: 30→34 solved if the percentages are interpreted with that denominator. [Tables 1–3](../pdfs/19-reflexion.md)
- MBPP Python decreases 80.1%→77.1%, a three-percentage-point regression; a knowledge-store answer must preserve this exception to the general improvement claim. [Table 1](../pdfs/19-reflexion.md)
- MBPP Rust improves 70.9%→75.4%, a 4.5-percentage-point increase on its translated subset. [Table 1](../pdfs/19-reflexion.md)
- LeetcodeHard Python improves 7.5%→15.0%, corresponding to 3→6 of 40 tasks: doubling the rate but adding only three successes. [§4.3; Table 1](../pdfs/19-reflexion.md)
- Generated-test diagnostics for HumanEval Python are TP .99, FN .40, FP .01, TN .60; for MBPP Python they are .84, .59, .16, .41. These are the table's conditional groupings, not a conventional normalized confusion matrix. [Table 2; §4.3, Analysis](../pdfs/19-reflexion.md)
- HumanEval Rust's corresponding values are .87, .37, .13, .63; MBPP Rust's are .84, .51, .16, .49. Test reliability depends on benchmark and language setting. [Table 2](../pdfs/19-reflexion.md)
- The prose gives more precise false-positive values of 16.3% for MBPP Python versus 1.4% for HumanEval Python, interpreted as incorrectness among implementations passing internal tests. The authors associate MBPP's high false-positive rate with its overall regression. [§4.3, Analysis](../pdfs/19-reflexion.md)
- The same prose calls HumanEval's baseline 82%, whereas Table 1 says 80.1% and Table 2 rounds to .80. Use table-specific values rather than silently substituting 82%. [§4.3, Analysis; Tables 1–2](../pdfs/19-reflexion.md)
- Without test generation, reflection scores 52% on the 50-problem Rust subset versus 60% baseline; inability to stop correctly can induce harmful edits. [Table 3; §4.3, Ablation study](../pdfs/19-reflexion.md)
- With tests but without self-reflection, performance remains 60%; full testing+reflection reaches 68%. This supports their interaction for this hard Rust setting, not a general theorem that direct debugging cannot work. [Table 3; §4.3, Ablation study](../pdfs/19-reflexion.md)
- starchat-beta HumanEval Python remains at .26 mean accuracy with and without Reflexion, averaged over eight trials; reported standard deviations are .00481 and .00305 respectively. Larger inference procedures alone did not improve this weaker model here. [Appendix A; Table 4](../pdfs/19-reflexion.md)

### What the appendix examples demonstrate

- The ALFWorld mug/lamp example shows a failure after wandering away from the desk with the lamp, followed by a short successful attempt that revisits the known desk, takes the mug, and uses the lamp. Memory can preserve a useful location even if the reflection's “find lamp first” explanation is not a complete causal account. [Appendix B; Figure 5](../pdfs/19-reflexion.md)
- Programming instructions constrain outputs to a function body with required indentation and feed back implementation, test outcomes, and reflection. Formatting contracts are part of making generated code executable. [Appendix C.1–C.2](../pdfs/19-reflexion.md)
- The HotpotQA search example changes from an unrelated actor to Sam Kelly and reaches the intended role; however, its displayed reflection recommends Gorden Kaye, the wrong earlier branch. The example therefore cannot cleanly establish that the shown reflection caused the successful search. [Appendix D.1; Figure 7](../pdfs/19-reflexion.md)
- The profession example narrows an overinclusive answer to the intersection “novelist”; it illustrates answer-set refinement after binary feedback rather than new external retrieval. [Appendix D.2](../pdfs/19-reflexion.md)
- The White Plains example changes answer granularity from one battle to the broader campaign named in the supplied context, illustrating reinterpreting the question with ground-truth supporting text. [Appendix D.3](../pdfs/19-reflexion.md)
- The EPM examples illustrate revisiting an earlier answer/context, but the musician example's accepted answer follows reasoning claiming both musicians had seven bands. Accepted output is not equivalent to a sound explanation. [Appendix D.4.1](../pdfs/19-reflexion.md)
- The degree example corrects a generic list of fields to the specific field “Engineering” explicitly present in context; it highlights entity-specific extraction rather than generalized degree knowledge. [Appendix D.4.2](../pdfs/19-reflexion.md)

## 5. Authors' conclusions

- The authors conclude that natural-language reflection can improve agent behavior over a small number of trials without finetuning, while exposing explicit memories that humans can inspect. [§§1, 7](../pdfs/19-reflexion.md)
- They attribute gains to turning sparse feedback into actionable lessons and preserving these lessons across attempts, rather than repeatedly sampling an unchanged policy context. [§3; §4.1–4.2](../pdfs/19-reflexion.md)
- They argue that programming performance is bounded by the quality and coverage of self-generated tests, and prefer false negatives to false positives because false positives can end a run incorrectly. [§4.3, Analysis](../pdfs/19-reflexion.md)
- They interpret the Rust ablation as evidence for explicit diagnosis between error detection and implementation repair on difficult tasks. The stronger statement about competing debugging methods remains an inference from this limited comparison. [§4.3, Ablation study](../pdfs/19-reflexion.md)
- They propose richer memory structures, natural-language value learning, and off-policy exploration as future directions rather than implemented results. [§§5, 7](../pdfs/19-reflexion.md)
- They suggest monitoring reflections may aid interpretability/safety, while acknowledging that stronger agents can also amplify misuse. No formal guarantee of faithful or safe reflection is established. [§6](../pdfs/19-reflexion.md)

## 6. Limitations and open questions

### Explicit source limitations

- Reflection can get trapped in local optima; WebShop is a concrete failed application where better exploration was needed. [§5; Appendix B.1](../pdfs/19-reflexion.md)
- Small sliding-window memories discard experiences and are limited by context capacity; database-backed memories are proposed but not evaluated. [§3; §5](../pdfs/19-reflexion.md)
- Test-driven evaluation is difficult for nondeterministic generators, impure/API-dependent functions, hardware-sensitive outputs, and concurrent behavior. These limitations matter particularly when transferring to GPU systems. [§5](../pdfs/19-reflexion.md)
- Success depends on model self-evaluation/heuristic quality and has no formal guarantee; starchat-beta supplies a negative capability result. [§1; Appendix A](../pdfs/19-reflexion.md)
- The reproducibility section recommends isolated execution because generated code has not been validated before execution. [§8](../pdfs/19-reflexion.md)

### Analyst caveats and unresolved source issues

- **Interpretation:** same-task retry improvement is not evidence of general transfer to new tasks, durable weight learning, or a reusable cross-task knowledge base. The reported memory is local to the repeated-task procedure. [§3; §4](../pdfs/19-reflexion.md)
- **Interpretation:** comparing final pass@1 with a one-generation baseline measures an entire inference workflow, including additional model calls and execution. It does not isolate equal-cost model quality or equal-token improvement. [§4.3; Table 1](../pdfs/19-reflexion.md)
- **Interpretation:** HotpotQA's ground-truth binary feedback is unavailable in many real research tasks; omitting answer text does not remove dependence on an oracle checker. [§4.2](../pdfs/19-reflexion.md)
- **Interpretation:** memory explanations should be treated as hypotheses. The misleading search reflection and internally inconsistent musician reasoning show why a successful answer cannot validate every stored lesson. [Appendix D.1; D.4.1](../pdfs/19-reflexion.md)
- **Unresolved:** the printed loop uses OR where the surrounding prose suggests bounded failure retries; the test-set notation lists indices 0…n while calling n the maximum test count. Preserve prose intent and verify executable code before reproduction. [§3, Figure 2; §4.3](../pdfs/19-reflexion.md)
- **Unresolved:** Appendix C.3 repeats an actor-style code-editing instruction under the reflection heading; C.4's “no self-reflection” template still lists reflection, and C.5's “no test generation” template still lists test feedback. These templates are not reliable standalone ablation specifications. [Appendix C.3–C.5](../pdfs/19-reflexion.md)
- **Unresolved:** main and appendix HotpotQA values, HumanEval baseline prose/table values, and zero-shot versus appendix example wording are not reconciled. Do not merge these into a single supposedly identical run. [§4.2–4.3; Tables 1, 5; Appendix C.2](../pdfs/19-reflexion.md)

## 7. Relevance to this project — interpretation

- **Proposal:** store each research/coding attempt's goal, executable evidence, failure classification, proposed explanation, and next experiment separately; a textual reflection should retain links to the observations that motivated it.
- **Proposal:** distinguish a learned factual observation (e.g., a compiler diagnostic or measured timing) from a model's causal hypothesis; the latter should remain revisable when the next attempt disagrees.
- **Proposal:** maintain separate development tests and final independent validation for kernels. Passing model-generated checks should never by itself authorize a correctness or speedup claim.
- **Proposal:** measure retry-budget curves, total tokens, execution time, final success, and regressions to determine whether reflection is worthwhile relative to resampling or direct repair at matched cost.
- **Proposal:** include an exploration escape rule when repeated reflections recommend the same unsuccessful idea; the WebShop failure is a reason to evaluate this, not proof of a particular solution.
- **Proposal:** test a bounded recent-memory buffer against a searchable evidence-linked archive. Reflexion motivates the question but does not establish that larger/vector memories improve this project.
- **Proposal:** evaluate reflection quality separately from final answer correctness and audit whether retained lessons actually predict outcomes on new tasks.

## 8. Retrieval notes

- **Aliases:** Reflexion; verbal reinforcement learning; reflective episodic memory; actor–evaluator–self-reflection; LeetcodeHardGym.
- **Questions answered:** What changes when the model weights stay fixed? What is the difference between raw trajectory memory and reflection? Why can self-tests hurt? What does Reflexion's pass@1 mean? Where did reflection fail?
- **Useful numerical anchors:** ALFWorld 130/134 cumulative; HumanEval Python 80.1%→91.0%; MBPP Python 80.1%→77.1%; hard Rust ablation 60/52/60/68%; WebShop no significant benefit. [§4; Tables 1–3; Appendix B.1](../pdfs/19-reflexion.md)
- [ReAct](11-react.md) explains the within-trial reasoning/action actor that Reflexion augments with across-trial memory.
- [Self-Refine](18-self-refine.md) explains iterative feedback/refinement and its own evaluator/stopping caveats; compare mechanisms and supervision conditions, not aggregate task scores.
- [SWE-agent](10-swe-agent.md) provides the distinct question of interface design for repository interaction; Reflexion does not evaluate that interface.
- [Toolformer](20-toolformer.md) is navigation for learning tool-use behavior through model finetuning, a different adaptation mechanism from textual memory.

## 9. Source coverage and verification notes

- Read main §§1–8 and Appendices A–D, including additional-model tables, negative WebShop result, programming templates, and example trajectories. Bibliography entries were used only as navigation, not independently verified studies. [Local full text](../pdfs/19-reflexion.md)
- Checked numerical claims against Tables 1–5 and the corresponding prose. Inspected PDF text on physical pages 4–7 to verify Algorithm 1 and main figure/table labels; source-linked Markdown labels are used elsewhere.
- Curve endpoints not explicitly stated in prose/tables are not guessed from extracted figure text. Exact performance for the GPT-classifier ALFWorld curve and individual trial-by-trial points is intentionally not asserted.
- Recorded printed algorithm/template inconsistencies and misleading example explanations rather than silently repairing them. No benchmark runs, code inspection, or independent model-contamination audit were performed.
