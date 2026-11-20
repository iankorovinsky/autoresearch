---
paper_id: "11"
title: "ReAct: Synergizing Reasoning and Acting in Language Models"
source_version: "2210.03629v3"
source_markdown: "../pdfs/11-react.md"
source_pdf: "../pdfs/11-react.pdf"
source_url: "https://arxiv.org/abs/2210.03629"
analysis_date: "2026-11-20"
analysis_status: "reviewed"
topics: ["reasoning-and-acting", "tool-use", "retrieval", "planning", "grounding", "few-shot-prompting", "agent-evaluation"]
summary: "Introduces interleaved language reasoning and environment actions, testing prompted and fine-tuned variants on knowledge reasoning and interactive decision tasks."
---

# ReAct: Synergizing Reasoning and Acting in Language Models

## 1. Paper identity and scope

- **Central question:** can a language model reason about what to do and then use the resulting observations to improve its subsequent reasoning, rather than treating reasoning and acting as separate pipelines? [§1](../pdfs/11-react.md)
- **Contribution:** a general prompting/control paradigm, ReAct, with experiments on HotpotQA, FEVER, ALFWorld, and WebShop; the paper also explores bootstrapped fine-tuning and human intervention. [§1; §§3–4; Appendix A.3](../pdfs/11-react.md)
- **Main setting:** frozen PaLM-540B with a few hand-annotated demonstrations; smaller PaLM fine-tuning and GPT-3 experiments are separate settings. [§2; §3.2; Appendix A.1](../pdfs/11-react.md)
- **Scope boundary:** these are question-answering, fact-verification, household simulation, and shopping-benchmark experiments; the paper does not measure code optimization, GPU speedup, or persistent learning across independent tasks. [§§3–4](../pdfs/11-react.md)

## 2. Concepts and terminology

### Reasoning and action in one context

- **Ordinary agent formulation:** at step (t), observations and earlier actions form context (c_t), and a policy chooses an environment action (a_t). [§2](../pdfs/11-react.md)
- **Augmented action space:** ReAct adds a language space of possible thoughts to the environment-action space, written in the paper as \(\hat{\mathcal A}=\mathcal A\cup\mathcal L\). A language action changes the model context rather than the external world. [§2](../pdfs/11-react.md)
- **Thought:** a generated natural-language reasoning trace used as working context for later decisions; it does not itself establish an external fact. [§2](../pdfs/11-react.md)
- **Action:** an executable environment command such as a search, lookup, navigation, or finish instruction. [§§2–4](../pdfs/11-react.md)
- **Observation:** information supplied by the environment after executing an action, distinct from model-generated thoughts. [§2; Figure 1](../pdfs/11-react.md)
- **Reason to act:** use language to decompose the objective, choose information to retrieve, plan subgoals, and select actions. [§§1–2](../pdfs/11-react.md)
- **Act to reason:** obtain new evidence from tools or environments, then use that evidence to update the reasoning state or answer. [§§1–2](../pdfs/11-react.md)
- **Thought functions:** goal decomposition, commonsense inference, extracting relevant observations, arithmetic/comparison, progress tracking, exception handling, and plan revision. [§2; §3.2](../pdfs/11-react.md)
- **Dense reasoning:** knowledge tasks alternate thought/action/observation steps, making reasoning explicit before each information-seeking action. [§2; §3.2](../pdfs/11-react.md)
- **Sparse reasoning:** long-horizon decision tasks permit multiple actions between thoughts; the model decides when reasoning is useful rather than mechanically thinking after every action. [§2; §4](../pdfs/11-react.md)
- **Closed loop:** observations can change the next plan; ReAct is not a fixed action sequence generated once and executed blindly. [§1; Figure 1](../pdfs/11-react.md)
- **Within-trajectory context:** the framework described here accumulates thoughts/actions/observations during an attempt. A persistent episodic reflection store between attempts is not part of the core formulation. [§2](../pdfs/11-react.md)

### Controlled alternatives and hybrid reasoning

- **Standard prompting:** examples contain questions and answers without explicit reasoning, actions, or observations. [§3.2](../pdfs/11-react.md)
- **CoT:** remove action/observation steps, retaining language reasoning based on internal model knowledge. [§3.2](../pdfs/11-react.md)
- **Act:** remove thought steps while keeping external interaction; this isolates the incremental role of explicit reasoning in an interactive agent. [§3.2; §4](../pdfs/11-react.md)
- **CoT self-consistency (CoT-SC):** sample 21 reasoning trajectories at temperature 0.7 and select the majority answer in the principal comparison. [§3.2](../pdfs/11-react.md)
- **ReAct → CoT-SC:** when ReAct has not answered within a step limit, switch to internal-knowledge reasoning with self-consistency. The limits are seven steps for HotpotQA and five for FEVER. [§3.2](../pdfs/11-react.md)
- **CoT-SC → ReAct:** if the most common answer among (n) CoT samples appears fewer than (n/2) times, switch to external retrieval. This is a heuristic confidence gate, not a calibrated probability of correctness. [§3.2](../pdfs/11-react.md)
- **Hybrid motivation:** CoT can formulate flexible reasoning but hallucinate facts; ReAct can ground facts but be constrained by poor retrieval or rigid trajectories. [§3.2–3.3](../pdfs/11-react.md)
- **ReAct-IM:** an Inner-Monologue-inspired ablation replaces varied sparse reasoning with dense goal/subgoal feedback, limiting reasoning about completion, next subgoals, and likely object locations. [§4; Appendix B.2](../pdfs/11-react.md)

### Tool interfaces and benchmark semantics

- **Wikipedia `search[entity]`:** returns the first five sentences of an exact entity page, or suggests the top five similar entities if lookup fails. [§3.1](../pdfs/11-react.md)
- **Wikipedia `lookup[string]`:** returns the next matching sentence within the current page, functioning like incremental in-page search. [§3.1](../pdfs/11-react.md)
- **`finish[answer]`:** terminates the knowledge task with the proposed answer or verification label. [§3.1](../pdfs/11-react.md)
- **Intentionally limited retrieval:** the API is weaker than strong lexical/neural retrievers and requires explicit entity identification/reformulation, exposing the interaction between reasoning and retrieval. [§3.1](../pdfs/11-react.md)
- **HotpotQA:** multi-hop question answering across two or more Wikipedia passages, evaluated by exact match (EM). The input is only the question, not the supporting passages. [§3.1; Table 1](../pdfs/11-react.md)
- **FEVER:** classify claims as SUPPORTS, REFUTES, or NOT ENOUGH INFO; the evaluation reports label accuracy in the question/claim-only setup. [§3.1; Table 1](../pdfs/11-react.md)
- **ALFWorld:** a text household environment with six task families—pick, clean, heat, cool, look, and pick two—requiring navigation, object manipulation, and ordered subgoals. [§4; Table 3](../pdfs/11-react.md)
- **ALFWorld state reasoning:** a task may involve more than 50 locations and more than 50 expert actions; plausible-location knowledge helps exploration but must be corrected by observations. [§4](../pdfs/11-react.md)
- **WebShop:** search products, inspect options, choose attributes, and complete a simulated purchase under user constraints, using noisy structured/unstructured product descriptions. [§4](../pdfs/11-react.md)
- **WebShop metrics:** score measures attribute coverage; success rate requires all constraints to be satisfied. Higher partial credit does not necessarily imply a completed correct purchase. [§4; Table 4](../pdfs/11-react.md)

## 3. Methodology and experimental setup

- **Few-shot knowledge prompts:** six HotpotQA and three FEVER training cases are randomly selected and manually annotated into ReAct trajectories; baselines are constructed by removing components. [§3.2; Appendix C.1–C.2](../pdfs/11-react.md)
- **Knowledge reasoning examples:** demonstrations teach multi-hop decomposition, disambiguation after failed search, comparisons, evidence extraction, and appropriate uncertainty when retrieved text does not establish the full claim. [Appendix C.1–C.2](../pdfs/11-react.md)
- **ALFWorld evaluation:** 134 unseen games, with task-specific prompting. Three annotated examples per task type produce six ordered two-example prompts. The same underlying trajectories form Act prompts. [§4](../pdfs/11-react.md)
- **ALFWorld comparator:** BUTLER is trained by imitation on (10^5) expert trajectories per task type; its reported best-of-eight trials differ from ReAct's best-of-six prompt configurations. [§4; Table 3](../pdfs/11-react.md)
- **ALFWorld decoding:** the table reports greedy decoding for the prompted methods, while BUTLER uses beam search. Results are not compute-matched training comparisons. [Table 3](../pdfs/11-react.md)
- **WebShop evaluation:** 500 test instructions, one-shot prompted examples; the environment contains 1.18 million products and roughly 12,000 human instructions. [§4](../pdfs/11-react.md)
- **WebShop baselines:** imitation learning on 1,012 annotated trajectories; IL+RL additionally uses 10,587 training instructions. Their scores are taken from the original WebShop work. [§4; Table 4](../pdfs/11-react.md)
- **Fine-tuning data:** 3,000 correct-answer model-generated trajectories, separately prepared for ReAct and baselines, train PaLM-8B and PaLM-62B. Correct-answer filtering does not necessarily validate every intermediate rationale. [§3.2](../pdfs/11-react.md)
- **Fine-tuning target:** decode complete trajectories, including thoughts, actions, and observations, conditioned on the question/claim. [§3.2](../pdfs/11-react.md)
- **Training schedules:** batch size 64; ReAct/Act use 4,000 steps for both sizes; Standard/CoT use 2,000 steps for 8B and 1,000 for 62B because longer training degraded those baselines. [Appendix B.1](../pdfs/11-react.md)
- **Manual diagnostic sample:** 50 successful and 50 unsuccessful HotpotQA trajectories from each of ReAct and CoT, totaling 200. Success is initially defined by EM, then the reasoning/facts are inspected. [§3.3; Table 2](../pdfs/11-react.md)
- **GPT-3 replication subset:** `text-davinci-002`, greedy decoding; 500 sampled HotpotQA validation questions and all 134 ALFWorld games, using the prompt set selected by PaLM performance. [Appendix A.1; Table 5](../pdfs/11-react.md)

## 4. Results and evidence

### Knowledge-intensive tasks

- **PaLM-540B HotpotQA EM:** Standard 28.7, CoT 29.4, CoT-SC 33.4, Act 25.7, ReAct 27.4, CoT-SC → ReAct 34.2, and ReAct → CoT-SC 35.1. ReAct alone beats Act but not CoT or Standard in this table. [Table 1](../pdfs/11-react.md)
- **PaLM-540B FEVER accuracy:** Standard 57.1, CoT 56.3, CoT-SC 60.4, Act 58.9, ReAct 60.9, CoT-SC → ReAct 64.6, and ReAct → CoT-SC 62.0. [Table 1](../pdfs/11-react.md)
- **Hybrid ordering is task-dependent:** external-first is strongest on HotpotQA, while confidence-gated internal-first is strongest on FEVER in this comparison. [§3.3; Table 1](../pdfs/11-react.md)
- **Specialized supervised gap:** Table 1 includes supervised reference results of 67.5 HotpotQA EM and 89.5 FEVER accuracy; the prompting methods remain well below these task-specialized systems. [Table 1; §3.3](../pdfs/11-react.md)
- **Sampling efficiency:** hybrid approaches reach the reported 21-sample CoT-SC performance with about three to five CoT samples, according to the scaling analysis; tool-call costs are additional and not eliminated by this comparison. [§3.3; Figure 2](../pdfs/11-react.md)
- **Correct answers with flawed justification:** among the 50 EM-success cases per method inspected, 6% of ReAct versus 14% of CoT have hallucinated facts/reasoning. These are conditional diagnostic rates, not overall false-positive rates over the benchmark. [Table 2; §3.3](../pdfs/11-react.md)
- **Failure taxonomy:** reasoning errors, failed/uninformative retrieval, hallucination, and label ambiguity. The authors include repetitive thought/action loops under reasoning error. [§3.3; Table 2](../pdfs/11-react.md)
- **Sampled failure distribution:** ReAct is reported as 47% reasoning error, 23% search error, 0% hallucination, and 29% label ambiguity; CoT as 16% reasoning error, 56% hallucination, and 28% label ambiguity. These small-sample values are not universal risk estimates. [Table 2](../pdfs/11-react.md)
- **Grounding does not remove reasoning failure:** an irrelevant or empty result can derail the trajectory, and greedy generation can get stuck repeating unproductive actions. [§3.3](../pdfs/11-react.md)
- **Time-varying labels:** the hotel-size example shows retrieved contemporary information disagreeing with an older dataset label; disagreement with EM can reflect label staleness rather than a model factual error. [Appendix A.2; Figure 4](../pdfs/11-react.md)
- **Fine-tuning reversal:** prompted ReAct is weakest among the four formats at PaLM-8B/62B, but after fine-tuning it is strongest; fine-tuned 8B ReAct exceeds the compared prompted 62B methods, and fine-tuned 62B exceeds the compared prompted 540B methods. [§3.3; Figure 3](../pdfs/11-react.md)
- **Authors' explanation of fine-tuning:** acting formats teach information-access behavior, whereas direct-answer/CoT training more readily memorizes answer facts; this is their interpretation of observed training behavior. [§3.3; Appendix B.1](../pdfs/11-react.md)

### Decision-making tasks and interventions

- **ALFWorld aggregate:** ReAct averages 57% across prompt configurations and reaches 71% in the best of six; Act's best-of-six is 45%, BUTLER's best-of-eight 37%, and BUTLERg's 22%. The often-quoted +34 points is 71 minus 37, a comparison of selected best trials. [Table 3](../pdfs/11-react.md)
- **Prompt robustness:** the worst ReAct prompt trial is reported at 48%, above the best Act 45%; relative gains across paired trials range from 33% to 90%, averaging 62%. [§4](../pdfs/11-react.md)
- **Task-family variation:** best ReAct scores are 92 pick, 58 clean, 96 heat, 86 cool, 78 look, and 41 pick-two; it is not best in every task family—BUTLER reaches 100 on cool. [Table 3](../pdfs/11-react.md)
- **ReAct-IM:** average 48% and best 53%, compared with ReAct average 57% and best 71%; the paper attributes weaknesses to insufficient completion tracking, next-subgoal selection, and commonsense exploration. [§4; Table 3; Appendix B.2](../pdfs/11-react.md)
- **WebShop:** ReAct score 66.6 and success 40.0%, versus Act 62.3/30.1%, IL 59.9/29.1%, and IL+RL 62.4/28.7%. [Table 4](../pdfs/11-react.md)
- **Human gap:** WebShop human experts score 82.1 with 59.6% success; the authors observe more extensive exploration and query reformulation in human trajectories. [§4; Table 4](../pdfs/11-react.md)
- **Interpreting the “10%” improvement:** ReAct's 40.0% versus Act's 30.1% is +9.9 percentage points; versus IL's 29.1% it is +10.9 points. Do not present these as a 10% relative improvement. [Table 4](../pdfs/11-react.md)
- **GPT-3 transfer:** Appendix A.1 reports HotpotQA 30.8 EM for GPT-3 versus 29.4 for PaLM on the 500-question subset; ALFWorld 78.4% versus 70.9%. This is a distinct subset/configuration from Table 1's 27.4 HotpotQA ReAct score. [Table 5](../pdfs/11-react.md)
- **Human thought editing:** one illustrative ALFWorld trajectory succeeds after a human removes one hallucinated thought and adds a later hint. This demonstrates feasibility, not a population-level intervention success rate. [Appendix A.3; Figure 5](../pdfs/11-react.md)

### What the example trajectories establish

- **HotpotQA demonstrations:** search disambiguation and finding a bridge entity make the reasoning/action dependency explicit; the agent need not retrieve every noun in the question. [Appendix C.1](../pdfs/11-react.md)
- **FEVER examples:** a narrowly changed date/location can require exact evidence; a confident CoT answer can miss the changed attribute, while retrieval can also fail on entity ambiguity. [Appendix D.1](../pdfs/11-react.md)
- **ALFWorld knife example:** ReAct tracks finding, taking, moving, cleaning, then placing; Act attempts cleaning without moving to the sink and repeats failing actions. [Appendix D.2.1–D.2.2](../pdfs/11-react.md)
- **ReAct-IM knife example:** a faulty subgoal phrasing treats an obtained knife as already clean, then repeatedly tries to place it; language state descriptions can themselves introduce errors. [Appendix D.2.3](../pdfs/11-react.md)
- **WebShop option example:** ReAct reasons over product options to choose the required flavor/pack size; the displayed trajectory scores 1.0 versus Act's 0.125. This is a case study, not the benchmark mean. [Appendix D.3; Table 10](../pdfs/11-react.md)
- **Correct answer can have an invalid rationale:** Appendix E's San Marco/About Time example reaches the right answer using wrong entity types; final-answer agreement alone does not validate a trajectory. [Appendix E.1](../pdfs/11-react.md)

## 5. Authors' conclusions

- Combining reasoning and action can produce stronger, more interpretable task-solving trajectories than isolating either capability, with gains varying by task and baseline. [§6; Tables 1, 3–4](../pdfs/11-react.md)
- External interaction improves factual grounding while language reasoning guides retrieval and long-horizon behavior; the two are complementary rather than universally interchangeable. [§1; §3.3](../pdfs/11-react.md)
- Sparse, flexible thoughts can be more useful for decision tasks than repetitive state/subgoal restatement. [§4; Appendix B.2](../pdfs/11-react.md)
- Combining ReAct with self-consistency is especially effective in the studied knowledge tasks; ReAct alone is not the best method in every setting. [§3.3](../pdfs/11-react.md)
- More high-quality trajectories, multitask training, and integration with reinforcement learning are proposed paths forward. [§6](../pdfs/11-react.md)

## 6. Limitations and open questions

### Explicit source limitations

- Large action spaces require more demonstrations than limited in-context windows can accommodate. [§6](../pdfs/11-react.md)
- Uninformative retrieval and failure to reformulate can trap the agent; Wikipedia access does not guarantee relevant evidence. [§3.3](../pdfs/11-react.md)
- Greedy decoding can yield repetitive thought/action loops; improved decoding is suggested but not established as a fix. [§3.3, footnote on reasoning errors](../pdfs/11-react.md)
- Most primary experiments use a then-nonpublic model, limiting direct reproduction; prompts and supplementary GPT-3 experiments are provided to help. [Reproducibility Statement](../pdfs/11-react.md)
- The action spaces are deliberately constrained: no real purchases and no editing Wikipedia. Broader real-world tool access would create risks absent from these benchmarks. [Ethics Statement](../pdfs/11-react.md)
- Human thought editing is explored through an example; systematic evaluation remains future work. [Appendix A.3](../pdfs/11-react.md)

### Analyst caveats

- **Analysis:** visible reasoning is useful for diagnosis but is not a proof that the text faithfully explains the model's internal computation or that its facts are correct. The paper itself includes correct-answer/invalid-reasoning examples. [Basis: Table 2; Appendix E.1](../pdfs/11-react.md)
- **Analysis:** prompt-selected best-of-six results should not be presented as the expected score of a randomly chosen prompt; the mean is separately available. [Basis: Table 3](../pdfs/11-react.md)
- **Analysis:** comparisons against imitation/RL methods vary in model scale, pretraining, task training, and decoding; they do not isolate the benefit of ReAct at matched total compute. [Basis: §4](../pdfs/11-react.md)
- **Analysis:** “0% hallucination” in the sampled ReAct failure category is a small diagnostic result, not a guarantee against hallucination, especially in decision tasks or new domains. [Basis: Table 2; Appendix A.3](../pdfs/11-react.md)
- **Source caution:** Appendix C.4's Table 7 is labeled Act/no thoughts but includes a `think` line in its displayed example. The formal Act ablation is defined as removing thoughts; the example inconsistency should not be silently used to redefine the method. [§3.2; Appendix C.4, Table 7](../pdfs/11-react.md)
- **Source caution:** some prose cross-references point to the wrong table/figure, and rounded diagnostic percentages do not sum perfectly; use the captioned tables and specified sample sizes. [§3.3–4; Tables 1–4](../pdfs/11-react.md)

## 7. Relevance to this project — interpretation

- **Proposal:** use a repeatable hypothesis → tool action → measured observation loop for kernel work; thoughts should choose the next experiment and interpret its output rather than stand in for an experiment. [Basis: §2](../pdfs/11-react.md)
- **Proposal:** record which statements come from execution/profiling and which are model hypotheses, making failed assumptions easier to diagnose later. [Basis: §1; §3.3](../pdfs/11-react.md)
- **Proposal:** allow sparse planning between routine operations, and trigger replanning when tool outputs contradict the current plan; test whether this outperforms mechanically verbose reasoning. [Basis: §2; §4](../pdfs/11-react.md)
- **Proposal:** detect repeated unproductive actions and budget exhaustion, then consider switching strategy; the paper motivates such switches but does not establish an optimal kernel-agent restart rule. [Basis: §3.2–3.3](../pdfs/11-react.md)
- **Scope warning:** a ReAct harness supplies an interaction pattern, not numerical correctness guarantees, compiler expertise, persistent memory, or autonomous improvement of its own weights. [Basis: §§2–4](../pdfs/11-react.md)

## 8. Retrieval notes

- **Aliases:** ReAct; Reason + Act; thought-action-observation; sparse thoughts; dense thoughts; Act-only; CoT-SC; ReAct-IM; tool-grounded reasoning.
- **Questions answered:** when to interleave retrieval and reasoning; why tool access alone can fail; how to compare Act, CoT, and ReAct; what “34-point ALFWorld improvement” means; whether ReAct guarantees factuality; how thought editing was evaluated.
- **Related analyses:** [SWE-agent](10-swe-agent.md) specializes the interaction interface for repository work; [Reflexion](19-reflexion.md) adds lessons across attempts; [Self-Refine](18-self-refine.md) focuses on iterative artifact revision; [Toolformer](20-toolformer.md) trains tool-call behavior through a different learning signal.

## 9. Source coverage and verification notes

- Covered main §§1–6 and reproducibility/ethics statements; Appendix A's GPT-3, time-varying facts, and human intervention; Appendix B's training and IM details; Appendix C's task prompts; Appendix D's task trajectories; Appendix E's error examples. [Local full text](../pdfs/11-react.md)
- Inspected the prompt and trajectory formats and their substantive differences; lengthy repeated environment observations are not reproduced as findings. Results are transcribed from captioned Tables 1–5 and the cited diagnostic sections, not estimated from images.
- The PaLM/GPT-3 HotpotQA scores refer to different evaluated subsets; ALFWorld average/best trials and WebShop score/success are explicitly separated. No experiments were re-run.
- This analysis is tied to v3. It describes reported evidence, not a claim that these models, environments, or API behaviors are current.
