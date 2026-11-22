---
paper_id: "29"
title: "ClarifyGPT: Empowering LLM-based Code Generation with Intention Clarification"
source_version: "2310.10996v1"
source_markdown: "../pdfs/29-clarifygpt.md"
source_pdf: "../pdfs/29-clarifygpt.pdf"
source_url: "https://arxiv.org/abs/2310.10996"
analysis_date: "2026-11-21"
analysis_status: "draft"
topics: ["code-generation", "requirement-clarification", "ambiguity-detection", "clarifying-questions", "user-simulation", "human-evaluation", "execution-feedback", "differential-testing", "evaluation", "measurement-validity"]
summary: "ClarifyGPT flags a requirement as ambiguous when sampled solutions disagree on generated test inputs, asks targeted questions and regenerates code; Pass@1 gains on HumanEval/MBPP use answers from humans or an LLM that both see ground-truth tests."
---

# ClarifyGPT: Empowering LLM-based Code Generation with Intention Clarification

## 1. Paper identity and scope

- This analysis covers arXiv `2310.10996v1`, a preprint by Mu, Shi, Wang, Yu, Zhang, Wang, Liu and Wang presenting ClarifyGPT, a prompting framework wrapped around an existing chat LLM. [Title; front matter](../pdfs/29-clarifygpt.md)
- The research problem is that LLMs generate code directly from unclear requirements instead of asking the user, which the authors split into two questions: when to ask clarifying questions, and what to ask. [Abstract; §1](../pdfs/29-clarifygpt.md)
- The claimed contributions are the framework, an LLM-based user-simulation method for automated evaluation, an evaluation on four benchmarks plus a human study, and a public dataset and code repository. [§1, contribution list](../pdfs/29-clarifygpt.md)
- The work studies general-purpose function-level Python code generation on HumanEval and MBPP variants. It contains no GPU kernels, no performance measurement and no formal specification; every result is a functional-correctness pass rate on unit tests. [§4.3; §4.4](../pdfs/29-clarifygpt.md)
- The framework changes no model weights. It is evaluated only with two OpenAI chat models, and the authors state that models without instruction tuning are unsuitable as its base. [§4.2; §6.2](../pdfs/29-clarifygpt.md)
- The source does not measure whether the ambiguity detector is accurate. No benchmark problem carries an ambiguity label, and the only reported outcome is end-task Pass@1 after the whole pipeline. [§4.4; §5.1–5.3](../pdfs/29-clarifygpt.md)
- The only component study varies the number of prompt demonstrations. There is no ablation that removes the consistency check, the mutation step or the reasoning step of the question prompt. [§4.1, RQ3; §5.3](../pdfs/29-clarifygpt.md)

## 2. Concepts and terminology

### Ambiguity and the consistency criterion

- The source treats a requirement as clear when it leaves no room for interpretation and as ambiguous when stakeholders can interpret it in different ways; the running example is sorting a list without a stated order. [§1; §3.2](../pdfs/29-clarifygpt.md)
- The operational definition is an assumption stated by the authors: if an LLM produces code solutions with different functionality for one requirement, the requirement admits several interpretations and needs clarification. [§3.2](../pdfs/29-clarifygpt.md)
- **Code consistency check:** sample `n` solutions from the LLM for the requirement, execute all of them on the generated test inputs, and compare outputs. Identical outputs mean "unambiguous"; non-identical outputs mean "ambiguous". [§1; §3.2](../pdfs/29-clarifygpt.md)
- **Code clustering:** for a requirement judged ambiguous, the sampled solutions are grouped by their test outputs. Figure 1 draws the decision as "#clusters<2?"; fewer than two clusters returns a candidate directly, otherwise the pipeline proceeds to question generation. [§3.2; Figure 1](../pdfs/29-clarifygpt.md)
- The criterion is behavioural agreement among the model's own samples, not agreement with ground truth. Only test inputs are generated; no reference tests or expected outputs are used at this stage. [§3.1; §3.2](../pdfs/29-clarifygpt.md)
- The decision rule is all-or-nothing as described: any output difference triggers clarification. The source gives no tolerance, vote share or minimum cluster size, and no rule for solutions that raise exceptions or time out. [§3.2; Figure 1](../pdfs/29-clarifygpt.md)

### Test inputs and mutation

- **Seed input:** a test input produced by the LLM from a few-shot prompt whose instruction asks for "complex, difficult, or corner-case" inputs. Seeds initialise a pool for mutation. [§3.1.1; Figure 3(a)](../pdfs/29-clarifygpt.md)
- **Type-aware mutation:** a fuzzing loop taken from prior work (cited as Liu et al. 2023b) that picks a random pool entry, applies one mutation matching its data type, and returns new inputs to the pool. [§3.1.2](../pdfs/29-clarifygpt.md)
- The mutation table gives: `int|float` returns x ± 1; `bool` a random boolean; `NoneType` returns None; `str` removes, repeats or replaces a substring; `List` removes, repeats, inserts or replaces an item; `Tuple` and `Set` mutate via the list form; `Dict` removes, updates or inserts a key/value pair. [Figure 2](../pdfs/29-clarifygpt.md)
### Prompts

- All four prompts share one layout: an instruction, few-shot demonstrations, and a query. [§3.1.1; §3.3; §3.4; §5.2](../pdfs/29-clarifygpt.md)
- **Reasoning-based question prompt:** the model is told to analyse the functionality of each inconsistent solution, compare the differences, decide which parts of the requirement are ambiguous, and then ask targeted questions. The authors relate this to Chain-of-Thought prompting, with the analysis as the intermediate reasoning. [§3.3; Figure 3(b)](../pdfs/29-clarifygpt.md)
- **Clarification:** one question paired with its answer. Clarifications are appended to the end of the original docstring to form the **refined requirement**, leaving the original text intact. [§3.4; Figure 3(d)](../pdfs/29-clarifygpt.md)

### Feedback sources and metric

- **ClarifyGPT (Human Feedback):** answers come from recruited participants. **ClarifyGPT (Simulated Feedback):** answers come from an LLM prompted to play the user. The two labels name different experiments with different coverage. [§5.1; §5.2; Table 3](../pdfs/29-clarifygpt.md)
- **Pass@k:** a problem counts as solved if any of `k` generated solutions passes all of its unit tests. The paper fixes `k = 1`. [§4.4](../pdfs/29-clarifygpt.md)
- Each approach is run three times and the average is reported. [§4.4](../pdfs/29-clarifygpt.md)
- "Improvement" percentages in the prose and tables are **relative** changes over the Default baseline's Pass@1, not percentage-point differences. Percentage-point values in this note are derived by subtraction. [Table 2 caption; Table 3 caption](../pdfs/29-clarifygpt.md)

## 3. Methodology and experimental setup

### Pipeline stages

- Stage 1, test-input generation: a few-shot prompt containing the requirement asks the LLM for seed inputs; type-aware mutation then expands the pool "until we attain the desired number of generated inputs". The number is not stated. [§3.1.1–3.1.2](../pdfs/29-clarifygpt.md)
- The stated reasons for combining the two steps are that an LLM can produce inputs satisfying semantic constraints (for example balanced parentheses) and that querying an LLM for large numbers of tests is slow and costly. [§3.1](../pdfs/29-clarifygpt.md)
- Stage 2, consistency check: `n` solutions are sampled, executed on the generated inputs, and compared. The value of `n` appears only as a symbol; no number is given anywhere in the text. [§1; §3.2](../pdfs/29-clarifygpt.md)
- If outputs are identical, "one of the sampled codes would be output as the final code solution"; Figure 1 labels this branch "Randomly Return One of the Candidate Codes". No question is asked. [§3.2; Figure 1](../pdfs/29-clarifygpt.md)
- If outputs differ, one solution is chosen at random from each output cluster, and these inconsistent solutions are placed in the question-generation prompt. [§3.2](../pdfs/29-clarifygpt.md)
- Stage 3, question generation: the query holds the requirement and the inconsistent solutions; the demonstration shows a per-solution analysis followed by numbered clarifying questions. [§3.3; Figure 3(b)](../pdfs/29-clarifygpt.md)
- Answer handling: answers are collected from a user (or simulator) and paired with their questions. The described pipeline has a single round of questions; no follow-up questions or re-check of the refined requirement are described. [§3.4; Figure 1](../pdfs/29-clarifygpt.md)
- Stage 4, enhanced generation: the refined requirement is passed to the same LLM with a prompt telling it to return only a correct, efficient Python function. The source describes one final solution and no second consistency check. [Abstract; §3.4; Figure 3(d)](../pdfs/29-clarifygpt.md)

### Models, prompts and configuration

- Base models are ChatGPT accessed as `gpt-3.5-turbo` and GPT-4 accessed as `gpt-4-turbo`, both through the OpenAI API. No dated snapshot identifier is given. [§4.2](../pdfs/29-clarifygpt.md)
- Sampling uses top-p 0.95 and frequency penalty 0. Temperature is 0 for all prompts except the sampling of code solutions, which uses 0.8. `max_tokens` is 800 for the question-asking prompt and 300 for other prompts, and outputs are truncated at five stop sequences (`\nclass`, `\ndef`, `\n#`, `\nif`, `\nprint`). [§4.6](../pdfs/29-clarifygpt.md)
- Demonstrations are hand-written from the first three problems of each benchmark, and each prompt carries three demonstrations; the authors attribute the limit to model input length. [§4.6](../pdfs/29-clarifygpt.md)
- No hardware, wall-clock time, token count or API cost is reported for any method. [§4.6; §5](../pdfs/29-clarifygpt.md)

### Benchmarks and baselines

- HumanEval has 164 hand-written Python problems with an average of 7.8 tests per problem and 67.7 prompt words on average (range 17–249). [§4.3; Table 1](../pdfs/29-clarifygpt.md)
- MBPP-sanitized has 427 problems, an average of 3.1 tests per problem (the prose says three test cases each) and 14.5 prompt words on average (range 7–47). [§4.3; Table 1](../pdfs/29-clarifygpt.md)
- HumanEval-ET and MBPP-ET keep the same problems and prompts and extend the tests, to averages of 107.5 and 101.7 tests per problem. They are a stricter grading of the same generation task, not additional independent problems. [§4.3; Table 1; §5.1 Setup](../pdfs/29-clarifygpt.md)
- Baselines are Default (the benchmark requirement given directly to the LLM), CoT with the same three demonstration seeds, and GPT-Engineer, an open-source tool that asks clarifying questions for every requirement, ambiguous or not. [§2.1; §4.5](../pdfs/29-clarifygpt.md)

### Human evaluation protocol (RQ1)

- The human study uses GPT-4 only, with cost of human participation given as the reason. [§4.1, RQ1](../pdfs/29-clarifygpt.md)
- ClarifyGPT is applied to the MBPP-sanitized requirements only. The resulting code is graded twice, with the MBPP-sanitized tests and with the MBPP-ET tests. [§5.1 Setup](../pdfs/29-clarifygpt.md)
- The consistency check flags 140 MBPP-sanitized problems as ambiguous, with an average of 2.85 clarifying questions per flagged problem. Derived: 140 of 427 is 32.8% of the benchmark under GPT-4. [§5.1 Setup; Table 1](../pdfs/29-clarifygpt.md)
- Ten participants took part: three Ph.D. students, two Master's students, two senior researchers and three industry developers, none a co-author. All have at least three years of Python experience and six have more than five. [§5.1 Setup](../pdfs/29-clarifygpt.md)
- Each participant received a task description, example questionnaires and a training exercise, then answered questions for 42 problems; each problem was answered by three participants. Derived: 10 × 42 = 420 = 140 × 3, so the assignment is internally consistent. [§5.1 Setup](../pdfs/29-clarifygpt.md)
- Each questionnaire contains the requirement, **the unit test cases with expected input-output examples**, and the generated questions. Participants therefore answer with the ground-truth behaviour in front of them. [§5.1 Setup](../pdfs/29-clarifygpt.md)
- The participants' answers are fed back into ClarifyGPT to generate final code, and the reported Pass@1 is the average over the three participants' answer sets. [§5.1 Setup](../pdfs/29-clarifygpt.md)

### Simulated-feedback protocol (RQ2) and demonstration study (RQ3)

- The simulator is an LLM given the clarifying questions and the problem's ground-truth test cases; the authors' stated rationale is that the tests encode the functionality the user wants. [§5.2 Setup](../pdfs/29-clarifygpt.md)
- The simulation prompt's instruction is to answer the clarifying questions "using the information provided in the given requirement and tests" and to reply only with the answers. Its query shows the requirement, assert-style test cases with expected outputs, and the questions. [Figure 3(c)](../pdfs/29-clarifygpt.md)
- The source does not say which model acts as the simulator, nor whether the original or the extended tests are supplied for the -ET benchmarks. [§5.2 Setup; Figure 3(c)](../pdfs/29-clarifygpt.md)
- RQ2 applies the three baselines and ClarifyGPT to both models on all four benchmarks and compares Pass@1. The source says only that "all baselines adopted the same experimental setup"; it does not describe who answers GPT-Engineer's questions. [§5.2 Setup](../pdfs/29-clarifygpt.md)
- RQ3 varies the number of demonstrations from zero to three for ClarifyGPT with simulated feedback, on both models and all four benchmarks, averaging three runs. [§5.3 Setup](../pdfs/29-clarifygpt.md)

## 4. Results and evidence

### Human feedback (GPT-4, MBPP-sanitized requirements)

- On MBPP-sanitized, Pass@1 is 70.96% for Default, 72.68% for CoT, 73.77% for GPT-Engineer and 80.80% for ClarifyGPT with human answers. On MBPP-ET the corresponding values are 51.52%, 53.79%, 54.96% and 60.19%. [Table 2](../pdfs/29-clarifygpt.md)
- The source reports relative improvements over Default of 13.87% (MBPP-sanitized) and 16.83% (MBPP-ET), and 15.35% "on average". [Table 2; §5.1](../pdfs/29-clarifygpt.md)
- Derived: the absolute gains over Default are 9.84 and 8.67 percentage points. A 9.84-point gain on 427 problems corresponds to about 42 problems, averaged over the three answer sets. [Table 2; Table 1](../pdfs/29-clarifygpt.md)
- Against the best baseline in each column (GPT-Engineer), the source reports relative gains of 9.53% and 9.52%. Derived: 7.03 and 5.23 percentage points. [§5.1 Results; Table 2](../pdfs/29-clarifygpt.md)
- The source does not report Pass@1 separately for the 140 flagged problems and the 287 unflagged ones, so the table does not show how much of the gain comes from clarified problems. [§5.1; Table 2](../pdfs/29-clarifygpt.md)
- The baseline rows of Table 2 are numerically identical to the GPT-4 baseline rows of Table 3 in the two MBPP columns, including GPT-Engineer. Only the ClarifyGPT row differs between the human and simulated conditions. [Table 2; Table 3](../pdfs/29-clarifygpt.md)

### Simulated feedback (both models, four benchmarks)

- ChatGPT, Default versus ClarifyGPT (Simulated): HumanEval 64.63 → 74.39, HumanEval-ET 57.32 → 64.84, MBPP-sanitized 65.57 → 74.08, MBPP-ET 46.68 → 55.58, four-benchmark average 58.55 → 67.22 (Pass@1 %, three-run mean). [Table 3](../pdfs/29-clarifygpt.md)
- GPT-4, Default versus ClarifyGPT (Simulated): HumanEval 78.86 → 87.80, HumanEval-ET 70.73 → 78.05, MBPP-sanitized 70.96 → 78.69, MBPP-ET 51.52 → 58.47, average 68.02 → 75.75. [Table 3](../pdfs/29-clarifygpt.md)
- Reported relative improvements over Default are 15.10%, 13.12%, 12.98% and 19.07% for ChatGPT (15.07% average) and 11.34%, 10.35%, 10.89% and 13.49% for GPT-4 (11.52% average). Derived: the absolute gains are 9.76, 7.52, 8.51 and 8.90 percentage points for ChatGPT and 8.94, 7.32, 7.73 and 6.95 for GPT-4. [Table 3; §5.2 Results](../pdfs/29-clarifygpt.md)
- CoT and GPT-Engineer average 61.21% and 61.33% with ChatGPT and 69.78% and 69.94% with GPT-4, between Default and ClarifyGPT in every benchmark column. [Table 3](../pdfs/29-clarifygpt.md)
- The prose reports "an average improvement of 11.45%, 8.65%, 6.95%, and 8.56% across the four benchmarks" over GPT-Engineer without saying what is averaged. [§5.2 Results](../pdfs/29-clarifygpt.md)
- For GPT-4 on the MBPP-sanitized requirements, human answers give 80.80% and 60.19% while simulated answers give 78.69% and 58.47%. Derived: human feedback is higher by 2.11 and 1.72 percentage points. This is the paper's only human-versus-simulated comparison (one model, one requirement set, two test suites); HumanEval and ChatGPT have no human condition. [Table 3; §5.1 Setup](../pdfs/29-clarifygpt.md)
- The number of problems flagged ambiguous and the number of questions asked are not reported for HumanEval or for ChatGPT. The 140-problem and 2.85-question figures belong to GPT-4 on MBPP-sanitized. [§5.1 Setup; §5.2](../pdfs/29-clarifygpt.md)

### Number of demonstrations (simulated feedback)

- Four-benchmark average Pass@1 for zero-, one-, two- and three-shot prompts is 59.77%, 64.07%, 65.84% and 67.22% with ChatGPT (Default 58.55%) and 68.59%, 72.14%, 73.80% and 75.75% with GPT-4 (Default 68.02%). [Table 4](../pdfs/29-clarifygpt.md)
- Zero-shot relative gains over Default are 1.4–2.8% per benchmark for ChatGPT and 0.0–1.6% for GPT-4; GPT-4 zero-shot on HumanEval-ET equals Default at 70.73%. [Table 4](../pdfs/29-clarifygpt.md)
- Derived: the zero-shot average gains are 1.22 percentage points (ChatGPT) and 0.57 (GPT-4), against 8.67 and 7.73 for three-shot. Most of the reported benefit depends on the hand-written demonstrations. [Table 4](../pdfs/29-clarifygpt.md)
- The one-shot setting gives average relative gains of 9.4% (ChatGPT) and 6.2% (GPT-4), which the authors describe as close to three-shot performance. [Table 4; §5.3 Results](../pdfs/29-clarifygpt.md)
- The demonstration count is described as varying "in the prompt"; the text does not say whether all four prompts (test-input, question, simulation, final generation) are reduced together. [§5.3 Setup](../pdfs/29-clarifygpt.md)

### Case study and cost-related information

- Case I (MBPP-sanitized problem 71, `comb_sort`, ChatGPT): Default fails the tests; GPT-Engineer asks five questions and passes; ClarifyGPT asks one question about ascending or descending order and passes. [§6.1; Figure 4](../pdfs/29-clarifygpt.md)
- The authors call some of GPT-Engineer's five questions uninformative and name two as examples (sorting-algorithm constraints and preferred programming language), since the function name and the Python signature already answer them. [§6.1; Figure 4](../pdfs/29-clarifygpt.md)
- Case II (HumanEval problem 1, `separate_paren_groups`, ChatGPT): Default passes; GPT-Engineer asks three questions and then fails; ClarifyGPT asks no question and passes. [§6.1; Figure 4](../pdfs/29-clarifygpt.md)
- The two cases are selected examples described as representative; they illustrate the mechanism and are not a measured rate of unnecessary questions. [§6.1](../pdfs/29-clarifygpt.md)
- Cost appears only qualitatively: extra questions increase input and output tokens and "operational expenses", and bulk LLM test generation is called slow and costly. The only quantity related to user burden is the 2.85-question average. [§3.1; §5.1 Setup; §6.1](../pdfs/29-clarifygpt.md)
- Derived from the pipeline description: an unambiguous requirement needs a seed-input call and `n` sampled solutions; an ambiguous one adds a question-generation call and a final-generation call. With `n` unreported, total calls per problem cannot be computed. [§3.1–3.4](../pdfs/29-clarifygpt.md)

## 5. Authors' conclusions

- The authors conclude that detecting ambiguous requirements and asking targeted questions improves code generation, citing gains for both models and all four benchmarks. [§5.2 Results; §7](../pdfs/29-clarifygpt.md)
- They attribute the advantage over GPT-Engineer to asking only when needed and to deriving questions from inconsistent solutions, arguing that asking about clear requirements burdens users and can harm output. [§5.2 Results; §6.1](../pdfs/29-clarifygpt.md)
- They state the consistency check identifies ambiguous requirements "effectively" and "without any supervised training". This is an inference from end-task gains; detection quality is not measured directly. [§5.2 Results](../pdfs/29-clarifygpt.md)
- They conclude the simulation is a good proxy for human feedback because the two conditions give close Pass@1 values, while also observing that simulated answers may sometimes not match user intent. [§5.2 Results; §6.3](../pdfs/29-clarifygpt.md)
- They call ClarifyGPT robust to the number of demonstrations and suggest one-shot use as a trade-off between effectiveness and efficiency. [§5.3 Results](../pdfs/29-clarifygpt.md)
- They argue the approach improves interpretability, because users can relate an added clarification to a change in the generated code, and that it spares users from reading generated code to find requirement gaps. [§6.2, Benefits](../pdfs/29-clarifygpt.md)
- They believe the framework generalises and can support practical development. The evidence base for this is two models and four benchmarks built from two problem sets. [§6.3; §7](../pdfs/29-clarifygpt.md)

## 6. Limitations and open questions

### Limitations and threats stated by the source

- The framework needs a model that can follow instructions and formulate questions; models without instruction tuning (InCoder and CodeGen are named) are excluded. [§6.2, Limitations](../pdfs/29-clarifygpt.md)
- Because the check requires generated test inputs and comparable outputs, the authors say ClarifyGPT "is not suitable for generating code with complex input (e.g., image or file)". They add that code which returns no output values, with deep-learning programs as the example, may also be subject to limitations. [§6.2, Limitations](../pdfs/29-clarifygpt.md)
- Threat 1, data leakage: the benchmarks may be in the models' training data, which could bias the assessment. The stated mitigation is the choice of hand-written or hand-verified, widely used benchmarks. [§6.3](../pdfs/29-clarifygpt.md)
- Threat 2, user simulation: low-fidelity simulated answers could mislead the evaluation. The stated mitigation is giving the simulator the questions and ground-truth tests, supported by the closeness of human and simulated results. [§6.3](../pdfs/29-clarifygpt.md)
- Threat 3, generalisability: addressed by using two chat models, four datasets and three-run averages. [§6.3](../pdfs/29-clarifygpt.md)

### Analyst caveats and missing evidence

- **Answerer has the oracle:** both human participants and the simulator see the problem's ground-truth tests with expected outputs, and the source does not say that any grading tests are withheld from them. The numbers measure the pipeline when the answerer knows the intended behaviour, not when a user has unstated or uncertain intent. [Basis: §5.1 Setup; §5.2 Setup; Figure 3(c)](../pdfs/29-clarifygpt.md)
- **Leakage mitigation is selection only:** choosing hand-written or hand-verified public benchmarks does not show that the models were not exposed to them, and no contamination test is reported. [Basis: §6.3](../pdfs/29-clarifygpt.md)
- **Information-channel confound:** Default and CoT receive only the requirement, while ClarifyGPT's refined prompt contains answers derived from the tests. There is no control that gives the baseline the same test information without questions, so the gain cannot be split between "asking" and "extra information". [Basis: §4.5; §5.2 Setup](../pdfs/29-clarifygpt.md)
- **Simulator fidelity is weakly tested:** the proxy claim rests on aggregate Pass@1 for GPT-4 on one requirement set. No answer-level agreement between humans and simulator, and no agreement among the three participants per problem, is reported. [Basis: §5.1 Setup; Table 3; §6.3](../pdfs/29-clarifygpt.md)
- **Detector validity is unmeasured:** sampled solutions can agree on a shared wrong reading, and can disagree because of bugs or unspecified corner cases instead of requirement ambiguity. No false-positive or false-negative rate is given. [Basis: §3.2](../pdfs/29-clarifygpt.md)
- **Unreported parameters:** `n`, the number of generated test inputs, the number of seeds, execution time limits and the handling of exceptions are absent, so the detector's sensitivity cannot be reproduced from the text. The models are named `gpt-4-turbo` and `gpt-3.5-turbo` without snapshot dates, so exact reproduction of the Pass@1 values is not assured either. The linked repository was not inspected for this note. [Basis: §3.1.2; §3.2; §4.2; §4.6](../pdfs/29-clarifygpt.md)
- **Unflagged path differs from Default:** an unflagged requirement returns a random solution sampled at temperature 0.8, whereas the configuration text implies other prompts run at temperature 0. Without a per-subset breakdown, part of any difference from Default could come from this change. [Basis: §3.2; §4.6](../pdfs/29-clarifygpt.md)
- **Demonstration overlap:** demonstrations come from the first three problems of each benchmark, and Table 1's problem counts are the full sets. The source does not say whether those problems are excluded from scoring. [Basis: §4.6; Table 1](../pdfs/29-clarifygpt.md)
- **No uncertainty estimates:** the words "significantly" and "substantially" are used without statistical tests; three-run averages are given without spread. [Basis: §4.4; §5.2 Results](../pdfs/29-clarifygpt.md)
- **Independence of evidence:** four benchmark columns rest on two requirement sets, so the "four benchmarks" are not four independent replications. [Basis: §4.3; Table 1](../pdfs/29-clarifygpt.md)- **Unexplained coincidence:** Table 4's ChatGPT one-shot values for MBPP-sanitized and MBPP-ET (70.96, 51.52) equal the GPT-4 Default values for the same columns. The source does not comment; this note does not assume an error. [Basis: Table 4](../pdfs/29-clarifygpt.md)

### Source inconsistencies

- **Source inconsistency:** §1 says the human study "improves the performance (Pass@1) of ChatGPT on MBPP-ET from 51.52% to 60.19%". Table 2, §4.1 and §5.1 attribute these values to GPT-4, the only model in the human study. The table-specific attribution is GPT-4. [§1; §4.1, RQ1; §5.1; Table 2](../pdfs/29-clarifygpt.md)
- **Source inconsistency:** §5.1 says ClarifyGPT (Human Feedback) "achieves the highest performance on all four benchmarks", but the human study and Table 2 cover two benchmarks. [§5.1 Results; Table 2](../pdfs/29-clarifygpt.md)
- **Source inconsistency:** §5.3 says ClarifyGPT "consistently outperforms the Default baseline across two LLMs and four benchmarks" for zero to three demonstrations, but Table 4 shows GPT-4 zero-shot HumanEval-ET at 70.73%, equal to Default, marked "0.0% -". [§5.3; Table 4](../pdfs/29-clarifygpt.md)
- **Source inconsistency (minor arithmetic):** §5.2 gives the average of the two models' relative improvements as 13.27%. Derived: the mean of the reported 11.52% and 15.07% is 13.30% (13.29% from unrounded table values). [§5.2, Answering RQ2; Table 3](../pdfs/29-clarifygpt.md)
- **Source inconsistency (minor arithmetic):** the GPT-Engineer comparison figures are 11.45%, 8.65%, 6.95% and 8.56%. Derived two-model means of the per-benchmark relative gains from Table 3 are 11.52%, 8.64%, 6.95% and 8.55%; the HumanEval value differs by 0.07. [§5.2 Results; Table 3](../pdfs/29-clarifygpt.md)
- **Source ambiguity:** the "Average" relative improvements (15.35%, 15.07%, 11.52%) equal the mean of per-benchmark relative gains. Derived: the relative change of the Average column itself is 15.12%, 14.81% and 11.37%. Either definition should be named when the number is reused. [Table 2; Table 3](../pdfs/29-clarifygpt.md)
- **Source ambiguity:** in Table 3 the Human Feedback row's Average (70.50) is over two benchmarks, in a column where all other rows average four. It is not comparable with 75.75. [Table 3](../pdfs/29-clarifygpt.md)
- **Internal tension:** §5.2 says the simulation "may generate user responses that do not fulfill the users' intentions", while §6.3 says the closeness of results is "proving" the simulation is a good proxy. [§5.2 Results; §6.3](../pdfs/29-clarifygpt.md)

## 7. Relevance to this project — interpretation

- **Proposal:** treat every item in this section as an untested design idea for the kernel-autoresearch pipeline. The source evidence comes from small Python functions, not GPU kernels, and nothing here describes current repository behaviour. [Basis: §4.3; §6.2, Limitations](../pdfs/29-clarifygpt.md)
- **Proposal:** in Stage 1, consider a behavioural trigger for clarification: draft several independent interpretations of the user's request (for example small reference implementations), run them on generated inputs, and ask the user only when their outputs separate into more than one cluster. [Basis: §3.2; Figure 1](../pdfs/29-clarifygpt.md)
- **Proposal:** do not treat agreement among drafts as proof that context is sufficient. Keep an explicit checklist for items that sampled agreement cannot reveal in kernel work, such as dtype, shape ranges, numerical tolerance, memory layout and target hardware. The checklist items are this note's suggestion; the source only supports the point that the check sees behavioural disagreement and nothing else. [Basis: §3.2; §6.2, Limitations](../pdfs/29-clarifygpt.md)
- **Proposal:** generate Stage 1 questions from concrete divergent artefacts. Show the model the disagreeing drafts, have it state how they differ, and only then write questions, so that each question maps to an observed behavioural difference. [Basis: §3.3; Figure 3(b)](../pdfs/29-clarifygpt.md)
- **Proposal:** store each clarification as a question–answer pair appended to the unchanged original request, and have both the natural-language spec and the Lean spec cite those pairs. This keeps a record of which spec clause came from the user and which from inference. [Basis: §3.4; Figure 3(d)](../pdfs/29-clarifygpt.md)
- **Proposal:** adapt the comparison step before using it on kernels. Exact output equality would flag floating-point reordering as "ambiguity", and the source itself excludes complex inputs and code without return values, so tensor-aware input generation and tolerance-based clustering would be needed and would have to be validated separately. [Basis: §3.2; Figure 2; §6.2, Limitations](../pdfs/29-clarifygpt.md)
- **Proposal:** log question burden as a Stage 1 metric: the fraction of tasks that trigger questions and the number of questions per triggered task, alongside downstream outcome. The source's only such figures are 140 of 427 problems and 2.85 questions per flagged problem. [Basis: §5.1 Setup](../pdfs/29-clarifygpt.md)
- **Proposal:** avoid an always-ask policy as the default and test it as a baseline instead. The source's evidence against always asking is one comparison system and two hand-picked cases, so the project should measure it on its own tasks. [Basis: §5.2 Results; §6.1; Figure 4](../pdfs/29-clarifygpt.md)
- **Proposal:** if Stage 1 is evaluated with a simulated user, record exactly what the simulator can see and run two conditions: a simulator holding the oracle (an upper bound, as in this source) and a simulator holding only what a real requester would know. Add an information-matched control, where the non-asking baseline receives the same extra facts without a question step, to separate the value of asking from the value of the information. [Basis: §4.5; §5.2 Setup; Figure 3(c); §6.3](../pdfs/29-clarifygpt.md)
- **Proposal:** treat few-shot examples in the question-generation prompt as a design variable to test. In this source the zero-shot variant gained 0.6–1.2 percentage points on average and the three-shot variant 7.7–8.7 (derived from Table 4), on different tasks from the project's. [Basis: Table 4; §5.3](../pdfs/29-clarifygpt.md)
- **Proposal:** in Stage 2, when a reference implementation is searched for and several candidates are found, the same output-clustering idea could flag that the candidates implement different functions and route the choice back to the user, instead of leaving it to an LLM judge alone. [Basis: §3.2](../pdfs/29-clarifygpt.md)

## 8. Retrieval notes

- Search aliases: ClarifyGPT, intention clarification, requirement clarification, clarifying questions for code generation, ambiguity detection, code consistency check, type-aware mutation, user simulation, simulated user feedback, GPT-Engineer baseline, MBPP-sanitized, MBPP-ET, HumanEval-ET.
- Questions answered: How does ClarifyGPT decide a requirement is ambiguous? What sample counts or thresholds does it state? How are questions generated and answers folded back into the prompt? What did the human study measure, with how many participants and problems? What does the simulated user have access to? How large are the Pass@1 gains, in relative and absolute terms? What limits does the source state for non-trivial inputs?
- Related local analyses, as navigation aids and not controlled comparisons: [Self-Refine](18-self-refine.md) and [Reflexion](19-reflexion.md) for feedback loops that act after generation instead of before it; [ReAct](11-react.md) for general interleaving of reasoning and actions; [From Generation to Judgment](13-generation-to-judgment.md) for LLMs used as evaluators, relevant to the LLM-as-user simulation here.
- Notes written in parallel and not read for this analysis: [AuxSpec](28-auxspec.md) and [Verina](26-verina.md) may be relevant to the specification-writing step that follows clarification in the project's Stage 1.

## 9. Source coverage and verification notes

- Read the complete local Markdown: abstract, §§1–7 including all subsections, Tables 1–4, and the reference list. The source has no appendix.
- Opened and inspected all four figure images. Figure 1 (pipeline and the "#clusters<2?" decision), Figure 2 (mutation table) and Figure 3 (four prompt templates, including the simulation prompt's instruction and query fields) are described from the images, since their content is not in the Markdown text.
- Figure 4 was inspected for the question lists, question counts and pass/fail stamps. The comparison operators in its small code snippets were not legible enough to verify, so this note makes no claim about the code shown there.
- Cross-checked Tables 2–4 and the surrounding prose against the text layer extracted from the 21-page PDF (tables on PDF pp. 12, 14 and 15); the values match the Markdown. The PDF pages were not rendered visually; figures were inspected only through the image assets. The PDF text also confirms that `n` and the number of test inputs are not stated, and that the §1 "ChatGPT on MBPP-ET" wording is in the original and is not a conversion error.
- Recomputed table averages, relative improvements and percentage-point differences. Results marked "Derived" are this note's arithmetic; discrepancies are listed in §6 and were not corrected.
- Conversion notes: the red colouring that marks relative improvements in Tables 2–4 is lost in the Markdown; one Table 4 cell reads "58.4713.5% ↑" with a missing space (also present in the PDF text layer); Table 3 uses a backslash for cells with no human-feedback result; Figure 2 is a table preserved only as an image.
- Not read or verified: the public repository cited as the dataset and code release, the GPT-Engineer issue cited in a footnote, and all cited prior work. Statements about those rely on the paper's descriptions.
- No experiments were reproduced. No model was queried, no benchmark was run, and no participant data was examined.
