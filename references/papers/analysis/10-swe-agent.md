---
paper_id: "10"
title: "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering"
source_version: "2405.15793v3"
source_markdown: "../pdfs/10-swe-agent.md"
source_pdf: "../pdfs/10-swe-agent.pdf"
source_url: "https://arxiv.org/abs/2405.15793"
analysis_date: "2026-11-20"
analysis_status: "reviewed"
topics: ["agent-computer-interface", "software-engineering", "tool-design", "context-management", "execution-feedback", "evaluation", "error-recovery"]
summary: "Studies how search, file viewing, editing guardrails, and context management affect fixed-model software-engineering agents on SWE-bench and HumanEvalFix."
---

# SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering

## 1. Paper identity and scope

- **Research question:** can an interface designed for language-model users improve autonomous software engineering without changing the model's weights? The unit of design is the entire agent–computer interaction, not only a prompt or a tool API. [§§1–2](../pdfs/10-swe-agent.md)
- **Contribution:** the agent-computer interface (ACI) concept, an implemented SWE-agent interface, empirical ablations, and behavioral analysis of complete issue-resolution trajectories. [§1; §7](../pdfs/10-swe-agent.md)
- **Task scope:** resolving GitHub issues in existing Python repositories, plus short-form debugging in Python, JavaScript, and Java. This is not a GPU-kernel optimization study. [§4; Appendix B.7](../pdfs/10-swe-agent.md)
- **Frozen evidence:** this note concerns arXiv v3 and its historical GPT-4 Turbo/Claude 3 Opus experiments; it does not describe a current SWE-agent release or contemporary benchmark leaderboard. [§4; local source metadata](../pdfs/10-swe-agent.md)

## 2. Concepts and terminology

### ACI as a complete interaction contract

- **ACI:** the abstraction between model and computer, defining available actions, command documentation, observation formatting, state presentation, and history management. [§2](../pdfs/10-swe-agent.md)
- **LM as a distinct end user:** an interface convenient for a human may be inefficient for a model because extra information consumes context and computation, and many simple operations require costly sequential model calls. [§2](../pdfs/10-swe-agent.md)
- **Fixed-model design:** interface changes aim to elicit an existing model's capabilities; the main system is not fine-tuned on SWE-bench. [§2](../pdfs/10-swe-agent.md)
- **Simple actions:** a small number of understandable arguments and concise documentation reduce the burden of choosing among many shell options. [§2, principle 1](../pdfs/10-swe-agent.md)
- **Compact actions:** useful higher-order operations should take few turns; an interface made of excessively granular commands can be simple locally but inefficient globally. [§2, principle 2](../pdfs/10-swe-agent.md)
- **Informative, concise feedback:** expose relevant state changes without flooding the context; an edit should immediately show the updated region. [§2, principle 3](../pdfs/10-swe-agent.md)
- **Guardrails:** prevent common mistakes from propagating and give actionable recovery feedback. They are error-management mechanisms, not proofs of functional correctness. [§2, principle 4; §3](../pdfs/10-swe-agent.md)
- **ReAct-style control loop:** each step generates a thought and a command, executes the command, and conditions the next step on the observation. Ordinary shell commands remain available alongside custom commands. [§3](../pdfs/10-swe-agent.md)

### Search, viewing, and editing

- **Repository localization:** identify relevant files, then relevant lines; issue descriptions, filenames, symbols, and stack traces provide starting signals. [§3; Appendix A.1](../pdfs/10-swe-agent.md)
- **Search commands:** `find_file` finds filenames; `search_dir` searches content across a directory; `search_file` searches within a file, defaulting to the current open file. [Table 4](../pdfs/10-swe-agent.md)
- **Summarized search:** display bounded results together rather than requiring repeated next/previous navigation. More than 50 matches triggers a request to narrow the query instead of dumping all matches. [§3; Appendix A.1](../pdfs/10-swe-agent.md)
- **Search tradeoff:** suppressing broad results protects context but may require an additional model call to formulate a narrower query. [Appendix A.1, Search & navigation](../pdfs/10-swe-agent.md)
- **Stateful file viewer:** `open`, `goto`, `scroll_up`, and `scroll_down` maintain the current file and location; default view length is 100 lines. [§3; Table 4](../pdfs/10-swe-agent.md)
- **Location grounding:** display full path, total line count, per-line numbers, and omitted-line counts so the agent knows both its current view and what it has not inspected. [§3; Appendix A.1](../pdfs/10-swe-agent.md)
- **Integrated edit:** replace an inclusive line range in the open file using start line, end line, and replacement text; refresh the view after applying the change. [§3; Figure 3; Table 4](../pdfs/10-swe-agent.md)
- **Why shell-only editing is hard:** whole-file rewriting is expensive and risks collateral changes; multi-line `sed` operations require fragile quoting, line arithmetic, and extra inspection commands. [§5.1; Appendix A.1](../pdfs/10-swe-agent.md)
- **Linting guardrail:** apply the tentative edit, run selected `flake8` checks, retain it if valid, otherwise revert it and expose error messages plus proposed and original snippets. [Appendix A.1, File editor](../pdfs/10-swe-agent.md)
- **Selected lint categories:** undefined names/exports, duplicate arguments, indentation errors, syntax errors, and inability to read the source file; the documented codes include F821, F822, F831, E111–E113, E999, and E902. [Appendix A.1](../pdfs/10-swe-agent.md)
- **Why show three pieces of feedback:** the error identifies the failure, the proposed edit helps avoid repeating it, and the original snippet prevents subsequent edits from targeting a reverted state. [Appendix A.1; Figure 11](../pdfs/10-swe-agent.md)
- **Guardrail ordering constraint:** removing an argument before removing its uses can be rejected; some changes must be reordered or combined into one edit. The paper explicitly acknowledges this flexibility-versus-protection tradeoff. [Appendix A.1; Figure 11](../pdfs/10-swe-agent.md)
- **Creation and submission:** `create` opens a new file; `submit` collects changes into a patch and ends the episode. The final patch is generated from the workspace state rather than requiring the model to produce a perfectly formatted patch directly. [Table 4; §4; Appendix A.2](../pdfs/10-swe-agent.md)

### Context, implementation, and configuration

- **Recent-observation retention:** keep the most recent five observations in full and collapse earlier observations into one-line omission notices. Thoughts and action history remain available. [§3; Appendix C, Collapsed Observation Template](../pdfs/10-swe-agent.md)
- **Stale-state reduction:** removing old observations reduces both tokens and contradictory/outdated views of files that have since changed. [Appendix C](../pdfs/10-swe-agent.md)
- **Silent-command feedback:** explicitly state when a command succeeded with no output; silence otherwise encourages redundant checks. [§3; Appendix C, Next Step Template](../pdfs/10-swe-agent.md)
- **Response-format recovery:** malformed responses trigger a retry; after recovery, repeated error messages are removed except the first. [§3](../pdfs/10-swe-agent.md)
- **Prompt sequence:** system instructions, optional successful demonstration, and issue description precede turn-specific observations; the system prompt and task context are not treated like disposable old command output. [Appendix C; Figures 26–32](../pdfs/10-swe-agent.md)
- **Demonstration representation:** a demonstration can be presented as separate historical turns or a single explicitly labeled message; the authors note model-dependent tradeoffs in these representations. [Appendix C, Demonstration Template](../pdfs/10-swe-agent.md)
- **Implementation components:** environment, agent/ACI logic, and logging; Docker contains execution, while logs preserve trajectories and final patches. [Appendix A.2](../pdfs/10-swe-agent.md)
- **Configuration dimensions:** prompt templates, command implementations, parsing/history control flow, and environment state variables. This makes ACI variants experimentally configurable. [Appendix A.3](../pdfs/10-swe-agent.md)
- **Command outputs:** custom Python/Bash commands communicate effects through standard output; documentation is extracted from command definitions, and state variables coordinate viewer/editor behavior. [Appendix A.3](../pdfs/10-swe-agent.md)

## 3. Methodology and experimental setup

- **Primary benchmark:** 2,294 SWE-bench test instances from 12 Python repositories; SWE-bench Lite contains 300 self-contained functional bug-fix tasks. [§4; Table 12](../pdfs/10-swe-agent.md)
- **Secondary benchmark:** HumanEvalFix has 164 problems per language; an instance starts with a buggy code file and available example tests, emphasizing editing rather than repository navigation. [Appendix B.7; Table 12](../pdfs/10-swe-agent.md)
- **Models:** GPT-4 Turbo `gpt-4-1106-preview` and Claude 3 Opus `claude-3-opus-20240229`; their reported context windows are 128k and 200k, with a warning that different tokenizers make token counts non-equivalent. [§4](../pdfs/10-swe-agent.md)
- **RAG baseline:** BM25 retrieves files from the issue text, after which a non-interactive model produces a patch. It does not receive iterative execution feedback. [§4](../pdfs/10-swe-agent.md)
- **Shell-only baseline:** an interactive Linux shell agent adapted from InterCode; it has execution feedback but lacks SWE-agent's custom ACI. Results distinguish with-demonstration and without-demonstration variants. [§4; Table 1](../pdfs/10-swe-agent.md)
- **Resolve/pass@1 metric:** proportion of tasks passing the benchmark's tests after applying the generated patch. This is test-suite success, not equivalence to a human patch or proof over all behaviors. [§4; Appendix D](../pdfs/10-swe-agent.md)
- **Cost metric:** §4 defines average API inference cost over successfully resolved instances; this differs from total campaign cost divided by successes and should not be silently interpreted as such. [§4](../pdfs/10-swe-agent.md)
- **Per-instance budget:** $4 of model inference; if exhausted, existing edits are automatically submitted. Historical API prices are part of the setting. [§4](../pdfs/10-swe-agent.md)
- **Design process:** manual inspection of selected development examples followed by a hyperparameter sweep, rather than automated optimization of the entire ACI. [§4; Appendix E.3](../pdfs/10-swe-agent.md)
- **Sweep:** 37 randomly selected development instances; vary temperature 0/0.2, viewer size 100/200, and full versus last-five observation history. Table 5 reports means over five samples. [Appendix B.1; Table 5](../pdfs/10-swe-agent.md)
- **Chosen setting:** temperature 0, 100-line window, last-five observations for both models; this gives GPT-4 Turbo 15.1% on the development sweep and ties for Claude's best 8.1%. These are development results, not the main test scores. [Appendix B.1; Table 5](../pdfs/10-swe-agent.md)
- **Ablation scope:** interface-component changes are evaluated on SWE-bench Lite using GPT-4 Turbo; they should not be treated as universal settings for all models and domains. [§5.1; Table 3](../pdfs/10-swe-agent.md)
- **Failure classifier:** GPT-4o `gpt-4o-2024-05-13` labels 248 unresolved Lite trajectories using trajectory, proposed patch, and gold patch; a 15-instance human-labeled sample checks agreement. [Appendix B.4](../pdfs/10-swe-agent.md)
- **Repeated-run study:** six GPT-4 Turbo runs on Lite measure both average single-run variation and pass@k across retries. [Appendix B.5; Table 10](../pdfs/10-swe-agent.md)

## 4. Results and evidence

### Main task results and ablations

- **Full SWE-bench:** GPT-4 Turbo SWE-agent resolves 12.47% (286/2,294), compared with GPT-4 Turbo RAG 1.31%; Claude 3 Opus SWE-agent resolves 10.46%, compared with Claude RAG 3.79%. [§5; Table 1](../pdfs/10-swe-agent.md)
- **SWE-bench Lite:** GPT-4 Turbo SWE-agent reaches 18.00% (54/300); shell-only with demonstration is 11.00%, shell-only without demonstration 7.33%, and GPT-4 Turbo RAG 2.67%. [Table 1](../pdfs/10-swe-agent.md)
- **Baseline-dependent improvement:** 18% versus 11% is +7 percentage points, approximately +64% relative; 18% versus 7.33% is +10.67 points. The introduction's roughly +10.7-point statement corresponds to the shell baseline without a demonstration. [§1; §5; Table 1](../pdfs/10-swe-agent.md)
- **Claude Lite:** SWE-agent 13.00% versus RAG 4.33%; transfer to a second model is demonstrated, although the interface was primarily developed with GPT-4 Turbo. [Table 1; §1](../pdfs/10-swe-agent.md)
- **Historical resolved-instance costs:** Table 1 gives full/Lite GPT-4 Turbo SWE-agent $1.59/$1.67 and Claude SWE-agent $2.59/$2.18; GPT-4 Turbo shell-only Lite is $1.46. [§4; Table 1](../pdfs/10-swe-agent.md)
- **HumanEvalFix:** Table 2 reports 87.7% Python, 89.7% JavaScript, and 87.9% Java for GPT-4 Turbo SWE-agent. The baseline GPT-4 row reports 47.0%, 48.2%, and 50.0%; baseline numbers are imported from prior work, not all re-run in the same interaction regime. [Table 2; Appendix B.7](../pdfs/10-swe-agent.md)
- **Editing:** removing the custom edit command reduces Lite resolution from 18.0% to 10.3%; keeping edit but removing linting gives 15.0%. [Table 3; §5.1](../pdfs/10-swe-agent.md)
- **Search:** summarized search gives 18.0%, no specialized search 15.7%, and iterative next/previous search 12.0%. More tools or more human-like navigation do not necessarily improve this agent. [Table 3; §5.1](../pdfs/10-swe-agent.md)
- **Mechanism behind iterative-search failures:** agents tend to inspect every result sequentially, consuming budget/context instead of refining the query or selecting promising matches. [§5.1; Figure 5](../pdfs/10-swe-agent.md)
- **Viewer size:** 100 lines gives 18.0%, 30 lines 14.3%, and full-file display 12.7%; both insufficient and excessive local context hurt in this setup. [Table 3](../pdfs/10-swe-agent.md)
- **History and demonstrations:** full observation history gives 15.0% versus 18.0% with last-five; removing the demonstration gives 16.3%. [Table 3](../pdfs/10-swe-agent.md)
- **Repeated runs:** six Lite runs range from 17.33% to 18.67%, with reported mean 17.94 and standard deviation 0.49; stable aggregate scores do not imply the same tasks are always solved. [Appendix B.5; Table 10](../pdfs/10-swe-agent.md)
- **Retry coverage:** pass@1 through pass@6 are 17.94%, 23.89%, 27.35%, 29.67%, 31.33%, and 32.67%. These are multi-attempt success measures, not an automatic mechanism for choosing the correct patch without a verifier. [Table 10](../pdfs/10-swe-agent.md)
- **Repository heterogeneity:** GPT-4 Turbo Lite rates range from zero on Flask/xarray to 33.33% on requests; several repositories have only 3–6 cases, so individual percentages have small denominators. [Table 6](../pdfs/10-swe-agent.md)
- **Issue age:** Table 7 shows no clear monotonic relationship between creation year and success; this is an observational contamination check, not proof that the models never saw benchmark material. [Appendix B.2; Table 7](../pdfs/10-swe-agent.md)

### Behavioral evidence and failure patterns

- **Early phase:** successful trajectories often begin with reproduction (`create → edit → python`) or broad-to-narrow localization; the reproduction triple occurs 156 times at turns 1–3 in the reported resolved GPT-4 trajectories. [§5.2; Appendix B.3.2; Table 8](../pdfs/10-swe-agent.md)
- **Later phase:** editing and execution dominate from around turn five onward, with additional localization when new failures reveal dependencies or other affected files. [§5.2; Appendix B.3.2](../pdfs/10-swe-agent.md)
- **Solved-trajectory lengths:** GPT-4 Turbo full-test successes average 14.71 turns, median 12, with 75% completed within 18; these statistics condition on successful tasks. [Appendix B.3.1; Figure 14](../pdfs/10-swe-agent.md)
- **Success versus prolonged failure:** successful GPT-4 runs generally submit earlier and more cheaply. The paper hypothesizes that simply extending the budget will not solve many stuck trajectories; that causal claim is not established by a budget-randomized experiment here. [§5.2; Appendix B.3.1–B.3.2](../pdfs/10-swe-agent.md)
- **Edit-error prevalence:** 1,185 of 2,294 GPT-4 trajectories (51.7%) have at least one failed lint-checked edit. Failed editing is common even when a suitable file has already been localized. [§5.2; Appendix B.3.3](../pdfs/10-swe-agent.md)
- **Recovery definition:** consecutive failed edits immediately followed by a successful edit count as recovery; switching to a non-edit action instead is classified as unsuccessful recovery for this analysis. [Appendix B.3.3](../pdfs/10-swe-agent.md)
- **Recovery probabilities:** the reported probability of eventual successful editing is 90.5% at zero previous failures and 57.2% after one failure; more consecutive failures correlate with lower recovery. This is syntactically accepted editing, not task resolution. [Figure 20; Appendix B.3.3](../pdfs/10-swe-agent.md)
- **Failure categories:** failed reproduction, failure to find the correct file, failure to find the edit location, overly specific implementation, incorrect implementation, budget exhaustion, failed edit recovery, premature abandonment, and other. [Table 9](../pdfs/10-swe-agent.md)
- **Failure proportions:** incorrect or overly specific implementations together account for 52.0%, and failed-edit recovery 23.4%, in the 248-trajectory classified sample. [Appendix B.4; Figure 8](../pdfs/10-swe-agent.md)
- **Classifier uncertainty:** reported 87% human/LM category agreement comes from only 15 manually labeled examples; failure proportions inherit classifier error and cannot be treated as independently audited labels for every run. [Appendix B.4](../pdfs/10-swe-agent.md)
- **Patch size:** generated patches tend to be larger than corresponding gold patches, partly because agents leave reproduction code; successful patches are smaller than the overall generated-patch distribution. Table 11 trims to values within the 90th percentile, so its means are not untrimmed population means. [Appendix B.6; Table 11](../pdfs/10-swe-agent.md)
- **Localization proxy:** file-set F1 against gold edited/removed files is 59.05% for GPT-4 Turbo SWE-agent versus 45.47% for the reported Claude 3 Opus BM25/RAG comparator; the comparison changes both scaffolding and model. [Appendix B.9](../pdfs/10-swe-agent.md)
- **Termination:** explicit submission, budget exit with changes, budget exit without changes, and malformed-response early exit are distinguished; 266 of 286 GPT-4 successes are intentionally submitted. [Table 13](../pdfs/10-swe-agent.md)
- **Requests case:** a bytes-to-string fix passes tests but duplicates an existing utility rather than reusing it; benchmark success can still leave maintainability concerns. [Appendix D, psf/requests-2317](../pdfs/10-swe-agent.md)
- **Pylint case:** a regex fix passes tests but differs in boundary behavior from the gold patch, illustrating the limits of existing test coverage. [Appendix D, pylint-dev/pylint-5859](../pdfs/10-swe-agent.md)
- **SymPy case:** repeated scrolling and speculative search consume the budget without a patch; understanding an inherited property would require more targeted localization. [Appendix D, sympy/sympy-21614](../pdfs/10-swe-agent.md)
- **Django case:** the agent edits a plausible but wrong abstraction and submits without adequate testing, showing that successful linting and plausible reasoning do not ensure functional repair. [Appendix D, django/django-14411](../pdfs/10-swe-agent.md)

## 5. Authors' conclusions

- Tailored ACIs can substantially improve fixed-model software-engineering performance by making actions reliable and feedback useful. [§7](../pdfs/10-swe-agent.md)
- Interface quality spans action granularity, output design, state grounding, error prevention, and history processing; treating these as unrelated prompt tricks misses their interaction. [§§1–3; §7](../pdfs/10-swe-agent.md)
- Human-oriented interface conventions are not automatically optimal for LMs, as the iterative-search comparison demonstrates. [§5.1](../pdfs/10-swe-agent.md)
- ACI design can be a systematic behavioral research method, analogous to HCI, rather than only an engineering convenience. [§7](../pdfs/10-swe-agent.md)
- More sophisticated tools, automated interface design, and transfer beyond programming are proposed research directions, not established results of this paper. [Appendix E.3](../pdfs/10-swe-agent.md)

## 6. Limitations and open questions

### Explicit source limitations

- The final toolkit is small; static/dynamic analysis, better fault localization, fuzzing, and browsing are suggested extensions. [Appendix E.3](../pdfs/10-swe-agent.md)
- Interface design and case inspection are manual; the paper does not demonstrate an autonomous ACI-improvement loop. [Appendix E.3](../pdfs/10-swe-agent.md)
- Transfer beyond programming is open, despite general motivation for ACIs. [Appendix E.3](../pdfs/10-swe-agent.md)
- Guardrails assume a well-formed starting codebase and can reject useful intermediate states, constraining edit order. [Appendix A.1](../pdfs/10-swe-agent.md)
- Docker isolation limits accidental side effects but is explicitly not equivalent to hardware virtualization or protection against deliberate container-escape exploits. [Appendix E.1](../pdfs/10-swe-agent.md)
- The authors identify risks from untrusted evaluation infrastructure, malicious task descriptions, and offensive uses of capable coding agents. [Appendix E.1](../pdfs/10-swe-agent.md)

### Analyst caveats and source inconsistencies

- **Analysis:** test-based resolve rate cannot certify maintainability, absence of regressions outside tests, or equivalence to the intended specification; Appendix D directly demonstrates this distinction. [Appendix D](../pdfs/10-swe-agent.md)
- **Analysis:** observations about early success versus late failure are confounded by task difficulty and trajectory quality; they do not establish that a better restart policy always dominates a larger budget. [§5.2; Appendix B.3](../pdfs/10-swe-agent.md)
- **Source inconsistency:** §5 says 88.3% HumanEvalFix, while Table 2 gives 87.7% Python, 89.7% JS, and 87.9% Java. Use the explicit language-specific table values and retain the discrepancy. [§5; Table 2](../pdfs/10-swe-agent.md)
- **Source inconsistency:** Appendix B.3.3 prints 113/286 as 31.5%; the arithmetic is approximately 39.5%. Do not reuse 31.5% as a verified derived statistic. [Appendix B.3.3](../pdfs/10-swe-agent.md)
- **Source inconsistency:** some appendix Claude counts describe 35 Lite successes while Table 1's 13% implies 39; sections may reflect different runs. Do not combine those denominators without checking underlying artifacts. [Table 1; Figure 14; Table 13](../pdfs/10-swe-agent.md)
- **Analysis:** the 248 unresolved failure-analysis cases correspond to 52 successes, not the headline run's 54; treat the classifier study as its own sample. [Table 1; Appendix B.4; Table 10](../pdfs/10-swe-agent.md)
- **Extraction/source caution:** some thousands separators render as `2true294`, and some prose references point to the wrong numbered figures/tables. This note uses 2,294 from Table 12 and cites actual table captions rather than reproducing malformed links. [§4; Table 12](../pdfs/10-swe-agent.md)

## 7. Relevance to this project — interpretation

- **Proposal:** design kernel-agent commands around meaningful operations—compile, execute correctness tests, benchmark, inspect profiler output—so a single action returns a compact, actionable result. This extrapolates the ACI principles; the paper does not test GPU kernels. [Basis: §2](../pdfs/10-swe-agent.md)
- **Proposal:** separate syntax/compilation errors, numerical failures, and performance feedback, and show the candidate state corresponding to each result so the agent does not reason from stale outputs. [Basis: §3; Appendix A.1](../pdfs/10-swe-agent.md)
- **Proposal:** experimentally compare history policies and observation sizes at matched budgets instead of assuming more logs or a longer context are always better. [Basis: Table 3](../pdfs/10-swe-agent.md)
- **Proposal:** record trajectories, tool calls, cost, and failure categories alongside final speedups; instrument repeated failed edits and ineffective searches to identify recoverable harness problems. [Basis: Appendix A.2; B.3–B.4](../pdfs/10-swe-agent.md)
- **Proposal:** retain independent correctness evaluation even if agent-written reproduction tests pass; test success and a compelling agent explanation are not sufficient for deployment. [Basis: Appendix D](../pdfs/10-swe-agent.md)
- **Open project hypothesis:** repeated failure could trigger a restart or a different search strategy, but its value should be measured rather than inferred from the paper's observational trajectory statistics. [Basis: §5.2](../pdfs/10-swe-agent.md)

## 8. Retrieval notes

- **Aliases:** SWE-agent; ACI; agent-computer interface; summarized search; linting guardrails; last-five observations; SWE-bench Lite; HumanEvalFix.
- **Questions answered:** why can tool-interface design change agent success; how should code-viewing/search/editing commands interact; what feedback helps recover from failed edits; what does context truncation retain; why do additional retries improve pass@k?
- **Related analyses:** [ReAct](11-react.md) for the thought/action/observation loop; [Reflexion](19-reflexion.md) for a different mechanism that stores lessons between attempts; [Self-Refine](18-self-refine.md) for artifact revision through self-feedback.
- **Navigation only:** [ISO-Bench](07-iso-bench.md) for repository-level inference optimization; [Robust Agentic CUDA](08-robust-agentic-cuda.md) for kernel-agent execution and verification. These pointers do not imply matched experimental comparisons.

## 9. Source coverage and verification notes

- Covered main §§1–7; Appendix A's interface, implementation, and configuration; Appendix B.1–B.9's sweeps, stratified results, trajectories, failures, retries, patches, datasets, and termination; Appendix C's prompt roles; Appendix D's four qualitative cases; Appendix E's impacts, reproducibility, and limitations. [Local full text](../pdfs/10-swe-agent.md)
- Read the substantive author analysis and inspected representative command/prompt/trajectory examples; the long repeated source-code and shell-output transcripts in Appendix D are not reproduced or independently executed. No claim is made to have re-run the experiments.
- Main numerical checks use Tables 1–3, 5–13 and the associated prose; inconsistencies are retained above rather than silently reconciled. The analysis does not infer new numbers from unlabeled graphical pixels.
- Bibliography entries provide context but are not independent evidence for claims about works not read in this assignment. These notes describe the frozen v3 source and should be rechecked if the source version changes.
