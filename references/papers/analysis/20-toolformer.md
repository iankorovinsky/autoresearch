---
paper_id: "20"
title: "Toolformer: Language Models Can Teach Themselves to Use Tools"
source_version: "2302.04761v1"
source_markdown: "../pdfs/20-toolformer.md"
source_pdf: "../pdfs/20-toolformer.pdf"
source_url: "https://arxiv.org/abs/2302.04761"
analysis_date: "2026-11-20"
analysis_status: "reviewed"
topics: ["tool-use", "self-supervised-learning", "synthetic-data", "api-selection", "retrieval", "language-modeling", "evaluation", "decoding"]
summary: "Learns API-call placement and arguments by filtering sampled tool annotations for next-token loss improvement, with controlled tool-disabled and decoding-policy evaluations."
---

# Toolformer: Language Models Can Teach Themselves to Use Tools

## Paper identity and scope

- Analyzes arXiv 2302.04761v1: self-supervised training for choosing tools, insertion positions, arguments, and incorporation of returned text. [Abstract; §§1–2](../pdfs/20-toolformer.md)
- The study fine-tunes GPT-J on tool-augmented language-modeling text, then evaluates prompted zero-shot downstream tasks. It changes model weights; it is not merely an inference-time prompting scaffold. [§§2,4.1–4.2](../pdfs/20-toolformer.md)
- Five tools cover factual QA, Wikipedia search, arithmetic, translation into English, and calendar date access. The study does not evaluate arbitrary software tools or GPU optimization. [§3](../pdfs/20-toolformer.md)
- “Self-supervised” describes call generation/filtering after a handful of human-written API demonstrations and hand-designed tool-specific heuristics; it does not mean no human input or no external pretrained models. [§1; Appendix A](../pdfs/20-toolformer.md)

## Concepts and terminology

- A call is a tool name plus text input; a result is also text. Calls and results are inserted into ordinary language sequences using delimiters and a result separator. [§2](../pdfs/20-toolformer.md)
- The implementation reuses existing token sequences for opening/closing brackets and the arrow, so extending the model vocabulary is unnecessary. [§2, footnote 1](../pdfs/20-toolformer.md)
- The pipeline samples candidate calls, executes them, filters by subsequent-token prediction benefit, merges retained calls into text, then fine-tunes with a standard language-modeling objective. [§2; Figure 2](../pdfs/20-toolformer.md)
- Candidate insertion probability is p_i = p_M(<API> | P(x), x_1:i−1). P(x) contains the complete annotation input, so the data-generation procedure can look ahead. [§2, Sampling API Calls; Appendix A.1, Machine Translation](../pdfs/20-toolformer.md)
- Sampling keeps positions exceeding τ_s, capped at the top k positions. Up to m candidate calls are sampled per position, with the closing delimiter as stop token. [§2, Sampling API Calls](../pdfs/20-toolformer.md)
- Candidates that never generate the closing delimiter are discarded; API execution can be a neural model, retrieval system, or program. [§2, footnote 2; Executing API Calls](../pdfs/20-toolformer.md)
- The filtering loss L_i(z) is weighted negative log likelihood of the original suffix tokens conditioned on prefix z and the preceding original text. The labels are existing corpus tokens, not human tool-use ratings. [§2, Filtering API Calls](../pdfs/20-toolformer.md)
- L_i+ supplies call and result. L_i− is the minimum loss of no call and a call with no returned result. Keeping L_i−−L_i+ ≥ τ_f requires improvement over the better of those two controls. [§2, Filtering API Calls](../pdfs/20-toolformer.md)
- The no-result control helps distinguish useful returned information from information already exposed by the generated question or expression. [§2, Filtering API Calls](../pdfs/20-toolformer.md)
- During filtering, the call/result is prepended rather than inserted at its final position because the not-yet-adapted model finds mid-text API syntax disruptive. Training later interleaves calls at the selected positions. [§2, footnote 3; Model Finetuning](../pdfs/20-toolformer.md)
- Weights are proportional to max(0,1−0.2t), normalized to sum to one. Only the next five positions have positive weight, favoring calls near immediately useful information. [§4.1, Dataset Generation](../pdfs/20-toolformer.md)
- The loss-improvement score is a usefulness proxy for predicting corpus text, not a correctness proof, tool-reliability estimate, or direct task-reward objective. [§2; §5, Data Quality](../pdfs/20-toolformer.md)
- At inference, decoding pauses after the result separator, the external API is executed, and its result plus closing delimiter is inserted before generation resumes. [§2, Inference](../pdfs/20-toolformer.md)
- The headline decoding policy forces a call whenever its opening token ranks within the top ten tokens, even if ordinary greedy decoding would choose another token. [§4.2; §5, Decoding Strategy](../pdfs/20-toolformer.md)
- At most one API call is allowed per downstream input to prevent repetitive calling. This cap is distinct from training texts that may contain several independently sampled calls. [§2, Model Finetuning; §4.2; §7](../pdfs/20-toolformer.md)
- Toolformer (disabled) is the same trained model with the API-opening token probability set to zero. It isolates access to tools from benefits retained in the adapted weights. [§4.1, Baseline Models](../pdfs/20-toolformer.md)

## Methodology and experimental setup

- Training uses a CCNet subset, with an unmodified GPT-J baseline and GPT-J+CC trained on text without calls. OPT-66B and original non-instruction-tuned GPT-3 davinci are larger comparators. [§4.1](../pdfs/20-toolformer.md)
- The paper calls its GPT-J model 6.7B parameters; retain that reported identity rather than treating these results as contemporary model comparisons. [§1; §8](../pdfs/20-toolformer.md)
- Default annotation settings are τ_s=0.05, τ_f=1.0, five positions, and five candidates per position. Calculator and translation use τ_s=0, twenty positions, ten candidates, and τ_f=0.5. [Appendix A](../pdfs/20-toolformer.md)
- Documents with no surviving API calls are removed from the training set. The authors acknowledge this changes the text distribution despite the conceptual claim that annotation preserves corpus content. [§4.1, footnote 4](../pdfs/20-toolformer.md)
- QA uses retrieval-augmented Atlas trained on Natural Questions: Atlas-large generates annotations, while Atlas-xxl serves inference requests. The external QA component changes size between phases. [§3; Appendix A.1](../pdfs/20-toolformer.md)
- The calculator supports addition, subtraction, multiplication, and division, rounds results to two decimal places, and returns no result for syntactically invalid equations. [§3; Appendix A.1](../pdfs/20-toolformer.md)
- Calculator annotation selects documents with three arithmetically related numbers in a 100-token window, number-bearing equation/total/average phrases, or three numbers with only a 1% sample retained for that last broad criterion. [Appendix A.1, Calculator](../pdfs/20-toolformer.md)
- Wikipedia search uses BM25 over the KILT Wikipedia dump and returns snippets. This is a fixed corpus search tool, not live unrestricted web browsing. [§3, Wikipedia Search](../pdfs/20-toolformer.md)
- Translation uses NLLB-600M and fastText language detection, always translating into English. The described NLLB system covers 200 languages. [§3; Appendix A.1](../pdfs/20-toolformer.md)
- Translation annotations target non-English ten-token chunks surrounded by English, with language confidence >0.8 and numeric/symbol-only chunks excluded. [Appendix A.1, Machine Translation](../pdfs/20-toolformer.md)
- Translation calls whose input appears only after the insertion point are removed, addressing one documented train/inference lookahead mismatch. [Appendix A.1, Machine Translation](../pdfs/20-toolformer.md)
- The calendar returns a date without arguments. During annotation the date is approximated from each document URL; only roughly 18% of documents have an extractable date. [§3; Appendix A.1, Calendar](../pdfs/20-toolformer.md)
- Training uses up to 25,000 examples per API, maximum sequence length 1,024, batch size 128, learning rate 1e−5, and 10% linear warmup. [§4.1; Appendix B](../pdfs/20-toolformer.md)
- Models train with ZeRO-3/BF16 on eight A100 40GB GPUs for up to 2,000 steps; perplexity on 1,000 CCNet development examples is checked every 500 steps to choose the checkpoint. [Appendix B](../pdfs/20-toolformer.md)
- Downstream zero-shot means natural-language task instructions without task-specific demonstrations. API-use demonstrations were still supplied during training-data construction. [§4.2; Appendices A.2,C](../pdfs/20-toolformer.md)
- LAMA uses SQuAD, Google-RE, and T-REx examples with the missing fact at the end; success means the correct word occurs within the first five generated words. Wikipedia search is disabled for this Wikipedia-derived benchmark. [§4.2.1](../pdfs/20-toolformer.md)
- Math evaluation covers ASDiv, SVAMP, and MAWPS. It extracts the first generated number, except an equation-containing prediction uses the first number after the equals sign. [§4.2.2, footnote 7](../pdfs/20-toolformer.md)
- WebQuestions, Natural Questions, and TriviaQA success means an answer appears within the first twenty generated words. The QA API is disabled, partly because Atlas was trained on Natural Questions. [§4.2.3](../pdfs/20-toolformer.md)
- MLQA supplies English context and a question in Spanish, German, Hindi, Vietnamese, Chinese, or Arabic, requesting an English answer and checking containment within ten generated words. [§4.2.4; Appendix C.4](../pdfs/20-toolformer.md)
- TempLAMA covers changing Wikidata facts for 2010–2020. Dateset supplies synthetic date-relative questions and an example-specific presumed current date returned by the calendar. [§4.2.5; Appendix D](../pdfs/20-toolformer.md)
- Dateset has 9,400 examples across seven template families: elapsed days (400), past calendar attributes (800), future attributes (800), weekday of an explicit date (400), nearby relative days (4,000), holiday attributes (1,800), and holiday intervals (1,200). [Appendix D; Table 11](../pdfs/20-toolformer.md)
- Dateset starts from 500 sampled current dates and nearby dates in a four-year range; holiday templates use United States federal holidays. Some explicit-date questions can be solved without current-date access. [Appendix D; Table 11](../pdfs/20-toolformer.md)
- Perplexity checks use WikiText and 10,000 held-out CCNet documents. Toolformer is evaluated with API calls disabled; enabled-call perplexity would require marginalizing over possible calls and is not reported. [§4.3, footnote 8](../pdfs/20-toolformer.md)
- Scaling experiments use GPT-2 variants of 124M, 355M, 775M, and 1.6B parameters plus GPT-J, with only QA, calculator, and search tools. [§4.4](../pdfs/20-toolformer.md)

## Results and evidence

- At filtering threshold 1.0, Table 2 lists 18,526 QA, 60,974 search, 994 calculator, 20,587 calendar, and 1,034 translation examples. These are generated-pool counts, before the per-API training cap. [Table 2; Appendix B](../pdfs/20-toolformer.md)
- At threshold 0.5 the respective counts are 51,987, 207,241, 3,680, 61,811, and 3,156; at 2.0 they are 5,135, 13,944, 138, 3,007, and 229. Higher required benefit sharply reduces available supervision. [Table 2](../pdfs/20-toolformer.md)
- LAMA Toolformer scores are 33.8/11.5/53.5 for SQuAD/Google-RE/T-REx, versus disabled 22.1/6.3/34.9 and GPT-3 26.8/7.0/39.8 under this evaluation. [Table 3](../pdfs/20-toolformer.md)
- Those LAMA gains over the strongest GPT-J-family no-tool baseline are 11.7, 5.2, and 18.6 percentage points. QA is used on 98.1% of examples, another tool on 0.7%, and none on 1.2%. [§4.2.1; Table 3](../pdfs/20-toolformer.md)
- Math scores are Toolformer 40.4/29.4/44.0 on ASDiv/SVAMP/MAWPS, disabled 14.8/6.3/15.0, GPT-J 7.5/5.2/9.9, and GPT-3 14.0/10.0/19.8. [Table 4](../pdfs/20-toolformer.md)
- The calculator is called on 97.9% of math examples. The authors hypothesize that tool-augmented training also improves internal arithmetic, explaining disabled-model gains; the paper does not isolate that mechanism with a separate data ablation. [§4.2.2](../pdfs/20-toolformer.md)
- QA scores are Toolformer 26.3/17.7/48.8 on WebQS/NQ/TriviaQA versus disabled 18.9/12.6/46.7. GPT-3 remains higher at 29.0/22.6/65.9. [Table 5](../pdfs/20-toolformer.md)
- Wikipedia search is used on 99.3% across the QA examples; the authors attribute remaining deficits partly to irrelevant retrieval and lack of interactive query reformulation. [§4.2.3](../pdfs/20-toolformer.md)
- MLQA Toolformer scores for Es/De/Hi/Vi/Zh/Ar are 20.6/13.5/1.4/10.6/16.8/3.7; all exceed its disabled counterpart, but German, Chinese, and Arabic remain below original GPT-J 16.5/18.2/8.2. [Table 6](../pdfs/20-toolformer.md)
- Translation use ranges from 63.8–94.9% for most languages, but only 7.3% for Hindi. Improvement with the tool does not imply reliable multilingual question answering. [§4.2.4](../pdfs/20-toolformer.md)
- All-English MLQA control scores for GPT-3 are 23.6–27.2 across the subsets, above its multilingual-question scores. The authors interpret English-output and language-understanding difficulty as a major factor. [Table 6; §4.2.4](../pdfs/20-toolformer.md)
- TempLAMA scores are Toolformer 16.3, disabled 12.7, GPT-J 13.7, and GPT-3 15.5. Calendar use is just 0.2%; the gain chiefly accompanies QA/search access. [Table 7; §4.2.5](../pdfs/20-toolformer.md)
- Dateset scores are Toolformer 27.3 versus disabled 5.9, GPT-J 3.9, and GPT-3 0.8; calendar is used on 54.8% of examples. These results use each example's synthetic current date. [Table 7; §4.2.5; Appendix D](../pdfs/20-toolformer.md)
- WikiText/CCNet perplexity is 9.9/10.6 for original GPT-J and 10.3/10.5 for both GPT-J+CC and disabled Toolformer. Equality is with the CCNet-adapted baseline, not preservation of every original-model metric. [Table 8](../pdfs/20-toolformer.md)
- Authors report substantial tool benefit emerging around 775M parameters, with search as an exception for smaller models. This is a pattern in the tested families and tools, not a universal capacity threshold. [§4.4; Figure 4](../pdfs/20-toolformer.md)
- On T-REx, no calls gives 34.9; ordinary greedy k=1 gives 47.8 with 40.3% calling; k=3 gives 52.9 with 82.8% calling; k=10 gives 53.5 with 98.1% calling. [Table 9](../pdfs/20-toolformer.md)
- On WebQS, k=0/1/3/10 gives 18.9/19.3/26.3/26.3, with calls on 0/8.5/99.3/100% of examples. The reported gain depends substantially on the nonstandard call-trigger rule. [Table 9](../pdfs/20-toolformer.md)
- At k=1, no-call subsets score 44.3 on T-REx and 19.9 on WebQS versus no-tool full-set scores 34.9 and 18.9. Authors interpret this as some selective calling calibration; the compared subsets are selected by the model. [§5; Table 9](../pdfs/20-toolformer.md)
- Table 10 qualitatively aligns large loss reductions with useful API responses, but a search about a song for an unrelated “fast train” phrase still gets a positive 0.92 score. The filter admits false positives. [§5, Data Quality; Table 10](../pdfs/20-toolformer.md)
- A calendar example judged intuitively useful has gain only 0.33, below the default 1.0 threshold. The qualitative usefulness column is not identical to the training retention decision. [Table 10; Appendix A](../pdfs/20-toolformer.md)

## Authors' conclusions

- The authors conclude sampled-and-filtered tool annotations can teach a language model when and how to use several APIs with little manual demonstration. [§8](../pdfs/20-toolformer.md)
- They conclude this training substantially improves zero-shot performance on selected tasks and can exceed much larger no-tool models in some settings. [§8; Tables 3–7](../pdfs/20-toolformer.md)
- They interpret the matching disabled-tool perplexity of Toolformer and GPT-J+CC as evidence that adding calls does not impose an additional core language-modeling penalty in those evaluations. [§4.3; Table 8](../pdfs/20-toolformer.md)
- They identify interactive retrieval, chained calls, iterative bootstrapping, and accounting for tool costs as future improvements, rather than demonstrated capabilities. [§§4.2.3,7](../pdfs/20-toolformer.md)
- They suggest imperfect tool annotations may teach the model not to follow every returned response blindly; this is a hypothesis from qualitative examples, not a controlled robustness result. [§5, Data Quality](../pdfs/20-toolformer.md)

## Limitations and open questions

- Explicit limitation: tools are sampled independently, so training lacks examples where one result supplies the next call's argument; the system does not learn tool chains in this study. [§7](../pdfs/20-toolformer.md)
- Explicit limitation: it cannot interactively browse retrieval results or refine unsuccessful search queries, limiting open-ended factual investigation. [§7; §4.2.3](../pdfs/20-toolformer.md)
- Explicit limitation: calling decisions are sensitive to prompt wording; the headline results use particular task prompts and a modified decoder. [§7; Appendix C; Table 9](../pdfs/20-toolformer.md)
- Explicit limitation: annotation can be sample-inefficient; more than a million processed documents yield only a few thousand useful calculator examples. [§7](../pdfs/20-toolformer.md)
- Explicit limitation: call selection does not account for tool-dependent computational cost. Small base-model parameter count is not total system size, latency, or inference cost. [§7; Appendix A.1](../pdfs/20-toolformer.md)
- Analyst caveat: two-decimal calculator outputs impose a numerical contract and can discard precision. Results do not establish suitability for exact arithmetic or kernel numerical verification. [§3, Calculator](../pdfs/20-toolformer.md)
- Analyst caveat: lenient answer-containment and numeric-extraction metrics differ from standard exact-match evaluations; scores should not be copied into incompatible leaderboards. [§4.2.1–4.2.4](../pdfs/20-toolformer.md)
- Analyst caveat: tool availability is manually restricted by benchmark. The “autonomous” selection claim operates within those supplied API permissions, not an identical unrestricted toolkit across every task. [§4.2.1; §4.2.3](../pdfs/20-toolformer.md)
- Analyst caveat: changes in tool implementation between annotation and inference, external retrieval knowledge, and forced calling all contribute to the system configuration; comparison with a no-tool large LM does not isolate parameter efficiency. [Appendix A.1; §4.2; Tables 3–7](../pdfs/20-toolformer.md)
- Analyst caveat: preserving the content of retained texts does not guarantee preservation of the training distribution or every capability; multilingual regressions and WikiText degradation relative to original GPT-J are explicit counterexamples to a broad preservation reading. [§4.1, footnote 4; Tables 6,8](../pdfs/20-toolformer.md)
- Analyst caveat: lower next-token loss can reward lexical overlap or irrelevant factual snippets. The paper does not quantify adversarial tool-output robustness or prove tool-return factuality. [§5, Data Quality; Table 10](../pdfs/20-toolformer.md)
- Source inconsistency: the calculator demonstration changes a sentence about three-times-fewer goals into a twenty-more-goals statement instead of only inserting a call. Treat the printed template as a documented noisy example, not a guaranteed content-preserving transformation. [Appendix A.2, Calculator](../pdfs/20-toolformer.md)
- Source ambiguity: Table 9 reports WebQS k=10 overall 26.3 and called-subset 26.4 despite a rounded 100.0% call rate. Rounding or aggregation may explain it; the underlying counts are not supplied here. [Table 9](../pdfs/20-toolformer.md)
- Analyst caveat: the paper provides no systematic downstream confidence intervals, repeated-training uncertainty, or per-tool cost/latency breakdown for the reported comparisons. [§§4–5; Appendix B](../pdfs/20-toolformer.md)

## Relevance to this project — interpretation

- Proposal: store useful compile/test/profile tool interactions as candidate training examples, with explicit success criteria and provenance. This extends the annotation idea; Toolformer does not test such GPU tools. [Basis: §2](../pdfs/20-toolformer.md)
- Proposal: compare a tool result against both a no-call and an input-only control when measuring its informational value, rather than crediting information already present in the request. [Basis: §2, Filtering API Calls](../pdfs/20-toolformer.md)
- Proposal: use executable correctness and measured performance objectives for kernel agents; future-token loss alone would not establish numerical correctness or useful speedup. [Basis: §2; §5, Data Quality](../pdfs/20-toolformer.md)
- Proposal: measure call rate, success conditional on calling, no-call success, and total compute costs, including external models and tool execution. [Basis: Table 9; §7](../pdfs/20-toolformer.md)
- Proposal: record call-trigger policies and input visibility, and explicitly train/evaluate chained interactions before expecting a compile–repair–profile workflow to emerge. [Basis: §4.2; Appendix A.1; §7](../pdfs/20-toolformer.md)

## Retrieval notes

- Aliases: Toolformer; self-supervised tool use; API-call annotation; loss-reduction filtering; tool-disabled ablation; top-k call triggering; Dateset.
- Answers questions about generating tool-use supervision, choosing where to call, controlling for query information, tool/model cost accounting, zero-shot definitions, and the limitations of independently sampled calls.
- Related navigation: [ReAct](11-react.md) studies reasoning/action interaction; [SWE-agent](10-swe-agent.md) studies tool-interface design; [Reflexion](19-reflexion.md) studies feedback memory. Their methods and evaluation regimes are distinct.

## Source coverage and verification notes

- Read the complete substantive main text §§1–8 and Appendices A–D, including the filtering equations, all eleven numbered tables, five annotation prompts, four downstream prompt families, and Dateset construction. [§§1–8; Appendices A–D](../pdfs/20-toolformer.md)
- Inspected PDF pages 1–3 for figure-contained predictions, pipeline diagram, QA prompt, and equations missing or incomplete in Markdown; inspected PDF page 8 for the perplexity/scaling discussion and page 15 for API implementation details.
- Quantitative bullets use text tables and their stated units/conditions. Figure 4's scaling interpretation follows its accompanying prose; the curves were not independently digitized. [§4.4; Figure 4](../pdfs/20-toolformer.md)
- No model training, tool execution, benchmark rerun, or independent audit of referenced external models was performed. Bibliographic entries and full demonstration transcripts remain in the original source.
