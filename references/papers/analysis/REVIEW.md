# Reading and review record

Review date: 2026-11-20. Scope: all 22 selected works at the versions in [the source manifest](../pdfs/manifest.json). This is a literature-reading and source-checking record, not a replication study.

## What was done

- Parallel analysts read each work's full substantive text, including relevant appendices, and wrote the matching analysis. Figure-contained prompts, code, taxonomies, and charts were inspected through PDFs or saved assets when needed. Each note records its own coverage and omissions.
- The primary reviewer integrated the notes, examined their structure and evidence framing, and spot-checked consequential claims against frozen sources. Separate second-reader checks covered the groups described below.
- Analyses distinguish concepts/methods, empirical results, authors' conclusions, analyst caveats, and proposed applications. Source contradictions and missing information remain explicit.
- All 22 analyses were checked for valid metadata, matching IDs and source versions, nine-section structure, existing local citation targets, valid cited physical PDF-page ranges, and accidental placeholders.
- All 21 PDF hashes were checked against the manifest. Original PDFs were not changed. The collection has one non-PDF article, represented by Markdown and preserved HTML.
- The four folder READMEs, analysis catalog, JSON topic index, and repository retrieval guidance were added and their local links checked.

`analysis_status: reviewed` means these reading, editorial, structural, and targeted source checks are complete. It does not certify every source assertion, independently validate every derivation, or imply that every figure was digitized. Coverage varies by note and is described in its final section.

## Targeted source-review scope

### Kernel benchmarks and optimization: 01–08

The primary review checked selected task definitions, correctness/performance distinctions, major result conditions, and source inconsistencies. Checks included KernelBench's best-so-far refinement and timing definitions; KernelBenchX's conditional speedups, category rates, and conflicting aggregate values; KernelBench-Verified's standard versus hidden-tested populations and audit denominator; PTXBench's runtime compute-or-movement qualification; TritonBench's printed call/execution discrepancy and retrieval conditions; MultiKernelBench's task/platform scope and category-subset gains; ISO-Bench's five-percent similarity band and selective correctness checks; and robust-kbench's soft-verifier meaning, generated-versus-hardware-evaluated budget, timing protocol, and environment discrepancy.

The assigned analysts additionally checked the details recorded in each note, including appendices and selected visual material. This second pass was a targeted audit, not an independent full reread of all eight sources.

### Numerical modeling, METR, and LARK: 09, 21, 22

A second reader checked the MMA model/architecture map, differential-test stopping criterion, elementary-error bounds, and workaround values. The notes retain the difference between observed agreement with hardware and formal proof, and between hardware fidelity and numerical accuracy.

For METR, the review checked geometric-mean/best-of-k/floor conventions, attempt budgets, task exclusions, headline model portfolios, outcome distributions, amortization assumptions, fine-tuning sample count, and cost shares. The note clarifies that the original 1.05× comparator and the 1.81× portfolio use different model pools.

For LARK, the review checked exact-arithmetic and unbounded-memory semantics, trusted concurrency reduction, tested reference formalization, proof assumptions, production adaptation, major evaluation tables, and the final corrected mutant count. Claims remain limited to the extracted/formalized semantics and enumerated trust boundaries.

### Agent methods: 10, 11, 18, 19, 20

A second reader checked SWE-agent's task denominators, resolved-instance cost definition, main/ablation tables, retry coverage, and error-recovery/classifier samples. ReAct checks covered main tables, diagnostic samples, environment counts, best-versus-average reporting, self-consistency budget, switching rules, and fine-tuning sample size.

Self-Refine checks covered primary and feedback-ablation tables, diagnostic/human samples, task-specific selection protocols, and oracle math results. Reflexion checks covered cumulative trial success, memory sizes, failure heuristics, programming/QA denominators, internal tests, conditional false-positive rates, regressions, and the negative WebShop result.

Toolformer checks covered sampling/filtering equations, prefix-versus-insertion behavior, inference call triggering and the single-call cap, all Tables 2–9, tool implementations between training/inference, training settings, prompts, and Dateset construction. The qualitative annotation filter is not treated as a correctness guarantee.

### Judge studies and reproducibility: 12, 14, 15, 16, 17

A second reader checked Prometheus's major quantitative tables, training settings, and human study; reproducible evaluation's table values, bootstrap distinctions, normalization/window descriptions, and hardware; G-Eval's three tables, sampling method, and bias discussion; MT-Bench's agreement denominators, tie handling, prompt/evaluation details, and trained-judge claims; and Fair Evaluators' tables, scoring/balancing formulas, human labels, random-order repetitions, sampling, entropy routing, and cost conditions.

The review found an additional G-Eval ambiguity: some printed AVG cells do not equal the arithmetic mean of the displayed component values. The analysis preserves the published values, labels them author-reported, and records the discrepancy rather than replacing them with reconstructed averages.

### Survey: 13

The assigned analyst read the full substantive survey and appendices, recovered the missing taxonomy from PDF page 33, and checked its training and benchmark inventories. The primary review checked its six prompting families, inventory/source labels, unresolved classifications, and consistent secondary-evidence framing. Dataset sizes and external-study claims remain survey-reported unless separately reviewed in their own local paper notes. The survey's bibliography was not independently verified item by item.

## Limits and unresolved issues

- No models were trained or evaluated, kernels executed, proof artifacts replayed, benchmarks reproduced, or external repositories audited in this work.
- Not every numerical statement received a fully independent second-reader check. The record above names the targeted scope; per-paper notes provide finer detail.
- Some source plots, example transcripts, and full code listings are summarized by their role rather than exhaustively reproduced. Original evidence remains in the source collection.
- Known source inconsistencies remain unresolved where the text does not support a reconciliation. A reviewed note can legitimately contain uncertain or contradictory source evidence.
- This review freezes the collected versions. It does not establish current model capabilities, current software behavior, current prices, or later paper corrections.
- Cross-paper links are retrieval aids. They do not imply matching experiments, a common leaderboard, or an independently validated cross-paper conclusion.

## Future updates

Record substantive changes here or in a dated successor record, naming the source version, affected analyses, exact check scope, and remaining uncertainty. Revalidate local links, source-version alignment, indexes, and source hashes after changes. Use [the collection maintenance procedure](../README.md) when adding or replacing source versions.
