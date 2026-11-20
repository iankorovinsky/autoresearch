# Papers and analyses

This collection separates preserved source evidence from derived reading notes. It covers the 21 entries supplied through Notion plus LARK, for 22 works total. Entry 21, *Measuring Automated Kernel Engineering*, is a METR web article rather than an arXiv paper. LARK is entry 22, arXiv 2611.02028v1.

## Directory roles

- [pdfs/README.md](pdfs/README.md) catalogs the source collection and explains conversion/provenance. Read full-text `.md` files for text retrieval; use original `.pdf`, HTML, and figure assets to resolve visual or extraction questions.
- [analysis/README.md](analysis/README.md) catalogs the analyses and routes questions by topic. Every note has nine consistent sections, metadata, and source locators.
- [pdfs/manifest.json](pdfs/manifest.json) is the source inventory: identifiers, frozen versions, URLs, original files, PDF hashes/page counts, and conversion details.
- [analysis/index.json](analysis/index.json) is the retrieval inventory: analysis/source paths, version, summary, topics, review status, section names, and descriptive size counts. Paths in this index are relative to `analysis/`.
- [analysis/REVIEW.md](analysis/REVIEW.md) records what reading and checking were performed and the limits of that review.

## File identity and formats

The stable two-digit ID and descriptive slug match across formats. For example, `01-kernelbench.pdf`, `01-kernelbench.md`, and `analysis/01-kernelbench.md` refer to the same collected work. An ID is an inventory key, not a ranking or publication order. Do not renumber existing entries when adding a paper.

Markdown is the primary LLM-readable format. It supports headings, bullet statements, code, TeX math, local links, and human review without requiring a custom XML schema. YAML frontmatter carries metadata; JSON provides a simple machine-readable catalog. Some complex full-text tables remain HTML to preserve merged cells. The original PDFs remain available when conversion loses visual information.

## Source versus interpretation

The text in `pdfs/` preserves paper content; the text in `analysis/` is a secondary analysis. Citation labels identify the linked artifact's section/table/figure. HTML and PDF figure numbering occasionally differ. A `PDF p.` reference means the one-based physical page, not necessarily the printed page number.

Each analysis separates concepts and methods, reported results, the authors' conclusions, explicit limitations and analyst caveats, and proposed project applications. Related-paper links support navigation; they do not by themselves establish a validated cross-paper comparison. Cross-paper hypotheses must be labeled as new synthesis and supported by the underlying studies.

## Adding or updating a work

1. Assign the next unused ID, verify the title/identity, and obtain an authoritative full-text source. Record exact version, URL, retrieval date, and any non-PDF status.
2. Preserve the original PDF/HTML and local assets; create readable Markdown with equations and code intact. Check table values and any figures essential to the analysis.
3. Add an entry to `pdfs/manifest.json`, including the PDF SHA-256 when a PDF exists. Do not invent a PDF for a web article.
4. Write a matching analysis using the structure and metadata in [the analysis guide](analysis/README.md). Read the substantive appendices and state any omissions.
5. Review consequential numbers, metric definitions, source locators, denominators, and guarantee boundaries. Retain source discrepancies rather than silently correcting the paper.
6. Update both catalogs and `analysis/index.json`; record the review scope in `analysis/REVIEW.md`. Check local links and ensure source/analysis versions match.

When a new source revision appears, preserve the old snapshot and its analysis or move both into an explicitly versioned archive before replacing the current pair. Update paths, hashes, versions, and indexes together. Never silently attach an old analysis to a new PDF. If a conversion is corrected without changing the paper version, record that separately from a source revision.
