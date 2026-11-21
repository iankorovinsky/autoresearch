# Development tooling implementation and validation

**Implemented and checked: 2026-11-21.** This records repository tooling work, not research/kernel performance experiments.

## Delivered

The executable root `dev` wrapper starts the independent `tooling/` package. It provides a task catalog and setup/check/test runner, an environment doctor, and an offline documentation checker. See the [usage guide](../../tooling/README.md) and [source package](../../tooling/src/repo_dev/).

The package has its own Python 3.12 environment and dependency lock. Its runtime dependency is `markdown-it-py`; it does not import the compute CLI or application packages. Compute remains in `compute/.venv`; frontend and data each use their own `node_modules` and npm lockfile. Child environment selection overrides inherited Python/Node import paths without changing the user's shell.

## Actual validation

- `./dev setup --dry-run --json` showed separate commands and working directories for all four components.
- `./dev setup --json` completed all four installations using existing lockfiles. Python environments were already synchronized; frontend and data were reinstalled by `npm ci` in their respective directories.
- `./dev doctor --json` passed local checks: both Python environments were isolated and synchronized; Node **24.21.0** was selected from the existing nvm installation despite Node **16.20.2** on the interactive shell PATH. Configuration names/status were reported without values. SSH was skipped by default.
- `./dev check --json` passed Python lint, frontend type checking, Prisma schema validation, and default documentation/index validation.
- Local suites passed: **31 tooling tests**, **33 compute tests**, **15 frontend tests**, and **7 data tests** (**86 total**). Compute's seven live tests were explicitly deselected. The tooling suite was rerun after the final process-group cleanup and malformed-manifest handling changes.
- Default documentation validation reported zero broken local links, missing index entries, invalid JSON/catalog paths, or unclosed fences. It was rerun after these usage and validation documents were added.

The new tests exercise Markdown/reference/HTML links, fragments and duplicate headings, code-block exclusions, JSON/JSONC parsing, duplicate/non-finite JSON values, artifact coverage, preserved-source exclusions, catalog paths/counts, symlink escape rejection, runtime selection, inherited-environment cleanup, secret redaction, configuration precedence, optional SSH behavior, process-group cleanup after an exited leader, dry runs, failure aggregation, and JSON/log stream separation.

## Boundaries and findings

No GPU allocation, kernel execution, Supabase connection, migration, deployment, or application-feature change was part of this validation. The data suite uses disposable in-memory PostgreSQL/PGlite fixtures. The optional SSH doctor path was tested with a controlled subprocess response; no live SSH check was needed for this tooling work.

The optional `--prose-paths` scan reported **five existing unresolved references** in the historical Modal guide: two helper scripts and three candidate-example paths. That guide is now labeled historical; examples/proposals are warnings rather than automatic document edits or default failures.

`npm ci` emitted existing dependency-audit warnings (one moderate finding for frontend; one moderate and three high findings for data) and install-script notices for Prisma packages. Type/schema checks and local tests passed. These messages were not independently investigated as a security audit, and package versions/lockfiles were not upgraded as part of the tooling task.

The doctor checks configuration presence/placeholders, not credential validity or remote service availability. The documentation checker validates local structure, not research claims or live URLs. Preserved paper Markdown is excluded by default; catalog JSON and project-authored analysis/docs remain checked. The wrapper may bootstrap its own locked environment on first invocation; diagnostics do not install or repair the other components.

Return to the [artifacts index](README.md).
