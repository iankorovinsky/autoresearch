# Repository development utilities

Use the executable `./dev` at the repository root for repeatable setup, local checks/tests, environment diagnostics, and offline documentation validation. These utilities are separate from the research implementation and the GPU compute CLI.

## Getting started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Node 24 (recommended). The frontend/data minimum is Node 22.12. The wrapper uses Python 3.12; uv can obtain it when needed.

From the repository root:

```bash
./dev list                   # Available named tasks
./dev setup --dry-run        # Preview commands and component directories
./dev setup                 # Install each component's locked dependencies
./dev doctor                # Check local environments and configuration
./dev check                 # Python lint, frontend types, Prisma schema, docs
./dev test                  # All local test suites; live GPU tests excluded
./dev docs                  # Documentation and index checks only
```

Select components to narrow a task:

```bash
./dev setup tooling compute
./dev test compute
./dev check frontend
./dev doctor frontend --profile app
./dev run frontend:dev
./dev run frontend:build
```

The components are `tooling`, `compute`, `frontend`, and `data`. `check` also includes repository documentation checks, even when a component is selected. `frontend:dev` and `frontend:build` are explicit named tasks; they are not part of the default setup/check/test sequence. No task runs database migrations, deploys code, creates a GPU environment, or submits GPU jobs. The data tests use disposable in-memory PGlite databases.

Task groups continue after a failed component so the final summary reports all failures. Use `--fail-fast` to stop at the first failure. Ctrl-C stops the active task's process group and returns 130. Other failures return 1; successful checks/tasks return 0. `--json` emits a machine-readable result on stdout and routes child-task logs to stderr:

```bash
./dev check --json > check-results.json
./dev test --fail-fast --json > test-results.json
./dev run data:check --dry-run
```

Redirect reports outside the repo, or review/ignore them deliberately; arbitrary generated result files are not added to Git automatically. The wrapper locates the repo relative to itself, so invoking it by absolute path also works from another directory.

## Dependency isolation

- This package has its own [pyproject.toml](pyproject.toml), [uv.lock](uv.lock), Python version file, and `tooling/.venv`. Its only runtime dependency is the Markdown parser; pytest/Ruff are development dependencies.
- The compute CLI continues using `compute/.venv`, `compute/pyproject.toml`, and `compute/uv.lock`. Tooling does not import `kernel_compute`, KernelBench, Modal, PyTorch, or frontend/database packages.
- Frontend and data each retain their own `package.json`, `package-lock.json`, and `node_modules`. The runner invokes `npm ci` from the selected package directory. It does not create a root Node workspace or root Python environment.
- Python child commands explicitly target the component's environment. Inherited active-environment/import-path overrides are cleared. Node child commands use the selected Node installation with inherited `NODE_PATH`/`NODE_OPTIONS` cleared. The runner also clears the live-GPU-test opt-in variable and explicitly excludes the `live` marker from compute tests.
- Python setup uses `uv sync --locked`; ordinary checks/tests use `uv run --locked --no-sync`. Setup installs dependencies; checks/tests do not implicitly sync another component's environment or rewrite lockfiles. Download caches may be reused by uv/npm without sharing the installed environments.

The first invocation of `./dev` bootstraps/synchronizes **only the tooling environment** before executing the requested command. This also applies to `doctor` and `--dry-run`; the diagnostic/dry-run operations themselves do not install other components. Subsequent invocations use that locked tooling environment.

Node selection is local to child processes, in this order:

1. `REPO_DEV_NODE_BIN`, if set, must point to a supported installation's directory containing both `node` and `npm`.
2. The newest installed Node 24 under `$NVM_DIR/versions/node/`, or `~/.nvm/versions/node/` by default.
3. A supported `node` and matching `npm` on PATH.

This makes the existing Node 24 installation usable even when the interactive shell still points at Node 16. It does not change your shell, nvm default, or global package installation. An invalid explicit override fails rather than silently selecting something else.

```bash
REPO_DEV_NODE_BIN="$HOME/.nvm/versions/node/v24.21.0/bin" ./dev doctor
```

## Environment doctor

```bash
./dev doctor
./dev doctor compute --json
./dev doctor frontend --profile app
./dev doctor --strict
./dev doctor compute --ssh
```

The doctor checks:

- Python version, expected environment prefix, disabled system/user site packages, lockfile presence, and an offline `uv sync --check --locked` dependency check.
- Effective Node version and matching npm, per-package installation, direct manifest/lock consistency, and local `npm ls` dependency health.
- Presence/non-placeholder values for the frontend's public Supabase settings and `APP_URL`, plus the optional server-only `DATABASE_URL` used for migrations. It reports **names and status only**, never values or raw subprocess output.

The default `local` profile treats missing app/database configuration as warnings because lint/type/schema/unit checks do not need deployed services. The `app` profile requires frontend settings. Database credentials remain optional: the utility never applies migrations. `--strict` makes warnings fail the command.

Configuration presence checks read simple dotenv assignments without executing/interpolating expressions. Frontend development precedence is process environment, `.env.development.local`, `.env.local`, `.env.development`, then `.env`; data uses the process environment over `.env`. Unexpanded `${...}` references are not considered configured. These checks do not authenticate credentials, test remote services, or fully emulate Next.js/dotenv expansion and multiline syntax.

There are **no default network probes**. `--ssh` explicitly invokes the existing compute environment's read-only `kcompute doctor`, which checks Waterloo authentication, scheduler visibility, and the prepared worker environment. It does not allocate a GPU. Remote output is reduced to a status; use the compute CLI directly for its detailed diagnostics. The probe has a bounded timeout.

## Documentation checker

```bash
./dev docs
./dev docs compute/README.md wiki/artifacts --json
./dev docs --prose-paths
./dev docs --prose-paths --strict
./dev docs --include-sources
```

Default scope: root README/agent instructions, `compute/`, `frontend/`, `data/`, `tooling/`, and `wiki/`. Directory scans skip generated/dependency/cache directories, hidden files, and nested Git checkouts such as the research submodules. Vendored KernelBench/Harbor code is outside the default roots. Preserved paper Markdown under `wiki/papers/pdfs/` is excluded except its README; use `--include-sources` to inspect converted source text explicitly. JSON inventories/metadata remain included by default.

Checks include:

- Missing local link/image targets, reference-style links, HTML `href`/`src` targets inside Markdown, and Markdown heading/explicit HTML anchors. Relative URLs, repository-root URLs, percent-encoded paths, and duplicate headings are handled. Links inside code blocks/inline code are not treated as real links.
- Local links escaping the repository and unclosed fenced code blocks.
- Invalid JSON, duplicate keys, and non-finite JSON constants. Files named `tsconfig*.json` and `.jsonc` accept comments/trailing commas; ordinary `.json` stays strict.
- Markdown artifacts missing from `wiki/artifacts/README.md`, including nested artifacts. Companion JSON/HTML files need not have separate index entries.
- Missing file paths in the paper source manifest and analysis retrieval index, and inconsistent analysis `entry_count`.

Artifact/catalog checks always run, including when specific files/directories are selected. Diagnostics include the file, containing block's starting line (for Markdown links), error code, and message. These checks never modify documentation or indexes.

`--prose-paths` additionally warns about unresolved path-shaped inline-code references. These may be examples/proposals, so they are advisory unless `--strict` is set. The existing historical Modal guide currently produces five such warnings; default Markdown-link validation passes. The checker does not inspect shell commands for every implied path, validate research claims, fetch external URLs, or check fragments inside PDFs/HTML files. Heading fragments use a GitHub-style slug approximation rather than a hosted renderer.

## Maintaining the tools

The task catalog is in [tasks.py](src/repo_dev/tasks.py). Add an explicit command/working-directory entry there rather than shell-concatenating user input. Keep dependency installation scoped to the component's directory and put network-dependent or mutating tasks behind explicit named commands.

```bash
./dev test tooling
./dev check tooling
```

Tests cover parsing and real failure behavior: code/fence exclusions, local links/anchors, JSON/JSONC, index coverage, source-snapshot exclusions, catalog consistency, environment selection/isolation, configuration redaction/precedence, optional SSH, process-group cleanup, dry runs, failure aggregation, and stdout/stderr separation.

See the [implementation validation artifact](../wiki/artifacts/development-tooling-validation.md) for the actual setup/check/test results. The parser follows the [markdown-it-py token API](https://markdown-it-py.readthedocs.io/en/latest/using.html); uv environment handling follows the [locking and syncing documentation](https://docs.astral.sh/uv/concepts/projects/sync/).
