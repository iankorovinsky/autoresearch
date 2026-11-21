# Auto Research frontend

A local research workspace inspired by the Silico screen recording: a compact project sidebar, a research conversation, and an inspectable experiment panel. Built with Next.js App Router, React, TypeScript, and Lucide icons. The visual system uses warm white panels, charcoal navigation and code surfaces, burnt-orange actions, and Geist typography with monospace labels. The sign-in screen includes a decorative sample workspace preview.

## Run locally

Use Node.js 22.12+ (Node 24 recommended; `.nvmrc` is included). The older Node 16 installation on this machine cannot run the Next.js toolchain.

```sh
cd frontend
nvm use
npm ci
npm run dev
```

Copy `.env.example` to `.env.local` and supply the Supabase public configuration. Keep `APP_URL=http://127.0.0.1:3000` for local development. Open http://127.0.0.1:3000; signed-out visitors are redirected to sign in. See [AUTH.md](AUTH.md) for email/Google setup and current delivery limitations.

```sh
npm run build     # TypeScript check + Next.js production build in .next/
npm start         # Serve the production build locally
npm run typecheck # Check TypeScript without building
npm run format    # Format source files
```

To enable local Codex chat, sign in with ChatGPT using `codex login` and set
`AUTORESEARCH_CODEX_ENABLED=1` in `.env.local`. The app uses `codex` from
your PATH by default. Set `AUTORESEARCH_CODEX_BIN` to the absolute path of a
compatible Codex CLI if you have multiple installations. Restart Next.js after
changing these settings. App Server checks the ChatGPT sign-in before sending a
turn, so model usage goes through that account.

To enable local Claude chat, set `AUTORESEARCH_CLAUDE_ENABLED=1` in `.env.local`
and restart Next.js. The route uses the [Claude Agent SDK](https://www.npmjs.com/package/@anthropic-ai/claude-agent-sdk),
which runs its bundled Claude Code binary as a subprocess with your existing
Claude Code sign-in, `CLAUDE_CODE_OAUTH_TOKEN`, or `ANTHROPIC_API_KEY`. Set
`AUTORESEARCH_CLAUDE_MODEL` to pin a model and `AUTORESEARCH_CLAUDE_BIN` to use a
specific Claude Code executable. The assistant can read repository files (Read,
Glob, Grep) but cannot edit files or run commands. Pick Claude or Codex with the
assistant button next to Send or in Workspace settings; the choice is saved per
account in this browser.

## Included

- Project and thread navigation, workspace search (Cmd/Ctrl+K), and new threads (Cmd/Ctrl+N). Search ranks thread titles, tasks, projects, and saved messages with TanStack match-sorter utilities. TanStack Virtual renders only the visible result rows; a memoized index and deferred updates keep typing responsive. Arrow keys select results, Enter opens the thread or matching message, and Escape closes search. Search currently covers browser-saved workspace data, not Supabase.

  Implementation follows the official [TanStack Virtual React API](https://tanstack.com/virtual/latest/docs/framework/react/react-virtual) and [fuzzy matching guide](https://tanstack.com/table/v8/docs/guide/fuzzy-filtering).
- Research composer with Shift+Enter for newlines, text/code attachments (up to 1 MB each), an auto-research preference, and cancellable streamed Codex replies when enabled.
- Overview, Timeline, Plan, Workers, Compute, Metrics, and Code views.
- Interactive runtime chart, candidate selection, code excerpt copying, and Markdown report downloads.
- Expandable experiment panel, collapsible sidebar, responsive stacked layout, keyboard focus states, and focus-contained dialogs.
- Supabase email/password authentication, verification resend, password recovery, cookie sessions, and protected workspace access.
- Threads and the GPU preference saved in this browser using localStorage, scoped by signed-in user ID.

## Demo boundary and backend integration

Codex App Server or the Claude Agent SDK handles chat and stores the
conversation in its own local session history. The Next.js backend stores the
mapping from workspace thread to provider conversation under
`frontend/.codex-local/` or `frontend/.claude-local/`, keyed by workspace account
and thread. Keep those directories and the provider session stores if you want
to resume a chat after restarting the backend. Each thread keeps separate Claude
and Codex conversations. The browser sends only the new prompt. The
backend opens Codex in read-only mode for this chat integration.

Seeded experiment measurements, workers, timelines, and code excerpts are
illustrative. The GPU and auto-research controls are preferences. Text and code
attachments stay in the browser and only their filenames are saved in messages.
The current report export contains the sample candidate summary.

## Verification

The production build type-checks the application. To check the Codex integration, start a new thread and ask Codex to remember a unique synthetic label. Wait for the streamed reply to complete, restart Next.js, and ask for the label in the same thread without including it in the follow-up. The frontend can be evaluated without any Python or GPU dependencies.

## Next.js structure and environment

`src/app/layout.tsx` defines the root document, metadata, and global styles. `src/app/page.tsx` renders the interactive workspace in `src/App.tsx`, a client component. Initial HTML uses stable defaults; local threads, preferences, and screen size are restored after hydration, before persistence runs.

Next.js automatically loads `frontend/.env.local`. The existing `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` names work with Next.js without a Vite compatibility mapping. The application uses Supabase Auth for identity and sessions. Research tables are not wired in yet. `APP_URL` sets the canonical auth redirect origin. Google sign-in is always offered; its credentials are configured in Supabase. Only publishable browser configuration belongs in `NEXT_PUBLIC_*` variables.

The `frontend/` directory belongs to the parent Auto Research Git repository. Source files, the dependency lockfile, and `.env.example` are eligible to commit. `.next/`, `node_modules/`, local environment files, and generated TypeScript build state are ignored.

## Interactive experiment charts

Overview and Metrics share sample measurements. Drag across the plot to zoom,
or use the From/To sliders. Switch between runtime and speedup, toggle the
baseline and best-so-far overlays, and pin two candidates to compare runtimes.
Best-so-far includes earlier experiments even when the viewport is zoomed.
Arrow keys and Home/End inspect points; Enter/Space pins them; Escape clears
comparison. Metrics also includes clickable speedup bars. The data-table toggle
provides accessible measurements and comparison buttons. CSV export contains
only the visible range and labels every row as sample data. Reset restores the
full range and clears comparisons. No backend or W&B requests are made.
