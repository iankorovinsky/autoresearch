# Auto Research frontend

A local research workspace inspired by the Silico screen recording: a compact project sidebar, a research conversation, and an inspectable experiment panel. Built with Next.js App Router, React, TypeScript, and Lucide icons. The visual system uses warm white surfaces, sage accents, and restrained typography.

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

## Included

- Project and thread navigation, thread search (Cmd/Ctrl+K), and new threads (Cmd/Ctrl+N).
- Research composer with Shift+Enter for newlines, text/code attachments (up to 1 MB each), a model preference, auto-research toggle, and cancellable simulated replies.
- Overview, Timeline, Plan, Workers, Compute, Metrics, and Code views.
- Interactive runtime chart, candidate selection, code excerpt copying, and Markdown report downloads.
- Expandable experiment panel, collapsible sidebar, responsive stacked layout, keyboard focus states, and focus-contained dialogs.
- Supabase email/password authentication, verification resend, password recovery, cookie sessions, and protected workspace access.
- Threads and model/GPU preferences saved in this browser using localStorage, scoped by signed-in user ID.

## Demo boundary and backend integration

**This is a frontend prototype.** All seeded experiment measurements, workers, timelines, and code excerpts are illustrative. Authentication connects to Supabase, but research data remains local. The app does not call an LLM, provision GPUs, query the research database, or execute kernels. New messages receive a clearly labeled simulated response; new threads display an empty experiment state instead of fabricated completed runs. Model, GPU, and auto-research controls are preferences for the future backend.

`src/data.ts` contains the typed thread/message models, sample threads, candidate results, and illustrative CUDA excerpt. Start the backend integration there, then replace the simulated response in `App.tsx` with an API/streaming adapter. Add server-supplied experiment IDs, run status, events, metrics, workers, and candidate artifacts. Credentials and database access should live on the server.

Attachments are read locally and displayed as context chips; only their filenames are retained in saved messages. No file content is transmitted. Report export currently exports the sample candidate summary. Fonts load from Google Fonts, with system fallbacks if offline.

Local browser storage is for prototype convenience, not a durable database. To reset the prototype, remove that user’s `ar:<user-id>:threads`, `ar:<user-id>:model`, and `ar:<user-id>:gpu` keys from this site's browser storage.

## Verification

The production build type-checks the application. Browser smoke checks covered all seven experiment tabs, creating and sending a new thread, the simulated response and empty experiment state, thread persistence after reload, and desktop/phone layouts. The frontend can be evaluated without any Python or GPU dependencies.

## Next.js structure and environment

`src/app/layout.tsx` defines the root document, metadata, and global styles. `src/app/page.tsx` renders the interactive workspace in `src/App.tsx`, a client component. Initial HTML uses stable defaults; local threads, preferences, and screen size are restored after hydration, before persistence runs.

Next.js automatically loads `frontend/.env.local`. The existing `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` names work with Next.js without a Vite compatibility mapping. The application uses Supabase Auth for identity and sessions. Research tables are not wired in yet. `APP_URL` sets the canonical auth redirect origin. Google sign-in is always offered; its credentials are configured in Supabase. Only publishable browser configuration belongs in `NEXT_PUBLIC_*` variables.

The `frontend/` directory belongs to the parent Auto Research Git repository. Source files, the dependency lockfile, and `.env.example` are eligible to commit. `.next/`, `node_modules/`, local environment files, and generated TypeScript build state are ignored.
