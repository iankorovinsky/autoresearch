import { createAgentRoute } from "../../../lib/agents/stream-route";
import { runCodexTurn } from "../../../lib/codex/app-server";

export const runtime = "nodejs";

export const POST = createAgentRoute({
  label: "Codex",
  enabledEnv: "AUTORESEARCH_CODEX_ENABLED",
  stateDir: ".codex-local",
  run: ({ cwd, input, savedId, signal, onText, save }) =>
    runCodexTurn({
      cwd,
      input,
      savedThreadId: savedId,
      signal,
      onText,
      saveThread: save,
    }),
});
