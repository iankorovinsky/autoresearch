import { createAgentRoute } from "../../../lib/agents/stream-route";
import { runClaudeTurn } from "../../../lib/claude/agent";

export const runtime = "nodejs";

export const POST = createAgentRoute({
  label: "Claude",
  enabledEnv: "AUTORESEARCH_CLAUDE_ENABLED",
  stateDir: ".claude-local",
  run: runClaudeTurn,
});
