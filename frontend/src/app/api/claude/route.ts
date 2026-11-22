import { createAgentRoute } from "../../../lib/agents/stream-route";
import { kernelToolsEnabled, runClaudeTurn } from "../../../lib/claude/agent";

export const runtime = "nodejs";

export const POST = createAgentRoute({
  label: "Claude",
  enabledEnv: "AUTORESEARCH_CLAUDE_ENABLED",
  stateDir: ".claude-local",
  run: runClaudeTurn,
  // Kernel evaluation and profiling run on a remote GPU and can take many minutes.
  timeoutMs: kernelToolsEnabled() ? 45 * 60_000 : 300_000,
});
