import type { AgentEvent } from "../agents/stream-route";

/** Progress text shown while one of the kernel tools runs on a remote GPU. */
const toolStatus: Record<string, string> = {
  mcp__autoresearch__evaluate_kernel:
    "Evaluating the kernel on a Modal GPU (correctness, then timing)…",
  mcp__autoresearch__profile_kernel:
    "Profiling the kernel with Nsight Compute on a Modal GPU…",
};

/**
 * The subset of Claude Agent SDK messages this mapper reads. Kept structural
 * so the pure mapping can be unit-tested without loading the SDK bundle.
 */
export type ClaudeMessageLike =
  | {
      type: "stream_event";
      parent_tool_use_id: string | null;
      session_id: string;
      event: {
        type: string;
        index?: number;
        delta?: { type: string; text?: string };
      };
    }
  | {
      type: "assistant";
      parent_tool_use_id: string | null;
      session_id: string;
      message: { content: { type: string; text?: string; name?: string }[] };
    }
  | { type: string; session_id?: string };

/**
 * Convert the main-thread text of Claude's streamed messages into the
 * delta/message events the browser renders for streamed replies.
 *
 * Deltas for one assistant message share an item ID derived from the session,
 * a per-session message counter, and the API content block index. Claude Code
 * emits the completed assistant message split into one message per content
 * block, so a text block's index there (0) can differ from its index in the
 * stream (for example 1 after a thinking block). The complete message is
 * therefore re-emitted under the IDs of the text blocks streamed for this
 * turn, in order, so the final text replaces the partial stream instead of
 * appearing next to it; then the counter moves on. Subagent output (non-null
 * `parent_tool_use_id`) and thinking are skipped. A call to one of the kernel
 * tools becomes a `status` event so the browser can say what is running.
 */
export function createClaudeTextMapper() {
  let turn = 0;
  /** Item IDs of text blocks streamed since the last complete message. */
  let streamed: string[] = [];
  return function claudeTextEvents(message: ClaudeMessageLike): AgentEvent[] {
    if (message.type === "stream_event" && "event" in message) {
      if (message.parent_tool_use_id !== null) return [];
      const { event } = message;
      if (
        event.type !== "content_block_delta" ||
        event.delta?.type !== "text_delta" ||
        !event.delta.text
      ) {
        return [];
      }
      const itemId = `${message.session_id}:${turn}:${event.index ?? 0}`;
      if (!streamed.includes(itemId)) streamed.push(itemId);
      return [{ type: "delta", itemId, text: event.delta.text }];
    }
    if (message.type === "assistant" && "message" in message) {
      if (message.parent_tool_use_id !== null) return [];
      const events: AgentEvent[] = [];
      let textBlock = 0;
      message.message.content.forEach((block, index) => {
        if (block.type === "tool_use") {
          const text = block.name ? toolStatus[block.name] : undefined;
          if (text) events.push({ type: "status", text });
          return;
        }
        if (block.type !== "text") return;
        events.push({
          type: "message",
          itemId:
            streamed[textBlock] ?? `${message.session_id}:${turn}:${index}`,
          text: block.text ?? "",
        });
        textBlock += 1;
      });
      turn += 1;
      streamed = [];
      return events;
    }
    return [];
  };
}
