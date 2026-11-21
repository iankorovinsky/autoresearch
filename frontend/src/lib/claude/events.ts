import type { AgentText } from "../agents/stream-route";

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
      message: { content: { type: string; text?: string }[] };
    }
  | { type: string; session_id?: string };

/**
 * Convert the main-thread text of Claude's streamed messages into the
 * delta/message events the browser already renders for Codex.
 *
 * Deltas for one assistant message share an item ID derived from the session
 * and a per-session message counter. When the complete assistant message
 * arrives, its text blocks are re-emitted as `message` events under the same
 * IDs so the final text wins over any partial stream, then the counter moves on.
 * Subagent output (non-null `parent_tool_use_id`), thinking, and tool blocks
 * are skipped.
 */
export function createClaudeTextMapper() {
  let turn = 0;
  return function claudeTextEvents(message: ClaudeMessageLike): AgentText[] {
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
      return [
        {
          type: "delta",
          itemId: `${message.session_id}:${turn}:${event.index ?? 0}`,
          text: event.delta.text,
        },
      ];
    }
    if (message.type === "assistant" && "message" in message) {
      if (message.parent_tool_use_id !== null) return [];
      const events: AgentText[] = [];
      message.message.content.forEach((block, index) => {
        if (block.type !== "text") return;
        events.push({
          type: "message",
          itemId: `${message.session_id}:${turn}:${index}`,
          text: block.text ?? "",
        });
      });
      turn += 1;
      return events;
    }
    return [];
  };
}
