import {
  query,
  type Options,
  type SDKMessage,
  type SDKResultMessage,
} from "@anthropic-ai/claude-agent-sdk";
import type { AgentTurn } from "../agents/stream-route";
import { createClaudeTextMapper } from "./events";

/** Tools the chat assistant may use: read the repository, never change it. */
const readOnlyTools = ["Read", "Glob", "Grep"];

const appendSystemPrompt = [
  "You are the assistant inside the Auto Research workspace, a tool for",
  "automated GPU kernel research. The working directory is the project",
  "repository: its wiki holds paper analyses and design artifacts, and",
  "AGENTS.md explains how to cite them. Answer in this chat. You can read",
  "files to ground an answer, but you cannot edit files or run commands here.",
].join(" ");

function resultFailure(result: SDKResultMessage, lastText: string) {
  if (/not logged in|please run \/login/i.test(lastText)) {
    return (
      "Claude Code is not signed in on this machine. Run `claude login`, or set " +
      "CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY in frontend/.env.local, then restart Next.js."
    );
  }
  if (result.subtype === "error_max_turns") {
    return "Claude stopped after reaching the turn limit for one reply.";
  }
  if (result.subtype === "error_max_budget_usd") {
    return "Claude stopped after reaching the spending limit for one reply.";
  }
  const detail =
    result.subtype === "success" ? "" : (result.errors ?? []).join(" ");
  return (
    detail ||
    "Claude could not complete the response. Check the local Claude Code sign-in and try again."
  );
}

function isMissingSession(error: unknown) {
  return (
    error instanceof Error &&
    /no conversation found|session.*not found|could not (find|resume)/i.test(
      error.message,
    )
  );
}

/**
 * Run one chat turn through the Claude Agent SDK, which drives the Claude
 * Code harness as a subprocess. The conversation lives in Claude Code's own
 * session store; this app saves only the session ID per workspace thread
 * and resumes it on the next turn. A saved session that no longer exists
 * (for example after Claude Code's retention window) starts a fresh one.
 */
export const runClaudeTurn: AgentTurn = async ({
  cwd,
  input,
  savedId,
  signal,
  onText,
  save,
}) => {
  signal.throwIfAborted();

  async function attempt(resume: string | undefined) {
    const controller = new AbortController();
    const stop = () => controller.abort();
    signal.addEventListener("abort", stop, { once: true });
    const toEvents = createClaudeTextMapper();
    const options: Options = {
      cwd,
      abortController: controller,
      includePartialMessages: true,
      permissionMode: "default",
      tools: readOnlyTools,
      allowedTools: readOnlyTools,
      settingSources: ["project"],
      maxTurns: 30,
      resume,
      systemPrompt: {
        type: "preset",
        preset: "claude_code",
        append: appendSystemPrompt,
      },
    };
    if (process.env.AUTORESEARCH_CLAUDE_MODEL) {
      options.model = process.env.AUTORESEARCH_CLAUDE_MODEL;
    }
    if (process.env.AUTORESEARCH_CLAUDE_BIN) {
      options.pathToClaudeCodeExecutable = process.env.AUTORESEARCH_CLAUDE_BIN;
    }

    let sessionId: string | undefined;
    let lastText = "";
    try {
      for await (const message of query({ prompt: input, options })) {
        signal.throwIfAborted();
        if (!sessionId && message.session_id) {
          sessionId = message.session_id;
          if (sessionId !== resume) await save(sessionId);
        }
        for (const event of toEvents(message)) {
          if (event.type === "message") lastText = event.text;
          onText(event);
        }
        if (message.type === "result") {
          if (message.subtype !== "success" || message.is_error) {
            throw new Error(resultFailure(message, lastText));
          }
          return;
        }
      }
      throw new Error("Claude disconnected before completing the response.");
    } finally {
      signal.removeEventListener("abort", stop);
      controller.abort();
    }
  }

  try {
    await attempt(savedId);
  } catch (error) {
    signal.throwIfAborted();
    if (savedId && isMissingSession(error)) {
      await attempt(undefined);
      return;
    }
    throw error instanceof Error
      ? error
      : new Error(
          "Could not start Claude. Check the Claude Agent SDK install.",
        );
  }
};

export type { SDKMessage };
