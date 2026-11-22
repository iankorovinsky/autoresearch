import { spawn } from "node:child_process";
import { createInterface } from "node:readline";

type Frame = {
  id?: number | string;
  method?: string;
  result?: {
    account?: { type: string } | null;
    thread?: { id: string };
    turn?: { id: string };
  };
  error?: { message?: string };
  params?: {
    threadId?: string;
    turnId?: string;
    itemId?: string;
    delta?: string;
    item?: { id: string; type: string; text?: string };
    turn?: { id: string; status: string; error?: { message?: string } };
  };
};

export type CodexText = {
  type: "delta" | "message";
  itemId: string;
  text: string;
};

function turnFailure(message = "") {
  if (message.includes("model_not_found")) {
    return "Codex rejected the configured model for your ChatGPT login. Check the CLI version and its model setting.";
  }
  return "Codex could not complete the response. Check the local Codex CLI and try again.";
}

export async function runCodexTurn({
  cwd,
  input,
  savedThreadId,
  signal,
  onText,
  saveThread,
}: {
  cwd: string;
  input: string;
  savedThreadId?: string;
  signal: AbortSignal;
  onText: (event: CodexText) => void;
  saveThread: (id: string) => Promise<void>;
}) {
  signal.throwIfAborted();
  const env = { ...process.env };
  delete env.OPENAI_API_KEY;
  delete env.CODEX_API_KEY;
  delete env.CODEX_INTERNAL_ORIGINATOR_OVERRIDE;

  const child = spawn(
    /* turbopackIgnore: true */ process.env.AUTORESEARCH_CODEX_BIN || "codex",
    ["app-server", "--listen", "stdio://", "-c", 'model_provider="openai"'],
    { cwd, env, stdio: ["pipe", "pipe", "ignore"] },
  );
  let launchFailed = false;
  child.on("error", () => {
    launchFailed = true;
  });
  child.stdin.on("error", () => child.kill());
  const exited = new Promise<void>((resolve) =>
    child.once("close", () => resolve()),
  );
  const lines = createInterface({ input: child.stdout });
  const frames = lines[Symbol.asyncIterator]();
  const stop = () => {
    child.kill();
  };
  signal.addEventListener("abort", stop, { once: true });

  function send(method: string, params?: object, id?: number) {
    signal.throwIfAborted();
    child.stdin.write(JSON.stringify({ method, params, id }) + "\n");
  }

  async function nextFrame(): Promise<Frame> {
    const next = await frames.next();
    signal.throwIfAborted();
    if (next.done) {
      await exited;
      throw new Error(
        launchFailed
          ? "Could not start Codex. Check AUTORESEARCH_CODEX_BIN and your PATH."
          : "Codex disconnected before completing the response.",
      );
    }
    return JSON.parse(next.value) as Frame;
  }

  async function request(id: number, method: string, params: object) {
    send(method, params, id);
    while (true) {
      const frame = await nextFrame();
      if (frame.id !== id) continue;
      if (frame.error || !frame.result) {
        throw new Error(turnFailure(frame.error?.message));
      }
      return frame.result;
    }
  }

  try {
    await request(1, "initialize", {
      clientInfo: { name: "autoresearch", version: "0.1.0" },
    });
    send("initialized");
    const { account } = await request(2, "account/read", {
      refreshToken: false,
    });
    if (account?.type !== "chatgpt") {
      throw new Error(
        "Sign in to the local Codex CLI with ChatGPT to use your plan.",
      );
    }

    const settings = { cwd, sandbox: "read-only", approvalPolicy: "never" };
    const { thread } = savedThreadId
      ? await request(3, "thread/resume", {
          ...settings,
          threadId: savedThreadId,
          excludeTurns: true,
        })
      : await request(3, "thread/start", settings);
    if (!thread?.id) throw new Error("Codex did not return a thread.");

    send(
      "turn/start",
      {
        threadId: thread.id,
        input: [{ type: "text", text: input }],
      },
      4,
    );

    // Notifications can arrive before the turn/start response.
    while (true) {
      const frame = await nextFrame();
      if (frame.id === 4 && frame.error) {
        throw new Error(turnFailure(frame.error.message));
      }
      const params = frame.params;
      if (params?.threadId !== thread.id) continue;

      if (
        frame.method === "item/agentMessage/delta" &&
        params.itemId &&
        params.delta
      ) {
        onText({ type: "delta", itemId: params.itemId, text: params.delta });
      } else if (
        frame.method === "item/completed" &&
        params.item?.type === "agentMessage"
      ) {
        onText({
          type: "message",
          itemId: params.item.id,
          text: params.item.text ?? "",
        });
      } else if (frame.method === "turn/completed") {
        if (params.turn?.status !== "completed") {
          throw new Error(turnFailure(params.turn?.error?.message));
        }
        if (!savedThreadId) await saveThread(thread.id);
        return;
      }
    }
  } finally {
    signal.removeEventListener("abort", stop);
    lines.close();
    child.stdin.end();
    const timeout = setTimeout(() => child.kill(), 2000);
    try {
      await exited;
    } finally {
      clearTimeout(timeout);
    }
  }
}
