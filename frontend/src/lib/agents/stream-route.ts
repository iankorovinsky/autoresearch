import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { createClient } from "../supabase/server";

/** A piece of assistant text streamed to the browser as NDJSON. */
export type AgentText = {
  type: "delta" | "message";
  itemId: string;
  text: string;
};

/** A short progress note, for example while a long tool call runs. */
export type AgentStatus = {
  type: "status";
  text: string;
};

export type AgentEvent = AgentText | AgentStatus;

export type AgentTurn = (args: {
  /** Directory the agent may read; the repository root in this app. */
  cwd: string;
  input: string;
  /** Provider conversation ID saved by an earlier turn of this thread. */
  savedId?: string;
  signal: AbortSignal;
  onText: (event: AgentEvent) => void;
  /** Persist the provider conversation ID for this workspace thread. */
  save: (id: string) => Promise<void>;
}) => Promise<void>;

const maxInputLength = 64 * 1024;
const defaultTimeoutMs = 300_000;

/**
 * Build the POST handler shared by the local agent routes. Each provider
 * keeps its own in-flight set and its own thread-to-conversation mapping
 * directory, keyed by the signed-in account and workspace thread.
 */
export function createAgentRoute({
  label,
  enabledEnv,
  stateDir,
  run,
  timeoutMs = defaultTimeoutMs,
}: {
  label: string;
  enabledEnv: string;
  stateDir: string;
  run: AgentTurn;
  /** Wall-clock limit for one reply; raise it when turns may run GPU jobs. */
  timeoutMs?: number;
}) {
  const activeThreads = new Set<string>();

  return async function POST(request: Request) {
    if (process.env[enabledEnv] !== "1") {
      return new Response(
        `Local ${label} is disabled. Set ${enabledEnv}=1 and restart the app.`,
        { status: 503 },
      );
    }
    const origin = request.headers.get("origin");
    const host = request.headers.get("host");
    if (
      !host ||
      !/^(127\.0\.0\.1|localhost)(:\d+)?$/.test(host) ||
      origin !== `http://${host}`
    ) {
      return new Response(`${label} is available from the local app only.`, {
        status: 403,
      });
    }

    const supabase = await createClient();
    const {
      data: { user },
      error,
    } = await supabase.auth.getUser();
    if (error || !user?.email_confirmed_at) {
      return new Response(`Sign in to Autoresearch to use ${label}.`, {
        status: 401,
      });
    }
    const body = await request.json().catch(() => null);
    const { threadId, input } = body ?? {};
    if (
      typeof threadId !== "string" ||
      !/^[a-zA-Z0-9-]{1,64}$/.test(threadId) ||
      typeof input !== "string" ||
      !input.trim() ||
      input.length > maxInputLength
    ) {
      return new Response("Provide a prompt of at most 65,536 characters.", {
        status: 400,
      });
    }

    const key = createHash("sha256")
      .update(`${user.id}:${threadId}`)
      .digest("hex");
    if (activeThreads.has(key)) {
      return new Response("A response is already running for this thread.", {
        status: 409,
      });
    }
    activeThreads.add(key);

    const directory = path.resolve(stateDir);
    const threadFile = path.join(directory, key);
    const cancel = new AbortController();
    const signal = AbortSignal.any([
      request.signal,
      cancel.signal,
      AbortSignal.timeout(timeoutMs),
    ]);
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      async start(controller) {
        const emit = (event: object) => {
          if (!cancel.signal.aborted) {
            controller.enqueue(encoder.encode(JSON.stringify(event) + "\n"));
          }
        };
        try {
          const savedId = await readFile(threadFile, "utf8").catch(
            (error: NodeJS.ErrnoException) => {
              if (error.code !== "ENOENT") throw error;
              return undefined;
            },
          );
          await run({
            cwd: path.resolve(".."),
            input,
            savedId: savedId?.trim() || undefined,
            signal,
            onText: emit,
            async save(id) {
              await mkdir(directory, { recursive: true, mode: 0o700 });
              await writeFile(threadFile + ".tmp", id, { mode: 0o600 });
              await rename(threadFile + ".tmp", threadFile);
            },
          });
          emit({ type: "done" });
        } catch (error) {
          if (!request.signal.aborted) {
            emit({
              type: "error",
              message: signal.aborted
                ? `The ${label} response timed out.`
                : error instanceof Error &&
                    !(error as NodeJS.ErrnoException).code
                  ? error.message
                  : `Could not access the local ${label} thread state.`,
            });
          }
        } finally {
          activeThreads.delete(key);
          if (!cancel.signal.aborted) controller.close();
        }
      },
      cancel() {
        cancel.abort();
      },
    });
    return new Response(stream, {
      headers: {
        "Content-Type": "application/x-ndjson; charset=utf-8",
        "Cache-Control": "no-store",
      },
    });
  };
}
