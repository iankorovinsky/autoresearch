import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { runCodexTurn } from "../../../lib/codex/app-server";
import { createClient } from "../../../lib/supabase/server";

export const runtime = "nodejs";

const activeThreads = new Set<string>();
const maxInputLength = 64 * 1024;

export async function POST(request: Request) {
  if (process.env.AUTORESEARCH_CODEX_ENABLED !== "1") {
    return new Response(
      "Local Codex is disabled. Set AUTORESEARCH_CODEX_ENABLED=1 and restart the app.",
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
    return new Response("Codex is available from the local app only.", {
      status: 403,
    });
  }

  const supabase = await createClient();
  const {
    data: { user },
    error,
  } = await supabase.auth.getUser();
  if (error || !user?.email_confirmed_at) {
    return new Response("Sign in to Autoresearch to use Codex.", {
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

  const stateDir = path.resolve(".codex-local");
  const threadFile = path.join(stateDir, key);
  const cancel = new AbortController();
  const signal = AbortSignal.any([
    request.signal,
    cancel.signal,
    AbortSignal.timeout(300_000),
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
        const savedThreadId = await readFile(threadFile, "utf8").catch(
          (error: NodeJS.ErrnoException) => {
            if (error.code !== "ENOENT") throw error;
            return undefined;
          },
        );
        await runCodexTurn({
          cwd: path.resolve(".."),
          input,
          savedThreadId,
          signal,
          onText: emit,
          async saveThread(id) {
            await mkdir(stateDir, { recursive: true, mode: 0o700 });
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
              ? "The Codex response timed out."
              : error instanceof Error && !(error as NodeJS.ErrnoException).code
                ? error.message
                : "Could not access the local Codex thread state.",
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
}
