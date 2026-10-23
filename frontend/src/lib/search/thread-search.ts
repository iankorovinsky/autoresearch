import { rankItem, rankings } from "@tanstack/match-sorter-utils";
import type { Thread } from "../../data";

export type SearchDocument = {
  id: string;
  threadId: string;
  title: string;
  project: string;
  kind: "thread" | "message";
  text: string;
  normalized: string;
  messageIndex?: number;
};
export type SearchResult = SearchDocument & { snippet: string; score: number };
export function normalizeSearch(value: string) {
  return value
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}

// Built only when threads change, never on individual keystrokes. The index
// belongs to this workspace instance and is not shared between signed-in users.
export function createSearchIndex(threads: Thread[]): SearchDocument[] {
  return threads.flatMap((thread) => [
    {
      id: `thread:${thread.id}`,
      threadId: thread.id,
      title: thread.title,
      project: thread.project,
      kind: "thread" as const,
      text: thread.task,
      normalized: normalizeSearch(
        `${thread.title} ${thread.task} ${thread.project}`,
      ),
    },
    ...thread.messages.map((message, messageIndex) => ({
      id: `message:${thread.id}:${messageIndex}`,
      threadId: thread.id,
      title: thread.title,
      project: thread.project,
      kind: "message" as const,
      text: message.text,
      normalized: normalizeSearch(message.text),
      messageIndex,
    })),
  ]);
}
function excerpt(text: string, tokens: string[]) {
  const collapsed = text.replace(/\s+/g, " ").trim();
  const normalized = normalizeSearch(collapsed);
  const match =
    tokens
      .map((token) => normalized.indexOf(token))
      .filter((index) => index >= 0)
      .sort((a, b) => a - b)[0] ?? 0;
  const start = Math.max(0, match - 35);
  return `${start ? "…" : ""}${collapsed.slice(start, start + 145)}${collapsed.length > start + 145 ? "…" : ""}`;
}
export function searchThreads(
  index: SearchDocument[],
  query: string,
): SearchResult[] {
  const normalized = normalizeSearch(query);
  if (!normalized)
    return index
      .filter((doc) => doc.kind === "thread")
      .map((doc) => ({ ...doc, snippet: doc.text, score: 0 }));
  const tokens = normalized.split(" ");
  const results: SearchResult[] = [];
  for (const doc of index) {
    let score = 0;
    let passed = true;
    for (const token of tokens) {
      const rank = rankItem(doc.normalized, token, {
        threshold: token.length === 1 ? rankings.CONTAINS : rankings.MATCHES,
      });
      if (!rank.passed) {
        passed = false;
        break;
      }
      score += rank.rank;
    }
    if (!passed) continue;
    // Prefer matching thread titles while retaining precise message matches.
    if (doc.kind === "thread") {
      const titleRank = rankItem(normalizeSearch(doc.title), normalized);
      score += titleRank.passed ? titleRank.rank : 0.25;
    }
    results.push({ ...doc, score, snippet: excerpt(doc.text, tokens) });
  }
  // Stable ties preserve the workspace's existing thread/message order.
  return results.sort((a, b) => b.score - a.score);
}
