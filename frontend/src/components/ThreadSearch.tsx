"use client";

import {
  useCallback,
  useDeferredValue,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { defaultRangeExtractor, useVirtualizer } from "@tanstack/react-virtual";
import type { Range } from "@tanstack/react-virtual";
import {
  ArrowDown,
  ArrowUp,
  CornerDownLeft,
  MessageSquare,
  Search,
  X,
} from "lucide-react";
import type { Thread } from "../data";
import { createSearchIndex, searchThreads } from "../lib/search/thread-search";
import type { SearchResult } from "../lib/search/thread-search";
import "./thread-search.css";

function Highlight({ text, query }: { text: string; query: string }) {
  const tokens = query
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((token) => token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!tokens.length) return text;
  const pattern = new RegExp(`(${tokens.join("|")})`, "gi");
  return text
    .split(pattern)
    .map((part, index) => (index % 2 ? <mark key={index}>{part}</mark> : part));
}

export default function ThreadSearch({
  threads,
  onSelect,
  onClose,
}: {
  threads: Thread[];
  onSelect: (result: SearchResult) => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const deferredQuery = useDeferredValue(query);
  const index = useMemo(() => createSearchIndex(threads), [threads]);
  const results = useMemo(
    () => searchThreads(index, deferredQuery),
    [index, deferredQuery],
  );
  const [cursor, setCursor] = useState(0);
  const selected = Math.min(cursor, Math.max(0, results.length - 1));
  const stale = query !== deferredQuery;
  const dialogRef = useRef<HTMLDialogElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const id = useId();
  const virtualizer = useVirtualizer({
    count: results.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 76,
    getItemKey: (index) => results[index].id,
    overscan: 4,
    // Keep the active option in the DOM for aria-activedescendant even when
    // keyboard navigation jumps beyond the current visible window.
    rangeExtractor: useCallback(
      (range: Range) =>
        Array.from(new Set([...defaultRangeExtractor(range), selected]))
          .filter((i) => i < results.length)
          .sort((a, b) => a - b),
      [selected, results.length],
    ),
  });
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const dialog = dialogRef.current;
    dialog?.showModal();
    inputRef.current?.focus();
    return () => {
      dialog?.close();
      previous?.focus();
    };
  }, []);
  useEffect(() => {
    if (results.length) virtualizer.scrollToIndex(selected, { align: "auto" });
  }, [selected, results.length, virtualizer]);
  function updateQuery(value: string) {
    setQuery(value);
    setCursor(0);
    scrollRef.current?.scrollTo({ top: 0 });
  }
  function pick(result: SearchResult) {
    if (!stale) onSelect(result);
  }
  return (
    <dialog
      ref={dialogRef}
      className="thread-search-dialog"
      aria-labelledby={`${id}-title`}
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          const rect = event.currentTarget.getBoundingClientRect();
          if (
            event.clientX < rect.left ||
            event.clientX > rect.right ||
            event.clientY < rect.top ||
            event.clientY > rect.bottom
          )
            onClose();
        }
      }}
    >
      <h2 id={`${id}-title`} className="search-sr-only">
        Search your workspace
      </h2>
      <div className="workspace-search-input">
        <Search size={20} />
        <input
          ref={inputRef}
          aria-label="Search threads and messages"
          role="combobox"
          aria-autocomplete="list"
          aria-expanded="true"
          aria-controls={`${id}-results`}
          aria-activedescendant={
            results.length ? `${id}-option-${selected}` : undefined
          }
          value={query}
          placeholder="Search threads and messages…"
          autoComplete="off"
          onChange={(event) => updateQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.nativeEvent.isComposing) return;
            if (event.key === "Escape") {
              event.preventDefault();
              event.stopPropagation();
              onClose();
              return;
            }
            if (event.key === "Tab") return;
            if (event.key === "ArrowDown" || event.key === "ArrowUp") {
              event.preventDefault();
              if (!stale && results.length)
                setCursor(
                  (current) =>
                    (Math.min(current, results.length - 1) +
                      (event.key === "ArrowDown" ? 1 : -1) +
                      results.length) %
                    results.length,
                );
            }
            if (event.key === "Enter") {
              event.preventDefault();
              if (!stale && results[selected]) pick(results[selected]);
            }
          }}
        />
        {query && (
          <button
            className="search-clear"
            aria-label="Clear search"
            onClick={() => {
              updateQuery("");
              inputRef.current?.focus();
            }}
          >
            <X size={14} />
          </button>
        )}
        <button
          className="search-escape"
          aria-label="Close search"
          onClick={onClose}
        >
          esc
        </button>
      </div>
      <div className="search-results-heading">
        <span>{query.trim() ? "SEARCH RESULTS" : "YOUR THREADS"}</span>
        <span role="status" aria-live="polite">
          {stale
            ? "Searching…"
            : `${results.length} ${results.length === 1 ? "result" : "results"}`}
        </span>
      </div>
      <div
        ref={scrollRef}
        id={`${id}-results`}
        className="workspace-search-results"
        role="listbox"
        aria-label="Search results"
        aria-busy={stale}
        style={{
          height: results.length ? Math.min(380, results.length * 76) : 150,
        }}
      >
        {results.length ? (
          <div
            style={{
              height: virtualizer.getTotalSize(),
              position: "relative",
              width: "100%",
            }}
          >
            {virtualizer.getVirtualItems().map((item) => {
              const result = results[item.index];
              return (
                <button
                  type="button"
                  role="option"
                  tabIndex={-1}
                  aria-selected={item.index === selected}
                  aria-posinset={item.index + 1}
                  aria-setsize={results.length}
                  id={`${id}-option-${item.index}`}
                  key={item.key}
                  className={`workspace-search-result ${item.index === selected ? "selected" : ""}`}
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    height: item.size,
                    transform: `translateY(${item.start}px)`,
                  }}
                  onMouseDown={(event) => event.preventDefault()}
                  onMouseEnter={() => setCursor(item.index)}
                  onClick={() => pick(result)}
                  disabled={stale}
                >
                  <span className={`search-result-icon ${result.kind}`}>
                    <MessageSquare size={17} />
                  </span>
                  <span className="search-result-copy">
                    <strong>
                      <Highlight text={result.title} query={deferredQuery} />
                    </strong>
                    <span>
                      <Highlight
                        text={result.snippet || result.project}
                        query={deferredQuery}
                      />
                    </span>
                    <small>
                      {result.project}
                      <i />{" "}
                      {result.kind === "thread"
                        ? "Thread"
                        : `Message ${(result.messageIndex ?? 0) + 1}`}
                    </small>
                  </span>
                  <CornerDownLeft className="search-result-enter" size={15} />
                </button>
              );
            })}
          </div>
        ) : (
          <div className="search-no-results">
            <Search size={25} />
            <strong>No matches found</strong>
            <p>Try a different title, topic, or phrase from a message.</p>
          </div>
        )}
      </div>
      <footer className="workspace-search-footer">
        <span>
          <kbd>
            <ArrowUp size={10} />
          </kbd>
          <kbd>
            <ArrowDown size={10} />
          </kbd>{" "}
          navigate{" "}
          <kbd>
            <CornerDownLeft size={10} />
          </kbd>{" "}
          open
        </span>
        <span>Searches this local workspace</span>
      </footer>
    </dialog>
  );
}
