"use client";

import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  Activity,
  ArrowDown,
  ArrowDownToLine,
  ArrowRight,
  ArrowUp,
  BookOpen,
  Check,
  CheckCheck,
  ChevronDown,
  ChevronRight,
  Circle,
  CircleCheck,
  Clock3,
  Code2,
  Cpu,
  Ellipsis,
  ExternalLink,
  FileCode2,
  FileText,
  FlaskConical,
  Folder,
  GitBranch,
  LayoutPanelLeft,
  LoaderCircle,
  Maximize2,
  MessageSquare,
  Microscope,
  PanelLeftClose,
  Play,
  Plus,
  Search,
  Settings2,
  Sparkles,
  Square,
  Terminal,
  X,
  Zap,
} from "lucide-react";
import { initialThreads, experiments, kernelCode } from "./data";
import type { Thread, Message } from "./data";
const tabs = [
  "Overview",
  "Timeline",
  "Plan",
  "Workers",
  "Compute",
  "Metrics",
  "Code",
] as const;
type Tab = (typeof tabs)[number];
function Brand({ small = false }: { small?: boolean }) {
  return (
    <span className={`brand-mark ${small ? "small" : ""}`} aria-hidden="true">
      <i />
      <i />
      <i />
    </span>
  );
}
function IconButton({
  label,
  children,
  onClick,
  className = "",
}: {
  label: string;
  children: ReactNode;
  onClick: () => void;
  className?: string;
}) {
  return (
    <button
      type="button"
      className={`icon-button ${className}`}
      title={label}
      aria-label={label}
      onClick={onClick}
    >
      {children}
    </button>
  );
}
function loadPreference(key: string, fallback: string) {
  try {
    return localStorage.getItem(key) || fallback;
  } catch {
    return fallback;
  }
}
function loadThreads(): Thread[] {
  try {
    const saved = localStorage.getItem("ar-threads-v1");
    return saved ? JSON.parse(saved) : initialThreads;
  } catch {
    return initialThreads;
  }
}
function Chart({ large = false }: { large?: boolean }) {
  const [point, setPoint] = useState<number | null>(null);
  const vals = [
    2.99, 2.73, 2.51, 2.12, 2.24, 1.93, 1.67, 1.79, 1.48, 1.61, 1.38, 1.24,
  ];
  return (
    <div className={`chart ${large ? "large" : ""}`}>
      <div className="chart-key">
        <span>
          <i className="dot olive" /> Candidate
        </span>
        <span>
          <i className="dash" /> PyTorch baseline
        </span>
        <span className="chart-unit">Runtime (ms) ↓</span>
      </div>
      <svg
        viewBox="0 0 620 210"
        role="img"
        aria-label="Candidate runtime improves from 2.99 milliseconds to 1.24 over 12 experiments, against a 2.99 millisecond PyTorch baseline"
        onMouseLeave={() => setPoint(null)}
      >
        {[3, 2.5, 2, 1.5, 1].map((n, i) => (
          <g key={n}>
            <text x="0" y={25 + i * 38}>
              {n.toFixed(1)}
            </text>
            <line
              className="grid-line"
              x1="35"
              x2="602"
              y1={21 + i * 38}
              y2={21 + i * 38}
            />
          </g>
        ))}
        <line className="baseline" x1="35" y1="23" x2="602" y2="23" />
        <defs>
          <linearGradient id="chart-fill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#7b8558" stopOpacity=".18" />
            <stop offset="100%" stopColor="#7b8558" stopOpacity=".015" />
          </linearGradient>
        </defs>
        <path
          d={`M 35 175 L ${vals.map((v, i) => `${35 + i * 51.2} ${21 + (3 - v) * 76}`).join(" L ")} L 598 175 Z`}
          fill="url(#chart-fill)"
        />
        <polyline
          points={vals
            .map((v, i) => `${35 + i * 51.2},${21 + (3 - v) * 76}`)
            .join(" ")}
          fill="none"
          stroke="#7d8657"
          strokeWidth="2.5"
          strokeLinejoin="round"
        />
        {vals.map((v, i) => (
          <g key={i} onMouseEnter={() => setPoint(i)}>
            <circle
              cx={35 + i * 51.2}
              cy={21 + (3 - v) * 76}
              r="13"
              fill="transparent"
            />
            <circle
              cx={35 + i * 51.2}
              cy={21 + (3 - v) * 76}
              r={point === i || i === 11 ? 4.5 : 3}
              fill={i === 11 ? "#7d8657" : "#fff"}
              stroke="#7d8657"
              strokeWidth="1.6"
            />
          </g>
        ))}
        {[1, 3, 6, 9, 12].map((n) => (
          <text key={n} x={35 + (n - 1) * 51.2} y="201" textAnchor="middle">
            {n.toString().padStart(2, "0")}
          </text>
        ))}
        {point !== null && (
          <g>
            <rect
              x={Math.min(495, 35 + point * 51.2)}
              y="60"
              width="106"
              height="26"
              rx="5"
              fill="#30382e"
            />
            <text
              className="chart-tooltip"
              x={Math.min(495, 35 + point * 51.2) + 53}
              y="77"
              textAnchor="middle"
            >
              Run {point + 1} · {vals[point]} ms
            </text>
          </g>
        )}
      </svg>
      <div className="chart-caption">Experiment</div>
    </div>
  );
}
export default function App() {
  const [threads, setThreads] = useState<Thread[]>(initialThreads);
  const [restored, setRestored] = useState(false);
  const [activeId, setActiveId] = useState("matmul");
  const active = threads.find((t) => t.id === activeId) ?? threads[0];
  const [tab, setTab] = useState<Tab>("Overview");
  const [sidebar, setSidebar] = useState(true);
  const [panel, setPanel] = useState(true);
  const [expanded, setExpanded] = useState(false);
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState(false);
  const [draft, setDraft] = useState("");
  const [auto, setAuto] = useState(true);
  const [running, setRunning] = useState(false);
  const [modal, setModal] = useState<"settings" | "experiments" | null>(null);
  const [toast, setToast] = useState("");
  const [attachments, setAttachments] = useState<
    { name: string; content: string }[]
  >([]);
  const [model, setModel] = useState("Claude Sonnet");
  const [modelOpen, setModelOpen] = useState(false);
  const [gpu, setGpu] = useState("NVIDIA T4");
  const [selectedExperiment, setSelectedExperiment] = useState("EXP-012");
  const dialogRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    setThreads(loadThreads());
    setModel(loadPreference("ar-model", "Claude Sonnet"));
    setGpu(loadPreference("ar-gpu", "NVIDIA T4"));
    setSidebar(window.innerWidth > 700);
    setRestored(true);
  }, []);
  useEffect(() => {
    if (!restored) return;
    try {
      localStorage.setItem("ar-threads-v1", JSON.stringify(threads));
    } catch {
      setToast(
        "Browser storage is unavailable; changes last for this session.",
      );
    }
  }, [threads, restored]);
  useEffect(() => {
    if (!restored) return;
    try {
      localStorage.setItem("ar-model", model);
      localStorage.setItem("ar-gpu", gpu);
    } catch {
      /* Preferences remain in session memory. */
    }
  }, [model, gpu, restored]);
  useEffect(() => {
    if (toast) {
      const t = setTimeout(() => setToast(""), 3500);
      return () => clearTimeout(t);
    }
  }, [toast]);
  useEffect(() => {
    if (active.messages.length > 1 || running)
      bottomRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
  }, [active.messages, running]);
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setSidebar(true);
        setSearch((s) => !s);
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "n") {
        e.preventDefault();
        newThread();
      }
      if (e.key === "Escape") {
        setModal(null);
        setModelOpen(false);
        setSearch(false);
      }
    };
    window.addEventListener("keydown", handler);
    return () => {
      window.removeEventListener("keydown", handler);
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);
  useEffect(() => {
    if (!modal) return;
    const previousFocus = document.activeElement as HTMLElement | null;
    const dialog = dialogRef.current;
    const focusable = () =>
      Array.from(
        dialog?.querySelectorAll<HTMLElement>(
          'button, select, input, [tabindex="0"]',
        ) ?? [],
      );
    focusable()[0]?.focus();
    const trapFocus = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const elements = focusable();
      const first = elements[0];
      const last = elements[elements.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", trapFocus);
    return () => {
      document.removeEventListener("keydown", trapFocus);
      previousFocus?.focus();
    };
  }, [modal]);
  function newThread() {
    const id = crypto.randomUUID();
    setThreads((ts) => [
      {
        id,
        title: "New research thread",
        task: "New experiment",
        project: "Kernel optimization",
        messages: [],
      },
      ...ts,
    ]);
    setActiveId(id);
    setExpanded(false);
    setPanel(false);
    setDraft("");
    setAttachments([]);
    setTimeout(() => inputRef.current?.focus(), 50);
  }
  function send() {
    if (!draft.trim() || running) return;
    const id = active.id;
    const message: Message = {
      role: "user",
      text: draft.trim(),
      files: attachments.map((f) => f.name),
    };
    setThreads((ts) =>
      ts.map((t) =>
        t.id === id
          ? {
              ...t,
              title: t.messages.length ? t.title : draft.trim().slice(0, 45),
              messages: [...t.messages, message],
            }
          : t,
      ),
    );
    setDraft("");
    setAttachments([]);
    setRunning(true);
    timerRef.current = setTimeout(() => {
      setThreads((ts) =>
        ts.map((t) =>
          t.id === id
            ? {
                ...t,
                messages: [
                  ...t.messages,
                  {
                    role: "assistant",
                    text:
                      "Your research request has been added to this local workspace. " +
                      (message.files?.length
                        ? `Attached context: ${message.files.join(", ")}. `
                        : "") +
                      "The next step is to establish a baseline, generate candidates, and validate correctness before comparing performance. Connect the research backend to execute this plan; the experiment results shown here are sample data.",
                  },
                ],
              }
            : t,
        ),
      );
      setRunning(false);
      setPanel(true);
    }, 1400);
  }
  function exportReport() {
    if (!initialThreads.some((thread) => thread.id === active.id)) {
      setToast("No completed experiment to export yet");
      return;
    }
    const blob = new Blob(
      [
        `# Auto Research — ${active.task}\n\nSample report — demo data, no evaluations were executed.\n\nBest speedup: 2.41x\nRuntime: 1.24 ms\nBaseline: 2.99 ms\nCorrectness: 5/5\n\n${experiments.map((e) => `${e.id}: ${e.name} — ${e.time} ms (${e.speed}x)`).join("\n")}`,
      ],
      { type: "text/markdown" },
    );
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "autoresearch-report.md";
    a.click();
    URL.revokeObjectURL(a.href);
    setToast("Report downloaded");
  }
  const isSeed = initialThreads.some((t) => t.id === active.id);
  return (
    <div
      className={`app ${!sidebar ? "sidebar-hidden" : ""} ${expanded ? "panel-expanded" : ""}`}
    >
      {sidebar && (
        <aside className="sidebar">
          <div className="brand">
            <Brand />
            <span>
              auto<span className="brand-light">research</span>
            </span>
            <IconButton
              label="Collapse sidebar"
              onClick={() => setSidebar(false)}
            >
              <PanelLeftClose size={15} />
            </IconButton>
          </div>
          <button className="new-thread" onClick={newThread}>
            <Plus size={17} />
            <span>New thread</span>
            <kbd>⌘ N</kbd>
          </button>
          <button
            className="nav-link"
            onClick={() => {
              setPanel(true);
              setExpanded(true);
              setTab("Overview");
            }}
          >
            <BookOpen size={16} /> Reports<span className="nav-count">3</span>
          </button>
          <div className="section-label">
            WORKSPACE
            <IconButton
              label="Workspace settings"
              onClick={() => setModal("settings")}
            >
              <Plus size={13} />
            </IconButton>
          </div>
          <button
            className="project selected"
            onClick={() => {
              setActiveId("matmul");
              setPanel(true);
              setExpanded(false);
            }}
          >
            <Folder size={15} />
            <span>Kernel optimization</span>
            <ChevronDown size={13} />
          </button>
          <button
            className="project subproject"
            onClick={() => {
              setActiveId("softmax");
              setPanel(true);
              setExpanded(false);
            }}
          >
            <span className="branch-line" />
            Fused operations
          </button>
          <div className="section-label threads-label">
            THREADS
            <IconButton
              label="Search threads"
              onClick={() => setSearch(!search)}
            >
              <Search size={13} />
            </IconButton>
          </div>
          {search && (
            <div className="search-field">
              <Search size={14} />
              <input
                autoFocus
                aria-label="Search threads"
                placeholder="Search threads…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
          )}
          <div className="thread-list">
            {threads
              .filter((t) =>
                t.title.toLowerCase().includes(query.toLowerCase()),
              )
              .map((t) => (
                <button
                  key={t.id}
                  className={`thread ${active.id === t.id ? "active" : ""}`}
                  onClick={() => {
                    setActiveId(t.id);
                    setExpanded(false);
                    setPanel(true);
                    setDraft("");
                    setAttachments([]);
                  }}
                >
                  <span
                    className={`thread-dot ${t.id === active.id ? "current" : ""}`}
                  />
                  <span>{t.title}</span>
                  {t.id === active.id && <Ellipsis size={14} />}
                </button>
              ))}
            {query &&
              !threads.some((t) =>
                t.title.toLowerCase().includes(query.toLowerCase()),
              ) && <p className="muted empty-search">No matching threads</p>}
          </div>
          <div className="sidebar-bottom">
            <div className="workspace-status">
              <span className="status-light" />
              <span>Local workspace</span>
              <span className="demo-tag">DEMO</span>
            </div>
            <button className="profile" onClick={() => setModal("settings")}>
              <div className="avatar">AR</div>
              <span>
                Research workspace<small>Personal workspace</small>
              </span>
              <Settings2 size={15} />
            </button>
          </div>
        </aside>
      )}
      <main className="main">
        <header className="topbar">
          <div className="breadcrumbs">
            {!sidebar && (
              <IconButton label="Open sidebar" onClick={() => setSidebar(true)}>
                <LayoutPanelLeft size={17} />
              </IconButton>
            )}
            <span>Kernel optimization</span>
            <ChevronRight size={13} />
            <span className="crumb-current">{active.title}</span>
          </div>
          <div className="top-actions">
            <span className="demo-indicator">
              <i /> Demo workspace
            </span>
            <IconButton
              label={panel ? "Hide experiment panel" : "Show experiment panel"}
              onClick={() => {
                setPanel(!panel);
                setExpanded(false);
              }}
            >
              <LayoutPanelLeft size={17} />
            </IconButton>
            <IconButton
              label="Workspace settings"
              onClick={() => setModal("settings")}
            >
              <Ellipsis size={19} />
            </IconButton>
          </div>
        </header>
        <div className={`workspace ${!panel ? "chat-only" : ""}`}>
          {!expanded && (
            <section className="conversation">
              <div className="conversation-heading">
                <span>
                  <MessageSquare size={14} /> Research thread
                </span>
                <span className="subtle">
                  {isSeed ? "12 experiments" : "New session"}
                </span>
              </div>
              <div className="messages">
                {active.messages.length === 0 ? (
                  <div className="welcome">
                    <Brand />
                    <h1>What will you discover?</h1>
                    <p>
                      Start with a question. Build a hypothesis.
                      <br />
                      Let the experiments do the talking.
                    </p>
                    <button
                      onClick={() =>
                        setDraft(
                          "Optimize a matrix multiplication kernel for NVIDIA T4. Compare tile sizes and validate correctness.",
                        )
                      }
                    >
                      <Cpu size={16} /> Optimize a GPU kernel
                      <ArrowRight size={15} />
                    </button>
                    <button
                      onClick={() =>
                        setDraft(
                          "Set up a benchmark suite to compare candidate kernels against the PyTorch baseline.",
                        )
                      }
                    >
                      <Activity size={16} /> Explore a benchmark
                      <ArrowRight size={15} />
                    </button>
                  </div>
                ) : (
                  <>
                    {active.messages.slice(0, 1).map((m, i) => (
                      <div className="user-message" key={i}>
                        <div className="message-eyebrow">
                          YOU<span>Just now</span>
                        </div>
                        <p>{m.text}</p>
                        {m.files?.map((f) => (
                          <span className="file-chip" key={f}>
                            <FileText size={12} />
                            {f}
                          </span>
                        ))}
                      </div>
                    ))}
                    {isSeed && (
                      <div className="assistant-message">
                        <div className="assistant-label">
                          <Brand small />
                          <strong>Auto Research</strong>
                          <span>Research complete</span>
                        </div>
                        <p>
                          I’ll explore tile configurations for{" "}
                          {active.task.toLowerCase()}, using PyTorch as the
                          baseline and verifying each candidate before measuring
                          performance.
                        </p>
                        <button
                          className="activity-disclosure"
                          onClick={() => {
                            setPanel(true);
                            setTab("Timeline");
                          }}
                        >
                          <CheckCheck size={15} />
                          <span>Explored 12 candidates over 4 iterations</span>
                          <ChevronRight size={14} />
                        </button>
                        <p className="result-lead">
                          A faster kernel, with correctness intact.
                        </p>
                        <p>
                          The best candidate uses <code>64 × 64</code> tiling
                          with shared memory, achieving a{" "}
                          <strong>2.41× speedup</strong> over the PyTorch
                          baseline.
                        </p>
                        <div className="findings">
                          <div>
                            <CircleCheck size={15} />
                            <span>All 5 correctness checks passed</span>
                          </div>
                          <div>
                            <Zap size={15} />
                            <span>
                              Runtime reduced from 2.99 ms to{" "}
                              <strong>1.24 ms</strong>
                            </span>
                          </div>
                          <div>
                            <GitBranch size={15} />
                            <span>
                              Best candidate preserved in{" "}
                              <code>candidate_012.py</code>
                            </span>
                          </div>
                        </div>
                        <button
                          className="experiment-card"
                          onClick={() => {
                            setPanel(true);
                            setTab("Overview");
                          }}
                        >
                          <div className="experiment-icon">
                            <FlaskConical size={21} />
                          </div>
                          <div>
                            <strong>{active.task}</strong>
                            <span>
                              <i className="dot olive" /> Completed{" "}
                              <span className="middle-dot">·</span> 12
                              experiments
                            </span>
                          </div>
                          <ArrowRight size={17} />
                          <div className="experiment-card-bottom">
                            <span>
                              Best speedup <strong>2.41×</strong>
                            </span>
                            <span>
                              View experiment <ArrowRight size={12} />
                            </span>
                          </div>
                        </button>
                        <p className="next-step">
                          Next, we could test larger matrix sizes or explore
                          register tiling to push performance further.
                        </p>
                        <div className="message-footer">
                          <span>1m 42s</span>
                          <span>·</span>
                          <span>12 tool calls</span>
                          <IconButton
                            label="Download research report"
                            onClick={exportReport}
                          >
                            <ArrowDownToLine size={13} />
                          </IconButton>
                        </div>
                      </div>
                    )}
                    {active.messages.slice(1).map((m, i) => (
                      <div
                        key={i}
                        className={
                          m.role === "user"
                            ? "user-message followup"
                            : "assistant-message followup"
                        }
                      >
                        <div className="message-eyebrow">
                          {m.role === "user" ? "YOU" : "AUTO RESEARCH"}
                        </div>
                        <p>{m.text}</p>
                        {m.files?.map((f) => (
                          <span className="file-chip" key={f}>
                            <FileText size={12} />
                            {f}
                          </span>
                        ))}
                      </div>
                    ))}
                  </>
                )}
                {running && (
                  <div className="thinking">
                    <LoaderCircle size={15} className="spin" /> Preparing a
                    research plan…
                  </div>
                )}
                <div ref={bottomRef} />
              </div>
              <div className="composer-wrap">
                {isSeed && active.messages.length === 1 && (
                  <button
                    className="suggestion"
                    onClick={() =>
                      setDraft(
                        "Test the best candidate on larger matrix sizes: 2048, 4096, and 8192.",
                      )
                    }
                  >
                    <Plus size={12} /> Test on larger matrices
                    <ArrowRight size={12} />
                  </button>
                )}
                <div className="composer">
                  {attachments.length > 0 && (
                    <div className="attached-files">
                      {attachments.map((f, i) => (
                        <span className="file-chip" key={i}>
                          <FileText size={12} />
                          {f.name}
                          <IconButton
                            label={`Remove ${f.name}`}
                            onClick={() =>
                              setAttachments((fs) =>
                                fs.filter((_, n) => n !== i),
                              )
                            }
                          >
                            <X size={12} />
                          </IconButton>
                        </span>
                      ))}
                    </div>
                  )}
                  <textarea
                    ref={inputRef}
                    rows={2}
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        send();
                      }
                    }}
                    placeholder="Ask a question or start an experiment…"
                    aria-label="Research prompt"
                  />
                  <div className="composer-toolbar">
                    <IconButton
                      label="Attach text or code files"
                      onClick={() => fileRef.current?.click()}
                    >
                      <Plus size={18} />
                    </IconButton>
                    <input
                      ref={fileRef}
                      type="file"
                      accept=".txt,.md,.py,.json,.csv,.cu,.cpp,.js,.ts"
                      multiple
                      hidden
                      onChange={async (e) => {
                        const files = Array.from(e.target.files ?? []);
                        for (const file of files) {
                          if (file.size > 1024 * 1024) {
                            setToast("Choose files smaller than 1 MB");
                            continue;
                          }
                          const content = await file.text();
                          setAttachments((fs) => [
                            ...fs,
                            { name: file.name, content },
                          ]);
                        }
                        e.target.value = "";
                      }}
                    />
                    <button
                      className={`auto-toggle ${auto ? "on" : ""}`}
                      role="switch"
                      aria-checked={auto}
                      onClick={() => setAuto(!auto)}
                    >
                      <span className="toggle-track">
                        <i />
                      </span>
                      Auto research
                    </button>
                    <div className="model-picker">
                      <button
                        className="model-button"
                        onClick={() => setModelOpen(!modelOpen)}
                      >
                        {model}
                        <ChevronDown size={11} />
                      </button>
                      {modelOpen && (
                        <div className="model-menu">
                          {["Claude Sonnet", "Claude Opus", "GPT-5"].map(
                            (m) => (
                              <button
                                key={m}
                                onClick={() => {
                                  setModel(m);
                                  setModelOpen(false);
                                }}
                              >
                                {m}
                                {model === m && <Check size={13} />}
                              </button>
                            ),
                          )}
                          <small>Model preference · demo only</small>
                        </div>
                      )}
                    </div>
                    <button
                      className={`send-button ${running ? "running" : ""}`}
                      disabled={!running && !draft.trim()}
                      aria-label={running ? "Stop response" : "Send message"}
                      onClick={() => {
                        if (running) {
                          if (timerRef.current) clearTimeout(timerRef.current);
                          setRunning(false);
                          setToast("Response stopped");
                        } else send();
                      }}
                    >
                      {running ? <Square size={12} /> : <ArrowUp size={17} />}
                    </button>
                  </div>
                </div>
                <p className="composer-note">
                  A little curiosity goes a long way.
                  <span>Sample data · no compute connected</span>
                </p>
              </div>
            </section>
          )}
          {panel && (
            <section className="experiment-panel">
              <div className="panel-heading">
                <div className="panel-title">
                  <span className="panel-flask">
                    <FlaskConical size={16} />
                  </span>
                  <strong>{active.task}</strong>
                </div>
                <div className="panel-actions">
                  <IconButton
                    label={
                      expanded
                        ? "Restore split view"
                        : "Expand experiment panel"
                    }
                    onClick={() => setExpanded(!expanded)}
                  >
                    <Maximize2 size={14} />
                  </IconButton>
                  <IconButton label="Download report" onClick={exportReport}>
                    <ArrowDownToLine size={15} />
                  </IconButton>
                </div>
              </div>
              <div className="panel-subheading">
                {isSeed ? (
                  <>
                    <span className="complete-badge">
                      <Check size={11} /> Completed
                    </span>
                    <span>EXP-012</span>
                    <span>·</span>
                    <span>Sample experiment</span>
                  </>
                ) : (
                  <span>Awaiting backend connection</span>
                )}
              </div>
              <nav className="tabs" aria-label="Experiment views">
                {tabs.map((t) => (
                  <button
                    key={t}
                    className={tab === t ? "active" : ""}
                    onClick={() => setTab(t)}
                  >
                    {t}
                  </button>
                ))}
              </nav>
              <div className="panel-content" key={`${tab}-${activeId}`}>
                {!isSeed ? (
                  <>
                    <SectionIntro
                      eyebrow="YOUR RESEARCH"
                      title="A new place to explore"
                      description="Your experiment results will appear here once the research backend is connected."
                    />
                    <div className="empty-experiment">
                      <FlaskConical size={32} strokeWidth={1.2} />
                      <h2>Ready for your first experiment</h2>
                      <p>
                        Keep shaping your question in the thread. This workspace
                        will bring together the plan, evaluations, metrics, and
                        code.
                      </p>
                    </div>
                    <DemoNotice />
                  </>
                ) : (
                  <>
                    {tab === "Overview" && (
                      <>
                        <div className="report-eyebrow">
                          <span>EXPERIMENT REPORT</span>
                          <span>Oct 16, 2026</span>
                        </div>
                        <h1>{active.task}</h1>
                        <p className="report-description">
                          Exploring tile sizes and memory access patterns to
                          find a faster, verified {active.task.toLowerCase()}{" "}
                          kernel.
                        </p>
                        <div className="stat-grid">
                          <div className="stat featured">
                            <span>
                              Best speedup
                              <Zap size={13} />
                            </span>
                            <strong>
                              2.41<span>×</span>
                            </strong>
                            <small>vs. PyTorch baseline</small>
                          </div>
                          <div className="stat">
                            <span>
                              Kernel runtime
                              <Clock3 size={13} />
                            </span>
                            <strong>
                              1.24<em>ms</em>
                            </strong>
                            <small>Baseline: 2.99 ms</small>
                          </div>
                          <div className="stat">
                            <span>
                              Correctness
                              <CircleCheck size={13} />
                            </span>
                            <strong>
                              5<span className="stat-divider">/5</span>
                            </strong>
                            <small>All checks passed</small>
                          </div>
                        </div>
                        <div className="block-heading">
                          <h2>Performance over iterations</h2>
                          <span className="subtle">12 experiments</span>
                        </div>
                        <Chart />
                        <div className="block-heading candidate-heading">
                          <h2>Top candidates</h2>
                          <button
                            className="text-button"
                            onClick={() => setModal("experiments")}
                          >
                            View all
                            <ArrowRight size={12} />
                          </button>
                        </div>
                        <CandidateTable
                          onSelect={(id) => {
                            setSelectedExperiment(id);
                            setTab("Code");
                          }}
                        />
                        <div className="insight">
                          <span className="insight-icon">
                            <Sparkles size={15} />
                          </span>
                          <div>
                            <strong>What made the difference</strong>
                            <p>
                              Larger tiles reduced global memory reads. Shared
                              memory reuse delivered the biggest improvement.
                            </p>
                          </div>
                        </div>
                        <div className="report-bottom">
                          <span>
                            <Cpu size={13} />
                            {gpu}
                          </span>
                          <span>
                            <Code2 size={13} />
                            CUDA
                          </span>
                          <span>
                            <Check size={13} />
                            FP32
                          </span>
                        </div>
                      </>
                    )}
                    {tab === "Timeline" && (
                      <>
                        <SectionIntro
                          eyebrow="RESEARCH LOG"
                          title="From question to result"
                          description="Every hypothesis, candidate, and evaluation in one place."
                        />
                        <div className="timeline">
                          {[
                            {
                              time: "14:32:00",
                              title: "Research started",
                              description:
                                "Established a PyTorch eager baseline of 2.99 ms.",
                              icon: <Play size={14} />,
                            },
                            {
                              time: "14:32:18",
                              title: "Initial candidates evaluated",
                              description:
                                "16 × 16 and 32 × 32 tile configurations passed correctness checks.",
                              icon: <Code2 size={14} />,
                            },
                            {
                              time: "14:32:46",
                              title: "Shared memory optimization",
                              description:
                                "Memory reuse improved runtime to 1.67 ms. Explored loop unrolling.",
                              icon: <Cpu size={14} />,
                            },
                            {
                              time: "14:33:21",
                              title: "New best candidate",
                              description:
                                "64 × 64 tiling achieved a 2.41× speedup with a 1.24 ms runtime.",
                              icon: <Zap size={14} />,
                            },
                            {
                              time: "14:33:42",
                              title: "Validation complete",
                              description:
                                "5/5 correctness trials passed. Best candidate and report saved.",
                              icon: <Check size={14} />,
                            },
                          ].map((e, i) => (
                            <div className="timeline-event" key={e.title}>
                              <span
                                className={`timeline-icon ${i === 4 ? "done" : ""}`}
                              >
                                {e.icon}
                              </span>
                              <div>
                                <small>{e.time}</small>
                                <h3>{e.title}</h3>
                                <p>{e.description}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                        <DemoNotice />
                      </>
                    )}
                    {tab === "Plan" && (
                      <>
                        <SectionIntro
                          eyebrow="RESEARCH PLAN"
                          title="A measured approach"
                          description="Optimize for speed. Verify at every step."
                        />
                        {[
                          "Establish the baseline",
                          "Explore tile configurations",
                          "Optimize memory access",
                          "Validate and compare",
                        ].map((s, i) => (
                          <div className="plan-step" key={s}>
                            <span className="step-number">0{i + 1}</span>
                            <div>
                              <h3>{s}</h3>
                              <p>
                                {
                                  [
                                    "Measure PyTorch eager across 100 timing trials. Fix the input shape, precision, and target GPU.",
                                    "Evaluate 16 × 16, 32 × 32, and 64 × 64 tile sizes with consistent benchmark settings.",
                                    "Use shared memory and loop unrolling to reduce redundant reads from global memory.",
                                    "Run five correctness trials, compare median runtimes, and retain the strongest candidate.",
                                  ][i]
                                }
                              </p>
                            </div>
                            <CircleCheck size={17} />
                          </div>
                        ))}
                        <div className="insight">
                          <Sparkles size={18} />
                          <div>
                            <strong>A good next question</strong>
                            <p>
                              Does the speedup hold at 2048 × 2048 and larger?
                            </p>
                            <button
                              className="text-button"
                              onClick={() => {
                                setExpanded(false);
                                setDraft(
                                  "Does the speedup hold at 2048 × 2048 and larger?",
                                );
                              }}
                            >
                              Explore in thread
                              <ArrowRight size={12} />
                            </button>
                          </div>
                        </div>
                      </>
                    )}
                    {tab === "Workers" && (
                      <>
                        <SectionIntro
                          eyebrow="AGENT WORKSPACE"
                          title="Research, in parallel"
                          description="A focused team for generation, evaluation, and review."
                        />
                        {[
                          {
                            name: "Researcher",
                            role: "Hypotheses & candidate generation",
                            icon: <Microscope size={20} />,
                            detail:
                              "Generated 12 candidate kernels across 4 iterations.",
                          },
                          {
                            name: "Evaluator",
                            role: "Benchmarking & correctness",
                            icon: <Terminal size={20} />,
                            detail:
                              "Completed 5 correctness and 100 timing trials per candidate.",
                          },
                          {
                            name: "Reviewer",
                            role: "Result analysis & synthesis",
                            icon: <FileText size={20} />,
                            detail:
                              "Compared results and prepared the experiment report.",
                          },
                        ].map((w) => (
                          <div className="worker-card" key={w.name}>
                            <div className="worker-icon">{w.icon}</div>
                            <div>
                              <h3>{w.name}</h3>
                              <small>{w.role}</small>
                              <p>{w.detail}</p>
                            </div>
                            <span className="complete-badge">
                              <Check size={10} /> Done
                            </span>
                          </div>
                        ))}
                        <DemoNotice />
                      </>
                    )}
                    {tab === "Compute" && (
                      <>
                        <SectionIntro
                          eyebrow="COMPUTE ENVIRONMENT"
                          title="Built for the benchmark"
                          description="A consistent environment for comparable results."
                        />
                        <div className="compute-card">
                          <div className="gpu-illustration">
                            <Cpu size={44} strokeWidth={1} />
                          </div>
                          <h2>{gpu}</h2>
                          <span className="neutral-badge">
                            Sample environment
                          </span>
                          <dl>
                            {[
                              ["Provider", "Modal"],
                              [
                                "Architecture",
                                gpu === "NVIDIA T4" ? "Turing" : "Ampere",
                              ],
                              ["Precision", "FP32"],
                              ["Correctness trials", "5"],
                              ["Performance trials", "100"],
                              ["Session duration", "1m 42s"],
                            ].map(([k, v]) => (
                              <div key={k}>
                                <dt>{k}</dt>
                                <dd>{v}</dd>
                              </div>
                            ))}
                          </dl>
                        </div>
                        <div className="insight">
                          <Cpu size={18} />
                          <div>
                            <strong>Ready when you are</strong>
                            <p>
                              Connect your backend to launch real GPU
                              evaluations. No compute resources are provisioned
                              in this demo.
                            </p>
                            <button
                              className="text-button"
                              onClick={() => setModal("settings")}
                            >
                              Environment settings
                              <ArrowRight size={12} />
                            </button>
                          </div>
                        </div>
                      </>
                    )}
                    {tab === "Metrics" && (
                      <>
                        <SectionIntro
                          eyebrow="PERFORMANCE METRICS"
                          title="Every iteration counts"
                          description="Compare candidate latency against the same baseline."
                        />
                        <div className="metric-summary">
                          <strong>
                            −58.5% <span>runtime</span>
                          </strong>
                          <span className="complete-badge">
                            <ArrowDown size={11} /> Lower is better
                          </span>
                        </div>
                        <Chart large />
                        <div className="block-heading">
                          <h2>Candidate speedup</h2>
                          <span className="subtle">Relative to PyTorch</span>
                        </div>
                        <div className="bar-chart">
                          {[...experiments].reverse().map((e) => (
                            <div key={e.id}>
                              <span>{e.id}</span>
                              <div>
                                <i
                                  style={{
                                    width: `${(Number(e.speed) / 2.6) * 100}%`,
                                  }}
                                />
                              </div>
                              <strong>{e.speed}×</strong>
                            </div>
                          ))}
                        </div>
                        <DemoNotice />
                      </>
                    )}
                    {tab === "Code" && (
                      <>
                        <SectionIntro
                          eyebrow="CANDIDATE SOURCE"
                          title={
                            selectedExperiment === "EXP-012"
                              ? "The best candidate"
                              : "Candidate preview"
                          }
                          description="Illustrative code excerpt. Connect the backend for complete executable candidates."
                        />
                        <div className="code-toolbar">
                          <span>
                            <FileCode2 size={14} />
                            {`candidate_${selectedExperiment.slice(-3)}.py`}
                          </span>
                          <button
                            className="text-button"
                            onClick={async () => {
                              try {
                                await navigator.clipboard.writeText(
                                  kernelCode.replace(
                                    "64",
                                    String(
                                      experiments.find(
                                        (e) => e.id === selectedExperiment,
                                      )?.tile ?? 64,
                                    ),
                                  ),
                                );
                                setToast("Code excerpt copied");
                              } catch {
                                setToast(
                                  "Clipboard unavailable in this browser",
                                );
                              }
                            }}
                          >
                            Copy code
                          </button>
                        </div>
                        <pre className="code-block">
                          {kernelCode
                            .replace(
                              "64",
                              String(
                                experiments.find(
                                  (e) => e.id === selectedExperiment,
                                )?.tile ?? 64,
                              ),
                            )
                            .split("\n")
                            .map((line, i) => (
                              <div key={i}>
                                <span className="line-number">{i + 1}</span>
                                <code
                                  className={
                                    line.trim().startsWith("#") ||
                                    line.trim().startsWith("//")
                                      ? "comment"
                                      : ""
                                  }
                                >
                                  {line || " "}
                                </code>
                              </div>
                            ))}
                        </pre>
                        <DemoNotice />
                      </>
                    )}
                  </>
                )}
              </div>
              <footer className="panel-footer">
                <span>
                  <i className="dot olive" /> Threads saved in this browser
                </span>
                <button onClick={() => setModal("experiments")}>
                  Experiment details
                  <ExternalLink size={11} />
                </button>
              </footer>
            </section>
          )}
        </div>
      </main>
      {toast && (
        <div className="toast" role="status">
          <CircleCheck size={16} />
          {toast}
        </div>
      )}
      {modal && (
        <div className="modal-backdrop" onClick={() => setModal(null)}>
          <div
            className={`modal ${modal === "experiments" ? "wide" : ""}`}
            ref={dialogRef}
            role="dialog"
            aria-modal="true"
            aria-label={
              modal === "settings" ? "Workspace settings" : "Experiment details"
            }
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-heading">
              <h2>
                {modal === "settings"
                  ? "Workspace settings"
                  : "Experiment details"}
              </h2>
              <IconButton label="Close dialog" onClick={() => setModal(null)}>
                <X size={18} />
              </IconButton>
            </div>
            {modal === "settings" ? (
              <>
                <p className="muted">
                  Your local research environment. Changes apply to this demo.
                </p>
                <label className="field-label">
                  Target GPU
                  <select value={gpu} onChange={(e) => setGpu(e.target.value)}>
                    <option>NVIDIA T4</option>
                    <option>NVIDIA A100</option>
                  </select>
                </label>
                <label className="field-label">
                  Preferred model
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                  >
                    <option>Claude Sonnet</option>
                    <option>Claude Opus</option>
                    <option>GPT-5</option>
                  </select>
                </label>
                <div className="connection-state">
                  <Circle size={12} />
                  <div>
                    <strong>Backend not connected</strong>
                    <p>
                      This workspace uses sample experiments. Thread messages
                      are saved in this browser.
                    </p>
                  </div>
                </div>
                <button
                  className="primary-button"
                  onClick={() => {
                    setModal(null);
                    setToast("Workspace preferences updated");
                  }}
                >
                  Done
                  <Check size={14} />
                </button>
              </>
            ) : (
              <>
                <p className="muted">
                  4 leading candidates from 12 sample experiments.
                </p>
                <CandidateTable
                  onSelect={(id) => {
                    setSelectedExperiment(id);
                    setTab("Code");
                    setPanel(true);
                    setModal(null);
                  }}
                />
                <dl className="detail-list">
                  <div>
                    <dt>Task</dt>
                    <dd>{active.task}</dd>
                  </div>
                  <div>
                    <dt>Hardware</dt>
                    <dd>{gpu}</dd>
                  </div>
                  <div>
                    <dt>Validation</dt>
                    <dd>5 correctness / 100 timing trials</dd>
                  </div>
                  <div>
                    <dt>Data source</dt>
                    <dd>Local demo fixtures</dd>
                  </div>
                </dl>
                <button className="primary-button" onClick={exportReport}>
                  Download report
                  <ArrowDownToLine size={14} />
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
function SectionIntro({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="section-intro">
      <span>{eyebrow}</span>
      <h1>{title}</h1>
      <p>{description}</p>
    </div>
  );
}
function DemoNotice() {
  return (
    <p className="demo-notice">
      <FlaskConical size={13} /> Sample data for exploring the workspace.
    </p>
  );
}
function CandidateTable({ onSelect }: { onSelect: (id: string) => void }) {
  return (
    <div className="candidate-table">
      <div className="table-header">
        <span>Configuration</span>
        <span>Runtime</span>
        <span>Speedup</span>
      </div>
      {experiments.map((e, i) => (
        <button
          className={`table-row ${i === 0 ? "best" : ""}`}
          onClick={() => onSelect(e.id)}
          key={e.id}
        >
          <div>
            <span className="candidate-id">
              {e.id}
              {i === 0 && <span className="best-label">BEST</span>}
            </span>
            <strong>{e.name}</strong>
          </div>
          <span>
            {e.time}
            <small> ms</small>
          </span>
          <span className="speedup">
            {e.speed}×{i === 0 ? <Zap size={12} /> : <Check size={12} />}
          </span>
        </button>
      ))}
    </div>
  );
}
