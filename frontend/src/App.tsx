import React, { useState, useRef, useEffect, useCallback } from "react";
import { PromptInputBox } from "@/components/ui/ai-prompt-box";
import { CostAnalytics } from "@/components/CostAnalytics";
import { ensureDemoAuthToken, getAuthToken } from "@/lib/auth";
import {
  clearDocuments,
  deleteDocument,
  listDocuments,
  type StoredDocument,
} from "@/lib/documents";
import {
  Sparkles,
  Bot,
  User,
  Plus,
  MessageSquare,
  Trash2,
  PanelLeftClose,
  PanelLeft,
  BarChart3,
  MessagesSquare,
  FileCheck2,
  FileText,
  AlertCircle,
  RefreshCw,
  X,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Message {
  role: "user" | "assistant";
  content: string;
  model?: string;
  streaming?: boolean;
}

interface Session {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const newSession = (): Session => ({
  id: crypto.randomUUID(),
  title: "New chat",
  messages: [],
  createdAt: Date.now(),
});

const loadSessions = (): Session[] => {
  try {
    const raw = localStorage.getItem("smartroute.sessions");
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
};

const saveSessions = (sessions: Session[]) => {
  localStorage.setItem("smartroute.sessions", JSON.stringify(sessions));
};

const formatBytes = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const formatUploadedAt = (value?: string | null) => {
  if (!value) return "Just now";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Recently";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
};

// ─── App Component ────────────────────────────────────────────────────────────

export default function App() {
  const [sessions, setSessions] = useState<Session[]>(() => {
    const saved = loadSessions();
    if (saved.length > 0) return saved;
    return [newSession()];
  });
  const [activeId, setActiveId] = useState<string>(() => {
    const saved = loadSessions();
    return saved.length > 0 ? saved[0].id : "";
  });

  // Navigation: 'chat' | 'analytics'
  const [currentView, setCurrentView] = useState<"chat" | "analytics">("chat");

  // Strategy and RAG state
  const [strategy, setStrategy] = useState<string>("cost_optimized");
  const [ragEnabled, setRagEnabled] = useState<boolean>(false);
  const [isUploadingDoc, setIsUploadingDoc] = useState<boolean>(false);
  const [uploadNotice, setUploadNotice] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [authNotice, setAuthNotice] = useState<string | null>(null);
  const [documents, setDocuments] = useState<StoredDocument[]>([]);
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(false);
  const [documentAction, setDocumentAction] = useState<string | null>(null);
  const [clearConfirm, setClearConfirm] = useState(false);

  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Sync active session id when sessions first load
  useEffect(() => {
    if (!activeId && sessions.length > 0) {
      setActiveId(sessions[0].id);
    }
  }, [activeId, sessions]);

  // Persist sessions
  useEffect(() => {
    saveSessions(sessions);
  }, [sessions]);

  // Auto-dismiss upload notice after 6 seconds
  useEffect(() => {
    if (uploadNotice) {
      const timer = setTimeout(() => setUploadNotice(null), 6000);
      return () => clearTimeout(timer);
    }
  }, [uploadNotice]);

  useEffect(() => {
    if (currentView === "chat") {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [sessions, activeId, currentView]);

  const refreshDocuments = useCallback(async (showError = false) => {
    setIsLoadingDocuments(true);
    try {
      const data = await listDocuments();
      setDocuments(data.documents);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to load documents";
      console.error("Document list error:", err);
      if (showError) {
        setUploadNotice({ type: "error", text: message });
      }
    } finally {
      setIsLoadingDocuments(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    ensureDemoAuthToken()
      .then(async () => {
        if (cancelled) return;
        setAuthNotice(null);
        await refreshDocuments();
      })
      .catch((err: unknown) => {
        console.error("Demo authentication error:", err);
        if (!cancelled) {
          setAuthNotice("Demo authentication is temporarily unavailable. Refresh to retry.");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [refreshDocuments]);

  useEffect(() => {
    if (!clearConfirm) return;
    const timer = setTimeout(() => setClearConfirm(false), 4000);
    return () => clearTimeout(timer);
  }, [clearConfirm]);

  const activeSession = sessions.find((s) => s.id === activeId);

  // ─── Session management ────────────────────────────────────────────────────

  const handleNewChat = () => {
    const s = newSession();
    setSessions((prev) => [s, ...prev]);
    setActiveId(s.id);
    setCurrentView("chat");
  };

  const handleSelectSession = (id: string) => {
    setActiveId(id);
    setCurrentView("chat");
  };

  const handleDeleteSession = (id: string) => {
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== id);
      if (next.length === 0) {
        const fresh = newSession();
        setActiveId(fresh.id);
        return [fresh];
      }
      if (id === activeId) setActiveId(next[0].id);
      return next;
    });
  };

  const handleDeleteDocument = async (filename: string) => {
    setDocumentAction(filename);
    try {
      await deleteDocument(filename);
      await refreshDocuments();
      setUploadNotice({ type: "success", text: `Removed ${filename} from storage and vector index.` });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Document delete failed";
      console.error("Document delete error:", err);
      setUploadNotice({ type: "error", text: message });
    } finally {
      setDocumentAction(null);
    }
  };

  const handleClearDocuments = async () => {
    if (documents.length === 0) return;
    if (!clearConfirm) {
      setClearConfirm(true);
      return;
    }

    setDocumentAction("clear-all");
    try {
      await clearDocuments();
      await refreshDocuments();
      setClearConfirm(false);
      setUploadNotice({
        type: "success",
        text: "Cleared all demo documents from storage and vector index.",
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Document clear failed";
      console.error("Document clear error:", err);
      setUploadNotice({ type: "error", text: message });
    } finally {
      setDocumentAction(null);
    }
  };

  // ─── Document Upload (RAG) ─────────────────────────────────────────────────

  const handleDocumentUpload = async (files: File[]) => {
    if (!files || files.length === 0) return;
    setIsUploadingDoc(true);
    setUploadNotice(null);

    const formData = new FormData();
    for (const file of files) {
      formData.append("files", file);
    }

    try {
      const token = getAuthToken() || (await ensureDemoAuthToken());
      setAuthNotice(null);
      const res = await fetch("/v1/documents/upload", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      if (!res.ok) {
        const errJson = (await res.json().catch(() => ({}))) as { detail?: string };
        throw new Error(errJson.detail || `Upload failed (HTTP ${res.status})`);
      }

      const data = await res.json();
      const count = data.documents?.length || files.length;
      const chunks = data.stats?.indexed_chunks || "multiple";
      await refreshDocuments();
      setUploadNotice({
        type: "success",
        text: `Successfully uploaded and indexed ${count} document(s) (${chunks} chunks in vector store).`,
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error";
      console.error("Document upload error:", err);
      setUploadNotice({
        type: "error",
        text: `Document indexing failed: ${message}`,
      });
    } finally {
      setIsUploadingDoc(false);
    }
  };

  // ─── Messaging ─────────────────────────────────────────────────────────────

  const handleSendMessage = async (rawMessage: string) => {
    if (!rawMessage.trim()) return;

    let token: string;
    try {
      token = getAuthToken() || (await ensureDemoAuthToken());
      setAuthNotice(null);
    } catch (err: unknown) {
      console.error("Demo authentication error:", err);
      setAuthNotice("Demo authentication is temporarily unavailable. Refresh to retry.");
      return;
    }

    const userMsg: Message = { role: "user", content: rawMessage };
    const asstMsg: Message = { role: "assistant", content: "", streaming: true };

    const isFirstMessage = !activeSession || activeSession.messages.length === 0;
    const newTitle = isFirstMessage
      ? rawMessage.slice(0, 36) + (rawMessage.length > 36 ? "…" : "")
      : undefined;

    setSessions((prev) =>
      prev.map((s) =>
        s.id === activeId
          ? {
              ...s,
              title: newTitle ?? s.title,
              messages: [...s.messages, userMsg, asstMsg],
            }
          : s
      )
    );

    setIsLoading(true);

    try {
      const res = await fetch("/v1/query/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          query: rawMessage,
          strategy,
          use_retrieval: ragEnabled,
          session_id: activeId,
        }),
      });

      if (!res.ok) throw new Error(`Status ${res.status}`);

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      let accumulated = "";
      let modelUsed = "";

      if (reader) {
        let buffer = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith("data: ")) {
              try {
                const parsed = JSON.parse(trimmed.slice(6));
                if (parsed.type === "chunk" && parsed.content) accumulated += parsed.content;
                if (parsed.type === "replace" && parsed.content !== undefined) accumulated = parsed.content;
                if (parsed.type === "done" && parsed.result?.model_used)
                  modelUsed = parsed.result.model_used;
              } catch {
                // Ignore partial JSON
              }
            }
          }

          setSessions((prev) =>
            prev.map((s) =>
              s.id === activeId
                ? {
                    ...s,
                    messages: s.messages.map((m, i, arr) =>
                      i === arr.length - 1 && m.role === "assistant"
                        ? { ...m, content: accumulated, model: modelUsed || m.model }
                        : m
                    ),
                  }
                : s
            )
          );
        }
      }

      setSessions((prev) =>
        prev.map((s) =>
          s.id === activeId
            ? {
                ...s,
                messages: s.messages.map((m, i, arr) =>
                  i === arr.length - 1 && m.role === "assistant"
                    ? {
                        ...m,
                        content: accumulated || "Response received.",
                        streaming: false,
                        model: modelUsed || "smartroute-gateway",
                      }
                    : m
                ),
              }
            : s
        )
      );
    } catch (err) {
      console.warn("Stream query failed:", err);
      setSessions((prev) =>
        prev.map((s) =>
          s.id === activeId
            ? {
                ...s,
                messages: s.messages.map((m, i, arr) =>
                  i === arr.length - 1 && m.role === "assistant"
                    ? {
                        ...m,
                        content:
                          "Unable to reach SmartRoute-AI right now. Please check the backend connection and try again.",
                        streaming: false,
                        model: "request-failed",
                      }
                    : m
                ),
              }
            : s
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const messages = activeSession?.messages ?? [];

  // ─── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="relative flex w-full h-screen overflow-hidden font-sans bg-[radial-gradient(125%_125%_at_50%_101%,rgba(245,87,2,1)_10.5%,rgba(245,120,2,1)_16%,rgba(245,140,2,1)_17.5%,rgba(245,170,100,1)_25%,rgba(238,174,202,1)_40%,rgba(202,179,214,1)_65%,rgba(148,201,233,1)_100%)]">

      {/* ── LEFT SIDEBAR ──────────────────────────────────────────────────── */}
      <AnimatePresence initial={false}>
        {sidebarOpen && (
          <motion.aside
            key="sidebar"
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 260, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: "easeInOut" }}
            className="flex-shrink-0 h-full overflow-hidden z-30"
          >
            <div className="flex flex-col h-full w-[260px] bg-black/35 backdrop-blur-2xl border-r border-white/10">
              {/* Sidebar Header */}
              <div className="flex items-center justify-between px-4 py-4 border-b border-white/10">
                <div className="flex items-center gap-2">
                  <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-black/40 border border-white/20">
                    <Sparkles className="h-3.5 w-3.5 text-orange-200" />
                  </div>
                  <span className="text-sm font-semibold text-white/90 tracking-tight">
                    SmartRoute<span className="text-orange-300">.AI</span>
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => setSidebarOpen(false)}
                  className="rounded-lg p-1.5 text-white/40 hover:text-white hover:bg-white/10 transition-colors"
                  title="Close sidebar"
                >
                  <PanelLeftClose className="h-4 w-4" />
                </button>
              </div>

              {/* Main Navigation Views: Chats vs Cost Analytics */}
              <div className="px-3 pt-3 pb-2 space-y-1 border-b border-white/5">
                <button
                  type="button"
                  onClick={() => setCurrentView("chat")}
                  className={`flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-xs font-medium transition-all ${
                    currentView === "chat"
                      ? "bg-white/15 text-white shadow-sm"
                      : "text-white/60 hover:bg-white/10 hover:text-white"
                  }`}
                >
                  <MessagesSquare className="h-4 w-4 text-orange-300" />
                  <span>Chats</span>
                </button>

                <button
                  type="button"
                  onClick={() => setCurrentView("analytics")}
                  className={`flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-xs font-medium transition-all ${
                    currentView === "analytics"
                      ? "bg-white/15 text-white shadow-sm"
                      : "text-white/60 hover:bg-white/10 hover:text-white"
                  }`}
                >
                  <BarChart3 className="h-4 w-4 text-emerald-400" />
                  <span>Cost Analytics</span>
                </button>
              </div>

              {/* New Chat Button */}
              <div className="px-3 py-2.5">
                <button
                  type="button"
                  onClick={handleNewChat}
                  className="flex w-full items-center gap-2 rounded-xl bg-white/10 hover:bg-white/20 border border-white/10 px-3 py-2 text-xs font-medium text-white transition-all shadow-sm"
                >
                  <Plus className="h-3.5 w-3.5" />
                  <span>New chat</span>
                </button>
              </div>

              {/* Session History List */}
              <div className="flex-1 overflow-y-auto px-2 pb-4 space-y-0.5 scrollbar-thin scrollbar-thumb-white/20 scrollbar-track-transparent">
                <div className="px-2 py-1 text-[11px] font-semibold text-white/40 uppercase tracking-wider">
                  History
                </div>
                {sessions.length === 0 ? (
                  <p className="text-center text-xs text-white/30 mt-6">No chats yet</p>
                ) : (
                  sessions.map((s) => (
                    <div
                      key={s.id}
                      onClick={() => handleSelectSession(s.id)}
                      className={`group relative flex items-center gap-2.5 w-full rounded-xl px-3 py-2 text-sm cursor-pointer transition-all ${
                        currentView === "chat" && s.id === activeId
                          ? "bg-white/15 text-white shadow-sm"
                          : "text-white/60 hover:bg-white/10 hover:text-white"
                      }`}
                    >
                      <MessageSquare className="h-3.5 w-3.5 flex-shrink-0 opacity-70" />
                      <span className="flex-1 truncate text-[13px]">{s.title}</span>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteSession(s.id);
                        }}
                        className="opacity-0 group-hover:opacity-100 flex-shrink-0 rounded p-0.5 text-white/40 hover:text-white transition-opacity"
                        title="Delete chat"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ))
                )}
              </div>

              {/* Knowledge Base */}
              <div className="border-t border-white/10 px-3 py-3">
                <div className="mb-2 flex items-center justify-between">
                  <div>
                    <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-white/55">
                      <FileText className="h-3.5 w-3.5 text-emerald-300" />
                      <span>Knowledge Base</span>
                    </div>
                    <div className="mt-0.5 text-[10px] text-white/35">
                      {documents.length} doc{documents.length === 1 ? "" : "s"} embedded
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => refreshDocuments(true)}
                    disabled={isLoadingDocuments}
                    className="rounded-lg p-1.5 text-white/40 transition-colors hover:bg-white/10 hover:text-white disabled:cursor-wait disabled:opacity-60"
                    title="Refresh documents"
                  >
                    <RefreshCw className={`h-3.5 w-3.5 ${isLoadingDocuments ? "animate-spin" : ""}`} />
                  </button>
                </div>

                <div className="max-h-40 space-y-1 overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-white/20 scrollbar-track-transparent">
                  {documents.length === 0 ? (
                    <div className="rounded-xl border border-dashed border-white/10 bg-black/15 px-3 py-3 text-[11px] leading-snug text-white/35">
                      Turn on RAG and upload a PDF, TXT, or MD file to show embedded documents here.
                    </div>
                  ) : (
                    documents.map((doc) => (
                      <div
                        key={doc.id}
                        className="group rounded-xl border border-white/10 bg-black/20 px-2.5 py-2 transition-colors hover:bg-white/10"
                      >
                        <div className="flex items-start gap-2">
                          <FileText className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-emerald-300" />
                          <div className="min-w-0 flex-1">
                            <div className="truncate text-[12px] font-medium text-white/85">
                              {doc.filename}
                            </div>
                            <div className="mt-0.5 flex items-center gap-1.5 text-[10px] text-white/40">
                              <span>{formatBytes(doc.size_bytes)}</span>
                              <span>•</span>
                              <span>{formatUploadedAt(doc.created_at)}</span>
                            </div>
                          </div>
                          <button
                            type="button"
                            onClick={() => handleDeleteDocument(doc.filename)}
                            disabled={documentAction === doc.filename}
                            className="rounded-md p-1 text-white/35 opacity-0 transition-all hover:bg-red-500/20 hover:text-red-200 group-hover:opacity-100 disabled:cursor-wait disabled:opacity-50"
                            title={`Delete ${doc.filename}`}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </div>

                <button
                  type="button"
                  onClick={handleClearDocuments}
                  disabled={documents.length === 0 || documentAction === "clear-all"}
                  className={`mt-2 flex w-full items-center justify-center gap-1.5 rounded-xl border px-3 py-2 text-[11px] font-medium transition-all disabled:cursor-not-allowed disabled:opacity-40 ${
                    clearConfirm
                      ? "border-red-400/40 bg-red-500/20 text-red-100"
                      : "border-white/10 bg-white/5 text-white/50 hover:bg-white/10 hover:text-white"
                  }`}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  <span>
                    {documentAction === "clear-all"
                      ? "Clearing..."
                      : clearConfirm
                      ? "Click again to clear all"
                      : "Clear all documents"}
                  </span>
                </button>
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* ── MAIN AREA ────────────────────────────────────────────────────── */}
      <div className="relative flex flex-1 flex-col h-full overflow-hidden">
        {/* Toggle sidebar button when closed */}
        {!sidebarOpen && (
          <button
            type="button"
            onClick={() => setSidebarOpen(true)}
            className="absolute top-4 left-4 z-20 rounded-xl bg-black/30 hover:bg-black/50 backdrop-blur-md border border-white/20 p-2 text-white/70 hover:text-white transition-all shadow-md"
            title="Open sidebar"
          >
            <PanelLeft className="h-4 w-4" />
          </button>
        )}

        {/* Upload Status Toast */}
        <AnimatePresence>
          {uploadNotice && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="absolute top-4 right-4 z-50 max-w-md"
            >
              <div
                className={`flex items-center gap-2.5 px-4 py-3 rounded-2xl backdrop-blur-xl border shadow-2xl text-xs ${
                  uploadNotice.type === "success"
                    ? "bg-emerald-950/80 border-emerald-500/40 text-emerald-200"
                    : "bg-red-950/80 border-red-500/40 text-red-200"
                }`}
              >
                {uploadNotice.type === "success" ? (
                  <FileCheck2 className="h-4 w-4 text-emerald-400 flex-shrink-0" />
                ) : (
                  <AlertCircle className="h-4 w-4 text-red-400 flex-shrink-0" />
                )}
                <span className="flex-1 leading-snug">{uploadNotice.text}</span>
                <button
                  type="button"
                  onClick={() => setUploadNotice(null)}
                  className="text-white/40 hover:text-white"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── CONDITIONALLY RENDER: Cost Analytics vs Chat Canvas ─────────── */}
        {/* Demo Auth Status Toast */}
        <AnimatePresence>
          {authNotice && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="absolute top-4 left-1/2 z-50 w-[calc(100%-2rem)] max-w-md -translate-x-1/2"
            >
              <div className="flex items-center gap-2.5 rounded-2xl border border-amber-400/40 bg-stone-950/85 px-4 py-3 text-xs text-amber-100 shadow-2xl backdrop-blur-xl">
                <AlertCircle className="h-4 w-4 flex-shrink-0 text-amber-300" />
                <span className="flex-1 leading-snug">{authNotice}</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {currentView === "analytics" ? (
          <CostAnalytics />
        ) : messages.length === 0 ? (
          /* ── Empty State ──────────────────────────────────────────────── */
          <div className="flex flex-1 flex-col items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="text-center px-6 mb-8"
            >
              <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-white drop-shadow-md mb-3">
                Where intelligence meets efficiency.
              </h1>
              <p className="text-sm sm:text-base text-white/75 max-w-md mx-auto drop-shadow">
                Intelligent LLM query routing, RAG retrieval, and real-time cost governance.
              </p>
            </motion.div>

            {/* Prompt Box Centered */}
            <div className="w-full max-w-2xl px-4">
              <PromptInputBox
                isLoading={isLoading}
                onSend={handleSendMessage}
                strategy={strategy}
                onStrategyChange={setStrategy}
                ragEnabled={ragEnabled}
                onRagChange={setRagEnabled}
                onUploadDocument={handleDocumentUpload}
                isUploadingDoc={isUploadingDoc}
              />
              <p className="mt-2.5 text-center text-[11px] text-white/50 drop-shadow">
                SmartRoute-AI • Fast, adaptive, cost-optimized routing
              </p>
            </div>
          </div>
        ) : (
          /* ── Active Conversation ───────────────────────────────────────── */
          <div className="flex flex-1 flex-col overflow-hidden">
            {/* Messages Scroll Area */}
            <div className="flex-1 overflow-y-auto py-6 scrollbar-thin scrollbar-thumb-white/20 scrollbar-track-transparent">
              <div className="mx-auto max-w-2xl px-4 space-y-4">
                <AnimatePresence initial={false}>
                  {messages.map((m, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`flex w-full gap-3 ${
                        m.role === "user" ? "justify-end" : "justify-start"
                      }`}
                    >
                      <div
                        className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm shadow-xl backdrop-blur-xl border ${
                          m.role === "user"
                            ? "bg-[#1F2023]/90 text-white border-white/10 rounded-br-sm"
                            : "bg-[#121316]/95 text-gray-100 border-white/10 rounded-bl-sm"
                        }`}
                      >
                        <div className="flex items-center gap-1.5 mb-1 text-[11px] text-white/50">
                          {m.role === "user" ? (
                            <>
                              <User className="h-3 w-3" />
                              <span>You</span>
                            </>
                          ) : (
                            <>
                              <Bot className="h-3 w-3 text-orange-300" />
                              <span>SmartRoute Assistant</span>
                              {m.model && (
                                <span className="ml-1 rounded bg-white/10 px-1 py-0.5 font-mono text-[9px] text-white/70">
                                  {m.model}
                                </span>
                              )}
                            </>
                          )}
                        </div>
                        <div className="whitespace-pre-wrap leading-relaxed">
                          {m.content}
                          {m.streaming && (
                            <span className="inline-block h-3.5 w-1 ml-1 bg-orange-300 animate-pulse align-middle" />
                          )}
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
                <div ref={messagesEndRef} />
              </div>
            </div>

            {/* Bottom Prompt Box */}
            <div className="px-4 pb-6 pt-2">
              <div className="mx-auto max-w-2xl">
                <PromptInputBox
                  isLoading={isLoading}
                  onSend={handleSendMessage}
                  strategy={strategy}
                  onStrategyChange={setStrategy}
                  ragEnabled={ragEnabled}
                  onRagChange={setRagEnabled}
                  onUploadDocument={handleDocumentUpload}
                  isUploadingDoc={isUploadingDoc}
                />
                <p className="mt-2 text-center text-[11px] text-white/50 drop-shadow">
                  SmartRoute-AI • Fast, adaptive, cost-optimized routing
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
