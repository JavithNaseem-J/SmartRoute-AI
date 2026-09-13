import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Header } from "./components/Header";
import { Sidebar } from "./components/Sidebar";
import { ChatMessage } from "./components/ChatMessage";
import { CanvasDrawer } from "./components/CanvasDrawer";
import { PromptInputBox } from "./components/ui/ai-prompt-box";
import { DemoOne } from "./components/ui/demo";
import {
  BudgetRecord,
  ChatMessage as ChatMessageType,
  ChatSession,
  DocumentItem,
  RoutingStrategy,
  StatsResponse,
} from "./types/chat";
import {
  AlertCircle,
  ArrowDown,
  Bot,
  CheckCircle2,
  Cpu,
  Layers,
  Sparkles,
  Zap,
} from "lucide-react";

const DEFAULT_DEV_JWT =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJkZXYtdXNlci0wMDEiLCJyb2xlIjoiYXV0aGVudGljYXRlZCIseyJleHAiOjE5OTk5OTk5OTl9.dev_signature";

function createNewSession(strategy: RoutingStrategy = "cost_optimized"): ChatSession {
  return {
    id: "session-" + Date.now() + "-" + Math.random().toString(36).substring(2, 7),
    title: "New Session",
    updatedAt: Date.now(),
    messages: [
      {
        id: "welcome-msg",
        role: "assistant",
        content:
          "Welcome to SmartRoute-AI Studio. Ask me anything — queries are dynamically classified and routed between fast economy models and high-reasoning frontier models for maximum performance and cost-efficiency.",
        timestamp: Date.now(),
        model_used: "smartroute-gateway",
      },
    ],
    strategy,
    useRetrieval: false,
  };
}

export default function App() {
  const [viewMode, setViewMode] = useState<"studio" | "demo">("studio");
  const [token, setToken] = useState<string>(
    () => localStorage.getItem("smartroute.jwt") || DEFAULT_DEV_JWT
  );
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [canvasOpen, setCanvasOpen] = useState(false);
  const [canvasContent, setCanvasContent] = useState<string>("");
  const [canvasTitle, setCanvasTitle] = useState<string>("Workspace Canvas");

  // Sessions state
  const [sessions, setSessions] = useState<ChatSession[]>(() => {
    try {
      const saved = localStorage.getItem("smartroute.sessions");
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      }
    } catch {
      // ignore
    }
    return [createNewSession()];
  });

  const [activeSessionId, setActiveSessionId] = useState<string>(() => sessions[0]?.id || "");

  // Health and telemetry
  const [health, setHealth] = useState("checking");
  const [ready, setReady] = useState("checking");
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [isIndexing, setIsIndexing] = useState(false);
  const [stats, setStats] = useState<StatsResponse>({});
  const [budget, setBudget] = useState<Record<string, BudgetRecord>>({});
  const [isBusy, setIsBusy] = useState(false);
  const [notice, setNotice] = useState<string>("");

  const chatContainerRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Active session helper
  const activeSession = useMemo(() => {
    return sessions.find((s) => s.id === activeSessionId) || sessions[0];
  }, [sessions, activeSessionId]);

  // Persist sessions
  useEffect(() => {
    localStorage.setItem("smartroute.sessions", JSON.stringify(sessions));
  }, [sessions]);

  // Save Token helper
  const handleSaveToken = useCallback((newToken: string) => {
    setToken(newToken);
    if (newToken) {
      localStorage.setItem("smartroute.jwt", newToken);
    } else {
      localStorage.removeItem("smartroute.jwt");
    }
  }, []);

  const authHeaders = useCallback((): Record<string, string> => {
    return token ? { Authorization: `Bearer ${token}` } : {};
  }, [token]);

  // Fetch status, docs, budget
  const refreshStatus = useCallback(async () => {
    try {
      const healthRes = await fetch("/health").then((r) => r.json());
      setHealth(healthRes.status || "healthy");
    } catch {
      setHealth("offline");
    }

    try {
      const readyRes = await fetch("/ready").then((r) => r.json());
      setReady(readyRes.status || "not_ready");
    } catch {
      setReady("offline");
    }
  }, []);

  const loadDocuments = useCallback(async () => {
    if (!token) return;
    try {
      const res = await fetch("/v1/documents", { headers: authHeaders() });
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
      }
    } catch (e) {
      console.warn("Failed to load documents:", e);
    }
  }, [token, authHeaders]);

  const loadMetrics = useCallback(async () => {
    if (!token) return;
    try {
      const [statsRes, budgetRes] = await Promise.all([
        fetch("/v1/stats?days=1", { headers: authHeaders() }).then((r) => (r.ok ? r.json() : {})),
        fetch("/v1/budget", { headers: authHeaders() }).then((r) => (r.ok ? r.json() : {})),
      ]);
      setStats(statsRes);
      setBudget(budgetRes);
    } catch (e) {
      console.warn("Failed to load metrics:", e);
    }
  }, [token, authHeaders]);

  useEffect(() => {
    refreshStatus();
    loadDocuments();
    loadMetrics();
    const interval = setInterval(() => {
      refreshStatus();
      loadMetrics();
    }, 20000);
    return () => clearInterval(interval);
  }, [refreshStatus, loadDocuments, loadMetrics]);

  // Scroll to bottom helper
  const scrollToBottom = useCallback(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [activeSession?.messages, scrollToBottom]);

  // Session Handlers
  const handleNewSession = () => {
    const fresh = createNewSession(activeSession?.strategy || "cost_optimized");
    setSessions((prev) => [fresh, ...prev]);
    setActiveSessionId(fresh.id);
  };

  const handleDeleteSession = (id: string) => {
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== id);
      if (next.length === 0) {
        const fresh = createNewSession();
        return [fresh];
      }
      return next;
    });
    if (activeSessionId === id) {
      const remaining = sessions.filter((s) => s.id !== id);
      setActiveSessionId(remaining[0]?.id || "");
    }
  };

  const handleStrategyChange = (newStrategy: RoutingStrategy) => {
    setSessions((prev) =>
      prev.map((s) => (s.id === activeSessionId ? { ...s, strategy: newStrategy } : s))
    );
  };

  const handleToggleRetrieval = () => {
    setSessions((prev) =>
      prev.map((s) => (s.id === activeSessionId ? { ...s, useRetrieval: !s.useRetrieval } : s))
    );
  };

  // Mode Toggle callback from PromptInputBox
  const handleModeToggle = (mode: "search" | "think" | "canvas", active: boolean) => {
    if (mode === "search") {
      setSessions((prev) =>
        prev.map((s) => (s.id === activeSessionId ? { ...s, useRetrieval: active } : s))
      );
      if (active) {
        setNotice("Knowledge search (RAG) enabled for subsequent queries.");
        setTimeout(() => setNotice(""), 4000);
      }
    } else if (mode === "think") {
      const targetStrategy: RoutingStrategy = active ? "quality_first" : "cost_optimized";
      handleStrategyChange(targetStrategy);
      if (active) {
        setNotice("Deep Think mode active: queries will route to high-tier reasoning models.");
        setTimeout(() => setNotice(""), 4000);
      }
    } else if (mode === "canvas") {
      setCanvasOpen(active);
    }
  };

  // Upload Documents Handler
  const handleUploadDocuments = async (files: FileList | File[]) => {
    if (!files || files.length === 0) return;
    const formData = new FormData();
    Array.from(files).forEach((f) => formData.append("files", f));

    try {
      setNotice("Uploading document to knowledge base...");
      const res = await fetch("/v1/documents/upload", {
        method: "POST",
        headers: authHeaders(),
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Upload failed");
      }

      setNotice("Document uploaded and indexed successfully!");
      loadDocuments();
      setTimeout(() => setNotice(""), 4000);
    } catch (e: any) {
      setNotice(`Upload error: ${e.message}`);
      setTimeout(() => setNotice(""), 6000);
    }
  };

  const handleDeleteDocument = async (filename: string) => {
    try {
      const res = await fetch(`/v1/documents/${encodeURIComponent(filename)}`, {
        method: "DELETE",
        headers: authHeaders(),
      });
      if (res.ok) {
        setDocuments((prev) => prev.filter((d) => d.filename !== filename));
      }
    } catch (e) {
      console.warn("Delete document failed:", e);
    }
  };

  const handleIndexDocuments = async () => {
    setIsIndexing(true);
    try {
      const res = await fetch("/v1/index", {
        method: "POST",
        headers: authHeaders(),
      });
      if (res.ok) {
        setNotice("Document index reloaded successfully.");
      }
    } catch (e: any) {
      setNotice(`Re-index error: ${e.message}`);
    } finally {
      setIsIndexing(false);
      setTimeout(() => setNotice(""), 4000);
    }
  };

  // Main Query Send Handler
  const handleSendMessage = async (rawMessage: string, files?: File[]) => {
    if (!rawMessage.trim() && (!files || files.length === 0)) return;

    // Clean prefix for backend query while detecting mode
    let queryText = rawMessage;
    let overrideRetrieval = activeSession?.useRetrieval ?? false;
    let overrideStrategy = activeSession?.strategy ?? "cost_optimized";

    if (rawMessage.startsWith("[Search: ")) {
      queryText = rawMessage.slice(9, -1);
      overrideRetrieval = true;
    } else if (rawMessage.startsWith("[Think: ")) {
      queryText = rawMessage.slice(8, -1);
      overrideStrategy = "quality_first";
    } else if (rawMessage.startsWith("[Canvas: ")) {
      queryText = rawMessage.slice(9, -1);
      setCanvasOpen(true);
    }

    const userMsgId = "user-" + Date.now();
    const assistantMsgId = "asst-" + Date.now();

    // Prepare attached file representations
    const attachedFiles = (files || []).map((f) => ({
      name: f.name,
      isImage: f.type.startsWith("image/"),
      url: f.type.startsWith("image/") ? URL.createObjectURL(f) : undefined,
    }));

    const userMessage: ChatMessageType = {
      id: userMsgId,
      role: "user",
      content: rawMessage,
      timestamp: Date.now(),
      files: attachedFiles,
    };

    const initialAssistantMessage: ChatMessageType = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      timestamp: Date.now(),
      streaming: true,
    };

    // Update active session with user and streaming assistant messages
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id !== activeSessionId) return s;
        // Auto-title session from first query
        const isFirst = s.messages.length <= 1;
        const newTitle = isFirst ? queryText.slice(0, 30) : s.title;
        return {
          ...s,
          title: newTitle,
          updatedAt: Date.now(),
          messages: [...s.messages, userMessage, initialAssistantMessage],
        };
      })
    );

    setIsBusy(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch("/v1/query/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...authHeaders(),
        },
        body: JSON.stringify({
          query: queryText,
          strategy: overrideStrategy,
          use_retrieval: overrideRetrieval,
          session_id: activeSessionId,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error (${response.status})`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let accumulatedText = "";
      let telemetry: Partial<ChatMessageType> = {};

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
                const data = JSON.parse(trimmed.slice(6));
                if (data.type === "chunk" && data.content) {
                  accumulatedText += data.content;
                } else if (data.type === "metadata" && data.data) {
                  telemetry = {
                    ...telemetry,
                    ...data.data,
                  };
                } else if (data.type === "done" && data.result) {
                  telemetry = {
                    ...telemetry,
                    model_used: data.result.model_used,
                    latency: data.result.latency,
                    cost: data.result.cost,
                    complexity: data.result.complexity,
                    confidence: data.result.confidence,
                    sources: data.result.sources,
                  };
                } else if (data.type === "error") {
                  telemetry.error = data.content;
                }
              } catch {
                // partial JSON, continue
              }
            }
          }

          // Update streaming state in active session
          setSessions((prev) =>
            prev.map((s) => {
              if (s.id !== activeSessionId) return s;
              return {
                ...s,
                messages: s.messages.map((m) =>
                  m.id === assistantMsgId
                    ? {
                        ...m,
                        content: accumulatedText,
                        streaming: true,
                        ...telemetry,
                      }
                    : m
                ),
              };
            })
          );
        }
      }

      // Finalize assistant message
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== activeSessionId) return s;
          return {
            ...s,
            messages: s.messages.map((m) =>
              m.id === assistantMsgId
                ? {
                    ...m,
                    content: accumulatedText || "No response received.",
                    streaming: false,
                    ...telemetry,
                  }
                : m
            ),
          };
        })
      );

      // If Canvas mode is active and code blocks exist, update Canvas content
      if (accumulatedText.includes("```")) {
        const match = accumulatedText.match(/```(?:[a-zA-Z0-9_-]+)?\n([\s\S]*?)```/);
        if (match && match[1]) {
          setCanvasContent(match[1]);
          setCanvasTitle(`Generated from "${queryText.slice(0, 25)}..."`);
          setCanvasOpen(true);
        }
      }

      loadMetrics();
    } catch (err: any) {
      if (err.name === "AbortError") {
        setNotice("Generation stopped by user.");
      } else {
        const errorMsg = err.message || "Failed to process query.";
        setSessions((prev) =>
          prev.map((s) => {
            if (s.id !== activeSessionId) return s;
            return {
              ...s,
              messages: s.messages.map((m) =>
                m.id === assistantMsgId
                  ? {
                      ...m,
                      content: "An error occurred while generating response.",
                      error: errorMsg,
                      streaming: false,
                    }
                  : m
              ),
            };
          })
        );
        setNotice(`API Error: ${errorMsg}`);
      }
      setTimeout(() => setNotice(""), 5000);
    } finally {
      setIsBusy(false);
      abortControllerRef.current = null;
    }
  };

  return (
    <div className="flex h-screen w-full flex-col bg-[#0F1012] text-gray-100 antialiased font-sans overflow-hidden">
      {/* Top Header */}
      <Header
        viewMode={viewMode}
        onViewModeChange={setViewMode}
        strategy={activeSession?.strategy || "cost_optimized"}
        onStrategyChange={handleStrategyChange}
        useRetrieval={activeSession?.useRetrieval || false}
        onToggleRetrieval={handleToggleRetrieval}
        health={health}
        ready={ready}
        token={token}
        onSaveToken={handleSaveToken}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        sidebarOpen={sidebarOpen}
      />

      {/* Global Notice Banner */}
      {notice && (
        <div className="z-30 flex items-center justify-between border-b border-indigo-500/30 bg-indigo-950/40 px-4 py-2 text-xs text-indigo-200 backdrop-blur-md animate-in fade-in-0 duration-200">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-4 w-4 text-indigo-400" />
            <span>{notice}</span>
          </div>
          <button
            type="button"
            onClick={() => setNotice("")}
            className="text-indigo-400 hover:text-white"
          >
            ✕
          </button>
        </div>
      )}

      {/* Main Viewport */}
      {viewMode === "demo" ? (
        <div className="relative flex-1 overflow-hidden">
          {/* Back button to Studio */}
          <div className="absolute top-4 left-4 z-50">
            <button
              type="button"
              onClick={() => setViewMode("studio")}
              className="flex items-center gap-2 rounded-xl border border-white/20 bg-black/50 px-3.5 py-2 text-xs font-medium text-white shadow-xl backdrop-blur-md hover:bg-black/80 transition-all"
            >
              ← Back to SmartRoute Studio
            </button>
          </div>
          <DemoOne />
        </div>
      ) : (
        <div className="relative flex flex-1 overflow-hidden">
          {/* Collapsible Sidebar */}
          <Sidebar
            isOpen={sidebarOpen}
            onClose={() => setSidebarOpen(false)}
            sessions={sessions}
            activeSessionId={activeSessionId}
            onSelectSession={(id) => {
              setActiveSessionId(id);
              setSidebarOpen(false);
            }}
            onNewSession={handleNewSession}
            onDeleteSession={handleDeleteSession}
            documents={documents}
            onUploadDocuments={handleUploadDocuments}
            onDeleteDocument={handleDeleteDocument}
            onIndexDocuments={handleIndexDocuments}
            budget={budget}
            stats={stats}
            isIndexing={isIndexing}
          />

          {/* Chat Stream & Prompt Container */}
          <main className="relative flex flex-1 flex-col overflow-hidden bg-[#101114]">
            {/* Messages Feed */}
            <div
              ref={chatContainerRef}
              className="flex-1 overflow-y-auto px-2 sm:px-4 py-6 scrollbar-thin scrollbar-thumb-[#2E3033] scrollbar-track-transparent"
            >
              <div className="mx-auto max-w-4xl space-y-4">
                {activeSession?.messages.map((msg) => (
                  <ChatMessage key={msg.id} message={msg} />
                ))}
              </div>
            </div>

            {/* Bottom Anchored AI Prompt Box */}
            <div className="w-full border-t border-[#232428] bg-[#121316]/95 p-3 sm:p-4 backdrop-blur-xl">
              <div className="mx-auto max-w-4xl">
                <PromptInputBox
                  isLoading={isBusy}
                  placeholder="Ask a query, upload a document, or toggle Search / Think..."
                  onSend={handleSendMessage}
                  onModeToggle={handleModeToggle}
                  onUploadDocument={handleUploadDocuments}
                />
                <div className="mt-2 flex items-center justify-between px-2 text-[11px] text-gray-500">
                  <div className="flex items-center gap-3">
                    <span className="flex items-center gap-1">
                      <Zap className="h-3 w-3 text-indigo-400" />
                      Dynamic Routing Gateway
                    </span>
                    <span className="hidden sm:inline">
                      Strategy:{" "}
                      <strong className="text-gray-400 font-mono capitalize">
                        {activeSession?.strategy.replace("_", " ")}
                      </strong>
                    </span>
                  </div>
                  <span className="hidden md:inline text-[10px]">
                    Press <kbd className="rounded bg-[#232428] px-1 py-0.5 font-mono">Enter</kbd> to
                    send, <kbd className="rounded bg-[#232428] px-1 py-0.5 font-mono">Shift+Enter</kbd>{" "}
                    for new line
                  </span>
                </div>
              </div>
            </div>
          </main>

          {/* Interactive Canvas Workspace Drawer */}
          <CanvasDrawer
            isOpen={canvasOpen}
            onClose={() => setCanvasOpen(false)}
            title={canvasTitle}
            content={canvasContent}
          />
        </div>
      )}
    </div>
  );
}
