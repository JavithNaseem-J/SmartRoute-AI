import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Database,
  FileText,
  KeyRound,
  RefreshCw,
  Trash2,
  UploadCloud
} from "lucide-react";
import { AiPromptBox, PromptMode } from "./components/ui/ai-prompt-box";
import { cn, formatCurrency } from "./lib/utils";

type DocumentRecord = {
  filename: string;
  size_bytes: number;
  modified_time?: number;
};

type StreamItem =
  | { type: "metadata"; data?: Record<string, unknown> }
  | { type: "chunk"; content?: string }
  | { type: "done"; result?: QueryResult }
  | { type: "error"; content?: string };

type QueryResult = {
  success?: boolean;
  error?: string;
  latency?: number;
  sources?: string[];
  model_used?: string;
  cost?: number;
  routing_info?: Record<string, unknown>;
};

const strategies = ["cost_optimized", "balanced", "quality_first"] as const;

function authHeaders(token: string): Record<string, string> {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function readJson(response: Response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || response.statusText);
  }
  return data;
}

export default function App() {
  const [token, setToken] = useState(() => localStorage.getItem("smartroute.jwt") || "");
  const [strategy, setStrategy] = useState<(typeof strategies)[number]>("cost_optimized");
  const [useRetrieval, setUseRetrieval] = useState(false);
  const [mode, setMode] = useState<PromptMode>("search");
  const [health, setHealth] = useState("checking");
  const [ready, setReady] = useState("checking");
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [answer, setAnswer] = useState("");
  const [metadata, setMetadata] = useState<Record<string, unknown>>({});
  const [result, setResult] = useState<QueryResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [stats, setStats] = useState<Record<string, unknown>>({});
  const [budget, setBudget] = useState<Record<string, { spent?: number; limit?: number }>>({});
  const abortRef = useRef<AbortController | null>(null);

  const apiReady = health === "healthy" || health === "starting";
  const hasToken = token.trim().length > 0;

  const saveToken = useCallback((value: string) => {
    setToken(value);
    localStorage.setItem("smartroute.jwt", value);
  }, []);

  const loadDocuments = useCallback(async () => {
    if (!hasToken) {
      return;
    }
    const data = await fetch("/v1/documents", { headers: authHeaders(token) }).then(readJson);
    setDocuments(data.documents || []);
  }, [hasToken, token]);

  const loadMetrics = useCallback(async () => {
    if (!hasToken) {
      return;
    }
    const [statsData, budgetData] = await Promise.all([
      fetch("/v1/stats?days=1", { headers: authHeaders(token) }).then(readJson),
      fetch("/v1/budget", { headers: authHeaders(token) }).then(readJson)
    ]);
    setStats(statsData);
    setBudget(budgetData);
  }, [hasToken, token]);

  const refreshStatus = useCallback(async () => {
    const healthData = await fetch("/health").then(readJson).catch(() => ({ status: "offline" }));
    setHealth(healthData.status || "offline");

    const readyData = await fetch("/ready").then(readJson).catch(() => ({ status: "not_ready" }));
    setReady(readyData.status || "not_ready");
  }, []);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  useEffect(() => {
    if (!hasToken) {
      return;
    }
    loadDocuments().catch((error) => setNotice(error.message));
    loadMetrics().catch((error) => setNotice(error.message));
  }, [hasToken, loadDocuments, loadMetrics]);

  const totalCost = useMemo(() => formatCurrency(stats.total_cost), [stats]);

  async function submitPrompt(prompt: string) {
    setBusy(true);
    setAnswer("");
    setMetadata({});
    setResult(null);
    setNotice("");

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch("/v1/query/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...authHeaders(token)
        },
        body: JSON.stringify({
          query: prompt,
          strategy,
          use_retrieval: useRetrieval,
          session_id: `web-${mode}`
        }),
        signal: controller.signal
      });

      if (!response.ok || !response.body) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || response.statusText);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
          const line = part.split("\n").find((entry) => entry.startsWith("data: "));
          if (!line) {
            continue;
          }
          const item = JSON.parse(line.slice(6)) as StreamItem;
          if (item.type === "metadata") {
            setMetadata(item.data || {});
          }
          if (item.type === "chunk") {
            setAnswer((current) => current + (item.content || ""));
          }
          if (item.type === "done") {
            setResult(item.result || {});
          }
          if (item.type === "error") {
            throw new Error(item.content || "Stream failed");
          }
        }
      }

      loadMetrics().catch(() => undefined);
    } catch (error) {
      if ((error as Error).name !== "AbortError") {
        setNotice((error as Error).message);
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  async function uploadFiles(files: FileList) {
    if (!hasToken) {
      setNotice("Add a JWT before uploading documents.");
      return;
    }

    const form = new FormData();
    Array.from(files).forEach((file) => form.append("files", file));
    setNotice("Uploading and indexing documents...");

    try {
      await fetch("/v1/documents/upload", {
        method: "POST",
        headers: authHeaders(token),
        body: form
      }).then(readJson);
      setNotice("Documents indexed.");
      await loadDocuments();
    } catch (error) {
      setNotice((error as Error).message);
    }
  }

  async function deleteDocument(filename: string) {
    await fetch(`/v1/documents/${encodeURIComponent(filename)}`, {
      method: "DELETE",
      headers: authHeaders(token)
    }).then(readJson);
    await loadDocuments();
  }

  return (
    <main className="app-shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">SmartRoute AI</p>
          <h1>Inference Gateway</h1>
        </div>
        <div className="status-strip">
          <StatusPill label="Health" value={health} good={apiReady} />
          <StatusPill label="Ready" value={ready} good={ready === "ready"} />
        </div>
      </section>

      <section className="console-grid">
        <aside className="side-panel">
          <div className="panel-section">
            <label className="field-label" htmlFor="jwt">
              <KeyRound size={15} />
              JWT
            </label>
            <textarea
              id="jwt"
              value={token}
              onChange={(event) => saveToken(event.target.value)}
              className="token-input"
              rows={4}
              placeholder="Paste Supabase JWT"
            />
          </div>

          <div className="panel-section">
            <label className="field-label" htmlFor="strategy">
              <Activity size={15} />
              Strategy
            </label>
            <select
              id="strategy"
              value={strategy}
              onChange={(event) => setStrategy(event.target.value as typeof strategy)}
              className="select-input"
            >
              {strategies.map((item) => (
                <option key={item} value={item}>
                  {item.replace("_", " ")}
                </option>
              ))}
            </select>

            <label className="toggle-row">
              <input
                type="checkbox"
                checked={useRetrieval}
                onChange={(event) => setUseRetrieval(event.target.checked)}
              />
              <span>Use retrieval</span>
            </label>
          </div>

          <div className="panel-section">
            <div className="section-head">
              <span>
                <Database size={15} />
                Documents
              </span>
              <button className="mini-button" type="button" onClick={() => loadDocuments()}>
                <RefreshCw size={14} />
              </button>
            </div>

            <div className="document-list">
              {documents.length === 0 ? (
                <p className="muted">No indexed documents.</p>
              ) : (
                documents.map((doc) => (
                  <div className="document-row" key={doc.filename}>
                    <FileText size={15} />
                    <span>{doc.filename}</span>
                    <button
                      className="mini-button danger"
                      type="button"
                      onClick={() => deleteDocument(doc.filename).catch((e) => setNotice(e.message))}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </aside>

        <section className="workbench">
          <AiPromptBox
            disabled={!hasToken}
            loading={busy}
            retrievalEnabled={useRetrieval}
            mode={mode}
            onModeChange={setMode}
            onSubmit={submitPrompt}
            onStop={() => abortRef.current?.abort()}
            onFilesSelected={uploadFiles}
          />

          {notice && (
            <div className="notice">
              <AlertCircle size={16} />
              <span>{notice}</span>
            </div>
          )}

          <article className={cn("response-panel", busy && "streaming")}>
            <div className="section-head">
              <span>
                <CheckCircle2 size={16} />
                Response
              </span>
              <span className="mono">{result?.latency ? `${result.latency.toFixed(2)}s` : "idle"}</span>
            </div>
            <div className="answer-text">{answer || "The next answer will stream here."}</div>
          </article>
        </section>

        <aside className="metrics-panel">
          <MetricCard label="Queries" value={String(stats.total_queries || 0)} />
          <MetricCard label="Total cost" value={totalCost} />
          <MetricCard label="Model" value={result?.model_used || "waiting"} />
          <MetricCard label="Last cost" value={formatCurrency(result?.cost || 0)} />

          <div className="panel-section">
            <div className="section-head">
              <span>
                <UploadCloud size={15} />
                Sources
              </span>
            </div>
            {(metadata.sources as string[] | undefined)?.length || result?.sources?.length ? (
              <ul className="source-list">
                {((metadata.sources as string[] | undefined) || result?.sources || []).map((source) => (
                  <li key={source}>{source}</li>
                ))}
              </ul>
            ) : (
              <p className="muted">No source attributions yet.</p>
            )}
          </div>

          <div className="panel-section">
            <div className="section-head">
              <span>Budget</span>
            </div>
            {Object.entries(budget)
              .filter(([key]) => !["alert_threshold", "timestamp"].includes(key))
              .map(([key, value]) => (
                <div className="budget-row" key={key}>
                  <span>{key}</span>
                  <span>
                    {formatCurrency(value.spent || 0)} / ${Number(value.limit || 0).toFixed(2)}
                  </span>
                </div>
              ))}
          </div>
        </aside>
      </section>
    </main>
  );
}

function StatusPill({ label, value, good }: { label: string; value: string; good: boolean }) {
  return (
    <div className={cn("status-pill", good && "good")}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
