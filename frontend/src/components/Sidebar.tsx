import React, { useRef } from "react";
import {
  DollarSign,
  FileText,
  MessageSquare,
  Plus,
  RefreshCw,
  Trash2,
  UploadCloud,
  X,
  TrendingDown,
  Layers,
} from "lucide-react";
import { BudgetRecord, ChatSession, DocumentItem, StatsResponse } from "../types/chat";

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  sessions: ChatSession[];
  activeSessionId: string;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
  documents: DocumentItem[];
  onUploadDocuments: (files: FileList) => void;
  onDeleteDocument: (filename: string) => void;
  onIndexDocuments: () => void;
  budget: Record<string, BudgetRecord>;
  stats: StatsResponse;
  isIndexing: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  onClose,
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  documents,
  onUploadDocuments,
  onDeleteDocument,
  onIndexDocuments,
  budget,
  stats,
  isIndexing,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  };

  const formatCurrency = (val?: number) => {
    if (val === undefined || isNaN(val)) return "$0.0000";
    return `$${val.toFixed(4)}`;
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm lg:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`fixed top-16 bottom-0 left-0 z-40 flex w-80 flex-col border-r border-[#2E3033] bg-[#121316] transition-transform duration-300 ease-in-out lg:static lg:translate-x-0 ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {/* Top Action: New Chat */}
        <div className="p-3 border-b border-[#2E3033] flex items-center justify-between gap-2">
          <button
            type="button"
            onClick={onNewSession}
            className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 px-4 py-2.5 text-xs font-semibold text-white shadow-md shadow-indigo-600/20 hover:from-indigo-500 hover:to-violet-500 transition-all"
          >
            <Plus className="h-4 w-4" />
            <span>New Chat Session</span>
          </button>

          <button
            type="button"
            onClick={onClose}
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-[#333333] text-gray-400 hover:bg-[#1F2023] hover:text-white lg:hidden"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Scrollable Middle: Sessions & Documents */}
        <div className="flex-1 overflow-y-auto p-3 space-y-6 scrollbar-thin scrollbar-thumb-[#333333] scrollbar-track-transparent">
          {/* Chat Sessions */}
          <div>
            <div className="flex items-center justify-between mb-2 px-1">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-400">
                Conversations ({sessions.length})
              </span>
            </div>

            <div className="space-y-1">
              {sessions.length === 0 ? (
                <p className="px-2 py-3 text-xs text-gray-500 text-center">
                  No previous conversations yet.
                </p>
              ) : (
                sessions.map((session) => (
                  <div
                    key={session.id}
                    className={`group relative flex items-center justify-between rounded-xl px-3 py-2 text-xs transition-all cursor-pointer ${
                      session.id === activeSessionId
                        ? "bg-[#1F2023] text-white border border-[#444444]"
                        : "text-gray-400 hover:bg-[#1A1B1E] hover:text-gray-200"
                    }`}
                    onClick={() => onSelectSession(session.id)}
                  >
                    <div className="flex items-center gap-2 overflow-hidden pr-2">
                      <MessageSquare className="h-3.5 w-3.5 flex-shrink-0 text-indigo-400" />
                      <span className="truncate">{session.title || "Untitled Chat"}</span>
                    </div>

                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteSession(session.id);
                      }}
                      className="opacity-0 group-hover:opacity-100 rounded p-1 text-gray-400 hover:text-red-400 transition-opacity"
                      title="Delete Session"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Knowledge Base Documents */}
          <div className="pt-2 border-t border-[#2E3033]">
            <div className="flex items-center justify-between mb-2 px-1">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-400 flex items-center gap-1.5">
                <Layers className="h-3.5 w-3.5 text-indigo-400" />
                Knowledge Base ({documents.length})
              </span>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={onIndexDocuments}
                  disabled={isIndexing}
                  className="rounded p-1 text-gray-400 hover:text-indigo-400 transition-colors"
                  title="Trigger Document Re-indexing"
                >
                  <RefreshCw className={`h-3.5 w-3.5 ${isIndexing ? "animate-spin text-indigo-400" : ""}`} />
                </button>
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="rounded p-1 text-gray-400 hover:text-indigo-400 transition-colors"
                  title="Upload Document (PDF, MD, TXT)"
                >
                  <UploadCloud className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.txt,.md,.docx,.json"
              className="hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files.length > 0) {
                  onUploadDocuments(e.target.files);
                  e.target.value = "";
                }
              }}
            />

            <div className="space-y-1">
              {documents.length === 0 ? (
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="border border-dashed border-[#333333] rounded-xl p-3 text-center cursor-pointer hover:border-indigo-500/50 transition-colors"
                >
                  <UploadCloud className="h-5 w-5 mx-auto text-gray-500 mb-1" />
                  <p className="text-[11px] text-gray-400">Click to upload knowledge docs</p>
                  <p className="text-[9px] text-gray-500">Supports PDF, MD, TXT</p>
                </div>
              ) : (
                documents.map((doc) => (
                  <div
                    key={doc.filename}
                    className="group flex items-center justify-between rounded-lg bg-[#1A1B1E] px-2.5 py-1.5 text-xs text-gray-300 border border-[#2E3033]"
                  >
                    <div className="flex items-center gap-2 overflow-hidden pr-2">
                      <FileText className="h-3.5 w-3.5 flex-shrink-0 text-emerald-400" />
                      <div className="overflow-hidden">
                        <p className="truncate text-gray-200 text-[11px]">{doc.filename}</p>
                        <p className="text-[9px] text-gray-500">{formatBytes(doc.size_bytes)}</p>
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={() => onDeleteDocument(doc.filename)}
                      className="opacity-0 group-hover:opacity-100 rounded p-1 text-gray-400 hover:text-red-400 transition-opacity"
                      title="Delete Document"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Bottom Section: Cost & Telemetry Cards */}
        <div className="p-3 border-t border-[#2E3033] bg-[#0E0F12] space-y-2.5 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
              Telemetry & Budget
            </span>
            <span className="text-[10px] text-emerald-400 font-mono flex items-center gap-0.5">
              <TrendingDown className="h-3 w-3" />
              {stats.savings_percentage ? `${stats.savings_percentage.toFixed(0)}% saved` : "active"}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div className="rounded-xl border border-[#2E3033] bg-[#16171B] p-2">
              <span className="text-[10px] text-gray-400 block">Total Spent</span>
              <strong className="text-sm font-semibold text-white font-mono">
                {formatCurrency(stats.total_cost || 0)}
              </strong>
            </div>

            <div className="rounded-xl border border-[#2E3033] bg-[#16171B] p-2">
              <span className="text-[10px] text-gray-400 block">Queries Routed</span>
              <strong className="text-sm font-semibold text-white font-mono">
                {stats.total_queries || 0}
              </strong>
            </div>
          </div>

          {/* Budget status progress bar */}
          {budget && Object.entries(budget).length > 0 && (
            <div className="space-y-1.5 pt-1">
              {Object.entries(budget)
                .filter(([k]) => !["alert_threshold", "timestamp"].includes(k))
                .slice(0, 2)
                .map(([name, rec]) => {
                  const spent = rec.spent || 0;
                  const limit = rec.limit || 10;
                  const pct = Math.min(100, Math.round((spent / limit) * 100));
                  return (
                    <div key={name} className="space-y-0.5">
                      <div className="flex justify-between text-[10px] text-gray-400">
                        <span className="capitalize">{name}</span>
                        <span className="font-mono">
                          ${spent.toFixed(2)} / ${limit.toFixed(2)}
                        </span>
                      </div>
                      <div className="h-1.5 w-full rounded-full bg-[#2E3033] overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            pct > 80 ? "bg-red-500" : pct > 50 ? "bg-amber-400" : "bg-emerald-500"
                          }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
            </div>
          )}
        </div>
      </aside>
    </>
  );
};
