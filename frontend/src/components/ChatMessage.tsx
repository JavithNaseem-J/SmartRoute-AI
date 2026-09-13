import React, { useState } from "react";
import {
  Bot,
  Check,
  Clock,
  Coins,
  Copy,
  Cpu,
  FileText,
  HelpCircle,
  Layers,
  Sparkles,
  User,
} from "lucide-react";
import { ChatMessage as ChatMessageType } from "../types/chat";

interface ChatMessageProps {
  message: ChatMessageType;
  onOpenImage?: (url: string) => void;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message, onOpenImage }) => {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formatCurrency = (val?: number) => {
    if (val === undefined || isNaN(val)) return "$0.0000";
    return `$${val.toFixed(5)}`;
  };

  return (
    <div
      className={`group relative flex w-full gap-3 py-4 px-3 sm:px-6 transition-colors ${
        isUser ? "bg-transparent" : "bg-[#18191D]/60 border-y border-[#232429]"
      }`}
    >
      {/* Avatar */}
      <div className="flex-shrink-0">
        {isUser ? (
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-tr from-gray-700 to-gray-600 text-white shadow-sm">
            <User className="h-4 w-4" />
          </div>
        ) : (
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-600 text-white shadow-md shadow-indigo-600/30">
            <Bot className="h-4 w-4" />
          </div>
        )}
      </div>

      {/* Content Body */}
      <div className="flex-1 space-y-2 overflow-hidden">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-gray-200">
              {isUser ? "You" : "SmartRoute Assistant"}
            </span>
            <span className="text-[10px] text-gray-500 font-mono">
              {new Date(message.timestamp).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </span>
          </div>

          {!isUser && message.content && (
            <button
              type="button"
              onClick={handleCopy}
              className="opacity-0 group-hover:opacity-100 flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-gray-400 hover:bg-[#2A2B30] hover:text-white transition-all"
              title="Copy message"
            >
              {copied ? (
                <>
                  <Check className="h-3.5 w-3.5 text-emerald-400" />
                  <span className="text-emerald-400">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="h-3.5 w-3.5" />
                  <span>Copy</span>
                </>
              )}
            </button>
          )}
        </div>

        {/* Attached Files / Previews */}
        {message.files && message.files.length > 0 && (
          <div className="flex flex-wrap gap-2 pb-1">
            {message.files.map((f, i) => (
              <div key={i} className="flex items-center">
                {f.isImage && f.url ? (
                  <div
                    onClick={() => onOpenImage?.(f.url!)}
                    className="h-16 w-16 rounded-xl overflow-hidden cursor-pointer border border-[#444444] hover:border-indigo-500 transition-colors"
                  >
                    <img src={f.url} alt={f.name} className="h-full w-full object-cover" />
                  </div>
                ) : (
                  <div className="flex items-center gap-1.5 rounded-lg border border-[#333333] bg-[#1F2023] px-2.5 py-1 text-xs text-gray-300">
                    <FileText className="h-3.5 w-3.5 text-indigo-400" />
                    <span className="truncate max-w-[150px]">{f.name}</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Message Text */}
        <div className="text-sm leading-relaxed text-gray-200 whitespace-pre-wrap break-words font-sans">
          {message.content}
          {message.streaming && (
            <span className="inline-block h-4 w-1.5 ml-1 bg-indigo-400 animate-pulse align-middle" />
          )}
        </div>

        {/* Error Banner if any */}
        {message.error && (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-2.5 text-xs text-red-300">
            <p className="font-medium">Query Error</p>
            <p className="text-[11px] text-red-400/90">{message.error}</p>
          </div>
        )}

        {/* Routing Telemetry Badges (Assistant only) */}
        {!isUser && (message.model_used || message.latency || message.cost) && (
          <div className="flex flex-wrap items-center gap-2 pt-2 text-[11px]">
            {message.model_used && (
              <div className="flex items-center gap-1.5 rounded-lg border border-indigo-500/30 bg-indigo-500/10 px-2.5 py-1 text-indigo-300 font-mono">
                <Cpu className="h-3 w-3" />
                <span>{message.model_used}</span>
              </div>
            )}

            {message.latency !== undefined && (
              <div className="flex items-center gap-1.5 rounded-lg border border-[#333333] bg-[#1A1B1E] px-2.5 py-1 text-gray-300 font-mono">
                <Clock className="h-3 w-3 text-amber-400" />
                <span>{message.latency.toFixed(2)}s</span>
              </div>
            )}

            {message.cost !== undefined && (
              <div className="flex items-center gap-1.5 rounded-lg border border-[#333333] bg-[#1A1B1E] px-2.5 py-1 text-gray-300 font-mono">
                <Coins className="h-3 w-3 text-emerald-400" />
                <span>{formatCurrency(message.cost)}</span>
              </div>
            )}

            {message.confidence !== undefined && (
              <div className="flex items-center gap-1.5 rounded-lg border border-[#333333] bg-[#1A1B1E] px-2.5 py-1 text-gray-300 font-mono">
                <Sparkles className="h-3 w-3 text-violet-400" />
                <span>{(message.confidence * 100).toFixed(0)}% conf</span>
              </div>
            )}

            {message.complexity && (
              <div className="flex items-center gap-1 rounded-lg border border-[#333333] bg-[#1A1B1E] px-2 py-1 text-gray-400 uppercase text-[10px] tracking-wider">
                <span>{message.complexity}</span>
              </div>
            )}
          </div>
        )}

        {/* Source Citations from RAG */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="rounded-xl border border-[#2E3033] bg-[#141518] p-2.5 mt-2 space-y-1.5">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-gray-300">
              <Layers className="h-3.5 w-3.5 text-emerald-400" />
              <span>Knowledge Sources ({message.sources.length})</span>
            </div>
            <ul className="space-y-1">
              {message.sources.map((source, idx) => (
                <li
                  key={idx}
                  className="flex items-center gap-1.5 text-[11px] text-gray-400 hover:text-gray-200"
                >
                  <span className="h-1 w-1 rounded-full bg-emerald-400" />
                  <span className="truncate">{source}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};
