import React, { useState } from "react";
import { Check, Code2, Copy, FileCode, Maximize2, Minimize2, Sparkles, X } from "lucide-react";

interface CanvasDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  content?: string;
  language?: string;
}

export const CanvasDrawer: React.FC<CanvasDrawerProps> = ({
  isOpen,
  onClose,
  title = "Interactive Artifact Workspace",
  content = "// SmartRoute-AI Canvas Workspace\n// When in Canvas mode, generated code snippets, diagrams, and markdown\n// will render in this live side-by-side workspace.\n\nfunction welcome() {\n  return 'Canvas Mode Active';\n}",
  language = "typescript",
}) => {
  const [copied, setCopied] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);

  if (!isOpen) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <aside
      className={`fixed top-16 bottom-0 right-0 z-30 flex flex-col border-l border-[#2E3033] bg-[#121316] transition-all duration-300 shadow-2xl ${
        isExpanded ? "w-full md:w-[75vw]" : "w-full md:w-[480px] xl:w-[560px]"
      }`}
    >
      {/* Canvas Header */}
      <div className="flex h-14 items-center justify-between border-b border-[#2E3033] bg-[#18191D] px-4">
        <div className="flex items-center gap-2 overflow-hidden">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-orange-500/10 text-orange-400 border border-orange-500/30">
            <FileCode className="h-4 w-4" />
          </div>
          <div className="overflow-hidden">
            <h3 className="text-xs font-semibold text-white truncate">{title}</h3>
            <span className="text-[10px] text-gray-400 font-mono uppercase">{language}</span>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={handleCopy}
            className="flex items-center gap-1 rounded-lg border border-[#333333] px-2.5 py-1 text-xs text-gray-300 hover:bg-[#2E3033] hover:text-white transition-colors"
          >
            {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
            <span>{copied ? "Copied" : "Copy"}</span>
          </button>

          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            className="hidden sm:flex rounded-lg p-1.5 text-gray-400 hover:bg-[#2E3033] hover:text-white transition-colors"
            title={isExpanded ? "Collapse" : "Expand"}
          >
            {isExpanded ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          </button>

          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1.5 text-gray-400 hover:bg-[#2E3033] hover:text-white transition-colors"
            title="Close Canvas"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Code / Artifact Editor Body */}
      <div className="flex-1 overflow-auto p-4 font-mono text-xs text-gray-200 bg-[#0E0F12] scrollbar-thin scrollbar-thumb-[#333333] scrollbar-track-transparent leading-relaxed whitespace-pre-wrap">
        {content}
      </div>

      {/* Footer Info */}
      <div className="flex items-center justify-between border-t border-[#2E3033] bg-[#141518] px-4 py-2 text-[11px] text-gray-400">
        <span className="flex items-center gap-1">
          <Sparkles className="h-3 w-3 text-orange-400" />
          Live Artifact Sync
        </span>
        <span>Canvas View</span>
      </div>
    </aside>
  );
};
