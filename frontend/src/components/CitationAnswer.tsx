import { useMemo, useState } from "react";
import { FileText, X } from "lucide-react";

import { legacySourcesToCitations } from "@/lib/format";
import type { Citation } from "@/types/chat";

interface CitationAnswerProps {
  content: string;
  citations?: Citation[];
  sources?: string[];
  streaming?: boolean;
}

const MARKER_PATTERN = /(\[C\d+\])/g;

export function CitationAnswer({
  content,
  citations,
  sources = [],
  streaming = false,
}: CitationAnswerProps) {
  const [activeId, setActiveId] = useState<string | null>(null);
  const evidence = useMemo(
    () => (citations === undefined ? legacySourcesToCitations(sources) : citations),
    [citations, sources]
  );
  const citationById = useMemo(
    () => new Map(evidence.map((citation) => [citation.id, citation])),
    [evidence]
  );
  const activeCitation = activeId ? citationById.get(activeId) : undefined;

  const toggleCitation = (citationId: string) => {
    setActiveId((current) => (current === citationId ? null : citationId));
  };

  return (
    <>
      <div className="whitespace-pre-wrap leading-relaxed">
        {content.split(MARKER_PATTERN).map((segment, index) => {
          const marker = segment.match(/^\[C(\d+)\]$/);
          if (!marker) return <span key={`${segment}-${index}`}>{segment}</span>;

          const citationId = `C${marker[1]}`;
          const citation = citationById.get(citationId);
          if (!citation) return null;

          const location = citation.page
            ? `page ${citation.page}`
            : citation.section
            ? `section ${citation.section}`
            : "document";
          return (
            <button
              key={`${citationId}-${index}`}
              type="button"
              onClick={() => toggleCitation(citationId)}
              className="mx-0.5 inline-flex align-super text-[10px] font-semibold text-emerald-300 transition-colors hover:text-emerald-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-300/70"
              aria-label={`Citation ${marker[1]}: ${citation.filename}, ${location}`}
              aria-expanded={activeId === citationId}
            >
              [{marker[1]}]
            </button>
          );
        })}
        {streaming && (
          <span className="ml-1 inline-block h-3.5 w-1 animate-pulse bg-orange-300 align-middle" />
        )}
      </div>

      {!streaming && evidence.length > 0 && (
        <div className="mt-2.5 flex flex-wrap items-center gap-1.5" aria-label="Answer citations">
          {evidence.map((citation, index) => {
            const citationNumber = citation.id.replace(/^C/, "");
            const location = citation.page
              ? `Page ${citation.page}`
              : citation.section
              ? citation.section
              : "Document source";
            return (
              <button
                key={`${citation.id}-${citation.filename}-${index}`}
                type="button"
                onClick={() => toggleCitation(citation.id)}
                className={`relative flex h-7 w-7 flex-none items-center justify-center rounded-md border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-300/70 ${
                  activeId === citation.id
                    ? "border-emerald-300/60 bg-emerald-400/15 text-emerald-100"
                    : "border-white/10 bg-white/5 text-white/45 hover:border-emerald-300/30 hover:bg-emerald-400/10 hover:text-emerald-200"
                }`}
                title={`${citation.filename} - ${location}`}
                aria-label={`Citation ${citationNumber}: ${citation.filename}, ${location}`}
                aria-expanded={activeId === citation.id}
              >
                <FileText className="h-3.5 w-3.5" />
                <span className="absolute -right-1 -top-1 flex h-3.5 min-w-3.5 items-center justify-center rounded-full bg-emerald-300 px-0.5 text-[8px] font-bold text-emerald-950">
                  {citationNumber}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {activeCitation && (
        <div className="relative mt-2.5 max-w-sm border-l-2 border-emerald-300/50 bg-white/[0.04] px-3 py-2.5 text-left">
          <button
            type="button"
            onClick={() => setActiveId(null)}
            className="absolute right-1.5 top-1.5 p-1 text-white/35 transition-colors hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-300/70"
            aria-label="Close citation details"
          >
            <X className="h-3 w-3" />
          </button>
          <div className="pr-6 text-[11px] font-semibold text-white/90 break-words">
            {activeCitation.filename}
          </div>
          <div className="mt-0.5 text-[10px] text-emerald-200/75">
            {activeCitation.page
              ? `Page ${activeCitation.page}`
              : activeCitation.section
              ? `Section: ${activeCitation.section}`
              : "Document source"}
          </div>
          {activeCitation.excerpt && (
            <p className="mt-2 text-[11px] leading-relaxed text-white/55">
              {activeCitation.excerpt}
            </p>
          )}
        </div>
      )}
    </>
  );
}
