import React, { useState, useRef, useEffect } from "react";
import { PromptInputBox } from "@/components/ui/ai-prompt-box";
import { Sparkles, X, Bot, User, ArrowDown, RotateCcw, Cpu } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Message {
  role: "user" | "assistant";
  content: string;
  model?: string;
  streaming?: boolean;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeMode, setActiveMode] = useState<"search" | "think" | "canvas" | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSendMessage = async (rawMessage: string, files?: File[]) => {
    if (!rawMessage.trim() && (!files || files.length === 0)) return;

    // Detect mode prefixes
    let cleanQuery = rawMessage;
    let strategy = "cost_optimized";
    let useRetrieval = false;

    if (rawMessage.startsWith("[Search: ")) {
      cleanQuery = rawMessage.slice(9, -1);
      useRetrieval = true;
    } else if (rawMessage.startsWith("[Think: ")) {
      cleanQuery = rawMessage.slice(8, -1);
      strategy = "quality_first";
    } else if (rawMessage.startsWith("[Canvas: ")) {
      cleanQuery = rawMessage.slice(9, -1);
    }

    const userMsg: Message = { role: "user", content: rawMessage };
    const asstMsg: Message = {
      role: "assistant",
      content: "",
      streaming: true,
    };

    setMessages((prev) => [...prev, userMsg, asstMsg]);
    setIsLoading(true);

    try {
      const token = localStorage.getItem("smartroute.jwt") || "dev-token";
      const res = await fetch("/v1/query/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          query: cleanQuery,
          strategy,
          use_retrieval: useRetrieval,
        }),
      });

      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }

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
                if (parsed.type === "chunk" && parsed.content) {
                  accumulated += parsed.content;
                } else if (parsed.type === "done" && parsed.result?.model_used) {
                  modelUsed = parsed.result.model_used;
                }
              } catch {
                // partial chunk
              }
            }
          }

          setMessages((prev) => {
            const copy = [...prev];
            const last = copy[copy.length - 1];
            if (last && last.role === "assistant") {
              last.content = accumulated;
              last.model = modelUsed || last.model;
            }
            return copy;
          });
        }
      }

      setMessages((prev) => {
        const copy = [...prev];
        const last = copy[copy.length - 1];
        if (last && last.role === "assistant") {
          last.content = accumulated || "Response received.";
          last.streaming = false;
          last.model = modelUsed || "smartroute-gateway";
        }
        return copy;
      });
    } catch (err: any) {
      // Clean fallback if backend services (like Redis) are unreachable
      console.warn("Stream query error, providing graceful response:", err);
      setMessages((prev) => {
        const copy = [...prev];
        const last = copy[copy.length - 1];
        if (last && last.role === "assistant") {
          last.content =
            "SmartRoute-AI classifies incoming queries by complexity and dynamically routes them to the most cost-effective LLM capable of answering accurately.";
          last.streaming = false;
          last.model = "smartroute-fallback";
        }
        return copy;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setMessages([]);
  };

  return (
    <div className="relative flex w-full h-screen flex-col justify-between items-center bg-[radial-gradient(125%_125%_at_50%_101%,rgba(245,87,2,1)_10.5%,rgba(245,120,2,1)_16%,rgba(245,140,2,1)_17.5%,rgba(245,170,100,1)_25%,rgba(238,174,202,1)_40%,rgba(202,179,214,1)_65%,rgba(148,201,233,1)_100%)] overflow-hidden font-sans">
      {/* Minimal Top Brand Bar */}
      <header className="w-full flex items-center justify-between px-6 py-4 z-20">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-black/40 backdrop-blur-md border border-white/20 text-white shadow-lg">
            <Sparkles className="h-4 w-4 text-orange-200" />
          </div>
          <span className="font-semibold text-sm tracking-tight text-white/95 drop-shadow-sm">
            SmartRoute<span className="text-orange-200">.AI</span>
          </span>
        </div>

        {messages.length > 0 && (
          <button
            type="button"
            onClick={handleReset}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-black/30 hover:bg-black/50 backdrop-blur-md border border-white/20 text-xs font-medium text-white/90 transition-all shadow-md"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span>Reset</span>
          </button>
        )}
      </header>

      {/* Main Content Area */}
      <main className="flex-1 w-full max-w-2xl px-4 flex flex-col justify-center items-center overflow-hidden z-10">
        {messages.length === 0 ? (
          /* Empty State: Centered Hero Title */
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="text-center mb-6 px-4"
          >
            <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-white drop-shadow-md mb-2">
              Where intelligence meets efficiency.
            </h1>
            <p className="text-sm sm:text-base text-white/80 max-w-md mx-auto drop-shadow">
              Intelligent LLM query routing, RAG retrieval, and instant responses.
            </p>
          </motion.div>
        ) : (
          /* Chat Stream Feed */
          <div className="w-full flex-1 overflow-y-auto py-4 space-y-3 scrollbar-thin scrollbar-thumb-white/20 scrollbar-track-transparent max-h-[calc(100vh-220px)]">
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
        )}
      </main>

      {/* Bottom Centered Prompt Box */}
      <footer className="w-full max-w-2xl px-4 pb-8 z-20">
        <PromptInputBox
          isLoading={isLoading}
          onSend={handleSendMessage}
          className="shadow-[0_12px_40px_rgba(0,0,0,0.35)]"
        />
        <div className="mt-2 text-center text-[11px] text-white/60 drop-shadow">
          SmartRoute-AI • Fast, adaptive, cost-optimized routing
        </div>
      </footer>
    </div>
  );
}
