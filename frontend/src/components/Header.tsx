import React, { useState } from "react";
import {
  Activity,
  CheckCircle2,
  Cpu,
  Database,
  Eye,
  KeyRound,
  LayoutGrid,
  Menu,
  Sparkles,
  X,
  Zap,
} from "lucide-react";
import { RoutingStrategy } from "../types/chat";

interface HeaderProps {
  viewMode: "studio" | "demo";
  onViewModeChange: (mode: "studio" | "demo") => void;
  strategy: RoutingStrategy;
  onStrategyChange: (strategy: RoutingStrategy) => void;
  useRetrieval: boolean;
  onToggleRetrieval: () => void;
  health: string;
  ready: string;
  token: string;
  onSaveToken: (token: string) => void;
  onToggleSidebar: () => void;
  sidebarOpen: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  viewMode,
  onViewModeChange,
  strategy,
  onStrategyChange,
  useRetrieval,
  onToggleRetrieval,
  health,
  ready,
  token,
  onSaveToken,
  onToggleSidebar,
}) => {
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [tokenInput, setTokenInput] = useState(token);

  const handleGenerateDevToken = async () => {
    // Generate a valid HMAC-SHA256 JWT using Web Crypto API with dev secret
    const secret = "super-secret-jwt-token-with-at-least-32-characters-long";
    const header = { alg: "HS256", typ: "JWT" };
    const payload = {
      sub: "dev-user-001",
      role: "authenticated",
      exp: Math.floor(Date.now() / 1000) + 60 * 60 * 24 * 30, // 30 days
    };

    const base64UrlEncode = (str: string) =>
      btoa(str).replace(/=/g, "").replace(/\+/g, "-").replace(/\//g, "_");

    const headerEnc = base64UrlEncode(JSON.stringify(header));
    const payloadEnc = base64UrlEncode(JSON.stringify(payload));
    const dataToSign = `${headerEnc}.${payloadEnc}`;

    try {
      const enc = new TextEncoder();
      const key = await crypto.subtle.importKey(
        "raw",
        enc.encode(secret),
        { name: "HMAC", hash: "SHA-256" },
        false,
        ["sign"]
      );
      const signature = await crypto.subtle.sign("HMAC", key, enc.encode(dataToSign));
      const sigArray = Array.from(new Uint8Array(signature));
      const sigString = String.fromCharCode.apply(null, sigArray);
      const sigEnc = base64UrlEncode(sigString);
      const fullToken = `${dataToSign}.${sigEnc}`;
      setTokenInput(fullToken);
    } catch (e) {
      console.error("Token generation error:", e);
      // Fallback standard dev bearer token
      setTokenInput("smartroute-dev-authenticated-bearer-token");
    }
  };

  const isHealthy = health === "healthy" || health === "ok";

  return (
    <>
      <header className="sticky top-0 z-40 flex h-16 w-full items-center justify-between border-b border-[#2E3033] bg-[#121316]/90 px-4 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onToggleSidebar}
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-[#333333] text-gray-300 transition-colors hover:bg-[#1F2023] hover:text-white"
            title="Toggle Sidebar"
          >
            <Menu className="h-5 w-5" />
          </button>

          <div className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-violet-600 to-indigo-500 shadow-md shadow-violet-500/20">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-semibold tracking-tight text-white md:text-base">
                  SmartRoute<span className="text-indigo-400">.AI</span>
                </h1>
                <span className="hidden rounded-md bg-indigo-500/10 px-1.5 py-0.5 text-[10px] font-medium text-indigo-300 border border-indigo-500/20 sm:inline-block">
                  v2.0
                </span>
              </div>
              <p className="text-[11px] text-gray-400 hidden sm:block">
                Intelligent LLM Router & Knowledge Gateway
              </p>
            </div>
          </div>
        </div>

        {/* View mode switcher */}
        <div className="flex items-center rounded-xl border border-[#333333] bg-[#1a1b1e] p-1">
          <button
            type="button"
            onClick={() => onViewModeChange("studio")}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
              viewMode === "studio"
                ? "bg-[#2E3033] text-white shadow-sm"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            <LayoutGrid className="h-3.5 w-3.5" />
            <span>Studio</span>
          </button>

          <button
            type="button"
            onClick={() => onViewModeChange("demo")}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
              viewMode === "demo"
                ? "bg-gradient-to-r from-orange-500 to-pink-500 text-white shadow-sm"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            <Eye className="h-3.5 w-3.5" />
            <span>Showcase</span>
          </button>
        </div>

        {/* Status & Settings */}
        <div className="flex items-center gap-2">
          {/* Strategy selector dropdown */}
          <div className="hidden lg:flex items-center rounded-lg border border-[#333333] bg-[#1a1b1e] px-2 py-1 text-xs">
            <Cpu className="h-3.5 w-3.5 text-gray-400 mr-1.5" />
            <select
              value={strategy}
              onChange={(e) => onStrategyChange(e.target.value as RoutingStrategy)}
              className="bg-transparent text-gray-200 focus:outline-none cursor-pointer text-xs"
            >
              <option value="cost_optimized" className="bg-[#1a1b1e] text-gray-200">
                Cost Optimized
              </option>
              <option value="balanced" className="bg-[#1a1b1e] text-gray-200">
                Balanced
              </option>
              <option value="quality_first" className="bg-[#1a1b1e] text-gray-200">
                Quality First
              </option>
            </select>
          </div>

          {/* RAG Toggle */}
          <button
            type="button"
            onClick={onToggleRetrieval}
            className={`hidden sm:flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors ${
              useRetrieval
                ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                : "border-[#333333] bg-[#1a1b1e] text-gray-400 hover:text-gray-200"
            }`}
            title="Toggle RAG Document Retrieval"
          >
            <Database className="h-3.5 w-3.5" />
            <span>RAG: {useRetrieval ? "On" : "Off"}</span>
          </button>

          {/* Health Indicator */}
          <div
            className={`flex items-center gap-1.5 rounded-lg border px-2 py-1 text-xs font-medium ${
              isHealthy
                ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
                : "border-amber-500/30 bg-amber-500/10 text-amber-400"
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                isHealthy ? "bg-emerald-400 animate-pulse" : "bg-amber-400"
              }`}
            />
            <span className="hidden md:inline">{isHealthy ? "Healthy" : "Offline"}</span>
          </div>

          {/* Auth Key Button */}
          <button
            type="button"
            onClick={() => {
              setTokenInput(token);
              setAuthModalOpen(true);
            }}
            className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors ${
              token
                ? "border-indigo-500/40 bg-indigo-500/10 text-indigo-300 hover:bg-indigo-500/20"
                : "border-red-500/40 bg-red-500/10 text-red-300 animate-pulse hover:bg-red-500/20"
            }`}
          >
            <KeyRound className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">{token ? "Token Active" : "Set Token"}</span>
          </button>
        </div>
      </header>

      {/* Auth Settings Dialog */}
      {authModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
          <div className="relative w-full max-w-md rounded-2xl border border-[#333333] bg-[#1F2023] p-6 shadow-2xl">
            <button
              type="button"
              onClick={() => setAuthModalOpen(false)}
              className="absolute right-4 top-4 rounded-full p-1 text-gray-400 hover:bg-[#2E3033] hover:text-white"
            >
              <X className="h-5 w-5" />
            </button>

            <div className="flex items-center gap-3 mb-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                <KeyRound className="h-5 w-5" />
              </div>
              <div>
                <h3 className="text-base font-semibold text-white">API Authentication</h3>
                <p className="text-xs text-gray-400">
                  SmartRoute-AI requires a JWT bearer token for API endpoints.
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1.5">
                  JWT Bearer Token
                </label>
                <textarea
                  value={tokenInput}
                  onChange={(e) => setTokenInput(e.target.value)}
                  rows={4}
                  placeholder="Paste your JWT token here..."
                  className="w-full rounded-xl border border-[#444444] bg-[#121316] p-3 text-xs text-gray-200 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 font-mono"
                />
              </div>

              <div className="rounded-xl border border-indigo-500/20 bg-indigo-500/5 p-3 text-xs text-indigo-200">
                <div className="flex items-center justify-between">
                  <span className="font-medium">Local Development?</span>
                  <button
                    type="button"
                    onClick={handleGenerateDevToken}
                    className="flex items-center gap-1 rounded-lg bg-indigo-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-indigo-500 transition-colors"
                  >
                    <Zap className="h-3.5 w-3.5" />
                    Auto-Generate Dev JWT
                  </button>
                </div>
                <p className="mt-1 text-[11px] text-gray-400">
                  Creates a signed token using the backend development secret key.
                </p>
              </div>

              <div className="flex items-center justify-end gap-2 pt-2 border-t border-[#333333]">
                {token && (
                  <button
                    type="button"
                    onClick={() => {
                      onSaveToken("");
                      setTokenInput("");
                      setAuthModalOpen(false);
                    }}
                    className="rounded-lg px-3 py-1.5 text-xs text-red-400 hover:bg-red-500/10 transition-colors"
                  >
                    Clear Token
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => {
                    onSaveToken(tokenInput.trim());
                    setAuthModalOpen(false);
                  }}
                  className="flex items-center gap-1.5 rounded-lg bg-indigo-600 px-4 py-2 text-xs font-medium text-white hover:bg-indigo-500 shadow-md shadow-indigo-600/20 transition-all"
                >
                  <CheckCircle2 className="h-4 w-4" />
                  Save & Apply
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
