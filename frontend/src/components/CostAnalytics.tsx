import React, { useState, useEffect, useCallback } from "react";
import {
  DollarSign,
  TrendingDown,
  Clock,
  Cpu,
  RefreshCw,
  Zap,
  ShieldCheck,
  Scale,
  Layers,
  AlertCircle,
  BarChart3,
} from "lucide-react";
import { motion } from "framer-motion";

interface StatsData {
  total_queries: number;
  total_cost: number;
  avg_cost_per_query: number;
  total_input_tokens: number;
  total_output_tokens: number;
  avg_latency: number;
  by_model: Record<string, { count: number; cost: number }>;
  by_complexity: Record<string, { count: number; cost: number }>;
  by_strategy: Record<string, { count: number; cost: number }>;
}

interface SavingsData {
  baseline_cost: number;
  actual_cost: number;
  savings: number;
  percentage: number;
}

interface BudgetData {
  daily: { spent: number; limit: number; remaining: number; percentage: number; alert: boolean };
  weekly: { spent: number; limit: number; remaining: number; percentage: number; alert: boolean };
  monthly: { spent: number; limit: number; remaining: number; percentage: number; alert: boolean };
}

const getAuthToken = () => localStorage.getItem("smartroute.jwt")?.trim() || null;

export function CostAnalytics() {
  const [days, setDays] = useState<number>(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [stats, setStats] = useState<StatsData | null>(null);
  const [savings, setSavings] = useState<SavingsData | null>(null);
  const [budget, setBudget] = useState<BudgetData | null>(null);

  const token = getAuthToken();

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      if (!token) {
        setStats(null);
        setSavings(null);
        setBudget(null);
        setError("Authentication token is missing. Sign in again to view analytics.");
        return;
      }

      const headers = { Authorization: `Bearer ${token}` };

      const [statsRes, savingsRes, budgetRes] = await Promise.all([
        fetch(`/v1/stats?days=${days}`, { headers }),
        fetch(`/v1/savings?days=${days}`, { headers }),
        fetch("/v1/budget", { headers }),
      ]);

      if (!statsRes.ok) throw new Error(`Stats returned ${statsRes.status}`);
      const statsJson = await statsRes.json();
      const savingsJson = savingsRes.ok ? await savingsRes.json() : null;
      const budgetJson = budgetRes.ok ? await budgetRes.json() : null;

      setStats(statsJson);
      setSavings(savingsJson);
      setBudget(budgetJson);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to load telemetry";
      console.error("Failed to load analytics:", err);
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [days, token]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Model friendly names & tier colors
  const formatModelName = (modelId: string) => {
    if (modelId.includes("liquid")) return { name: "Liquid LFM 2.5", tier: "Simple", color: "bg-emerald-400" };
    if (modelId.includes("nemotron")) return { name: "Nvidia Nemotron", tier: "Medium", color: "bg-blue-400" };
    if (modelId.includes("gpt-oss")) return { name: "OpenAI GPT-OSS 20B", tier: "Medium", color: "bg-blue-400" };
    if (modelId.includes("gemma")) return { name: "Google Gemma 31B", tier: "Complex", color: "bg-purple-400" };
    if (modelId.includes("free")) return { name: "OpenRouter Free Fallback", tier: "Fallback", color: "bg-amber-400" };
    return { name: modelId, tier: "Custom", color: "bg-gray-400" };
  };

  const totalQueries = stats?.total_queries || 0;

  return (
    <div className="flex flex-col h-full overflow-y-auto p-6 space-y-6 scrollbar-thin scrollbar-thumb-white/20 scrollbar-track-transparent">
      {/* Header bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-2 border-b border-white/10">
        <div>
          <div className="flex items-center gap-2">
            <BarChart3 className="h-6 w-6 text-orange-300" />
            <h1 className="text-2xl font-bold text-white tracking-tight">Cost & Routing Analytics</h1>
          </div>
          <p className="text-xs sm:text-sm text-white/70 mt-1">
            Real-time inference spend, query complexity telemetry, and model savings.
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Time range selector */}
          <div className="flex items-center bg-black/40 border border-white/10 rounded-xl p-1 text-xs">
            <button
              onClick={() => setDays(1)}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                days === 1 ? "bg-white/20 text-white font-medium shadow-sm" : "text-white/60 hover:text-white"
              }`}
            >
              Today
            </button>
            <button
              onClick={() => setDays(7)}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                days === 7 ? "bg-white/20 text-white font-medium shadow-sm" : "text-white/60 hover:text-white"
              }`}
            >
              7 Days
            </button>
            <button
              onClick={() => setDays(30)}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                days === 30 ? "bg-white/20 text-white font-medium shadow-sm" : "text-white/60 hover:text-white"
              }`}
            >
              30 Days
            </button>
          </div>

          <button
            onClick={fetchData}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-black/40 hover:bg-black/60 border border-white/15 text-white/80 hover:text-white text-xs transition-all shadow-sm"
            title="Refresh analytics"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 p-4 rounded-xl bg-red-500/20 border border-red-500/30 text-red-200 text-sm">
          <AlertCircle className="h-4 w-4 flex-shrink-0 text-red-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Hero KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Spend */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-2xl bg-black/35 backdrop-blur-xl border border-white/10 p-5 shadow-xl"
        >
          <div className="flex items-center justify-between text-white/60 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Total Spend</span>
            <div className="p-2 rounded-lg bg-emerald-500/20 text-emerald-400">
              <DollarSign className="h-4 w-4" />
            </div>
          </div>
          <div className="text-3xl font-bold text-white tracking-tight">
            ${stats ? stats.total_cost.toFixed(4) : "0.0000"}
          </div>
          <div className="mt-2 text-xs text-white/50">
            Avg ${(stats?.avg_cost_per_query || 0).toFixed(4)} / query ({totalQueries} queries)
          </div>
        </motion.div>

        {/* Cost Savings */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
          className="rounded-2xl bg-black/35 backdrop-blur-xl border border-white/10 p-5 shadow-xl"
        >
          <div className="flex items-center justify-between text-white/60 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Cost Savings</span>
            <div className="p-2 rounded-lg bg-orange-500/20 text-orange-300">
              <TrendingDown className="h-4 w-4" />
            </div>
          </div>
          <div className="text-3xl font-bold text-white tracking-tight">
            ${savings ? savings.savings.toFixed(4) : "0.0000"}
          </div>
          <div className="mt-2 text-xs text-emerald-300 font-medium">
            {savings ? `${savings.percentage.toFixed(1)}%` : "0.0%"} saved vs. static baseline
          </div>
        </motion.div>

        {/* Avg Latency */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="rounded-2xl bg-black/35 backdrop-blur-xl border border-white/10 p-5 shadow-xl"
        >
          <div className="flex items-center justify-between text-white/60 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Avg Latency</span>
            <div className="p-2 rounded-lg bg-blue-500/20 text-blue-400">
              <Clock className="h-4 w-4" />
            </div>
          </div>
          <div className="text-3xl font-bold text-white tracking-tight">
            {stats ? `${stats.avg_latency.toFixed(2)}s` : "0.00s"}
          </div>
          <div className="mt-2 text-xs text-white/50">
            End-to-end routing + LLM streaming
          </div>
        </motion.div>

        {/* Total Tokens */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="rounded-2xl bg-black/35 backdrop-blur-xl border border-white/10 p-5 shadow-xl"
        >
          <div className="flex items-center justify-between text-white/60 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Total Tokens</span>
            <div className="p-2 rounded-lg bg-purple-500/20 text-purple-400">
              <Cpu className="h-4 w-4" />
            </div>
          </div>
          <div className="text-3xl font-bold text-white tracking-tight">
            {stats
              ? (stats.total_input_tokens + stats.total_output_tokens).toLocaleString()
              : "0"}
          </div>
          <div className="mt-2 text-xs text-white/50">
            In: {(stats?.total_input_tokens || 0).toLocaleString()} • Out:{" "}
            {(stats?.total_output_tokens || 0).toLocaleString()}
          </div>
        </motion.div>
      </div>

      {/* Second Row: Spend by Model & Spend by Complexity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Breakdown */}
        <div className="rounded-2xl bg-black/35 backdrop-blur-xl border border-white/10 p-5 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Layers className="h-4 w-4 text-orange-300" />
              <h2 className="text-sm font-semibold text-white tracking-wide uppercase">
                Spend & Volume by Model
              </h2>
            </div>
            <span className="text-xs text-white/50">{Object.keys(stats?.by_model || {}).length} models tracked</span>
          </div>

          {!stats || Object.keys(stats.by_model).length === 0 ? (
            <div className="py-8 text-center text-xs text-white/40">
              No queries logged in this timeframe yet.
            </div>
          ) : (
            <div className="space-y-3.5">
              {Object.entries(stats.by_model).map(([modelId, data]) => {
                const info = formatModelName(modelId);
                const percent = totalQueries > 0 ? (data.count / totalQueries) * 100 : 0;
                return (
                  <div key={modelId} className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className={`w-2 h-2 rounded-full ${info.color}`} />
                        <span className="font-medium text-white/90">{info.name}</span>
                        <span className="rounded bg-white/10 px-1.5 py-0.5 text-[10px] text-white/60">
                          {info.tier}
                        </span>
                      </div>
                      <div className="flex items-center gap-3 text-white/70">
                        <span>{data.count} queries ({percent.toFixed(0)}%)</span>
                        <span className="font-semibold text-white">${data.cost.toFixed(4)}</span>
                      </div>
                    </div>
                    <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${info.color} rounded-full transition-all duration-500`}
                        style={{ width: `${Math.max(percent, 2)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Complexity Breakdown */}
        <div className="rounded-2xl bg-black/35 backdrop-blur-xl border border-white/10 p-5 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-orange-300" />
              <h2 className="text-sm font-semibold text-white tracking-wide uppercase">
                Spend by Query Complexity
              </h2>
            </div>
            <span className="text-xs text-white/50">Semantic Classifier</span>
          </div>

          {!stats || Object.keys(stats.by_complexity).length === 0 ? (
            <div className="py-8 text-center text-xs text-white/40">
              No complexity telemetry logged in this timeframe yet.
            </div>
          ) : (
            <div className="space-y-4">
              {(["simple", "medium", "complex"] as const).map((tier) => {
                const tierData = stats.by_complexity[tier] || { count: 0, cost: 0 };
                const percent = totalQueries > 0 ? (tierData.count / totalQueries) * 100 : 0;
                const tierColor =
                  tier === "simple"
                    ? "bg-emerald-400"
                    : tier === "medium"
                    ? "bg-blue-400"
                    : "bg-purple-400";

                return (
                  <div key={tier} className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2 capitalize">
                        <span className={`w-2 h-2 rounded-full ${tierColor}`} />
                        <span className="font-medium text-white/90">{tier} Queries</span>
                      </div>
                      <div className="flex items-center gap-3 text-white/70">
                        <span>{tierData.count} queries ({percent.toFixed(0)}%)</span>
                        <span className="font-semibold text-white">${tierData.cost.toFixed(4)}</span>
                      </div>
                    </div>
                    <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${tierColor} rounded-full transition-all duration-500`}
                        style={{ width: `${Math.max(percent, tierData.count > 0 ? 2 : 0)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Third Row: Strategy & Budget Tracking */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Strategy Breakdown */}
        <div className="rounded-2xl bg-black/35 backdrop-blur-xl border border-white/10 p-5 shadow-xl">
          <div className="flex items-center gap-2 mb-4">
            <Scale className="h-4 w-4 text-orange-300" />
            <h2 className="text-sm font-semibold text-white tracking-wide uppercase">
              Routing Strategies Used
            </h2>
          </div>

          {!stats || Object.keys(stats.by_strategy).length === 0 ? (
            <div className="py-6 text-center text-xs text-white/40">
              No strategy telemetry recorded yet.
            </div>
          ) : (
            <div className="space-y-3">
              {Object.entries(stats.by_strategy).map(([strategy, data]) => {
                const percent = totalQueries > 0 ? (data.count / totalQueries) * 100 : 0;
                return (
                  <div key={strategy} className="flex items-center justify-between p-3 rounded-xl bg-white/5 text-xs">
                    <div>
                      <div className="font-medium text-white capitalize">
                        {strategy.replace("_", " ")}
                      </div>
                      <div className="text-[11px] text-white/50">{data.count} queries routed</div>
                    </div>
                    <div className="text-right">
                      <div className="font-semibold text-white">${data.cost.toFixed(4)}</div>
                      <div className="text-[11px] text-emerald-300">{percent.toFixed(0)}% of traffic</div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Budget Status */}
        <div className="rounded-2xl bg-black/35 backdrop-blur-xl border border-white/10 p-5 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-orange-300" />
              <h2 className="text-sm font-semibold text-white tracking-wide uppercase">
                Budget Governance Limits
              </h2>
            </div>
            {budget?.daily.alert ? (
              <span className="rounded bg-red-500/20 px-2 py-0.5 text-[11px] text-red-300 font-medium">
                Daily Threshold Exceeded
              </span>
            ) : (
              <span className="rounded bg-emerald-500/20 px-2 py-0.5 text-[11px] text-emerald-300 font-medium">
                Within Budget
              </span>
            )}
          </div>

          <div className="space-y-4">
            {/* Daily */}
            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-white/70">Daily Limit ($10.00 USD)</span>
                <span className="text-white font-medium">
                  ${budget ? budget.daily.spent.toFixed(4) : "0.0000"} / ${budget?.daily.limit.toFixed(2) || "10.00"}
                </span>
              </div>
              <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                <div
                  className="h-full bg-orange-400 rounded-full transition-all duration-500"
                  style={{ width: `${Math.min(budget?.daily.percentage || 0, 100)}%` }}
                />
              </div>
            </div>

            {/* Weekly */}
            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-white/70">Weekly Limit ($50.00 USD)</span>
                <span className="text-white font-medium">
                  ${budget ? budget.weekly.spent.toFixed(4) : "0.0000"} / ${budget?.weekly.limit.toFixed(2) || "50.00"}
                </span>
              </div>
              <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-400 rounded-full transition-all duration-500"
                  style={{ width: `${Math.min(budget?.weekly.percentage || 0, 100)}%` }}
                />
              </div>
            </div>

            {/* Monthly */}
            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-white/70">Monthly Limit ($200.00 USD)</span>
                <span className="text-white font-medium">
                  ${budget ? budget.monthly.spent.toFixed(4) : "0.0000"} / ${budget?.monthly.limit.toFixed(2) || "200.00"}
                </span>
              </div>
              <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                <div
                  className="h-full bg-purple-400 rounded-full transition-all duration-500"
                  style={{ width: `${Math.min(budget?.monthly.percentage || 0, 100)}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
