export type RoutingStrategy = "cost_optimized" | "balanced" | "quality_first";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  files?: Array<{ name: string; url?: string; isImage?: boolean }>;
  model_used?: string;
  latency?: number;
  cost?: number;
  confidence?: number;
  complexity?: string;
  sources?: string[];
  streaming?: boolean;
  error?: string;
}

export interface ChatSession {
  id: string;
  title: string;
  updatedAt: number;
  messages: ChatMessage[];
  strategy: RoutingStrategy;
  useRetrieval: boolean;
}

export interface DocumentItem {
  filename: string;
  size_bytes: number;
  modified_time?: number;
}

export interface BudgetRecord {
  spent?: number;
  limit?: number;
  alert_threshold?: number;
}

export interface StatsResponse {
  total_queries?: number;
  total_cost?: number;
  estimated_baseline_cost?: number;
  savings_amount?: number;
  savings_percentage?: number;
  [key: string]: unknown;
}
