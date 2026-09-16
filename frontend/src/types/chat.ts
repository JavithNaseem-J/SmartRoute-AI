export interface Message {
  role: "user" | "assistant";
  content: string;
  model?: string;
  sources?: string[];
  streaming?: boolean;
}

export interface Session {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
}
