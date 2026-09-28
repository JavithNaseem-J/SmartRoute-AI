export interface Citation {
  id: string;
  filename: string;
  page?: number | null;
  section?: string | null;
  excerpt: string;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  model?: string;
  citations?: Citation[];
  streaming?: boolean;
}

export interface Session {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
}
