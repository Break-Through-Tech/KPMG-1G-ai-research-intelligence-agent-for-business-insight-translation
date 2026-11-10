export type Source = {
  arxivId: string;
  title: string;
  pages: string;
  score: number;
  passage: string;
  cited: boolean;
};

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  warning?: string;
  createdAt: number;
};

export type Chat = {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: number;
};
