import type { Source } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

/** Calls src/api.py, which runs Retriever + summarize() from the pipeline. */
export async function getReply(
  question: string
): Promise<{ content: string; sources: Source[]; warning?: string }> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch {
    throw new Error("Can't reach the API. Is uvicorn running on port 8000?");
  }

  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ?? `API returned ${res.status}`);

  const unknown: string[] = data.unknownCitations ?? [];
  return {
    content: data.answer,
    sources: data.sources ?? [],
    warning: unknown.length ? `Cites papers that weren't retrieved: ${unknown.join(", ")}` : undefined,
  };
}
