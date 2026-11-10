"use client";
import { useEffect, useRef } from "react";
import type { Message, Source } from "@/lib/types";
import SourceCard from "./SourceCard";

const STARTERS = [
  "Can AI agents design better training algorithms, and how would you benchmark that?",
  "How can we learn reusable, auditable models of how employees do their work from screen activity?",
  "When is it worth paying for a better estimate before routing a query to a specialist model?",
];

const CITE = /\[arXiv:([^\]]+)\]/g;

type Part = { text: string } | { cite: string };

/** Split an answer into text and citations; back-to-back repeats of one paper collapse. */
function parseAnswer(content: string): { parts: Part[]; order: string[] } {
  const parts: Part[] = [];
  const order: string[] = [];
  let last = 0;
  let prevCite: string | null = null;
  for (const m of content.matchAll(CITE)) {
    const between = content.slice(last, m.index);
    const id = m[1].trim();
    if (prevCite === id && between.trim() === "") {
      // duplicate right after the same citation: drop it and the whitespace between
    } else {
      if (between) parts.push({ text: between });
      parts.push({ cite: id });
      if (!order.includes(id)) order.push(id);
    }
    prevCite = id;
    last = m.index! + m[0].length;
  }
  if (last < content.length) parts.push({ text: content.slice(last) });
  return { parts, order };
}

function groupByPaper(sources: Source[]): Map<string, Source[]> {
  const map = new Map<string, Source[]>();
  for (const s of sources) map.set(s.arxivId, [...(map.get(s.arxivId) ?? []), s]);
  return map;
}

function AssistantMessage({ m }: { m: Message }) {
  const { parts, order } = parseAnswer(m.content);
  const byPaper = groupByPaper(m.sources ?? []);
  const anchor = (id: string) => `cite-${m.id}-${order.indexOf(id) + 1}`;
  const cited = order.filter((id) => byPaper.has(id));
  const uncited = [...byPaper.keys()].filter((id) => !order.includes(id));

  return (
    <div className="space-y-2">
      <div className="whitespace-pre-wrap font-serif text-[16px] leading-relaxed">
        {parts.map((p, i) =>
          "text" in p ? (
            <span key={i}>{p.text}</span>
          ) : (
            <sup key={i}>
              <button
                onClick={() => document.getElementById(anchor(p.cite))?.scrollIntoView({ behavior: "smooth", block: "center" })}
                disabled={!byPaper.has(p.cite)}
                title={byPaper.get(p.cite)?.[0].title ?? `arXiv:${p.cite} (not retrieved)`}
                className="mx-0.5 rounded bg-black/5 px-1 font-sans text-[11px] text-black/60 hover:bg-black/10 disabled:cursor-default disabled:hover:bg-black/5"
              >
                {order.indexOf(p.cite) + 1}
              </button>
            </sup>
          )
        )}
      </div>
      {m.warning && <div className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-900">{m.warning}</div>}
      {cited.map((id) => (
        <SourceCard key={id} n={order.indexOf(id) + 1} anchor={anchor(id)} sources={byPaper.get(id)!} />
      ))}
      {uncited.length > 0 && (
        <>
          <div className="pt-1 text-xs text-black/40">Also retrieved</div>
          {uncited.map((id) => <SourceCard key={id} sources={byPaper.get(id)!} />)}
        </>
      )}
    </div>
  );
}

type Props = {
  messages: Message[];
  loading: boolean;
  error: string | null;
  onRetry: () => void;
  onStarter: (q: string) => void;
};

export default function MessageList({ messages, loading, error, onRetry, onStarter }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading, error]);

  if (messages.length === 0 && !loading) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center px-6 text-center">
        <p className="text-lg text-black/80">Ask about recent AI research</p>
        <p className="mt-1 text-sm text-black/50">Get findings and business implications, cited to arXiv papers.</p>
        <div className="mt-6 grid w-full max-w-2xl gap-2">
          {STARTERS.map((q) => (
            <button
              key={q}
              onClick={() => onStarter(q)}
              className="rounded-xl border border-black/10 bg-white px-4 py-3 text-left text-sm text-black/70 hover:border-black/25"
            >{q}</button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-3xl space-y-5 px-6 py-6">
        {messages.map((m) =>
          m.role === "user" ? (
            <div key={m.id} className="flex flex-col items-end gap-1.5">
              {m.attachments?.map((name, i) => (
                <div key={i} className="flex max-w-[75%] items-center gap-1.5 rounded-lg border border-black/10 bg-white px-2.5 py-1.5 text-xs text-black/70">
                  <span aria-hidden>📄</span>
                  <span className="truncate">{name}</span>
                </div>
              ))}
              <div className="max-w-[75%] whitespace-pre-wrap rounded-2xl bg-white px-4 py-2 text-[15px] shadow-[0_0_0_0.5px_rgba(0,0,0,0.12)]">
                {m.content}
              </div>
            </div>
          ) : (
            <AssistantMessage key={m.id} m={m} />
          )
        )}
        {loading && <div className="animate-pulse text-sm text-black/45">Searching papers and writing an answer…</div>}
        {error && (
          <div className="flex items-center gap-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
            {error}
            <button onClick={onRetry} className="ml-auto underline">Retry</button>
          </div>
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}
