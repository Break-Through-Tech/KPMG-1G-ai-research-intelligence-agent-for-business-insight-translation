"use client";
import { useEffect, useRef } from "react";
import type { Message } from "@/lib/types";
import SourceCard from "./SourceCard";

const STARTERS = [
  "Can AI agents design better training algorithms, and how would you benchmark that?",
  "How can we learn reusable, auditable models of how employees do their work from screen activity?",
  "When is it worth paying for a better estimate before routing a query to a specialist model?",
];

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
            <div key={m.id} className="flex justify-end">
              <div className="max-w-[75%] whitespace-pre-wrap rounded-2xl bg-white px-4 py-2 text-[15px] shadow-[0_0_0_0.5px_rgba(0,0,0,0.12)]">
                {m.content}
              </div>
            </div>
          ) : (
            <div key={m.id} className="space-y-2">
              <div className="whitespace-pre-wrap font-serif text-[16px] leading-relaxed">{m.content}</div>
              {m.warning && (
                <div className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-900">{m.warning}</div>
              )}
              {m.sources?.map((s, i) => <SourceCard key={i} source={s} />)}
            </div>
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
