"use client";
import { useState } from "react";
import type { Source } from "@/lib/types";

export default function SourceCard({ source }: { source: Source }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-lg border border-black/10 bg-white px-3 py-2 text-xs text-black/60">
      <div className="flex items-center gap-2">
        <a
          href={`https://arxiv.org/abs/${source.arxivId}`}
          target="_blank"
          rel="noreferrer"
          className="font-medium text-black/80 hover:underline"
        >
          {source.title}
        </a>
        <span className="text-black/40">arXiv:{source.arxivId} · p.{source.pages}</span>
        <span className={`ml-auto rounded px-1.5 py-0.5 ${source.cited ? "bg-green-50 text-green-800" : "bg-black/5 text-black/45"}`}>
          {source.cited ? "cited" : "retrieved"}
        </span>
      </div>
      <button onClick={() => setOpen(!open)} className="mt-1 block w-full text-left">
        <span className={open ? "" : "line-clamp-2"}>"{source.passage}"</span>
      </button>
    </div>
  );
}
