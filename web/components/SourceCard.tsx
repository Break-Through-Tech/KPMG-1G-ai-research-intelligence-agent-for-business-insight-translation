"use client";
import { useState } from "react";
import type { Source } from "@/lib/types";

const ARXIV_ID = /^\d{4}\.\d{4,5}(v\d+)?$/;

type Props = {
  /** Citation number; omitted for papers that were retrieved but never cited. */
  n?: number;
  /** Anchor id so citation numbers in the answer can scroll here. */
  anchor?: string;
  /** All retrieved passages for one paper. */
  sources: Source[];
};

export default function SourceCard({ n, anchor, sources }: Props) {
  const [open, setOpen] = useState(false);
  const first = sources[0];
  const isArxiv = ARXIV_ID.test(first.arxivId);
  const pages = Array.from(new Set(sources.map((s) => s.pages))).join(", ");
  return (
    <div id={anchor} className="scroll-mt-20 rounded-lg border border-black/10 bg-white px-3 py-2 text-xs text-black/60">
      <div className="flex items-baseline gap-2">
        {n !== undefined && (
          <span className="rounded bg-black/5 px-1.5 py-0.5 text-[11px] font-medium text-black/70">{n}</span>
        )}
        {isArxiv ? (
          <a
            href={`https://arxiv.org/abs/${first.arxivId}`}
            target="_blank"
            rel="noreferrer"
            className="font-medium text-black/80 hover:underline"
          >
            {first.title}
          </a>
        ) : (
          <span className="font-medium text-black/80">{first.title}</span>
        )}
      </div>
      <div className="mt-0.5 text-black/40">
        {isArxiv ? `arXiv:${first.arxivId} · ` : "Uploaded PDF · "}p.{pages}
      </div>
      <button onClick={() => setOpen(!open)} className="mt-1.5 text-black/50 hover:text-black/80">
        {open ? "▾" : "▸"} {sources.length} passage{sources.length === 1 ? "" : "s"}
      </button>
      {open && (
        <ul className="mt-1.5 space-y-1.5">
          {sources.map((s, i) => (
            <li key={i} className="border-l-2 border-black/10 pl-2">
              <span className="text-black/40">p.{s.pages}</span> "{s.passage}"
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
