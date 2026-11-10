"use client";
import { useState } from "react";
import type { Chat } from "@/lib/types";

type Props = {
  chats: Chat[];
  activeId: string | null;
  onNew: () => void;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
};

export default function Sidebar({ chats, activeId, onNew, onSelect, onRename, onDelete }: Props) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  const commit = (id: string) => {
    if (draft.trim()) onRename(id, draft.trim());
    setEditingId(null);
  };

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-black/10 bg-[#efefec] p-3">
      <div className="px-2 pb-3 text-[15px] font-medium">KPMG AI Assistant</div>
      <button
        onClick={onNew}
        className="mb-4 rounded-lg border border-black/15 bg-white px-3 py-2 text-left text-sm hover:bg-black/5"
      >
        + New chat
      </button>

      <div className="px-2 pb-1 text-xs text-black/45">Recent chats</div>
      <nav className="flex-1 space-y-0.5 overflow-y-auto">
        {chats.length === 0 && (
          <p className="px-2 py-2 text-sm text-black/45">Your chats will show up here.</p>
        )}
        {chats.map((c) => (
          <div
            key={c.id}
            className={`group flex items-center rounded-lg px-2 py-1.5 text-sm ${
              c.id === activeId ? "bg-black/[0.08]" : "hover:bg-black/5"
            }`}
          >
            {editingId === c.id ? (
              <input
                autoFocus
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onBlur={() => commit(c.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commit(c.id);
                  if (e.key === "Escape") setEditingId(null);
                }}
                className="w-full rounded border border-black/20 bg-white px-1 py-0.5"
              />
            ) : (
              <>
                <button onClick={() => onSelect(c.id)} className="flex-1 truncate text-left">
                  {c.title}
                </button>
                <div className="hidden gap-1 group-hover:flex">
                  <button
                    title="Rename"
                    onClick={() => { setEditingId(c.id); setDraft(c.title); }}
                    className="rounded px-1 text-black/50 hover:text-black"
                  >✎</button>
                  <button
                    title="Delete"
                    onClick={() => { if (confirm(`Delete "${c.title}"?`)) onDelete(c.id); }}
                    className="rounded px-1 text-black/50 hover:text-red-700"
                  >✕</button>
                </div>
              </>
            )}
          </div>
        ))}
      </nav>

      <div className="mt-3 flex items-center gap-2 border-t border-black/10 px-2 pt-3 text-sm">
        <div className="flex h-7 w-7 items-center justify-center rounded-full bg-blue-100 text-xs text-blue-800">C</div>
        Consultant
      </div>
    </aside>
  );
}
