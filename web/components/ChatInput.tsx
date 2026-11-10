"use client";
import { useState } from "react";

export default function ChatInput({ disabled, onSend }: { disabled: boolean; onSend: (text: string) => void }) {
  const [text, setText] = useState("");

  const send = () => {
    if (!text.trim() || disabled) return;
    onSend(text.trim());
    setText("");
  };

  return (
    <div className="border-t border-black/10 bg-[#f7f7f5] px-6 py-4">
      <div className="mx-auto max-w-3xl">
        <div className="flex items-end gap-2 rounded-2xl border border-black/15 bg-white p-2">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
            rows={1}
            placeholder="Ask about recent AI research..."
            className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] outline-none"
          />
          <button
            onClick={send}
            disabled={disabled || !text.trim()}
            className="rounded-lg bg-black px-3 py-1.5 text-sm text-white disabled:opacity-30"
          >Send</button>
        </div>
        <p className="mt-2 text-center text-xs text-black/40">Answers come only from the indexed arXiv papers. Check citations before sharing.</p>
      </div>
    </div>
  );
}
