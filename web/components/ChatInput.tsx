"use client";
import { useRef, useState } from "react";

type Props = {
  disabled: boolean;
  uploading: boolean;
  onSend: (text: string) => void;
  onUpload: (files: File[]) => void;
};

export default function ChatInput({ disabled, uploading, onSend, onUpload }: Props) {
  const [text, setText] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const send = () => {
    if (!text.trim() || disabled) return;
    onSend(text.trim());
    setText("");
  };

  return (
    <div className="border-t border-black/10 bg-[#f7f7f5] px-6 py-4">
      <div className="mx-auto max-w-3xl">
        <div className="flex items-end gap-2 rounded-2xl border border-black/15 bg-white p-2">
          <button
            onClick={() => fileRef.current?.click()}
            disabled={uploading}
            title="Add PDFs to the paper index"
            className="rounded-lg px-2 py-1.5 text-lg leading-none text-black/50 hover:bg-black/5 disabled:opacity-30"
          >+</button>
          <input
            ref={fileRef}
            type="file"
            accept="application/pdf,.pdf"
            multiple
            className="hidden"
            onChange={(e) => {
              const files = Array.from(e.target.files ?? []);
              if (files.length) onUpload(files);
              e.target.value = "";
            }}
          />
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
