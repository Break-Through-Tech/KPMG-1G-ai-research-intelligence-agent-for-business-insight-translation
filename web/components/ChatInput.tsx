"use client";
import { useRef, useState } from "react";

export type Attachment = {
  id: string;
  name: string;
  status: "uploading" | "ready" | "error";
  error?: string;
};

type Props = {
  disabled: boolean;
  attachments: Attachment[];
  onRemove: (id: string) => void;
  onSend: (text: string) => void;
  onUpload: (files: File[]) => void;
};

export default function ChatInput({ disabled, attachments, onRemove, onSend, onUpload }: Props) {
  const [text, setText] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const uploading = attachments.some((a) => a.status === "uploading");
  const blocked = disabled || uploading;

  const send = () => {
    if (!text.trim() || blocked) return;
    onSend(text.trim());
    setText("");
  };

  return (
    <div className="border-t border-black/10 bg-[#f7f7f5] px-6 py-4">
      <div className="mx-auto max-w-3xl">
        <div className="rounded-2xl border border-black/15 bg-white p-2">
        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 px-1 pb-2">
            {attachments.map((a) => (
              <div
                key={a.id}
                title={a.error}
                className={`flex max-w-full items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs ${
                  a.status === "error"
                    ? "border-red-200 bg-red-50 text-red-800"
                    : "border-black/10 bg-[#f7f7f5] text-black/70"
                }`}
              >
                <span aria-hidden>📄</span>
                <span className="max-w-[14rem] truncate">{a.name}</span>
                {a.status === "uploading" && (
                  <>
                    <span className="h-3 w-3 animate-spin rounded-full border-2 border-black/20 border-t-black/60" />
                    <span className="text-black/50">Adding…</span>
                  </>
                )}
                {a.status === "error" && <span>{a.error}</span>}
                <button
                  onClick={() => onRemove(a.id)}
                  aria-label={`Remove ${a.name}`}
                  className="ml-0.5 leading-none text-black/40 hover:text-black/70"
                >×</button>
              </div>
            ))}
          </div>
        )}
        <div className="flex items-end gap-2">
          <button
            onClick={() => fileRef.current?.click()}
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
            disabled={blocked || !text.trim()}
            className="rounded-lg bg-black px-3 py-1.5 text-sm text-white disabled:opacity-30"
          >Send</button>
        </div>
        </div>
        <p className="mt-2 text-center text-xs text-black/40">Answers come only from the indexed arXiv papers. Check citations before sharing.</p>
      </div>
    </div>
  );
}
