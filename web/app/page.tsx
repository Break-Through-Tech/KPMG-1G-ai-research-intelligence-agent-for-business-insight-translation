"use client";
import { useEffect, useState } from "react";
import Sidebar from "@/components/Sidebar";
import MessageList from "@/components/MessageList";
import ChatInput, { type Attachment } from "@/components/ChatInput";
import { getReply, listPapers, uploadPdfs, type Paper } from "@/lib/assistant";
import { loadChats, saveChats, uid } from "@/lib/storage";
import type { Chat, Message } from "@/lib/types";

const newChat = (): Chat => ({ id: uid(), title: "New chat", messages: [], updatedAt: Date.now() });

export default function Home() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    const saved = loadChats();
    setChats(saved);
    setActiveId(saved[0]?.id ?? null);
    setHydrated(true);
    listPapers().then(setPapers);
  }, []);

  const uploading = attachments.some((a) => a.status === "uploading");

  // The API saves files under a sanitized name, so compare on the same normalization.
  const norm = (name: string) => name.replace(/\.pdf$/i, "").replace(/[^\w.\- ]+/g, "").trim().toLowerCase();

  const patch = (ids: string[], fn: (a: Attachment) => Attachment) =>
    setAttachments((prev) => prev.map((a) => (ids.includes(a.id) ? fn(a) : a)));

  const upload = async (files: File[]) => {
    const pdfs = files.filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    const rejected: Attachment[] = files
      .filter((f) => !pdfs.includes(f))
      .map((f) => ({ id: uid(), name: f.name, status: "error", error: "Not a PDF" }));
    const chips: Attachment[] = pdfs.map((f) => ({ id: uid(), name: f.name, status: "uploading" }));
    setAttachments((prev) => [...prev, ...rejected, ...chips]);
    if (chips.length === 0) return;
    const ids = chips.map((c) => c.id);
    try {
      const { added, skipped } = await uploadPdfs(pdfs);
      const addedKeys = new Set(added.map((p) => norm(p.file)));
      const skippedKeys = new Set(skipped.map((p) => norm(p.file)));
      patch(ids, (a) =>
        skippedKeys.has(norm(a.name)) && !addedKeys.has(norm(a.name))
          ? { ...a, status: "error", error: "No usable text" }
          : { ...a, status: "ready" }
      );
      setPapers(await listPapers());
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Upload failed";
      patch(ids, (a) => ({ ...a, status: "error", error: msg }));
    }
  };

  useEffect(() => {
    if (hydrated) saveChats(chats);
  }, [chats, hydrated]);

  const active = chats.find((c) => c.id === activeId) ?? null;
  const sorted = [...chats].sort((a, b) => b.updatedAt - a.updatedAt);

  const update = (id: string, fn: (c: Chat) => Chat) =>
    setChats((prev) => prev.map((c) => (c.id === id ? { ...fn(c), updatedAt: Date.now() } : c)));

  // The pipeline answers one question at a time, so only the latest question is sent.
  const ask = async (chatId: string, question: string) => {
    setLoading(true);
    setError(null);
    try {
      const reply = await getReply(question);
      const msg: Message = { id: uid(), role: "assistant", createdAt: Date.now(), ...reply };
      update(chatId, (c) => ({ ...c, messages: [...c.messages, msg] }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const send = (text: string) => {
    let chat = active;
    if (!chat) {
      chat = newChat();
      const c = chat;
      setChats((prev) => [c, ...prev]);
      setActiveId(c.id);
    }
    const names = attachments.filter((a) => a.status === "ready").map((a) => a.name);
    const userMsg: Message = {
      id: uid(),
      role: "user",
      content: text,
      createdAt: Date.now(),
      ...(names.length ? { attachments: names } : {}),
    };
    setAttachments([]);
    update(chat.id, (c) => ({
      ...c,
      title: c.messages.length === 0 ? text.slice(0, 40) : c.title,
      messages: [...c.messages, userMsg],
    }));
    ask(chat.id, text);
  };

  const retry = () => {
    const last = [...(active?.messages ?? [])].reverse().find((m) => m.role === "user");
    if (active && last) ask(active.id, last.content);
  };

  return (
    <div className="flex h-screen">
      <Sidebar
        chats={sorted}
        activeId={activeId}
        onNew={() => { const c = newChat(); setChats((p) => [c, ...p]); setActiveId(c.id); setError(null); }}
        onSelect={(id) => { setActiveId(id); setError(null); }}
        onRename={(id, title) => update(id, (c) => ({ ...c, title }))}
        papers={papers}
        onDelete={(id) => {
          setChats((p) => p.filter((c) => c.id !== id));
          if (id === activeId) setActiveId(null);
        }}
      />
      <main
        className={`relative flex min-w-0 flex-1 flex-col ${dragging ? "bg-blue-50/60" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={(e) => { if (e.currentTarget === e.target) setDragging(false); }}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!uploading) upload(Array.from(e.dataTransfer.files));
        }}
      >
        {dragging && (
          <div className="pointer-events-none absolute inset-4 z-10 flex items-center justify-center rounded-2xl border-2 border-dashed border-blue-300 text-blue-800">
            Drop PDFs to add them to the index
          </div>
        )}
        <header className="border-b border-black/10 px-6 py-3 text-[15px] font-medium">
          {active?.title ?? "New chat"}
        </header>
        <MessageList
          messages={active?.messages ?? []}
          loading={loading}
          error={error}
          onRetry={retry}
          onStarter={send}
        />
        <ChatInput
          disabled={loading}
          attachments={attachments}
          onRemove={(id) => setAttachments((prev) => prev.filter((a) => a.id !== id))}
          onSend={send}
          onUpload={upload}
        />
      </main>
    </div>
  );
}
