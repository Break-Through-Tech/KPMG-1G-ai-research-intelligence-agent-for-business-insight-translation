"use client";
import { useEffect, useState } from "react";
import Sidebar from "@/components/Sidebar";
import MessageList from "@/components/MessageList";
import ChatInput from "@/components/ChatInput";
import { getReply } from "@/lib/assistant";
import { loadChats, saveChats, uid } from "@/lib/storage";
import type { Chat, Message } from "@/lib/types";

const newChat = (): Chat => ({ id: uid(), title: "New chat", messages: [], updatedAt: Date.now() });

export default function Home() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const saved = loadChats();
    setChats(saved);
    setActiveId(saved[0]?.id ?? null);
    setHydrated(true);
  }, []);

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
    const userMsg: Message = { id: uid(), role: "user", content: text, createdAt: Date.now() };
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
        onDelete={(id) => {
          setChats((p) => p.filter((c) => c.id !== id));
          if (id === activeId) setActiveId(null);
        }}
      />
      <main className="flex min-w-0 flex-1 flex-col">
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
        <ChatInput disabled={loading} onSend={send} />
      </main>
    </div>
  );
}
