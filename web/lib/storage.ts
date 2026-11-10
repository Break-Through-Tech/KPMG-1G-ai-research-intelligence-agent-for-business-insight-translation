import type { Chat } from "./types";

const KEY = "kpmg-assistant:chats:v2";

export function loadChats(): Chat[] {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Chat[]) : [];
  } catch {
    return [];
  }
}

export function saveChats(chats: Chat[]): boolean {
  try {
    localStorage.setItem(KEY, JSON.stringify(chats));
    return true;
  } catch {
    // Usually the ~5MB quota. Large documents are the likely cause.
    return false;
  }
}

export const uid = () =>
  Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
