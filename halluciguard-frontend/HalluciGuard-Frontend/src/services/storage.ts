import type { Conversation, DataMode } from "@/types";

const KEY = "halluciguard.conversations.v1";
const MODE_KEY = "halluciguard.mode";
const THEME_KEY = "halluciguard.theme";

export function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Conversation[];
    if (!Array.isArray(parsed)) return [];
    // A request cannot survive a page reload; mark leftovers as interrupted rather than spinning forever.
    return parsed.map((c) => ({
      ...c,
      messages: c.messages.map((m) =>
        m.role === "assistant" && m.state === "processing"
          ? { ...m, state: "failed" as const, error: "This request was interrupted by a page reload. Retry to verify it again.", content: "" }
          : m,
      ),
    }));
  } catch {
    return [];
  }
}

export function saveConversations(conversations: Conversation[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(conversations));
  } catch {
    /* storage full or unavailable: the session still works in memory */
  }
}

export const loadMode = (fallback: DataMode): DataMode => {
  try {
    const v = localStorage.getItem(MODE_KEY);
    return v === "demo" || v === "live" ? v : fallback;
  } catch {
    return fallback;
  }
};
export const saveMode = (mode: DataMode) => {
  try {
    localStorage.setItem(MODE_KEY, mode);
  } catch {
    /* ignore */
  }
};
export const loadTheme = (): "dark" | "light" => {
  try {
    return localStorage.getItem(THEME_KEY) === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
};
export const saveTheme = (t: "dark" | "light") => {
  try {
    localStorage.setItem(THEME_KEY, t);
  } catch {
    /* ignore */
  }
};
