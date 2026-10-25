import { createContext, useCallback, useContext, useEffect, useMemo, useReducer, useRef, useState, type ReactNode } from "react";
import type { AssistantMessage, Conversation, DataMode, UserMessage, VerificationResult } from "@/types";
import { createApi, config, type HalluciGuardApi } from "@/services/api";
import { createDemoConversations } from "@/services/mock";
import { loadConversations, loadMode, saveConversations, saveMode } from "@/services/storage";
import { makeId, titleFromPrompt } from "@/lib/utils";

/* ------------------------------------------------------------------ reducer */

type Action =
  | { type: "replace"; conversations: Conversation[] }
  | { type: "addTurn"; conversationId: string; isNew: boolean; user: UserMessage; assistant: AssistantMessage }
  | { type: "resolve"; conversationId: string; assistantId: string; analysis: VerificationResult }
  | { type: "fail"; conversationId: string; assistantId: string; error: string }
  | { type: "restart"; conversationId: string; assistantId: string }
  | { type: "rename"; conversationId: string; title: string }
  | { type: "remove"; conversationId: string };

const mapAssistant = (state: Conversation[], cid: string, aid: string, fn: (m: AssistantMessage) => AssistantMessage) =>
  state.map((c) =>
    c.id !== cid
      ? c
      : { ...c, updatedAt: Date.now(), messages: c.messages.map((m) => (m.role === "assistant" && m.id === aid ? fn(m) : m)) },
  );

export function chatReducer(state: Conversation[], action: Action): Conversation[] {
  switch (action.type) {
    case "replace":
      return action.conversations;
    case "addTurn": {
      if (action.isNew) {
        return [
          { id: action.conversationId, title: titleFromPrompt(action.user.content), createdAt: Date.now(), updatedAt: Date.now(), messages: [action.user, action.assistant] },
          ...state,
        ];
      }
      return state.map((c) =>
        c.id === action.conversationId ? { ...c, updatedAt: Date.now(), messages: [...c.messages, action.user, action.assistant] } : c,
      );
    }
    case "resolve":
      return mapAssistant(state, action.conversationId, action.assistantId, (m) => ({
        ...m,
        state: "done",
        error: undefined,
        analysis: action.analysis,
        // The chat always shows the FINAL corrected answer; the initial answer is only a fallback and is flagged in the UI.
        content: action.analysis.finalResponse ?? action.analysis.initialResponse ?? "",
        createdAt: Date.now(),
      }));
    case "fail":
      return mapAssistant(state, action.conversationId, action.assistantId, (m) => ({ ...m, state: "failed", error: action.error, content: "", analysis: undefined }));
    case "restart":
      return mapAssistant(state, action.conversationId, action.assistantId, (m) => ({ ...m, state: "processing", error: undefined, content: "", analysis: undefined }));
    case "rename":
      return state.map((c) => (c.id === action.conversationId ? { ...c, title: action.title.trim() || c.title } : c));
    case "remove":
      return state.filter((c) => c.id !== action.conversationId);
  }
}

/* ------------------------------------------------------------------ context */

export interface VerificationSelection {
  conversationId: string;
  responseId: string;
}

interface ChatContextValue {
  conversations: Conversation[]; // newest activity first
  active?: Conversation;
  activeId: string | null;
  mode: DataMode;
  busy: boolean;
  selection: VerificationSelection | null;
  setMode(mode: DataMode): void;
  newChat(): void;
  selectConversation(id: string): void;
  renameConversation(id: string, title: string): void;
  deleteConversation(id: string): void;
  sendPrompt(text: string): Promise<boolean>;
  retry(assistantId: string): void;
  openVerification(conversationId: string, responseId: string): void;
  closeVerification(): void;
}

const ChatContext = createContext<ChatContextValue | null>(null);

export function useChat(): ChatContextValue {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChat must be used inside <ChatProvider>");
  return ctx;
}

interface ProviderProps {
  children: ReactNode;
  /** Inject an API (tests). Defaults to the live/demo API chosen by the mode toggle. */
  api?: HalluciGuardApi;
  /** Inject starting conversations (tests). Defaults to localStorage. */
  initialConversations?: Conversation[];
  /** Skip localStorage persistence (tests). */
  persist?: boolean;
}

export function ChatProvider({ children, api: injectedApi, initialConversations, persist = true }: ProviderProps) {
  const [mode, setModeState] = useState<DataMode>(() => (injectedApi ? injectedApi.mode : persist ? loadMode(config.startInDemo ? "demo" : "live") : "live"));
  const [conversations, dispatch] = useReducer(chatReducer, undefined, () => {
    if (initialConversations) return initialConversations;
    const stored = persist ? loadConversations() : [];
    // Starting in demo mode with nothing saved: seed sample conversations (all flagged as demo data).
    return !stored.length && persist && !injectedApi && mode === "demo" ? createDemoConversations() : stored;
  });
  const [activeId, setActiveId] = useState<string | null>(null);
  const [selection, setSelection] = useState<VerificationSelection | null>(null);

  const apiRef = useRef<HalluciGuardApi>(injectedApi ?? createApi(mode));
  useEffect(() => {
    if (!injectedApi) apiRef.current = createApi(mode);
  }, [mode, injectedApi]);

  const pending = useRef(new Set<string>()); // conversation ids with an in-flight request (synchronous duplicate guard)

  useEffect(() => {
    if (persist) saveConversations(conversations);
  }, [conversations, persist]);

  const sorted = useMemo(() => [...conversations].sort((a, b) => b.updatedAt - a.updatedAt), [conversations]);
  const active = useMemo(() => conversations.find((c) => c.id === activeId), [conversations, activeId]);
  const busy = !!active?.messages.some((m) => m.role === "assistant" && m.state === "processing");

  const run = useCallback(
    async (
      conversationId: string,
      assistantId: string,
      query: string,
      chatHistory: { role: "user" | "assistant"; content: string }[] = [],
    ) => {
      pending.current.add(conversationId);

      try {
        const analysis = await apiRef.current.verify({
          query,
          conversationId,
          responseId: assistantId,
          chatHistory,
        });

        dispatch({
          type: "resolve",
          conversationId,
          assistantId,
          analysis,
        });
      } catch (err) {
        dispatch({
          type: "fail",
          conversationId,
          assistantId,
          error:
            err instanceof Error
              ? err.message
              : "Unknown error while contacting the backend.",
        });
      } finally {
        pending.current.delete(conversationId);
      }
    },
    [],
  );

  const sendPrompt = useCallback(
    async (text: string) => {
      const prompt = text.trim();
      if (!prompt) return false;
      const isNew = !activeId || !conversations.some((c) => c.id === activeId);
      const conversationId = isNew ? makeId("conv") : (activeId as string);
      if (pending.current.has(conversationId)) return false; // already verifying a prompt in this conversation
      pending.current.add(conversationId);

      const now = Date.now();
      const user: UserMessage = { id: makeId("msg"), role: "user", content: prompt, createdAt: now };
      const assistant: AssistantMessage = { id: makeId("resp"), role: "assistant", replyTo: user.id, content: "", state: "processing", createdAt: now };
      dispatch({ type: "addTurn", conversationId, isNew, user, assistant });
      if (isNew) setActiveId(conversationId);
      const previousConversation = conversations.find(
        (conversation) => conversation.id === conversationId,
      );
      const chatHistory = (previousConversation?.messages ?? [])
        .filter(
          (message) =>
            message.role === "user" ||
            (message.role === "assistant" && message.state === "done"),
        )
        .map((message) => ({
          role: message.role,
          content: message.content,
        }));
      await run(conversationId, assistant.id, prompt,chatHistory,);
      return true;
    },
    [activeId, conversations, run],
  );

  const retry = useCallback(
    (assistantId: string) => {
      if (!active || pending.current.has(active.id)) return;
      const idx = active.messages.findIndex((m) => m.id === assistantId);
      const reply = active.messages[idx];
      if (idx < 1 || reply?.role !== "assistant") return;
      const user = active.messages.find((m) => m.id === reply.replyTo);
      if (!user || user.role !== "user") return;
      dispatch({ type: "restart", conversationId: active.id, assistantId });
      void run(active.id, assistantId, user.content);
    },
    [active, run],
  );

  const setMode = useCallback(
    (next: DataMode) => {
      setModeState(next);
      if (persist) saveMode(next);
      // First time in demo mode with nothing to look at: seed sample conversations (all flagged as demo data).
      if (next === "demo" && conversations.length === 0) dispatch({ type: "replace", conversations: createDemoConversations() });
    },
    [conversations.length, persist],
  );

  const value: ChatContextValue = {
    conversations: sorted,
    active,
    activeId,
    mode,
    busy,
    selection,
    setMode,
    newChat: () => {
      setActiveId(null);
    },
    selectConversation: (id) => setActiveId(id),
    renameConversation: (id, title) => dispatch({ type: "rename", conversationId: id, title }),
    deleteConversation: (id) => {
      dispatch({ type: "remove", conversationId: id });
      setSelection((s) => (s?.conversationId === id ? null : s));
      setActiveId((cur) => (cur === id ? null : cur));
    },
    sendPrompt,
    retry,
    openVerification: (conversationId, responseId) => setSelection({ conversationId, responseId }),
    closeVerification: () => setSelection(null),
  };

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}
