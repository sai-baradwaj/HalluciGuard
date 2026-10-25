import { useMemo } from "react";
import { useChat } from "./useChat";
import type { AssistantMessage, Conversation } from "@/types";

export interface ResolvedVerification {
  conversation: Conversation;
  message: AssistantMessage;
  query: string;
  /** Other assistant responses in the same conversation, for prev/next. */
  siblings: AssistantMessage[];
  index: number;
}

/**
 * Resolves the currently selected (conversationId, responseId) against live state on every render.
 * Nothing is cached per-selection, so switching responses can never show another response's analysis.
 */
export function useVerification(): ResolvedVerification | null {
  const { selection, conversations } = useChat();
  return useMemo(() => {
    if (!selection) return null;
    const conversation = conversations.find((c) => c.id === selection.conversationId);
    if (!conversation) return null;
    const siblings = conversation.messages.filter((m): m is AssistantMessage => m.role === "assistant");
    const index = siblings.findIndex((m) => m.id === selection.responseId);
    if (index < 0) return null;
    const message = siblings[index];
    const query = conversation.messages.find((m) => m.id === message.replyTo)?.content ?? message.analysis?.query ?? "";
    return { conversation, message, query, siblings, index };
  }, [selection, conversations]);
}
