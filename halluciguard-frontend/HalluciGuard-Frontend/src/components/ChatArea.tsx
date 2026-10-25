import { useEffect, useRef, useState } from "react";
import { ChatMessage } from "./ChatMessage";
import { EmptyState } from "./EmptyState";
import { MessageComposer } from "./MessageComposer";
import { PromptHistory, type PromptLine } from "./PromptHistory";
import { scrollToMessage } from "@/lib/dom";
import { useChat } from "@/hooks/useChat";

export function ChatArea() {
  const { active, activeId, busy, sendPrompt, openVerification, retry } = useChat();
  const [draft, setDraft] = useState("");
  const scroller = useRef<HTMLDivElement>(null);
  const lastActive = useRef<string | null>(null);
  const messages = active?.messages ?? [];
  const tail = messages[messages.length - 1];

  const promptLines: PromptLine[] = messages.flatMap((m) =>
    m.role === "user" ? [{ id: m.id, content: m.content, targetId: messages.find((r) => r.role === "assistant" && r.replyTo === m.id)?.id ?? m.id }] : [],
  );

  // Jump to the end when switching conversations; glide when new messages arrive.
  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    const switched = lastActive.current !== activeId;
    lastActive.current = activeId;
    el.scrollTo({ top: el.scrollHeight, behavior: switched ? "auto" : "smooth" });
  }, [activeId, messages.length, tail?.role === "assistant" ? tail.state : undefined]);

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      <div ref={scroller} className="thin-scroll min-h-0 flex-1 overflow-y-auto" aria-live="polite">
        {messages.length === 0 ? (
          <EmptyState onPick={setDraft} />
        ) : (
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 py-6 pl-4 pr-11 sm:pl-6 sm:pr-14">
            {messages.map((m) => (
              <ChatMessage key={m.id} message={m} conversationId={active!.id} onVerify={openVerification} onRetry={retry} />
            ))}
          </div>
        )}
      </div>
      <PromptHistory prompts={promptLines} onNavigate={scrollToMessage} />
      <MessageComposer value={draft} onChange={setDraft} onSend={sendPrompt} busy={busy} />
    </div>
  );
}
