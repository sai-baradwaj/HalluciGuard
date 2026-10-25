import { AlertTriangle, Loader2 } from "lucide-react";
import type { AssistantMessage, Message } from "@/types";
import { ShieldMark } from "./Logo";
import { Markdown } from "./Markdown";
import { MessageActions } from "./MessageActions";
import { DemoBadge, StatusBadge } from "./StatusBadge";
import { formatTime } from "@/lib/utils";

interface Props {
  message: Message;
  conversationId: string;
  onVerify(conversationId: string, responseId: string): void;
  onRetry(responseId: string): void;
}

export function ChatMessage({ message, conversationId, onVerify, onRetry }: Props) {
  if (message.role === "user") {
    return (
      <article id={`message-${message.id}`} data-testid="user-message" className="flex scroll-mt-6 justify-end rounded-2xl">
        <div className="max-w-[88%] rounded-2xl rounded-tr-md border border-[#15517b] bg-[#0b3155] px-4 py-3 text-sm leading-relaxed text-[#e4f1ff] [.light_&]:border-line-strong [.light_&]:bg-panel3 [.light_&]:text-fg">
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        </div>
      </article>
    );
  }
  return <AssistantBlock message={message} conversationId={conversationId} onVerify={onVerify} onRetry={onRetry} />;
}

function AssistantBlock({ message, conversationId, onVerify, onRetry }: Omit<Props, "message"> & { message: AssistantMessage }) {
  const analysis = message.analysis;
  const status = message.state === "processing" ? "processing" : message.state === "failed" ? "failed" : analysis?.status;

  return (
    <article id={`message-${message.id}`} data-testid="assistant-message" data-response-id={message.id} className="flex scroll-mt-6 gap-3 rounded-2xl">
      <div className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-xl border border-accent/50 bg-accent/10">
        <ShieldMark className="size-5" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <span className="text-sm font-semibold">HalluciGuard</span>
          <span className="text-[11px] text-muted">{formatTime(message.createdAt)}</span>
          {status && <StatusBadge status={status} />}
          {analysis?.source === "demo" && <DemoBadge />}
        </div>

        <div className="rounded-2xl rounded-tl-md border border-line bg-panel px-4 py-3.5">
          {message.state === "processing" && (
            <p className="flex items-center gap-2 text-sm text-muted">
              <Loader2 className="size-4 animate-spin text-accent-text" />
              Generating an answer and checking it against evidence…
            </p>
          )}
          {message.state === "failed" && (
            <div role="alert" className="flex gap-2.5 text-sm text-danger">
              <AlertTriangle className="mt-0.5 size-4 shrink-0" />
              <div>
                <p className="font-medium">This prompt couldn’t be verified.</p>
                <p className="mt-1 break-words text-[13px] text-soft">{message.error}</p>
              </div>
            </div>
          )}
          {message.state === "done" && (
            <>
              {message.content ? <Markdown>{message.content}</Markdown> : <p className="text-sm text-muted">The backend returned no answer text.</p>}
              {analysis && !analysis.finalResponse && analysis.initialResponse && (
                <p className="mt-3 flex items-center gap-2 border-t border-line pt-3 text-xs text-warning">
                  <AlertTriangle className="size-3.5 shrink-0" />
                  The backend returned no corrected answer, so the initial answer is shown unverified.
                </p>
              )}
            </>
          )}
        </div>

        <MessageActions message={message} conversationId={conversationId} onVerify={onVerify} onRetry={onRetry} />
      </div>
    </article>
  );
}
