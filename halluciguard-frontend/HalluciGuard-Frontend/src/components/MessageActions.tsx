import { useState } from "react";
import { Check, Copy, RotateCcw, Share2 } from "lucide-react";
import type { AssistantMessage } from "@/types";
import { Button } from "@/components/ui/button";
import { VerifyResponseButton } from "./VerifyResponseButton";
import { useToast } from "./Toast";
import { copyText } from "@/lib/utils";
import { shareText } from "@/lib/share";

interface Props {
  message: AssistantMessage;
  conversationId: string;
  onVerify(conversationId: string, responseId: string): void;
  onRetry(responseId: string): void;
}

export function MessageActions({ message, conversationId, onVerify, onRetry }: Props) {
  const toast = useToast();
  const [copied, setCopied] = useState(false);
  const hasText = message.state === "done" && !!message.content;

  const copy = async () => {
    if (await copyText(message.content)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } else toast("Couldn’t copy to the clipboard.", "error");
  };
  const share = async () => {
    const outcome = await shareText("HalluciGuard response", message.content);
    if (outcome === "copied") toast("Response copied to the clipboard.");
    if (outcome === "failed") toast("Sharing isn’t available here.", "error");
  };

  return (
    <div className="mt-3 flex flex-wrap items-center gap-1.5" role="toolbar" aria-label="Response actions">
      <Button variant="ghost" size="sm" onClick={copy} disabled={!hasText} aria-label="Copy response">
        {copied ? <Check /> : <Copy />} {copied ? "Copied" : "Copy"}
      </Button>
      <Button variant="ghost" size="sm" onClick={share} disabled={!hasText} aria-label="Share response">
        <Share2 /> Share
      </Button>
      {message.state === "failed" && (
        <Button variant="ghost" size="sm" onClick={() => onRetry(message.id)}>
          <RotateCcw /> Retry
        </Button>
      )}
      <VerifyResponseButton responseId={message.id} conversationId={conversationId} processing={message.state === "processing"} onOpen={onVerify} />
    </div>
  );
}
