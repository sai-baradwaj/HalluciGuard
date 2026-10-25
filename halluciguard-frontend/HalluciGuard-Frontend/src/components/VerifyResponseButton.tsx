import { Loader2, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Props {
  responseId: string;
  conversationId: string;
  processing: boolean;
  onOpen(conversationId: string, responseId: string): void;
}

/** Bound to ONE assistant response: it only ever opens the analysis stored under its own responseId. */
export function VerifyResponseButton({ responseId, conversationId, processing, onOpen }: Props) {
  return (
    <Button
      variant="verify"
      size="sm"
      data-response-id={responseId}
      disabled={processing}
      onClick={() => onOpen(conversationId, responseId)}
      title={processing ? "Analysis is still running" : "Compare the initial and final answers, metrics, claims and citations"}
    >
      {processing ? <Loader2 className="animate-spin" /> : <ShieldCheck />}
      {processing ? "Verifying…" : "Verify Response"}
    </Button>
  );
}
