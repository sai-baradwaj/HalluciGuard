import { AlertTriangle, BarChart3, ChevronLeft, ChevronRight, FileText, ListChecks, Loader2, PenLine, RotateCcw, X } from "lucide-react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { Dialog, DialogOverlay, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { DemoBadge, StatusBadge } from "@/components/StatusBadge";
import { useChat } from "@/hooks/useChat";
import { useVerification } from "@/hooks/useVerification";
import { Section } from "./Section";
import { ResponseComparison } from "./ResponseComparison";
import { MetricsComparison } from "./MetricsComparison";
import { ClaimVerification } from "./ClaimVerification";
import { CitationsPanel } from "./CitationsPanel";
import type { AssistantMessage } from "@/types";

/**
 * The lower "verification" half of the wireframe, shown as a bottom sheet over the chat.
 * It is driven by (conversationId, responseId) and re-resolves against live state, so it can never
 * show another response's analysis. It only opens when a Verify Response button selects something.
 */
export function VerificationModal() {
  const { selection, openVerification, closeVerification, retry } = useChat();
  const resolved = useVerification();
  const open = !!selection && !!resolved;

  const go = (delta: -1 | 1) => {
    if (!resolved) return;
    const next = resolved.siblings[resolved.index + delta];
    if (next) openVerification(resolved.conversation.id, next.id);
  };

  return (
    <Dialog open={open} onOpenChange={(o) => !o && closeVerification()}>
      <DialogPrimitive.Portal>
        <DialogOverlay />
        <DialogPrimitive.Content
          data-testid="verification-modal"
          data-response-id={resolved?.message.id}
          aria-describedby="verification-desc"
          className="anim-sheet fixed inset-x-0 bottom-0 top-0 z-50 mx-auto flex w-full flex-col overflow-hidden border border-line-strong bg-bg shadow-2xl outline-none md:top-[4vh] md:max-w-[1280px] md:rounded-t-3xl"
        >
          {resolved && (
            <>
              <header className="flex flex-wrap items-start gap-x-3 gap-y-2 border-b border-line bg-panel px-4 py-3 sm:px-6">
                <div className="min-w-0 flex-1 basis-40">
                  <DialogTitle className="text-base font-semibold">Response verification</DialogTitle>
                  <DialogDescription id="verification-desc" className="mt-0.5 line-clamp-2 text-xs text-muted">
                    <span className="sr-only">Question: </span>
                    {resolved.query}
                  </DialogDescription>
                </div>
                <Button variant="outline" size="icon-sm" onClick={closeVerification} aria-label="Close verification" className="order-2 md:order-3">
                  <X />
                </Button>
                <div className="order-3 flex w-full flex-wrap items-center gap-2 md:order-2 md:w-auto md:justify-end">
                  <StatusFor message={resolved.message} />
                  {resolved.message.analysis?.source === "demo" && <DemoBadge />}
                  {resolved.siblings.length > 1 && (
                    <div className="ml-auto flex items-center gap-0.5 md:ml-0" aria-label="Switch response">
                      <Button variant="ghost" size="icon-sm" onClick={() => go(-1)} disabled={resolved.index === 0} aria-label="Previous response">
                        <ChevronLeft />
                      </Button>
                      <span className="min-w-[3.5rem] text-center text-xs tabular-nums text-muted">
                        {resolved.index + 1} of {resolved.siblings.length}
                      </span>
                      <Button variant="ghost" size="icon-sm" onClick={() => go(1)} disabled={resolved.index === resolved.siblings.length - 1} aria-label="Next response">
                        <ChevronRight />
                      </Button>
                    </div>
                  )}
                </div>
              </header>

              {/* key = response id: all local state (tabs, expanded claims) resets when the response changes */}
              <div key={resolved.message.id} className="thin-scroll min-h-0 flex-1 overflow-y-auto">
                <Body message={resolved.message} query={resolved.query} onRetry={() => retry(resolved.message.id)} />
              </div>
            </>
          )}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </Dialog>
  );
}

function StatusFor({ message }: { message: AssistantMessage }) {
  const s = message.state === "processing" ? "processing" : message.state === "failed" ? "failed" : message.analysis?.status;
  return s ? <StatusBadge status={s} /> : null;
}

function Body({ message, query, onRetry }: { message: AssistantMessage; query: string; onRetry(): void }) {
  if (message.state === "processing") {
    return (
      <div className="grid h-full min-h-[50vh] place-items-center p-8" role="status">
        <div className="text-center">
          <Loader2 className="mx-auto size-8 animate-spin text-accent-text" />
          <p className="mt-4 text-sm font-medium">Verification in progress</p>
          <p className="mt-1 max-w-sm text-xs text-muted">The initial answer, metrics, claims and citations will appear here when the backend finishes.</p>
        </div>
      </div>
    );
  }

  if (message.state === "failed") {
    return (
      <div className="mx-auto max-w-xl p-6 sm:p-10" role="alert">
        <div className="rounded-2xl border border-danger/40 bg-danger/10 p-5">
          <p className="flex items-center gap-2 text-sm font-semibold text-danger">
            <AlertTriangle className="size-4" /> Verification failed
          </p>
          <p className="mt-2 break-words text-sm text-soft">{message.error ?? "The backend did not return a result."}</p>
          <p className="mt-2 text-xs text-muted">No answer, metrics, claims or citations exist for this response, so none are shown.</p>
          <Button variant="outline" size="sm" className="mt-4" onClick={onRetry}>
            <RotateCcw /> Retry this prompt
          </Button>
        </div>
      </div>
    );
  }

  const analysis = message.analysis;
  if (!analysis) {
    return <p className="p-8 text-center text-sm text-muted">No analysis data is available for this response.</p>;
  }


  return (
    <div>
      <div className="mx-auto max-w-[1180px] space-y-5 px-4 py-5 sm:px-8">
        {analysis.source === "demo" && (
          <p className="rounded-xl border border-warning/40 bg-warning/10 px-4 py-2.5 text-xs text-warning">
            Demo data: this analysis was generated locally to show the interface. It is not a real evaluation of any model answer.
          </p>
        )}
    
        {query && (
          <div className="rounded-xl border border-line bg-panel px-4 py-3">
            <h3 className="text-xs font-semibold text-muted">Question</h3>
            <p className="mt-1 whitespace-pre-wrap break-words text-sm">{query}</p>
          </div>
        )}

        <Section title="Initial and final response" icon={<FileText />}>
          <ResponseComparison analysis={analysis} />
        </Section>

        {(analysis.correctionSummary || analysis.notes) && (
          <Section title="Correction summary" icon={<PenLine />}>
            {analysis.correctionSummary?.text && <p className="text-sm leading-relaxed text-soft">{analysis.correctionSummary.text}</p>}
            {analysis.correctionSummary && analysis.correctionSummary.changes.length > 0 && (
              <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-soft marker:text-accent-text">
                {analysis.correctionSummary.changes.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            )}
            {analysis.notes && <p className="mt-3 whitespace-pre-wrap border-l-2 border-accent pl-3 text-sm text-soft">{analysis.notes}</p>}
          </Section>
        )}

        <Section title="Initial and final metrics" icon={<BarChart3 />}>
          <MetricsComparison initial={analysis.metrics.initial} final={analysis.metrics.final} />
        </Section>

        <Section title="Claim verification" icon={<ListChecks />}>
          <ClaimVerification claims={analysis.claims} citations={analysis.citations} />
        </Section>
      </div>
      <CitationsPanel citations={analysis.citations} claims={analysis.claims} />
    </div>
  );
}
