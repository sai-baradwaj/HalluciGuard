import { useMemo, useState } from "react";
import { Check, Copy } from "lucide-react";
import type { MetricSet, VerificationResult } from "@/types";
import { Markdown, type Highlight } from "@/components/Markdown";
import { Button } from "@/components/ui/button";
import { copyText, cn } from "@/lib/utils";
import { formatMetric } from "@/lib/metrics";
import { useMediaQuery } from "@/hooks/useMediaQuery";

function Scores({ set }: { set: MetricSet }) {
  const f = set.known.factuality;
  const r = set.known.hallucination_risk;
  if (!f && !r) return <span className="text-xs text-muted">Scores not provided</span>;
  return (
    <div className="flex flex-wrap gap-1.5 text-xs">
      {f && <span className="rounded-md border border-line bg-panel2 px-2 py-0.5">Factuality <b className="ml-1 text-fg">{formatMetric(f)}</b></span>}
      {r && <span className="rounded-md border border-line bg-panel2 px-2 py-0.5">Hallucination risk <b className="ml-1 text-fg">{formatMetric(r)}</b></span>}
    </div>
  );
}

interface PanelProps {
  title: string;
  tone: "initial" | "final";
  text?: string;
  metrics: MetricSet;
  highlights: Highlight[];
  emptyText: string;
}

function ResponsePanel({ title, tone, text, metrics, highlights, emptyText }: PanelProps) {
  const [copied, setCopied] = useState(false);
  return (
    <div
      data-testid={`${tone}-panel`}
      className={cn("flex h-full min-w-0 flex-col rounded-2xl border bg-panel", tone === "final" ? "border-success/40" : "border-line")}
    >
      <div className="flex min-h-[3.25rem] flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-2.5">
        <h4 className="text-sm font-semibold">{title}</h4>
        <Button
          variant="ghost"
          size="sm"
          disabled={!text}
          aria-label={`Copy ${tone} response`}
          onClick={async () => {
            if (text && (await copyText(text))) {
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            }
          }}
        >
          {copied ? <Check /> : <Copy />} {copied ? "Copied" : "Copy"}
        </Button>
      </div>
      <div className="border-b border-line px-4 py-2.5">
        <Scores set={metrics} />
      </div>
      <div className="thin-scroll max-h-[26rem] min-h-[10rem] flex-1 overflow-y-auto px-4 py-4">
        {text ? <Markdown highlights={highlights}>{text}</Markdown> : <p className="text-sm text-muted">{emptyText}</p>}
      </div>
    </div>
  );
}

export function ResponseComparison({ analysis }: { analysis: VerificationResult }) {
  const wide = useMediaQuery("(min-width: 768px)");
  const [tab, setTab] = useState<"initial" | "final">("initial");

  const initialHighlights = useMemo<Highlight[]>(
    () => analysis.claims.filter((c) => c.status === "contradicted" || c.status === "unsupported").map((c) => ({ text: c.text, status: c.status })),
    [analysis.claims],
  );
  const finalHighlights = useMemo<Highlight[]>(
    () => analysis.claims.filter((c) => c.corrected && c.correctedText).map((c) => ({ text: c.correctedText as string, status: "corrected" })),
    [analysis.claims],
  );

  const initial = (
    <ResponsePanel title="Initial response" tone="initial" text={analysis.initialResponse} metrics={analysis.metrics.initial} highlights={initialHighlights} emptyText="The backend did not return the initial answer." />
  );
  const final = (
    <ResponsePanel title="Final corrected response" tone="final" text={analysis.finalResponse} metrics={analysis.metrics.final} highlights={finalHighlights} emptyText="The backend did not return a corrected answer." />
  );

  return (
    <div>
      {wide ? (
        <div className="grid grid-cols-2 items-stretch gap-4">
          {initial}
          {final}
        </div>
      ) : (
        <>
          <div role="tablist" aria-label="Response version" className="mb-3 grid grid-cols-2 rounded-xl border border-line bg-panel2 p-1">
            {(["initial", "final"] as const).map((t) => (
              <button
                key={t}
                role="tab"
                aria-selected={tab === t}
                onClick={() => setTab(t)}
                className={cn("rounded-lg py-2 text-sm font-medium", tab === t ? "bg-panel3 text-fg shadow-[inset_0_0_0_1px_var(--line-strong)]" : "text-muted")}
              >
                {t === "initial" ? "Initial" : "Final corrected"}
              </button>
            ))}
          </div>
          {tab === "initial" ? initial : final}
        </>
      )}
      {(initialHighlights.length > 0 || finalHighlights.length > 0) && (
        <p className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted">
          {initialHighlights.length > 0 && (
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-4 rounded-sm bg-danger/30 shadow-[inset_0_-2px_var(--danger)]" /> Contradicted or unsupported claim
            </span>
          )}
          {finalHighlights.length > 0 && (
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-4 rounded-sm bg-success/30 shadow-[inset_0_-2px_var(--success)]" /> Corrected claim
            </span>
          )}
          <span>Highlights appear where the claim text matches the answer word for word.</span>
        </p>
      )}
    </div>
  );
}
