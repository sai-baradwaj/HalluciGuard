import { useMemo, useState } from "react";
import { ChevronDown } from "lucide-react";
import type { Citation, Claim, ClaimStatus } from "@/types";
import { ClaimStatusBadge, CLAIM_STATUS } from "@/components/StatusBadge";
import { citationsForClaim } from "@/lib/linking";
import { isHttpUrl, cn } from "@/lib/utils";

function ClaimRow({ claim, citations, open, onToggle }: { claim: Claim; citations: Citation[]; open: boolean; onToggle(): void }) {
  const linked = citationsForClaim(claim, citations);
  const panelId = `claim-panel-${claim.id}`;
  return (
    <li data-testid="claim-row" className="rounded-xl border border-line bg-panel2">
      <button type="button" aria-expanded={open} aria-controls={panelId} onClick={onToggle} className="flex w-full items-start gap-3 px-4 py-3 text-left">
        <ChevronDown className={cn("mt-0.5 size-4 shrink-0 text-muted transition-transform", open && "rotate-180")} />
        <div className="min-w-0 flex-1">
          <p className={cn("text-sm leading-relaxed", !open && "line-clamp-2")}>{claim.text}</p>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <ClaimStatusBadge status={claim.status} />
            <span className="text-[11px] text-muted">
              {claim.corrected === true ? "Corrected in the final answer" : claim.corrected === false ? "Not corrected" : "Correction status not provided"}
            </span>
          </div>
        </div>
      </button>
      {open && (
        <div id={panelId} className="space-y-3 border-t border-line px-4 py-3 pl-11 text-sm">
          <div>
            <h5 className="mb-1 text-xs font-semibold text-muted">Explanation</h5>
            <p className="leading-relaxed text-soft">{claim.explanation ?? <span className="text-muted">No explanation was provided.</span>}</p>
          </div>
          {claim.correctedText && (
            <div>
              <h5 className="mb-1 text-xs font-semibold text-muted">Corrected wording</h5>
              <p className="rounded-lg border border-success/30 bg-success/10 px-3 py-2 leading-relaxed text-soft">{claim.correctedText}</p>
            </div>
          )}
          <div>
            <h5 className="mb-1 text-xs font-semibold text-muted">Supporting evidence</h5>
            {claim.evidence.length ? (
              <ul className="space-y-2">
                {claim.evidence.map((e, i) => (
                  <li key={i} className="border-l-2 border-accent pl-3 italic leading-relaxed text-soft">{e}</li>
                ))}
              </ul>
            ) : (
              <p className="text-muted">No evidence was provided for this claim.</p>
            )}
          </div>
          {linked.length > 0 && (
            <div>
              <h5 className="mb-1 text-xs font-semibold text-muted">Sources</h5>
              <ul className="flex flex-wrap gap-2">
                {linked.map((c) => (
                  <li key={c.id}>
                    {isHttpUrl(c.url) ? (
                      <a href={c.url} target="_blank" rel="noopener noreferrer" className="rounded-full border border-accent/40 bg-accent/10 px-2.5 py-1 text-xs text-accent-text hover:bg-accent/20">
                        {c.title ?? c.domain ?? c.url}
                      </a>
                    ) : (
                      <span className="rounded-full border border-line px-2.5 py-1 text-xs text-muted">{c.title ?? `Source ${c.id}`}</span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </li>
  );
}

export function ClaimVerification({ claims, citations }: { claims: Claim[]; citations: Citation[] }) {
  const [openIds, setOpenIds] = useState<Set<string>>(new Set());
  const counts = useMemo(() => {
    const m = new Map<ClaimStatus, number>();
    claims.forEach((c) => m.set(c.status, (m.get(c.status) ?? 0) + 1));
    return [...m.entries()];
  }, [claims]);

  if (!claims.length) return <p className="text-sm text-muted">The backend did not return extracted claims for this response.</p>;

  const allOpen = openIds.size === claims.length;
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap gap-1.5 text-xs text-muted">
          <span>{claims.length} claims:</span>
          {counts.map(([status, n]) => (
            <span key={status} className={cn("rounded-full border px-2 py-0.5", CLAIM_STATUS[status].cls)}>
              {n} {CLAIM_STATUS[status].label}
            </span>
          ))}
        </div>
        <button type="button" className="text-xs font-medium text-accent-text" onClick={() => setOpenIds(allOpen ? new Set() : new Set(claims.map((c) => c.id)))}>
          {allOpen ? "Collapse all" : "Expand all"}
        </button>
      </div>
      <ul className="space-y-2">
        {claims.map((claim) => (
          <ClaimRow
            key={claim.id}
            claim={claim}
            citations={citations}
            open={openIds.has(claim.id)}
            onToggle={() =>
              setOpenIds((prev) => {
                const next = new Set(prev);
                if (next.has(claim.id)) next.delete(claim.id);
                else next.add(claim.id);
                return next;
              })
            }
          />
        ))}
      </ul>
    </div>
  );
}
