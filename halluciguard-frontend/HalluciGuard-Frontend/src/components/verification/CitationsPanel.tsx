import { ExternalLink, Link2 } from "lucide-react";
import type { Citation, Claim } from "@/types";
import { claimsForCitation } from "@/lib/linking";
import { isHttpUrl } from "@/lib/utils";

/** Full-width dark block at the bottom of the verification view, as in the wireframe. Always dark, in both themes. */
export function CitationsPanel({ citations, claims }: { citations: Citation[]; claims: Claim[] }) {
  return (
    <section aria-labelledby="citations-heading" className="border-t border-line-strong bg-cite px-4 py-6 text-[#d4e7fb] sm:px-8">
      <div className="mx-auto max-w-[1180px]">
        <h3 id="citations-heading" className="mb-4 flex items-center gap-2 text-sm font-semibold text-[#e4f1ff]">
          <Link2 className="size-4 text-[#36dcf3]" /> Citations
          <span className="rounded-full border border-[#143c61] px-2 py-0.5 text-[11px] font-normal text-[#8aa9c9]">{citations.length}</span>
        </h3>

        {citations.length === 0 ? (
          <p className="text-sm text-[#8aa9c9]">No citations available for this response.</p>
        ) : (
          <ol className="grid gap-3 lg:grid-cols-2">
            {citations.map((c, i) => {
              const supports = claimsForCitation(c, claims);
              return (
                <li key={`${c.id}-${i}`} id={`citation-${c.id}`} data-testid="citation" className="flex flex-col gap-2.5 rounded-xl border border-[#143c61] bg-[#071a30] p-4">
                  <div className="flex items-start justify-between gap-3">
                    <h4 className="min-w-0 text-sm font-semibold leading-snug text-[#e4f1ff]">
                      {isHttpUrl(c.url) ? (
                        <a href={c.url} target="_blank" rel="noopener noreferrer" className="inline-flex items-start gap-1.5 hover:text-[#36dcf3]">
                          <span className="break-words">{c.title ?? c.domain ?? c.url}</span>
                          <ExternalLink className="mt-0.5 size-3.5 shrink-0" aria-hidden />
                        </a>
                      ) : (
                        <span className="break-words">{c.title ?? `Source ${i + 1}`}</span>
                      )}
                    </h4>
                    {c.domain && <span className="shrink-0 rounded-full border border-[#143c61] bg-[#0a203b] px-2 py-0.5 text-[11px] text-[#8aa9c9]">{c.domain}</span>}
                  </div>

                  {c.url && (
                    <p className="break-all text-xs">
                      {isHttpUrl(c.url) ? (
                        <a href={c.url} target="_blank" rel="noopener noreferrer" className="text-[#36dcf3] underline underline-offset-2">
                          {c.url}
                        </a>
                      ) : (
                        <span className="text-[#8aa9c9]">{c.url}</span>
                      )}
                    </p>
                  )}

                  {c.excerpt && <blockquote className="border-l-2 border-[#08c9ed] pl-3 text-[13px] italic leading-relaxed text-[#c8def3]">{c.excerpt}</blockquote>}

                  <div className="text-xs text-[#8aa9c9]">
                    <span className="font-medium">Supports:</span>{" "}
                    {supports.length ? (
                      <ul className="mt-1 space-y-1">
                        {supports.map((cl) => (
                          <li key={cl.id} className="line-clamp-2 text-[#c8def3]">• {cl.text}</li>
                        ))}
                      </ul>
                    ) : (
                      "No specific claim was linked to this source."
                    )}
                  </div>
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </section>
  );
}
