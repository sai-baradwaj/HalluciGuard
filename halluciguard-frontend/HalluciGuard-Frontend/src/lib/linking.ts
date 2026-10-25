import type { Citation, Claim } from "@/types";

const norm = (s: string) => s.toLowerCase().replace(/\s+/g, " ").trim();

/** Claims a citation supports, using references on either side (ids, urls, or claim text). */
export function claimsForCitation(citation: Citation, claims: Claim[]): Claim[] {
  return claims.filter((claim) => {
    const byCitation = claim.citationRefs.some((ref) => ref === citation.id || (citation.url && ref === citation.url));
    const byClaimRef = citation.claimRefs.some((ref) => ref === claim.id || norm(ref) === norm(claim.text));
    return byCitation || byClaimRef;
  });
}

/** Citations a claim references. */
export function citationsForClaim(claim: Claim, citations: Citation[]): Citation[] {
  return citations.filter((c) => claimsForCitation(c, [claim]).length > 0);
}
