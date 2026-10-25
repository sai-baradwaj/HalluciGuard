import { normalizeBackendResult, normalizeClaimStatus, parseMetricSet } from "@/services/normalize";

const ctx = { responseId: "r", conversationId: "c", query: "q", source: "backend" } as const;

describe("normalizeBackendResult", () => {
  it("reads the existing backend field names (base_answer / rag_answer / quality_pass)", () => {
    const r = normalizeBackendResult(
      { base_answer: "A", rag_answer: "B", base_metrics: { factuality: 0.4 }, final_metrics: { factuality: 0.8 }, citations: ["https://x.org/p"], quality_pass: true },
      ctx,
    );
    expect(r.initialResponse).toBe("A");
    expect(r.finalResponse).toBe("B");
    expect(r.metrics.final.known.factuality).toEqual({ value: 0.8, unit: "ratio" });
    expect(r.citations[0]).toMatchObject({ url: "https://x.org/p", domain: "x.org" });
    expect(r.status).toBe("corrected"); // passed, and the final answer differs from the initial one
  });

  it("reads the suggested contract (initial_response / final_response / metrics.initial|final)", () => {
    const r = normalizeBackendResult({ initial_response: "A", final_response: "A", verification_status: "completed", metrics: { initial: { factuality: 0.6 }, final: { factuality: 0.6 } } }, ctx);
    expect(r.status).toBe("verified");
    expect(r.metrics.initial.known.factuality?.value).toBe(0.6);
  });

  it("never invents values: missing data stays empty and no status is claimed", () => {
    const r = normalizeBackendResult({ answer: "only an answer" }, ctx);
    expect(r.finalResponse).toBeUndefined();
    expect(r.status).toBeUndefined();
    expect(r.claims).toEqual([]);
    expect(r.citations).toEqual([]);
    expect(r.metrics.initial.known).toEqual({});
  });

  it("maps explicit failure / processing / quality gate states", () => {
    expect(normalizeBackendResult({ answer: "x", status: "failed" }, ctx).status).toBe("failed");
    expect(normalizeBackendResult({ answer: "x", status: "running" }, ctx).status).toBe("processing");
    expect(normalizeBackendResult({ answer: "x", quality_pass: false }, ctx).status).toBe("review");
  });

  it("unwraps a {result: ...} envelope", () => {
    expect(normalizeBackendResult({ result: { base_answer: "A", rag_answer: "B" } }, ctx).finalResponse).toBe("B");
  });
});

describe("parseMetricSet", () => {
  it("distinguishes normalized scores from percentages", () => {
    const s = parseMetricSet({ factuality: 0.62, relevance: 88, completeness_pct: 0.5, evidence_coverage: { value: 0.9, unit: "percent" } });
    expect(s.known.factuality?.unit).toBe("ratio");
    expect(s.known.relevance).toEqual({ value: 88, unit: "percent" });
    expect(s.known.completeness).toEqual({ value: 0.5, unit: "percent" });
    expect(s.known.evidence_coverage?.unit).toBe("percent");
  });

  it("accepts array form and keeps unknown metrics as extras", () => {
    const s = parseMetricSet([{ name: "Factuality", score: 0.5 }, { metric: "Tone fit", value: 0.7 }]);
    expect(s.known.factuality?.value).toBe(0.5);
    expect(s.extra[0].name).toBe("Tone fit");
  });

  it("skips values that cannot be interpreted as scores", () => {
    expect(parseMetricSet({ factuality: "N/A", relevance: 4000, completeness: -1 }).known).toEqual({});
  });
});

describe("normalizeClaimStatus", () => {
  it.each([
    ["SUPPORTED", "verified"],
    ["contradicted", "contradicted"],
    ["partially_supported", "partially_supported"],
    ["not enough info", "unable_to_verify"],
    ["unsupported", "unsupported"],
    ["???", "unknown"],
  ])("%s -> %s", (raw, expected) => expect(normalizeClaimStatus(raw)).toBe(expected));
});
