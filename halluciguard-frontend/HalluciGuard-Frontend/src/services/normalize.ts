/**
 * Maps a raw HalluciGuard backend payload onto the typed VerificationResult.
 *
 * The existing backend contract (POST /verify) uses names like base_answer / rag_answer / base_metrics /
 * final_metrics / citations / quality_pass. The suggested contract in the product brief (initial_response,
 * final_response, metrics.{initial,final}, claims, correction_summary) is accepted as well. This file is the
 * ONLY place that knows field names, so adapting to a changed backend means editing the alias lists below.
 *
 * Rule: nothing is invented. A field the backend did not send stays undefined / empty.
 */
import type {
  Citation,
  Claim,
  ClaimStatus,
  CorrectionSummary,
  MetricKey,
  MetricSet,
  MetricUnit,
  MetricValue,
  VerificationResult,
  VerificationStatus,
} from "@/types";
import { domainOf, isHttpUrl } from "@/lib/utils";

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);

function pick(obj: unknown, keys: string[]): unknown {
  if (!isObj(obj)) return undefined;
  for (const k of keys) {
    const v = obj[k];
    if (v !== undefined && v !== null && v !== "") return v;
  }
  return undefined;
}

/** Returns a non-empty string, reading common wrapper shapes like {text} / {answer}. */
function asText(v: unknown): string | undefined {
  if (typeof v === "string") return v.trim() ? v : undefined;
  if (isObj(v)) return asText(pick(v, ["answer", "text", "content", "response", "output"]));
  return undefined;
}

function asStringList(v: unknown): string[] {
  if (v === undefined || v === null) return [];
  const list = Array.isArray(v) ? v : [v];
  return list
    .map((item) => {
      if (typeof item === "string") return item.trim();
      if (typeof item === "number") return String(item);
      if (isObj(item)) return asText(pick(item, ["text", "snippet", "excerpt", "content", "quote", "evidence"])) ?? "";
      return "";
    })
    .filter(Boolean);
}

/* ------------------------------------------------------------------ metrics */

const METRIC_ALIASES: Record<MetricKey, string[]> = {
  factuality: ["factuality", "factuality_score", "factual_accuracy", "factual_score", "factualness"],
  hallucination_risk: [
    "hallucination_risk",
    "hallucination_score",
    "hallucination_rate",
    "hallucination",
    "hallucination_probability",
  ],
  completeness: ["completeness", "completeness_score"],
  logical_consistency: ["logical_consistency", "logical_consistency_score", "consistency", "logic"],
  relevance: ["relevance", "relevance_score", "answer_relevance", "relevancy"],
  evidence_coverage: ["evidence_coverage", "evidence_coverage_score", "citation_coverage", "source_coverage"],
};

const aliasToKey = new Map<string, MetricKey>();
for (const [key, aliases] of Object.entries(METRIC_ALIASES) as [MetricKey, string[]][]) {
  for (const a of aliases) aliasToKey.set(a, key);
}

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");

function unitFromHint(hint: unknown): MetricUnit | undefined {
  if (typeof hint !== "string") return undefined;
  const h = hint.toLowerCase();
  if (h === "percent" || h === "percentage" || h === "%" || h === "pct") return "percent";
  if (h === "ratio" || h === "score" || h === "normalized" || h === "normalised" || h === "fraction") return "ratio";
  return undefined;
}

function parseMetricValue(raw: unknown, nameHint: string): MetricValue | undefined {
  let num: number | undefined;
  let unit: MetricUnit | undefined;
  let explicit = false;

  if (typeof raw === "number") num = raw;
  else if (typeof raw === "string" && raw.trim() !== "" && Number.isFinite(Number(raw.replace("%", "")))) {
    num = Number(raw.replace("%", ""));
    if (raw.includes("%")) {
      unit = "percent";
      explicit = true;
    }
  } else if (isObj(raw)) {
    const pct = raw.percentage ?? raw.percent;
    const val = raw.score ?? raw.value;
    if (typeof pct === "number") {
      num = pct;
      unit = "percent";
      explicit = true;
    } else if (typeof val === "number") {
      num = val;
    }
    const hinted = unitFromHint(raw.unit);
    if (hinted) {
      unit = hinted;
      explicit = true;
    }
  }

  if (num === undefined || !Number.isFinite(num) || num < 0) return undefined;
  if (/_(percent|pct|percentage)$/.test(nameHint)) {
    unit = "percent";
    explicit = true;
  }
  if (!unit) unit = num <= 1 ? "ratio" : "percent";
  if (unit === "ratio" && num > 1 && !explicit) unit = "percent";
  if (unit === "percent" && num > 100) return undefined; // cannot be interpreted as a score
  return { value: num, unit };
}

export function parseMetricSet(raw: unknown): MetricSet {
  const set: MetricSet = { known: {}, extra: [] };
  const entries: Array<[string, unknown]> = [];

  if (Array.isArray(raw)) {
    raw.forEach((item, i) => {
      if (!isObj(item)) return;
      const name = pick(item, ["name", "metric", "label", "key"]);
      entries.push([typeof name === "string" ? name : `Metric ${i + 1}`, item]);
    });
  } else if (isObj(raw)) {
    // Allow {metrics: {...}} wrappers
    const inner = isObj(raw.metrics) ? raw.metrics : raw;
    for (const [k, v] of Object.entries(inner)) entries.push([k, v]);
  }

  for (const [name, value] of entries) {
    const key = slug(name);
    const stripped = key.replace(/_(percent|pct|percentage)$/, "");
    const parsed = parseMetricValue(value, key);
    if (!parsed) continue;
    const canonical = aliasToKey.get(stripped);
    if (canonical) {
      if (!set.known[canonical]) set.known[canonical] = parsed;
    } else {
      set.extra.push({ name: name.replace(/_/g, " "), value: parsed });
    }
  }
  return set;
}

/* ------------------------------------------------------------------- claims */

export function normalizeClaimStatus(raw: unknown): ClaimStatus {
  if (typeof raw === "boolean") return raw ? "verified" : "unsupported";
  if (typeof raw !== "string") return "unknown";
  const s = raw.toLowerCase().replace(/[_-]+/g, " ").trim();
  if (!s) return "unknown";
  if (/contradict|refut|incorrect|false|disproved/.test(s)) return "contradicted";
  if (/partial|mixed|somewhat/.test(s)) return "partially_supported";
  if (/unable|cannot|can't|insufficient|inconclusive|not enough|unknown|undetermined|\bnei\b/.test(s)) return "unable_to_verify";
  if (/unsupport|not supported|no support|no evidence|ungrounded|unverified|hallucinat/.test(s)) return "unsupported";
  if (/support|verified|true|correct|entail|confirm|valid|accurate|pass/.test(s)) return "verified";
  return "unknown";
}

function parseClaims(raw: unknown): Claim[] {
  if (!Array.isArray(raw)) return [];
  const claims: Claim[] = [];
  raw.forEach((item, i) => {
    if (typeof item === "string") {
      if (item.trim()) claims.push({ id: String(i + 1), text: item.trim(), status: "unknown", evidence: [], citationRefs: [] });
      return;
    }
    if (!isObj(item)) return;
    const text = asText(pick(item, ["claim", "text", "statement", "content", "claim_text"]));
    if (!text) return;
    const idRaw = pick(item, ["id", "claim_id"]);
    const corrected = pick(item, ["corrected", "was_corrected", "is_corrected", "revised"]);
    const refs = pick(item, ["citation_ids", "citations", "source_ids", "source_urls", "sources", "supported_by"]);
    const refList = (Array.isArray(refs) ? refs : refs !== undefined ? [refs] : [])
      .map((r) => (typeof r === "number" ? String(r) : typeof r === "string" ? r : isObj(r) ? String(pick(r, ["id", "citation_id", "url"]) ?? "") : ""))
      .filter(Boolean);
    claims.push({
      id: idRaw !== undefined ? String(idRaw) : String(i + 1),
      text,
      status: normalizeClaimStatus(pick(item, ["status", "verdict", "label", "verification_status", "result", "support"])),
      corrected: typeof corrected === "boolean" ? corrected : undefined,
      correctedText: asText(pick(item, ["corrected_claim", "corrected_text", "revised_claim", "correction"])),
      explanation: asText(pick(item, ["explanation", "reason", "reasoning", "rationale", "justification", "notes"])),
      evidence: asStringList(pick(item, ["evidence", "supporting_evidence", "evidence_excerpt", "evidence_text", "snippets"])),
      citationRefs: refList,
    });
  });
  return claims;
}

/* ---------------------------------------------------------------- citations */

function parseCitations(raw: unknown): Citation[] {
  const list = Array.isArray(raw) ? raw : raw !== undefined && raw !== null ? [raw] : [];
  const out: Citation[] = [];
  list.forEach((item, i) => {
    if (typeof item === "string") {
      const s = item.trim();
      if (!s) return;
      out.push(isHttpUrl(s) ? { id: String(i + 1), url: s, title: s, domain: domainOf(s), claimRefs: [] } : { id: String(i + 1), title: s, claimRefs: [] });
      return;
    }
    if (!isObj(item)) return;
    const idRaw = pick(item, ["id", "citation_id"]);
    const urlRaw = pick(item, ["url", "href", "link", "source_url"]);
    const sourceRaw = pick(item, ["source"]);
    const url = isHttpUrl(urlRaw) ? urlRaw : isHttpUrl(sourceRaw) ? sourceRaw : undefined;
    const titleRaw = pick(item, ["title", "name", "citation", "source_title"]) ?? (typeof sourceRaw === "string" && !isHttpUrl(sourceRaw) ? sourceRaw : undefined);
    const refs = pick(item, ["claims", "claim_ids", "supports", "supported_claims", "claim"]);
    const claimRefs = (Array.isArray(refs) ? refs : refs !== undefined ? [refs] : [])
      .map((r) => (typeof r === "number" ? String(r) : typeof r === "string" ? r : isObj(r) ? String(pick(r, ["id", "claim_id", "text", "claim"]) ?? "") : ""))
      .filter(Boolean);
    const domainRaw = pick(item, ["domain", "website", "host"]);
    const citation: Citation = {
      id: idRaw !== undefined ? String(idRaw) : String(i + 1),
      title: typeof titleRaw === "string" ? titleRaw : undefined,
      url,
      domain: typeof domainRaw === "string" ? domainRaw : url ? domainOf(url) : undefined,
      excerpt: asText(pick(item, ["excerpt", "snippet", "evidence", "content", "text", "quote"])),
      claimRefs,
    };
    if (citation.title || citation.url || citation.excerpt) out.push(citation);
  });
  return out;
}

/* ------------------------------------------------------ correction summary */

function parseCorrectionSummary(raw: unknown): CorrectionSummary | undefined {
  if (raw === undefined || raw === null) return undefined;
  if (typeof raw === "string") return raw.trim() ? { text: raw, changes: [] } : undefined;
  if (Array.isArray(raw)) {
    const changes = asStringList(raw);
    return changes.length ? { changes } : undefined;
  }
  if (isObj(raw)) {
    const text = asText(pick(raw, ["summary", "text", "description", "overview"]));
    const changes = asStringList(pick(raw, ["changes", "corrections", "items", "details"]));
    return text || changes.length ? { text, changes } : undefined;
  }
  return undefined;
}

/* ------------------------------------------------------------------- status */

const sameText = (a: string, b: string) => a.replace(/\s+/g, " ").trim() === b.replace(/\s+/g, " ").trim();

function deriveStatus(
  data: unknown,
  initial: string | undefined,
  final: string | undefined,
  claims: Claim[],
  summary: CorrectionSummary | undefined,
): { status?: VerificationStatus; qualityPass?: boolean } {
  const statusRaw = pick(data, ["verification_status", "status", "state"]);
  const statusStr = typeof statusRaw === "string" ? statusRaw.toLowerCase().replace(/[\s-]+/g, "_") : undefined;
  const passRaw = pick(data, ["quality_pass", "passed"]);
  let qualityPass = typeof passRaw === "boolean" ? passRaw : undefined;
  if (qualityPass === undefined && (statusStr === "pass" || statusStr === "passed")) qualityPass = true;

  if (statusStr && /^(failed|error|errored|failure)$/.test(statusStr)) return { status: "failed", qualityPass };
  if (statusStr && /^(processing|pending|running|in_progress|queued)$/.test(statusStr)) return { status: "processing", qualityPass };

  const completed = statusStr ? /^(completed|complete|done|success|succeeded|verified|passed|pass|ok)$/.test(statusStr) : false;
  if (statusStr === "fail") return { status: "review", qualityPass: false };
  if (qualityPass === false) return { status: "review", qualityPass };

  if (completed || qualityPass === true) {
    const corrected =
      (initial !== undefined && final !== undefined && !sameText(initial, final)) ||
      claims.some((c) => c.corrected === true) ||
      (summary?.changes.length ?? 0) > 0;
    return { status: corrected ? "corrected" : "verified", qualityPass };
  }
  return { status: undefined, qualityPass };
}

/* -------------------------------------------------------------------- main */

export interface NormalizeContext {
  responseId: string;
  conversationId?: string;
  query: string;
  source: "backend" | "demo";
}

export function normalizeBackendResult(rawInput: unknown, ctx: NormalizeContext): VerificationResult {
  // Unwrap common envelopes like { result: {...} } / { data: {...} }
  let data: unknown = rawInput;
  if (isObj(data) && !pick(data, ["base_answer", "rag_answer", "initial_response", "final_response", "final_answer", "answer", "response"])) {
    const inner = pick(data, ["result", "data"]);
    if (isObj(inner)) data = inner;
  }

  const initialResponse = asText(
    pick(data, ["initial_response", "base_answer", "initial_answer", "original_answer", "original_response", "llm_response", "baseline_answer", "response", "answer"]),
  );
  const finalResponse = asText(
    pick(data, ["final_response", "rag_answer", "final_answer", "verified_answer", "corrected_answer", "corrected_response", "output"]),
  );

  const metricsContainer = pick(data, ["metrics"]);
  const initialMetricsRaw =
    pick(data, ["base_metrics", "initial_metrics", "original_metrics"]) ?? (isObj(metricsContainer) ? metricsContainer.initial : undefined);
  const finalMetricsRaw =
    pick(data, ["final_metrics", "rag_metrics", "verified_metrics"]) ?? (isObj(metricsContainer) ? metricsContainer.final : undefined);

  const claims = parseClaims(pick(data, ["claims", "claim_verification", "claim_results", "verified_claims", "claim_checks"]));
  const citations = parseCitations(pick(data, ["citations", "sources", "references"]));
  const correctionSummary = parseCorrectionSummary(pick(data, ["correction_summary", "corrections", "correction_notes", "changes"]));
  const notesRaw = pick(data, ["verification_notes", "notes", "message"]);
  const notes = typeof notesRaw === "string" ? notesRaw : Array.isArray(notesRaw) ? asStringList(notesRaw).join("\n") || undefined : undefined;

  const { status, qualityPass } = deriveStatus(data, initialResponse, finalResponse, claims, correctionSummary);

  return {
    responseId: ctx.responseId,
    conversationId: ctx.conversationId,
    query: ctx.query,
    initialResponse,
    finalResponse,
    status,
    qualityPass,
    metrics: { initial: parseMetricSet(initialMetricsRaw), final: parseMetricSet(finalMetricsRaw) },
    claims,
    citations,
    correctionSummary,
    notes,
    source: ctx.source,
  };
}
