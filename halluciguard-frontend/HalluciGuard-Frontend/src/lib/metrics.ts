import type { MetricKey, MetricSet, MetricValue } from "@/types";

export interface MetricDef {
  key: MetricKey;
  label: string;
  better: "higher" | "lower";
}

/** Rows shown in the comparison table, in the order the wireframe lists them. */
export const METRIC_DEFS: MetricDef[] = [
  { key: "factuality", label: "Factuality", better: "higher" },
  { key: "hallucination_risk", label: "Hallucination Risk", better: "lower" },
  { key: "completeness", label: "Completeness", better: "higher" },
  { key: "logical_consistency", label: "Logical Consistency", better: "higher" },
  { key: "relevance", label: "Relevance", better: "higher" },
  { key: "evidence_coverage", label: "Evidence Coverage", better: "higher" },
];

export type Outcome = "improved" | "regressed" | "unchanged";

export interface MetricRow {
  id: string;
  label: string;
  /** undefined for extra backend metrics where we do not know which direction is desirable. */
  better?: "higher" | "lower";
  initial?: MetricValue;
  final?: MetricValue;
  delta?: MetricValue; // signed difference in the shared unit
  outcome?: Outcome;
  note?: string;
}

const round = (n: number, digits: number) => Number(n.toFixed(digits));

export function compareMetric(
  initial: MetricValue | undefined,
  final: MetricValue | undefined,
  better?: "higher" | "lower",
): Pick<MetricRow, "delta" | "outcome" | "note"> {
  if (!initial || !final) return {};
  if (initial.unit !== final.unit) return { note: "Not comparable: initial and final use different units" };
  const digits = initial.unit === "ratio" ? 2 : 1;
  const diff = round(final.value - initial.value, digits);
  const delta: MetricValue = { value: diff, unit: initial.unit };
  if (diff === 0) return { delta, outcome: "unchanged" };
  if (!better) return { delta };
  const improved = better === "higher" ? diff > 0 : diff < 0;
  return { delta, outcome: improved ? "improved" : "regressed" };
}

export function buildMetricRows(initial: MetricSet, final: MetricSet): MetricRow[] {
  const rows: MetricRow[] = METRIC_DEFS.map((def) => {
    const i = initial.known[def.key];
    const f = final.known[def.key];
    return { id: def.key, label: def.label, better: def.better, initial: i, final: f, ...compareMetric(i, f, def.better) };
  });

  const extras = new Map<string, MetricRow>();
  for (const e of initial.extra) extras.set(e.name.toLowerCase(), { id: `extra-${e.name}`, label: capitalize(e.name), initial: e.value });
  for (const e of final.extra) {
    const k = e.name.toLowerCase();
    const row = extras.get(k) ?? { id: `extra-${e.name}`, label: capitalize(e.name) };
    row.final = e.value;
    extras.set(k, row);
  }
  for (const row of extras.values()) rows.push({ ...row, ...compareMetric(row.initial, row.final) });
  return rows;
}

export const capitalize = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) : s);

export function formatMetric(v?: MetricValue): string {
  if (!v) return "N/A";
  return v.unit === "ratio" ? v.value.toFixed(2) : `${round(v.value, 1)}%`;
}

export function formatDelta(d?: MetricValue): string {
  if (!d) return "N/A";
  const sign = d.value > 0 ? "+" : d.value < 0 ? "−" : "";
  const abs = Math.abs(d.value);
  return d.unit === "ratio" ? `${sign}${abs.toFixed(2)}` : `${sign}${round(abs, 1)} pts`;
}

/** 0–100 for drawing bars. */
export const toPercent = (v: MetricValue) => (v.unit === "ratio" ? v.value * 100 : v.value);

export const hasAnyMetric = (set: MetricSet) => Object.keys(set.known).length > 0 || set.extra.length > 0;
