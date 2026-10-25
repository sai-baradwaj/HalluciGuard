import { buildMetricRows, compareMetric, formatDelta, formatMetric } from "@/lib/metrics";
import { parseMetricSet } from "@/services/normalize";

describe("compareMetric", () => {
  it("treats a higher factuality as an improvement and a lower risk as an improvement", () => {
    expect(compareMetric({ value: 0.5, unit: "ratio" }, { value: 0.9, unit: "ratio" }, "higher").outcome).toBe("improved");
    expect(compareMetric({ value: 0.7, unit: "ratio" }, { value: 0.1, unit: "ratio" }, "lower").outcome).toBe("improved");
    expect(compareMetric({ value: 0.1, unit: "ratio" }, { value: 0.4, unit: "ratio" }, "lower").outcome).toBe("regressed");
  });

  it("calculates a delta only when both values exist and share a unit", () => {
    expect(compareMetric(undefined, { value: 0.9, unit: "ratio" }, "higher").delta).toBeUndefined();
    const mixed = compareMetric({ value: 0.5, unit: "ratio" }, { value: 80, unit: "percent" }, "higher");
    expect(mixed.delta).toBeUndefined();
    expect(mixed.note).toMatch(/different units/);
  });

  it("reports unchanged values", () => {
    expect(compareMetric({ value: 0.5, unit: "ratio" }, { value: 0.5, unit: "ratio" }, "higher").outcome).toBe("unchanged");
  });
});

describe("formatting and rows", () => {
  it("formats scores and percentages differently and N/A when missing", () => {
    expect(formatMetric({ value: 0.62, unit: "ratio" })).toBe("0.62");
    expect(formatMetric({ value: 62, unit: "percent" })).toBe("62%");
    expect(formatMetric(undefined)).toBe("N/A");
    expect(formatDelta({ value: 0.29, unit: "ratio" })).toBe("+0.29");
    expect(formatDelta({ value: -18, unit: "percent" })).toBe("−18 pts");
  });

  it("always lists the six standard metrics, with N/A for those the backend omitted", () => {
    const rows = buildMetricRows(parseMetricSet({ factuality: 0.5 }), parseMetricSet({ factuality: 0.9 }));
    expect(rows.map((r) => r.label)).toEqual(["Factuality", "Hallucination Risk", "Completeness", "Logical Consistency", "Relevance", "Evidence Coverage"]);
    expect(rows[0].delta?.value).toBe(0.4);
    expect(rows[1].initial).toBeUndefined();
  });
});
