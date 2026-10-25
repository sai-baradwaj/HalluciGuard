import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import type { MetricSet, MetricValue } from "@/types";
import { buildMetricRows, formatDelta, formatMetric, hasAnyMetric, toPercent, type MetricRow } from "@/lib/metrics";
import { cn } from "@/lib/utils";

function ValueCell({ v }: { v?: MetricValue }) {
  if (!v) return <span className="text-muted">N/A</span>;
  return (
    <div>
      <span className="font-medium tabular-nums">{formatMetric(v)}</span>
      <span className="ml-1.5 text-[10px] text-muted">{v.unit === "ratio" ? "score 0–1" : "percent"}</span>
    </div>
  );
}

function ChangeCell({ row }: { row: MetricRow }) {
  if (!row.delta) return <span className="text-muted">{row.note ? <span className="text-xs">{row.note}</span> : "N/A"}</span>;
  const { outcome } = row;
  const Icon = row.delta.value > 0 ? ArrowUp : row.delta.value < 0 ? ArrowDown : Minus;
  // Direction of the number and whether that is good are different things (risk going down is an improvement).
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 font-medium tabular-nums",
        outcome === "improved" && "text-success",
        outcome === "regressed" && "text-danger",
        (!outcome || outcome === "unchanged") && "text-muted",
      )}
    >
      <Icon className="size-4" />
      {formatDelta(row.delta)}
      {outcome && <span className="text-[11px] font-normal">{outcome === "improved" ? "Improved" : outcome === "regressed" ? "Regressed" : "No change"}</span>}
    </span>
  );
}

export function MetricsComparison({ initial, final }: { initial: MetricSet; final: MetricSet }) {
  const rows = buildMetricRows(initial, final);
  const any = hasAnyMetric(initial) || hasAnyMetric(final);
  const chartRows = rows.filter((r) => r.initial || r.final);

  return (
    <div>
      {!any && <p className="mb-3 rounded-lg border border-line bg-panel2 px-3 py-2 text-xs text-muted">The backend did not return evaluation metrics for this response. No scores are estimated by the frontend.</p>}

      <div className="overflow-x-auto rounded-xl border border-line">
        <table className="w-full min-w-[34rem] border-collapse text-sm" data-testid="metrics-table">
          <thead>
            <tr className="bg-panel2 text-left text-xs text-muted">
              <th scope="col" className="px-4 py-2.5 font-medium">Metric</th>
              <th scope="col" className="px-4 py-2.5 font-medium">Initial value</th>
              <th scope="col" className="px-4 py-2.5 font-medium">Final value</th>
              <th scope="col" className="px-4 py-2.5 font-medium">Change</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id} className="border-t border-line">
                <th scope="row" className="px-4 py-3 text-left font-medium">
                  {row.label}
                  {row.better && (
                    <span className="mt-0.5 flex items-center gap-1 text-[11px] font-normal text-muted">
                      {row.better === "higher" ? <ArrowUp className="size-3" /> : <ArrowDown className="size-3" />}
                      {row.better === "higher" ? "Higher is better" : "Lower is better"}
                    </span>
                  )}
                </th>
                <td className="px-4 py-3"><ValueCell v={row.initial} /></td>
                <td className="px-4 py-3"><ValueCell v={row.final} /></td>
                <td className="px-4 py-3"><ChangeCell row={row} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[11px] text-muted">Values come from the backend. “Score 0–1” values are normalized; “percent” values are 0–100. Changes are only calculated when both values use the same unit.</p>

      {chartRows.length > 0 && (
        <figure className="mt-5" aria-label="Initial versus final metrics chart">
          <figcaption className="mb-3 flex flex-wrap items-center gap-4 text-xs text-muted">
            <span className="flex items-center gap-1.5"><span className="size-2.5 rounded-sm bg-[#4f86b5]" /> Initial</span>
            <span className="flex items-center gap-1.5"><span className="size-2.5 rounded-sm bg-success" /> Final</span>
            <span>Bars show each value on a 0–100% scale.</span>
          </figcaption>
          <div className="space-y-3" aria-hidden>
            {chartRows.map((row) => (
              <div key={row.id} className="grid grid-cols-[7.5rem_1fr] items-center gap-3 text-xs sm:grid-cols-[10rem_1fr]">
                <span className="truncate text-soft">{row.label}</span>
                <div className="space-y-1">
                  {([["initial", row.initial, "bg-[#4f86b5]"], ["final", row.final, "bg-success"]] as const).map(([k, v, color]) => (
                    <div key={k} className="flex items-center gap-2">
                      <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-panel3">
                        {v && <div className={cn("h-full rounded-full", color)} style={{ width: `${Math.min(100, Math.max(0, toPercent(v)))}%` }} />}
                      </div>
                      <span className="w-12 text-right tabular-nums text-muted">{formatMetric(v)}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </figure>
      )}
    </div>
  );
}
