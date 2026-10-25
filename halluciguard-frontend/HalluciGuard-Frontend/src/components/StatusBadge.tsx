import { AlertTriangle, FlaskConical, Loader2, PenLine, ShieldCheck, XCircle } from "lucide-react";
import type { ClaimStatus, VerificationStatus } from "@/types";
import { cn } from "@/lib/utils";

const base = "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium leading-5";

const STATUS: Record<VerificationStatus, { label: string; cls: string; icon: React.ReactNode }> = {
  processing: { label: "Verification in progress", cls: "border-accent/40 bg-accent/10 text-accent-text", icon: <Loader2 className="size-3 animate-spin" /> },
  verified: { label: "Verified", cls: "border-success/40 bg-success/10 text-success", icon: <ShieldCheck className="size-3" /> },
  corrected: { label: "Corrected", cls: "border-accent/40 bg-accent/10 text-accent-text", icon: <PenLine className="size-3" /> },
  failed: { label: "Verification failed", cls: "border-danger/40 bg-danger/10 text-danger", icon: <XCircle className="size-3" /> },
  review: { label: "Quality check not passed", cls: "border-warning/40 bg-warning/10 text-warning", icon: <AlertTriangle className="size-3" /> },
};

export function StatusBadge({ status, className }: { status: VerificationStatus; className?: string }) {
  const s = STATUS[status];
  return (
    <span className={cn(base, s.cls, className)} data-testid="status-badge">
      {s.icon}
      {s.label}
    </span>
  );
}

export function DemoBadge({ className }: { className?: string }) {
  return (
    <span className={cn(base, "border-warning/40 bg-warning/10 text-warning", className)} title="This result was generated locally by the demo adapter. It is not a real evaluation.">
      <FlaskConical className="size-3" />
      Demo data
    </span>
  );
}

export const CLAIM_STATUS: Record<ClaimStatus, { label: string; cls: string }> = {
  verified: { label: "Verified", cls: "border-success/40 bg-success/10 text-success" },
  unsupported: { label: "Unsupported", cls: "border-warning/40 bg-warning/10 text-warning" },
  contradicted: { label: "Contradicted", cls: "border-danger/40 bg-danger/10 text-danger" },
  partially_supported: { label: "Partially Supported", cls: "border-accent/40 bg-accent/10 text-accent-text" },
  unable_to_verify: { label: "Unable to Verify", cls: "border-line-strong bg-panel3 text-muted" },
  unknown: { label: "Status not provided", cls: "border-line bg-panel3 text-muted" },
};

export function ClaimStatusBadge({ status }: { status: ClaimStatus }) {
  const s = CLAIM_STATUS[status];
  return <span className={cn(base, s.cls)}>{s.label}</span>;
}
