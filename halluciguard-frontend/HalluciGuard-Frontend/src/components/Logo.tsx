import { cn } from "@/lib/utils";

export function ShieldMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 40 44" aria-hidden="true" className={cn("fill-none stroke-accent", className)} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 3 35 9v11c0 10-6 16-15 21C11 36 5 30 5 20V9Z" />
      <path d="m12 21 5 5 11-12" />
    </svg>
  );
}

export function Logo({ compact }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <ShieldMark className="h-9 w-8 shrink-0" />
      {!compact && (
        <div className="leading-tight">
          <div className="text-lg font-bold tracking-tight">
            Halluci<span className="text-accent-text">Guard</span>
          </div>
          <div className="text-[10px] tracking-wide text-muted">Detect · Verify · Trust</div>
        </div>
      )}
    </div>
  );
}
