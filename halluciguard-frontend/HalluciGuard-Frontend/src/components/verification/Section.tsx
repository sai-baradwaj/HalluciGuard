import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Section({ title, icon, aside, children, className, id }: { title: string; icon?: ReactNode; aside?: ReactNode; children: ReactNode; className?: string; id?: string }) {
  return (
    <section id={id} className={cn("rounded-2xl border border-line bg-panel p-4 sm:p-5", className)}>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <h3 className="flex items-center gap-2 text-sm font-semibold">
          {icon && <span className="text-accent-text [&_svg]:size-4">{icon}</span>}
          {title}
        </h3>
        {aside}
      </div>
      {children}
    </section>
  );
}
