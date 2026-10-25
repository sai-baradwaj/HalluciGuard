import { ArrowUpRight } from "lucide-react";
import { ShieldMark } from "./Logo";

const SUGGESTIONS = [
  { title: "Urban air pollution", hint: "Explore common causes and evidence", prompt: "What are the main causes of urban air pollution?" },
  { title: "Cleaner cities", hint: "Review practical approaches", prompt: "How can cities reduce air pollution?" },
  { title: "Air quality and health", hint: "Ask about health impacts", prompt: "What is the impact of air pollution on health?" },
];

export function EmptyState({ onPick }: { onPick(prompt: string): void }) {
  return (
    <div className="mx-auto flex max-w-2xl flex-col items-center px-4 pt-[8vh] text-center">
      <ShieldMark className="mb-5 h-14 w-12" />
      <h2 className="text-2xl font-semibold tracking-tight">Answers you can trust.</h2>
      <p className="mt-2 max-w-md text-sm leading-relaxed text-muted">
        Ask a question. HalluciGuard checks the answer against evidence, corrects it, and lets you inspect exactly what changed.
      </p>
      <div className="mt-8 grid w-full gap-3 text-left sm:grid-cols-3">
        {SUGGESTIONS.map((s) => (
          <button
            key={s.title}
            type="button"
            onClick={() => onPick(s.prompt)}
            className="group relative rounded-xl border border-line bg-panel p-4 transition-colors hover:border-accent"
          >
            <ArrowUpRight className="absolute right-3 top-3 size-4 text-accent-text" />
            <div className="mt-4 text-sm font-semibold">{s.title}</div>
            <div className="mt-1 text-xs leading-relaxed text-muted">{s.hint}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
