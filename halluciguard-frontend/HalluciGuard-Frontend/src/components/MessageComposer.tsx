import { useEffect, useRef } from "react";
import { ArrowUp, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  value: string;
  onChange(value: string): void;
  onSend(text: string): Promise<boolean> | boolean;
  busy: boolean;
}

/** Wide rounded composer. Enter sends, Shift+Enter adds a line. The attachment button from the wireframe is
 *  omitted because the backend's /verify contract accepts text only. */
export function MessageComposer({ value, onChange, onSend, busy }: Props) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const sending = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  const canSend = !!value.trim() && !busy;

  const submit = async () => {
    const text = value.trim();
    if (!text || busy || sending.current) return;
    sending.current = true; // synchronous guard: two quick Enter presses can't submit twice
    try {
      onChange("");
      await onSend(text);
    } finally {
      sending.current = false;
      ref.current?.focus();
    }
  };

  return (
    <div className="px-3 pb-3 pt-2 sm:px-6">
      <div className="mx-auto w-full max-w-3xl">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
          className="flex items-end gap-2 rounded-3xl border border-line-strong bg-panel py-2 pl-5 pr-2 shadow-lg focus-within:border-accent"
        >
          <label htmlFor="prompt" className="sr-only">
            Ask HalluciGuard a question
          </label>
          <textarea
            id="prompt"
            ref={ref}
            rows={1}
            value={value}
            maxLength={6000}
            placeholder="Ask HalluciGuard a question…"
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                void submit();
              }
            }}
            className="max-h-[200px] min-h-[36px] flex-1 resize-none bg-transparent py-2 text-sm leading-6 text-fg outline-none placeholder:text-muted"
          />
          <button
            type="submit"
            disabled={!canSend}
            aria-label={busy ? "Verifying" : "Send"}
            className={cn("btn-accent grid size-9 shrink-0 place-items-center rounded-full transition disabled:opacity-40")}
          >
            {busy ? <Loader2 className="size-4 animate-spin" /> : <ArrowUp className="size-5" strokeWidth={2.5} />}
          </button>
        </form>
        <p className="mt-2 text-center text-[11px] text-muted">Enter to send · Shift+Enter for a new line · Review verification results critically.</p>
      </div>
    </div>
  );
}
