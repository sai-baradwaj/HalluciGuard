import { useEffect, useRef, useState } from "react";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

export interface PromptLine {
  /** Id of the user message (the prompt). */
  id: string;
  content: string;
  /** Id of the element to scroll to: the assistant response for this prompt (or the prompt itself if none yet). */
  targetId: string;
}

interface Props {
  /** Prompts of the ACTIVE conversation only, oldest first, so the rail reads top-to-bottom like the chat. */
  prompts: PromptLine[];
  onNavigate(targetId: string): void;
}

const VISIBLE_ROWS = 5;
const ROW_PX = 24;
const PREVIEW_CHARS = 90;

const preview = (text: string) => {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length > PREVIEW_CHARS ? `${clean.slice(0, PREVIEW_CHARS).trimEnd()}…` : clean;
};

/**
 * Slim rail on the right edge of the chat, vertically centred. Only short lines are visible;
 * hovering a line reveals a few words of the prompt and clicking it jumps to that prompt's response.
 * Five lines show at once (the newest); older ones are reached by scrolling the rail.
 */
export function PromptHistory({ prompts, onNavigate }: Props) {
  const touch = useMediaQuery("(hover: none)");
  const [openId, setOpenId] = useState<string | null>(null);
  const listRef = useRef<HTMLUListElement>(null);

  // Start at the newest prompts (bottom); older ones are above.
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [prompts.length]);

  // Switching conversations must not leave a stale popover open.
  useEffect(() => setOpenId(null), [prompts[0]?.id]);

  if (!prompts.length) return null;

  return (
    <nav aria-label="Prompt history" className="pointer-events-none absolute right-1.5 top-1/2 z-10 -translate-y-1/2 sm:right-3">
      <ul
        ref={listRef}
        className="thin-scroll pointer-events-auto flex flex-col overflow-y-auto overflow-x-visible rounded-full py-1"
        style={{ maxHeight: VISIBLE_ROWS * ROW_PX + 8, scrollbarWidth: "none" }}
      >
        {prompts.map((p) => {
          const open = openId === p.id;
          const line = (
            <button
              type="button"
              aria-label={p.content}
              aria-expanded={touch ? open : undefined}
              onClick={() => (touch ? setOpenId(open ? null : p.id) : onNavigate(p.targetId))}
              className="group flex w-9 items-center justify-end pr-1.5 outline-none"
              style={{ height: ROW_PX }}
            >
              <span className={cn("h-0.5 rounded-full transition-all", open ? "w-6 bg-accent" : "w-3.5 bg-line-strong group-hover:w-6 group-hover:bg-accent group-focus-visible:w-6 group-focus-visible:bg-accent")} />
            </button>
          );
          return (
            <li key={p.id} data-testid="prompt-row" className="relative shrink-0">
              {touch ? (
                <>
                  {line}
                  {open && (
                    <div className="anim-fade absolute right-full top-1/2 mr-2 w-56 -translate-y-1/2 rounded-lg border border-line-strong bg-panel3 p-2.5 text-xs leading-relaxed shadow-xl">
                      <button
                        type="button"
                        onClick={() => {
                          onNavigate(p.targetId);
                          setOpenId(null);
                        }}
                        className="block w-full break-words text-left text-fg"
                      >
                        {preview(p.content)}
                      </button>
                      <button type="button" aria-label="Close preview" onClick={() => setOpenId(null)} className="mt-1.5 text-[11px] text-muted underline">
                        Close
                      </button>
                    </div>
                  )}
                </>
              ) : (
                <Tooltip>
                  <TooltipTrigger asChild>{line}</TooltipTrigger>
                  <TooltipContent side="left" sideOffset={6} className="max-w-[16rem]">
                    {preview(p.content)}
                  </TooltipContent>
                </Tooltip>
              )}
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
