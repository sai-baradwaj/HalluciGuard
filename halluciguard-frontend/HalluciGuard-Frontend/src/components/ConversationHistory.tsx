import { useEffect, useRef, useState } from "react";
import { MessageSquare, MoreHorizontal, Pencil, Trash2 } from "lucide-react";
import type { Conversation } from "@/types";
import { cn } from "@/lib/utils";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

interface Props {
  conversations: Conversation[];
  activeId: string | null;
  onSelect(id: string): void;
  onRename(id: string, title: string): void;
  onDelete(id: string): void;
}

/** Five rows are visible at once; the rest are reached by scrolling the list. */
const VISIBLE_ROWS = 5;
const ROW_PX = 36;

export function ConversationHistory({ conversations, activeId, onSelect, onRename, onDelete }: Props) {
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<Conversation | null>(null);

  if (!conversations.length) {
    return <p className="px-2 py-3 text-xs leading-relaxed text-muted">No conversations yet. Ask a question to start one.</p>;
  }

  return (
    <>
      <ul
        aria-label="Conversations"
        className="thin-scroll flex flex-col gap-0.5 overflow-y-auto pr-1"
        style={{ maxHeight: VISIBLE_ROWS * ROW_PX + (VISIBLE_ROWS - 1) * 2 }}
      >
        {conversations.map((c) => (
          <li key={c.id} data-testid="conversation-row" className="shrink-0">
            {renamingId === c.id ? (
              <RenameInput
                initial={c.title}
                onDone={(title) => {
                  if (title !== null) onRename(c.id, title);
                  setRenamingId(null);
                }}
              />
            ) : (
              <div
                className={cn(
                  "group flex h-9 items-center rounded-lg border border-transparent pr-1 hover:bg-panel3",
                  c.id === activeId && "border-line-strong bg-panel3 shadow-[inset_2px_0_var(--accent)]",
                )}
              >
                <button
                  type="button"
                  onClick={() => onSelect(c.id)}
                  aria-current={c.id === activeId ? "true" : undefined}
                  title={c.title}
                  className="flex min-w-0 flex-1 items-center gap-2 px-2.5 text-left text-[13px]"
                >
                  <MessageSquare className="size-3.5 shrink-0 text-muted" />
                  <span className="truncate">{c.title}</span>
                </button>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button
                      type="button"
                      aria-label={`Options for ${c.title}`}
                      className="grid size-7 shrink-0 place-items-center rounded-md text-muted opacity-0 hover:bg-panel2 hover:text-fg focus-visible:opacity-100 group-hover:opacity-100 data-[state=open]:opacity-100 [@media(hover:none)]:opacity-100"
                    >
                      <MoreHorizontal className="size-4" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="start">
                    <DropdownMenuItem onSelect={() => setRenamingId(c.id)}>
                      <Pencil /> Rename
                    </DropdownMenuItem>
                    <DropdownMenuItem danger onSelect={() => setDeleting(c)}>
                      <Trash2 /> Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            )}
          </li>
        ))}
      </ul>

      <Dialog open={!!deleting} onOpenChange={(open) => !open && setDeleting(null)}>
        <DialogContent>
          <DialogTitle className="text-base font-semibold">Delete this conversation?</DialogTitle>
          <DialogDescription className="mt-2 text-sm text-muted">
            “{deleting?.title}” and its verification results will be removed from this browser. This can’t be undone.
          </DialogDescription>
          <div className="mt-5 flex justify-end gap-2">
            <DialogClose asChild>
              <Button variant="outline">Cancel</Button>
            </DialogClose>
            <Button
              variant="danger"
              onClick={() => {
                if (deleting) onDelete(deleting.id);
                setDeleting(null);
              }}
            >
              Delete conversation
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

function RenameInput({ initial, onDone }: { initial: string; onDone(title: string | null): void }) {
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    ref.current?.focus();
    ref.current?.select();
  }, []);
  return (
    <input
      ref={ref}
      defaultValue={initial}
      aria-label="Conversation name"
      maxLength={80}
      onBlur={(e) => onDone(e.currentTarget.value)}
      onKeyDown={(e) => {
        if (e.key === "Enter") onDone(e.currentTarget.value);
        if (e.key === "Escape") onDone(null);
      }}
      className="h-9 w-full rounded-lg border border-accent bg-panel px-2.5 text-[13px] outline-none"
    />
  );
}
