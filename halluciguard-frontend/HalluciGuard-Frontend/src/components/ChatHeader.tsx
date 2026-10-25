import { Menu, Moon, Share2, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ShieldMark } from "./Logo";
import { DemoBadge } from "./StatusBadge";
import { useChat } from "@/hooks/useChat";
import { useToast } from "./Toast";
import { shareText } from "@/lib/share";
import { cn } from "@/lib/utils";
import type { Conversation, DataMode } from "@/types";

function transcript(c: Conversation): string {
  const lines = [`# ${c.title}`, ""];
  for (const m of c.messages) {
    if (m.role === "user") lines.push(`**You:** ${m.content}`, "");
    else if (m.state === "done" && m.content) lines.push(`**HalluciGuard (verified answer):**`, m.content, "");
  }
  return lines.join("\n").trim();
}

interface Props {
  theme: "dark" | "light";
  onToggleTheme(): void;
  onOpenSidebar(): void;
}

export function ChatHeader({ theme, onToggleTheme, onOpenSidebar }: Props) {
  const { active, mode, setMode } = useChat();
  const toast = useToast();
  const canShare = !!active?.messages.some((m) => m.role === "assistant" && m.state === "done");

  const share = async () => {
    if (!active) return;
    const outcome = await shareText(active.title, transcript(active));
    if (outcome === "copied") toast("Conversation copied to the clipboard as Markdown.");
    if (outcome === "failed") toast("Sharing isn’t available here.", "error");
  };

  return (
    <header className="flex h-14 shrink-0 items-center gap-2 border-b border-line px-3 sm:px-5">
      <Button variant="ghost" size="icon" className="md:hidden" onClick={onOpenSidebar} aria-label="Open sidebar">
        <Menu />
      </Button>
      <div className="flex min-w-0 items-center gap-2">
        <ShieldMark className="h-6 w-5 shrink-0 md:hidden" />
        <h1 className="truncate text-sm font-semibold">{active?.title ?? "New conversation"}</h1>
        {mode === "demo" && <DemoBadge className="hidden sm:inline-flex" />}
      </div>

      <div className="ml-auto flex items-center gap-1.5">
        <ModeSwitch mode={mode} onChange={setMode} />
        <Button variant="outline" size="sm" onClick={share} disabled={!canShare} aria-label="Share conversation" className="hidden sm:inline-flex">
          <Share2 /> Share
        </Button>
        <Button variant="ghost" size="icon" onClick={share} disabled={!canShare} aria-label="Share conversation" className="sm:hidden">
          <Share2 />
        </Button>
        <Button variant="ghost" size="icon" onClick={onToggleTheme} aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}>
          {theme === "dark" ? <Sun /> : <Moon />}
        </Button>
      </div>
    </header>
  );
}

function ModeSwitch({ mode, onChange }: { mode: DataMode; onChange(m: DataMode): void }) {
  const opt = (value: DataMode, label: string) => (
    <button
      type="button"
      role="radio"
      aria-checked={mode === value}
      onClick={() => onChange(value)}
      className={cn("rounded-md px-2.5 py-1 text-xs font-medium transition-colors", mode === value ? "bg-panel3 text-fg shadow-[inset_0_0_0_1px_var(--line-strong)]" : "text-muted hover:text-fg")}
    >
      {label}
    </button>
  );
  return (
    <div role="radiogroup" aria-label="Data source" className="flex rounded-lg border border-line bg-panel2 p-0.5">
      {opt("live", "Live backend")}
      {opt("demo", "Demo")}
    </div>
  );
}
