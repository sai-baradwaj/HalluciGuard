import { PanelLeftClose, PanelLeftOpen, Plus, X } from "lucide-react";
import { Logo, ShieldMark } from "./Logo";
import { ConversationHistory } from "./ConversationHistory";
import { Button } from "@/components/ui/button";
import { useChat } from "@/hooks/useChat";
import { cn } from "@/lib/utils";

interface Props {
  collapsed: boolean;
  onToggleCollapsed(): void;
  mobileOpen: boolean;
  onMobileClose(): void;
}

export function Sidebar({ collapsed, onToggleCollapsed, mobileOpen, onMobileClose }: Props) {
  const { conversations, activeId, newChat, selectConversation, renameConversation, deleteConversation } = useChat();

  const startNew = () => {
    newChat();
    onMobileClose();
  };

  return (
    <>
      {mobileOpen && <div className="anim-fade fixed inset-0 z-30 bg-[#020814]/70 md:hidden" onClick={onMobileClose} aria-hidden />}
      <aside
        aria-label="Conversations and prompts"
        className={cn(
          "z-40 flex flex-col border-r border-line bg-panel/95 backdrop-blur transition-transform md:static md:z-auto md:translate-x-0 md:transition-none",
          "fixed inset-y-0 left-0 w-[290px]",
          mobileOpen ? "translate-x-0" : "-translate-x-full",
          collapsed ? "md:w-16" : "md:w-[280px]",
        )}
      >
        {/* Expanded layout (always used in the mobile drawer) */}
        <div className={cn("flex h-full min-h-0 flex-col p-3", collapsed && "md:hidden")}>
          <div className="flex items-center justify-between px-1 pb-3">
            <Logo />
            <Button variant="ghost" size="icon-sm" onClick={onMobileClose} aria-label="Close sidebar" className="md:hidden">
              <X />
            </Button>
            <Button variant="ghost" size="icon-sm" onClick={onToggleCollapsed} aria-label="Collapse sidebar" className="hidden md:inline-flex">
              <PanelLeftClose />
            </Button>
          </div>

          <Button variant="accent" onClick={startNew} className="h-10 w-full">
            <Plus /> New Chat
          </Button>

          <section className="mt-5 min-h-0">
            <h2 className="mb-1.5 flex items-center justify-between px-2 text-xs font-semibold text-muted">
              Chat history <span className="font-normal">{conversations.length}</span>
            </h2>
            <ConversationHistory
              conversations={conversations}
              activeId={activeId}
              onSelect={(id) => {
                selectConversation(id);
                onMobileClose();
              }}
              onRename={renameConversation}
              onDelete={deleteConversation}
            />
          </section>
        </div>

        {/* Collapsed rail (desktop only) */}
        {collapsed && (
          <div className="hidden h-full flex-col items-center gap-3 py-3 md:flex">
            <ShieldMark className="h-9 w-8" />
            <Button variant="accent" size="icon" onClick={startNew} aria-label="New Chat" title="New Chat">
              <Plus />
            </Button>
            <Button variant="ghost" size="icon" onClick={onToggleCollapsed} aria-label="Expand sidebar" title="Expand sidebar">
              <PanelLeftOpen />
            </Button>
          </div>
        )}
      </aside>
    </>
  );
}
