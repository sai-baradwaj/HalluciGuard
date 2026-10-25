import { useState } from "react";
import { ChatProvider } from "@/hooks/useChat";
import type { HalluciGuardApi } from "@/services/api";
import type { Conversation } from "@/types";
import { ToastProvider } from "@/components/Toast";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Sidebar } from "@/components/Sidebar";
import { ChatHeader } from "@/components/ChatHeader";
import { ChatArea } from "@/components/ChatArea";
import { VerificationModal } from "@/components/verification/VerificationModal";
import { useTheme } from "@/hooks/useTheme";

function Shell() {
  const { theme, toggle } = useTheme();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="flex h-dvh w-full overflow-hidden">
      <Sidebar
        collapsed={collapsed}
        onToggleCollapsed={() => setCollapsed((c) => !c)}
        mobileOpen={mobileOpen}
        onMobileClose={() => setMobileOpen(false)}
      />
      <main className="flex min-w-0 flex-1 flex-col">
        <ChatHeader theme={theme} onToggleTheme={toggle} onOpenSidebar={() => setMobileOpen(true)} />
        <ChatArea />
      </main>
      <VerificationModal />
    </div>
  );
}

export interface AppProps {
  api?: HalluciGuardApi;
  initialConversations?: Conversation[];
  persist?: boolean;
}

export default function App(props: AppProps) {
  return (
    <ToastProvider>
      <TooltipProvider delayDuration={120} skipDelayDuration={300}>
        <ChatProvider {...props}>
          <Shell />
        </ChatProvider>
      </TooltipProvider>
    </ToastProvider>
  );
}
