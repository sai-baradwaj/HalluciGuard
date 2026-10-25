import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from "react";
import { cn } from "@/lib/utils";

type Tone = "info" | "error";
const Ctx = createContext<(message: string, tone?: Tone) => void>(() => {});
export const useToast = () => useContext(Ctx);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toast, setToast] = useState<{ message: string; tone: Tone } | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const notify = useCallback((message: string, tone: Tone = "info") => {
    setToast({ message, tone });
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setToast(null), tone === "error" ? 7000 : 2600);
  }, []);
  return (
    <Ctx.Provider value={notify}>
      {children}
      {toast && (
        <div
          role="status"
          className={cn(
            "anim-fade fixed bottom-24 left-1/2 z-[70] w-[min(560px,calc(100%-2rem))] -translate-x-1/2 rounded-xl border px-4 py-3 text-sm shadow-2xl",
            toast.tone === "error" ? "border-danger/50 bg-[#351724] text-[#ffe0e5]" : "border-line-strong bg-panel3 text-fg",
          )}
        >
          {toast.message}
        </div>
      )}
    </Ctx.Provider>
  );
}
