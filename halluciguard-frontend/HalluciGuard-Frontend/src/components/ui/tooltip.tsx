import * as React from "react";
import * as T from "@radix-ui/react-tooltip";
import { cn } from "@/lib/utils";

export const TooltipProvider = T.Provider;
export const Tooltip = T.Root;
export const TooltipTrigger = T.Trigger;

export const TooltipContent = React.forwardRef<HTMLDivElement, React.ComponentPropsWithoutRef<typeof T.Content>>(
  ({ className, sideOffset = 8, ...props }, ref) => (
    <T.Portal>
      <T.Content
        ref={ref}
        sideOffset={sideOffset}
        className={cn(
          "anim-fade z-[60] max-w-xs whitespace-pre-wrap break-words rounded-lg border border-line-strong bg-panel3 px-3 py-2 text-xs leading-relaxed text-fg shadow-xl",
          className,
        )}
        {...props}
      />
    </T.Portal>
  ),
);
TooltipContent.displayName = "TooltipContent";
