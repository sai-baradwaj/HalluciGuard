import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-colors disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        accent: "btn-accent font-semibold shadow-[0_6px_22px_#00c9e51a] hover:brightness-110",
        outline: "border border-line bg-panel2 text-fg hover:bg-panel3 hover:border-line-strong",
        ghost: "text-muted hover:bg-panel3 hover:text-fg",
        verify: "border border-accent/40 bg-accent/10 text-accent-text hover:bg-accent/20",
        danger: "bg-danger text-white hover:brightness-110",
      },
      size: { sm: "h-8 px-3 text-xs", md: "h-9 px-4", icon: "size-9", "icon-sm": "size-8" },
    },
    defaultVariants: { variant: "outline", size: "md" },
  },
);

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, asChild, ...props }, ref) => {
  const Comp = asChild ? Slot : "button";
  return <Comp ref={ref} className={cn(buttonVariants({ variant, size }), className)} {...props} />;
});
Button.displayName = "Button";
