import { useCallback, useEffect, useState } from "react";
import { loadTheme, saveTheme } from "@/services/storage";

export function useTheme() {
  const [theme, setTheme] = useState<"dark" | "light">(() => loadTheme());
  useEffect(() => {
    document.documentElement.classList.toggle("light", theme === "light");
    saveTheme(theme);
  }, [theme]);
  const toggle = useCallback(() => setTheme((t) => (t === "dark" ? "light" : "dark")), []);
  return { theme, toggle };
}
