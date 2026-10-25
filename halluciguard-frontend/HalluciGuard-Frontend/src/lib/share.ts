import { copyText } from "./utils";

export type ShareOutcome = "shared" | "copied" | "failed";

/** Uses the native share sheet where available (mostly mobile), otherwise copies to the clipboard. */
export async function shareText(title: string, text: string): Promise<ShareOutcome> {
  if (typeof navigator !== "undefined" && typeof navigator.share === "function") {
    try {
      await navigator.share({ title, text });
      return "shared";
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return "failed";
      /* fall through to clipboard */
    }
  }
  return (await copyText(text)) ? "copied" : "failed";
}
