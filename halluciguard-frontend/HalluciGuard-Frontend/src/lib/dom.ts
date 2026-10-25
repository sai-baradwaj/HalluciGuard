/** Scroll to a message in the chat and briefly ring it so the target is obvious. */
export function scrollToMessage(messageId: string): void {
  const el = document.getElementById(`message-${messageId}`);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("ring-2", "ring-accent");
  window.setTimeout(() => el.classList.remove("ring-2", "ring-accent"), 1600);
}
