import { memo, useMemo } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { visit, SKIP } from "unist-util-visit";
import type { Root, Text, Parent } from "mdast";
import { isHttpUrl } from "@/lib/utils";

export interface Highlight {
  text: string;
  status: string; // becomes data-status on <mark>, styled in index.css
}

/**
 * remark plugin: wraps verbatim occurrences of each highlight inside a single text node in <mark>.
 * Works on the AST (no raw HTML), so it is safe for untrusted model output.
 */
function remarkHighlight(highlights: Highlight[]) {
  const usable = highlights.filter((h) => h.text.trim().length >= 8);
  return () => (tree: Root) => {
    if (!usable.length) return;
    visit(tree, "text", (node: Text, index, parent: Parent | undefined) => {
      if (!parent || index === undefined) return;
      const lower = node.value.toLowerCase();
      const hits: Array<{ start: number; end: number; status: string }> = [];
      for (const h of usable) {
        const needle = h.text.trim().toLowerCase();
        const start = lower.indexOf(needle);
        if (start >= 0 && !hits.some((x) => start < x.end && start + needle.length > x.start)) hits.push({ start, end: start + needle.length, status: h.status });
      }
      if (!hits.length) return;
      hits.sort((a, b) => a.start - b.start);
      const parts: Array<Text | Parent> = [];
      let cursor = 0;
      for (const hit of hits) {
        if (hit.start > cursor) parts.push({ type: "text", value: node.value.slice(cursor, hit.start) });
        parts.push({
          type: "highlight",
          data: { hName: "mark", hProperties: { "data-status": hit.status } },
          children: [{ type: "text", value: node.value.slice(hit.start, hit.end) }],
        } as unknown as Parent);
        cursor = hit.end;
      }
      if (cursor < node.value.length) parts.push({ type: "text", value: node.value.slice(cursor) });
      parent.children.splice(index, 1, ...(parts as Parent["children"]));
      return [SKIP, index + parts.length];
    });
  };
}

const components: Components = {
  a: ({ href, children }) =>
    isHttpUrl(href) ? (
      <a href={href} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    ) : (
      <span>{children}</span>
    ),
  table: ({ children }) => (
    <div className="table-wrap">
      <table>{children}</table>
    </div>
  ),
};

interface Props {
  children: string;
  highlights?: Highlight[];
  className?: string;
}

export const Markdown = memo(function Markdown({ children, highlights, className }: Props) {
  const plugins = useMemo(() => [remarkGfm, remarkHighlight(highlights ?? [])], [highlights]);
  return (
    <div className={`md ${className ?? ""}`}>
      <ReactMarkdown remarkPlugins={plugins} components={components}>
        {children}
      </ReactMarkdown>
    </div>
  );
});
