// Shared markdown renderer. Wrap in a `.prose-spec` element (which sets the
// font-size and typography) and this fills it with rendered markdown.
// remark-gfm is what makes tables / strikethrough / task-lists render — the
// skeleton spec.md leads with a section table, so it's not optional here.
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function Markdown({ children }: { children: string }) {
  return <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>;
}
