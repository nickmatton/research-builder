// Tiny wrapper around highlight.js. We import only the `common` bundle
// (35 languages: js/ts/py/rs/go/json/yaml/md/bash/sql/css/html/...) — the
// `core` bundle is too narrow, the full bundle ships 190 langs we don't
// need. File extension → language id; auto-detect as a fallback.

import hljs from "highlight.js/lib/common";

const EXT_LANG: Record<string, string> = {
  ts: "typescript",
  tsx: "typescript",
  js: "javascript",
  jsx: "javascript",
  mjs: "javascript",
  cjs: "javascript",
  py: "python",
  rs: "rust",
  go: "go",
  java: "java",
  c: "c",
  h: "c",
  cpp: "cpp",
  cc: "cpp",
  hpp: "cpp",
  cs: "csharp",
  rb: "ruby",
  php: "php",
  swift: "swift",
  kt: "kotlin",
  scala: "scala",
  sh: "bash",
  bash: "bash",
  zsh: "bash",
  fish: "bash",
  json: "json",
  yml: "yaml",
  yaml: "yaml",
  toml: "ini",
  ini: "ini",
  md: "markdown",
  mdx: "markdown",
  html: "xml",
  htm: "xml",
  xml: "xml",
  css: "css",
  scss: "scss",
  less: "less",
  sql: "sql",
  dockerfile: "dockerfile",
  makefile: "makefile",
  txt: "plaintext",
  log: "plaintext",
};

const FILENAME_LANG: Record<string, string> = {
  dockerfile: "dockerfile",
  makefile: "makefile",
  cmakelists: "cmake",
  rakefile: "ruby",
  gemfile: "ruby",
};

export function detectLanguage(path: string): string | null {
  const name = path.toLowerCase().split("/").pop() ?? "";
  if (FILENAME_LANG[name]) return FILENAME_LANG[name];
  const dot = name.lastIndexOf(".");
  if (dot < 0) return null;
  return EXT_LANG[name.slice(dot + 1)] ?? null;
}

export function highlight(code: string, path: string): { html: string; lang: string } {
  // Long files in the file viewer can blow past the JS budget if we
  // tokenize the whole thing synchronously; cap at 200KB. That's
  // comfortably above any reasonable source file.
  const truncated = code.length > 200_000;
  const src = truncated ? code.slice(0, 200_000) : code;

  const lang = detectLanguage(path);
  try {
    if (lang && lang !== "plaintext" && hljs.getLanguage(lang)) {
      const result = hljs.highlight(src, { language: lang, ignoreIllegals: true });
      return {
        html: result.value + (truncated ? "\n\n[...truncated]" : ""),
        lang: lang,
      };
    }
    // Auto-detect for unknown extensions (e.g. README, scripts without .sh).
    const result = hljs.highlightAuto(src);
    return {
      html: result.value + (truncated ? "\n\n[...truncated]" : ""),
      lang: result.language ?? "plaintext",
    };
  } catch {
    // Fall back to escaped raw text if highlighting throws.
    return { html: escapeHtml(src), lang: "plaintext" };
  }
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
