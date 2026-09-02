/**
 * Markdown + KaTeX rendering for board cards.
 * `marked`, DOMPurify and KaTeX's auto-render are loaded from CDN in index.html.
 * Generated markdown is sanitized before insertion.
 */
const KATEX_DELIMITERS = [
  { left: "$$", right: "$$", display: true },
  { left: "\\[", right: "\\]", display: true },
  { left: "$", right: "$", display: false },
  { left: "\\(", right: "\\)", display: false },
];

export function renderMarkdownInto(el, markdown) {
  const raw = window.marked ? window.marked.parse(markdown || "") : escapeHtml(markdown || "");
  el.innerHTML = window.DOMPurify ? window.DOMPurify.sanitize(raw) : raw;
  if (window.renderMathInElement) {
    window.renderMathInElement(el, { delimiters: KATEX_DELIMITERS, throwOnError: false, strict: "ignore" });
  }
}

export function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
