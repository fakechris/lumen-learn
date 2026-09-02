/**
 * Hand-drawn annotations (circle / highlight) anchored to a snippet inside a
 * rendered card.
 *
 * Locating the snippet:
 *   1. KaTeX formulas keep their TeX source in <annotation>; match there and
 *      use the rendered formula's box.
 *   2. Otherwise search the card's text nodes (whitespace-insensitive) and
 *      measure a DOM Range.
 *   3. Fall back to the whole content box.
 */

const norm = (s) => String(s).replace(/\s+/g, "");

function unionRects(rects) {
  let l = Infinity, t = Infinity, r = -Infinity, b = -Infinity;
  for (const rc of rects) {
    if (rc.width === 0 && rc.height === 0) continue;
    l = Math.min(l, rc.left); t = Math.min(t, rc.top);
    r = Math.max(r, rc.right); b = Math.max(b, rc.bottom);
  }
  if (!isFinite(l)) return null;
  return { left: l, top: t, width: r - l, height: b - t };
}

function findInKatex(root, snippet) {
  const target = norm(snippet);
  for (const k of root.querySelectorAll(".katex")) {
    const ann = k.querySelector("annotation");
    if (ann && norm(ann.textContent).includes(target)) {
      // Display math is a full-width block; measure the glyph boxes instead.
      const bases = k.querySelectorAll(".katex-html > .base");
      return unionRects(Array.from(bases, (b) => b.getBoundingClientRect())) || k.getBoundingClientRect();
    }
  }
  return null;
}

function findInText(root, snippet) {
  const target = norm(snippet);
  if (!target) return null;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (n) => (n.parentElement?.closest(".katex-mathml") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
  });
  const chars = []; // { node, offset }
  let text = "";
  let n;
  while ((n = walker.nextNode())) {
    const s = n.textContent;
    for (let i = 0; i < s.length; i++) {
      if (/\s/.test(s[i])) continue;
      chars.push({ node: n, offset: i });
      text += s[i];
    }
  }
  const idx = text.indexOf(target);
  if (idx < 0) return null;
  const start = chars[idx], end = chars[idx + target.length - 1];
  const range = document.createRange();
  range.setStart(start.node, start.offset);
  range.setEnd(end.node, end.offset + 1);
  return unionRects(Array.from(range.getClientRects()));
}

export function locateSnippet(cardEl, snippet) {
  const content = cardEl.querySelector(".wb-card-content") || cardEl;
  const rect = findInKatex(content, snippet) || findInText(content, snippet) || content.getBoundingClientRect();
  const base = cardEl.getBoundingClientRect();
  return { x: rect.left - base.left, y: rect.top - base.top, w: rect.width, h: rect.height };
}

function wobblyEllipse(cx, cy, rx, ry, seed) {
  // 4 cubic segments with slight irregularity so it reads as hand-drawn.
  const k = 0.5523;
  const j = (i) => 1 + 0.06 * Math.sin(seed * 7 + i * 2.3);
  const p = (a, b) => `${a.toFixed(1)} ${b.toFixed(1)}`;
  const x0 = cx + rx * j(0), y0 = cy;
  return [
    `M ${p(x0, y0)}`,
    `C ${p(cx + rx, cy + ry * k * j(1))} ${p(cx + rx * k, cy + ry * j(1))} ${p(cx, cy + ry * j(2))}`,
    `C ${p(cx - rx * k * j(2), cy + ry)} ${p(cx - rx * j(3), cy + ry * k)} ${p(cx - rx * j(3), cy)}`,
    `C ${p(cx - rx, cy - ry * k * j(4))} ${p(cx - rx * k, cy - ry * j(4))} ${p(cx, cy - ry * j(5))}`,
    `C ${p(cx + rx * k * j(5), cy - ry)} ${p(cx + rx * j(6) + 6, cy - ry * k)} ${p(x0 + 4, y0 + 3)}`,
  ].join(" ");
}

/**
 * Draw a decoration on `cardEl`. Returns the SVG element. The stroke animates
 * over `durationMs` starting immediately.
 */
export function drawDecoration(cardEl, { kind, snippet, color }, durationMs = 800) {
  const box = locateSnippet(cardEl, snippet);
  let svg = cardEl.querySelector("svg.wb-annotations");
  if (!svg) {
    svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "wb-annotations");
    cardEl.appendChild(svg);
  }
  svg.setAttribute("width", cardEl.offsetWidth);
  svg.setAttribute("height", cardEl.offsetHeight);

  if (kind === "highlight") {
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", box.x - 3); rect.setAttribute("y", box.y - 2);
    rect.setAttribute("width", box.w + 6); rect.setAttribute("height", box.h + 4);
    rect.setAttribute("rx", 4);
    rect.setAttribute("fill", color || "rgba(255, 213, 79, 0.55)");
    rect.style.transformOrigin = `${box.x}px ${box.y}px`;
    rect.style.transform = "scaleX(0)";
    rect.style.transition = `transform ${durationMs}ms ease-out`;
    svg.appendChild(rect);
    requestAnimationFrame(() => { rect.style.transform = "scaleX(1)"; });
    return svg;
  }

  const padX = Math.max(10, box.w * 0.12), padY = Math.max(8, box.h * 0.35);
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", wobblyEllipse(box.x + box.w / 2, box.y + box.h / 2, box.w / 2 + padX, box.h / 2 + padY, box.x + box.y));
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", color || "#e05656");
  path.setAttribute("stroke-width", "2.6");
  path.setAttribute("stroke-linecap", "round");
  svg.appendChild(path);
  const len = path.getTotalLength();
  path.style.strokeDasharray = `${len}`;
  path.style.strokeDashoffset = `${len}`;
  path.style.transition = `stroke-dashoffset ${durationMs}ms cubic-bezier(.4,0,.2,1)`;
  requestAnimationFrame(() => { path.style.strokeDashoffset = "0"; });
  return svg;
}
