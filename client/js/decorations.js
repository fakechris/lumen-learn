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
  if (kind === "spotlight") return drawSpotlight(cardEl, snippet);
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

/** B1: dim everything except the snippet — SVG mask cutout over the viewport.
 *  No backdrop-filter (it breaks inside transformed ancestors). Auto-fades. */
export function drawSpotlight(cardEl, snippet, holdMs = 5000) {
  const box = locateSnippet(cardEl, snippet);
  const cardRect = cardEl.getBoundingClientRect();
  const vp = { x: cardRect.left + box.x, y: cardRect.top + box.y, w: box.w, h: box.h };
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.style.cssText = "position:fixed;inset:0;width:100vw;height:100vh;z-index:40;pointer-events:none;transition:opacity .5s ease";
  const mid = "hk-spot-mask";
  svg.innerHTML = `
    <defs>
      <mask id="${mid}">
        <rect x="0" y="0" width="100%" height="100%" fill="white"/>
        <rect class="hole" x="${vp.x - 14}" y="${vp.y - 12}" width="${vp.w + 28}" height="${vp.h + 24}" rx="14"
              fill="black" style="transition: all .45s cubic-bezier(.2,.8,.2,1)"/>
      </mask>
    </defs>
    <rect x="0" y="0" width="100%" height="100%" fill="rgba(43,43,43,.36)" mask="url(#${mid})"/>
    <rect class="ring" x="${vp.x - 6}" y="${vp.y - 4}" width="${vp.w + 12}" height="${vp.h + 8}" rx="12"
          fill="none" stroke="#e05656" stroke-width="2.2" stroke-dasharray="6 5" style="transition: all .45s cubic-bezier(.2,.8,.2,1)"/>`;
  document.body.appendChild(svg);
  // settle the hole tighter right after paint
  requestAnimationFrame(() => {
    const hole = svg.querySelector(".hole"), ring = svg.querySelector(".ring");
    if (hole) { hole.setAttribute("x", vp.x - 2); hole.setAttribute("y", vp.y); hole.setAttribute("width", vp.w + 4); hole.setAttribute("height", vp.h + 2); }
    if (ring) { ring.setAttribute("x", vp.x - 4); ring.setAttribute("y", vp.y - 2); ring.setAttribute("width", vp.w + 8); ring.setAttribute("height", vp.h + 4); }
  });
  const fade = () => { svg.style.opacity = "0"; setTimeout(() => svg.remove(), 550); };
  const timer = setTimeout(fade, holdMs);
  svg.addEventListener("click", () => { clearTimeout(timer); fade(); });
  return svg;
}
