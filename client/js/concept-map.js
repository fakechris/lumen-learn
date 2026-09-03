/**
 * Concept map (E3): the course's concepts as a hand-drawn graph on paper.
 *
 * Layout: prerequisite depth decides the column (a learning path reads left to
 * right); inside a column nodes are grouped by unit; a short relaxation keeps
 * circles and labels apart. Rendering: SVG with wobbly ink circles, curved
 * edges with hand-drawn arrowheads, handwriting labels. Node fill = mastery
 * (composite of the sessions that teach it). Click = open the session.
 */
const NS = "http://www.w3.org/2000/svg";
const UNIT_INK = ["#7c3aed", "#2563eb", "#15803d", "#d97706", "#be185d", "#0f766e"];
const COL_W = 215, ROW_H = 138, PAD_X = 90, PAD_Y = 70;

const esc = (t) => String(t == null ? "" : t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function el(tag, attrs = {}, text) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  return n;
}

function seeded(seed) {
  let h = 2166136261;
  for (const c of String(seed)) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); }
  return () => { h = Math.imul(h ^ (h >>> 15), 2246822507); h = Math.imul(h ^ (h >>> 13), 3266489909); return ((h >>>= 0) % 10000) / 10000; };
}

/** a circle drawn by hand: 10 wobbly anchor points joined by smooth curves, closed with a slight overlap */
export function inkCircle(cx, cy, r, seed) {
  const rnd = seeded(seed), n = 10, pts = [];
  for (let i = 0; i <= n + 1; i++) {
    const a = (i / n) * Math.PI * 2 - Math.PI / 2 + (i === n + 1 ? 0.35 : 0);
    const rr = r * (1 + (rnd() - 0.5) * 0.09);
    pts.push([cx + Math.cos(a) * rr, cy + Math.sin(a) * rr]);
  }
  let d = `M ${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const mx = (pts[i][0] + pts[i + 1][0]) / 2, my = (pts[i][1] + pts[i + 1][1]) / 2;
    d += ` Q ${pts[i][0].toFixed(1)} ${pts[i][1].toFixed(1)} ${mx.toFixed(1)} ${my.toFixed(1)}`;
  }
  return d;
}

function wrap(label, max = 7) {
  if (label.length <= max) return [label];
  const cut = label.search(/[：:，,、/ ]/);
  if (cut > 1 && cut < label.length - 1 && cut <= max + 2) return [label.slice(0, cut), label.slice(cut + 1)].map((s) => s.trim());
  return [label.slice(0, max), label.slice(max, max * 2 - 1) + (label.length > max * 2 - 1 ? "…" : "")];
}

/** prerequisite depth via longest path; back edges (cycles) are ignored in insertion order */
export function layout(map) {
  const nodes = map.nodes.map((n) => ({ ...n }));
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  // a pair connected both ways keeps one edge (prerequisite wins), so the map stays a learning path
  const pairSeen = new Map();
  const edgesAll = [];
  for (const e of [...map.edges].sort((a, b) => (a.type === "prerequisite" ? 0 : 1) - (b.type === "prerequisite" ? 0 : 1))) {
    if (!byId[e.source] || !byId[e.target]) continue;
    const key = [e.source, e.target].sort().join("|");
    if (pairSeen.has(key)) continue;
    pairSeen.set(key, e);
    edgesAll.push(e);
  }
  const dirEdges = edgesAll.filter((e) => e.type !== "contrast");
  const indeg = Object.fromEntries(nodes.map((n) => [n.id, 0]));
  const out = Object.fromEntries(nodes.map((n) => [n.id, []]));
  for (const e of dirEdges) { out[e.source].push(e.target); indeg[e.target]++; }
  // longest-path layering (Kahn); when a cycle blocks progress, cut it at the node with the fewest
  // remaining incoming edges so every node still gets a column
  const depth = Object.fromEntries(nodes.map((n) => [n.id, 0]));
  const remaining = new Set(nodes.map((n) => n.id));
  const queue = nodes.filter((n) => indeg[n.id] === 0).map((n) => n.id);
  while (remaining.size) {
    if (!queue.length) {
      const cut = [...remaining].sort((a, b) => indeg[a] - indeg[b])[0];
      indeg[cut] = 0;
      queue.push(cut);
    }
    const id = queue.shift();
    if (!remaining.has(id)) continue;
    remaining.delete(id);
    for (const t of out[id]) {
      if (!remaining.has(t)) continue;
      depth[t] = Math.max(depth[t], depth[id] + 1);
      if (--indeg[t] <= 0) queue.push(t);
    }
  }
  const MAX_ROWS = 6;
  const units = [...new Set(nodes.map((n) => n.unit || ""))];
  // multi-unit courses read unit by unit (left → right); inside a unit prerequisites go top → bottom.
  // single-unit courses use prerequisite depth as the column.
  const cols = {};
  const unitMajor = units.length > 1;
  for (const n of nodes) (cols[unitMajor ? units.indexOf(n.unit || "") : depth[n.id]] ||= []).push(n);
  // a crowded column wraps into sub-columns of MAX_ROWS
  const placed = [];
  let x = PAD_X;
  Object.keys(cols).map(Number).sort((a, b) => a - b).forEach((d) => {
    const list = cols[d];
    list.sort((a, b) => (unitMajor ? depth[a.id] - depth[b.id] : units.indexOf(a.unit || "") - units.indexOf(b.unit || "")) || b.weight - a.weight);
    for (let start = 0; start < list.length; start += MAX_ROWS) {
      const chunk = list.slice(start, start + MAX_ROWS);
      chunk.forEach((n, i) => { n.x = x + (i % 2 ? 22 : 0); n.row = i; n.rows = chunk.length; placed.push(n); });
      x += COL_W;
    }
  });
  const maxRows = Math.min(MAX_ROWS, Math.max(...Object.values(cols).map((c) => c.length)));
  const H = PAD_Y * 2 + Math.max(1, maxRows - 1) * ROW_H;
  for (const n of placed) {
    n.y = H / 2 - ((n.rows - 1) * ROW_H) / 2 + n.row * ROW_H;
    n.r = 26 + Math.min(3, n.weight || 1) * 5;
    n.unitIndex = Math.max(0, units.indexOf(n.unit || ""));
  }
  // relaxation: keep circles + labels apart (label = ~36px below the circle)
  for (let it = 0; it < 120; it++) {
    for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      const dx = b.x - a.x, dy = b.y - a.y, dist = Math.hypot(dx, dy) || 0.01;
      const min = a.r + b.r + 44;
      if (dist < min) {
        const push = (min - dist) / 2 * 0.5, ux = dx / dist, uy = dy / dist;
        a.x -= ux * push * 0.3; a.y -= uy * push; b.x += ux * push * 0.3; b.y += uy * push;
      }
    }
  }
  const W = x - COL_W + PAD_X + 60;
  const minY = Math.min(...nodes.map((n) => n.y - n.r - 20)), maxY = Math.max(...nodes.map((n) => n.y + n.r + 60));
  for (const n of nodes) n.y += PAD_Y - minY;
  return { nodes, edges: edgesAll, W, H: maxY - minY + PAD_Y * 2, units };
}

function masteryFill(c) {
  if (c == null) return "#fbfaf6";
  if (c < 25) return "#e8eef9";
  if (c < 50) return "#ece6fb";
  if (c < 75) return "#e3f4ea";
  return "#c7ecd6";
}

/**
 * @param container  element to render into
 * @param map        ConceptMap JSON from the server
 * @param opts       { mastery: {sessionId -> {composite}}, sessionTitle: id -> title, onOpen(node) }
 */
export function renderConceptMap(container, map, opts = {}) {
  const { nodes, edges, W, H, units } = layout(map);
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const mastery = opts.mastery || {};
  container.innerHTML = "";
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "concept-map" });
  svg.style.width = `${W}px`;
  svg.style.height = `${H}px`;
  const defs = el("defs");
  defs.innerHTML = `<marker id="cm-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
      <path d="M 1 1.5 L 9 5 L 1 8.5 L 3.2 5 Z" fill="#6b6257"/></marker>`;
  svg.appendChild(defs);

  // edges under the nodes
  const g = el("g");
  for (const e of edges) {
    const a = byId[e.source], b = byId[e.target];
    const dx = b.x - a.x, dy = b.y - a.y, dist = Math.hypot(dx, dy) || 1;
    const ux = dx / dist, uy = dy / dist;
    const x1 = a.x + ux * (a.r + 3), y1 = a.y + uy * (a.r + 3);
    const x2 = b.x - ux * (b.r + 5), y2 = b.y - uy * (b.r + 5);
    const bend = e.type === "contrast" ? 0 : 0.14;
    const mx = (x1 + x2) / 2 - uy * dist * bend, my = (y1 + y2) / 2 + ux * dist * bend;
    const path = el("path", { d: `M ${x1} ${y1} Q ${mx} ${my} ${x2} ${y2}`, fill: "none", "stroke-linecap": "round" });
    if (e.type === "contrast") { path.setAttribute("stroke", "#b9b3a6"); path.setAttribute("stroke-dasharray", "2 6"); path.setAttribute("stroke-width", "1.8"); }
    else if (e.type === "part_of" || e.type === "kind_of") { path.setAttribute("stroke", "#9a938a"); path.setAttribute("stroke-dasharray", "7 5"); path.setAttribute("stroke-width", "1.6"); path.setAttribute("marker-end", "url(#cm-arrow)"); }
    else { path.setAttribute("stroke", "#6b6257"); path.setAttribute("stroke-width", e.type === "prerequisite" ? "2" : "1.6"); path.setAttribute("marker-end", "url(#cm-arrow)"); }
    path.dataset.from = e.source; path.dataset.to = e.target;
    g.appendChild(path);
    if (e.relation) {
      const t = el("text", { x: mx, y: my - 6, "text-anchor": "middle", class: "cm-rel" }, e.relation);
      t.dataset.from = e.source; t.dataset.to = e.target;
      g.appendChild(t);
    }
  }
  svg.appendChild(g);

  // nodes
  const tip = document.createElement("div");
  tip.className = "cm-tip";
  container.appendChild(svg);
  container.appendChild(tip);
  for (const n of nodes) {
    const comps = (n.sessions || []).map((s) => mastery[s] && mastery[s].composite).filter((c) => c != null);
    const comp = comps.length ? comps.reduce((a, b) => a + b, 0) / comps.length : null;
    const node = el("g", { class: "cm-node" });
    node.appendChild(el("path", { d: inkCircle(n.x, n.y, n.r, n.id), fill: masteryFill(comp), stroke: UNIT_INK[n.unitIndex % UNIT_INK.length], "stroke-width": "2.2", "stroke-linejoin": "round" }));
    if (comp != null) node.appendChild(el("text", { x: n.x, y: n.y + 5, "text-anchor": "middle", class: "cm-score" }, Math.round(comp)));
    const lines = wrap(n.label);
    lines.forEach((ln, i) => node.appendChild(el("text", { x: n.x, y: n.y + n.r + 18 + i * 19, "text-anchor": "middle", class: "cm-label" + (i ? " more" : "") }, ln)));
    node.addEventListener("mouseenter", () => {
      const sess = (n.sessions || []).map((s) => `<li>${esc((opts.sessionTitle && opts.sessionTitle(s)) || s)}</li>`).join("");
      tip.innerHTML = `<b>${esc(n.label)}</b><p>${esc(n.summary)}</p>${sess ? `<ul>${sess}</ul>` : ""}<i>点击开始学习</i>`;
      tip.style.display = "block";
      const k = svg.getBoundingClientRect().width / W;
      tip.style.left = `${(n.x + n.r + 12) * k - container.scrollLeft}px`;
      tip.style.top = `${(n.y - 10) * k - container.scrollTop}px`;
      svg.querySelectorAll("[data-from]").forEach((p) => p.classList.toggle("lit", p.dataset.from === n.id || p.dataset.to === n.id));
    });
    node.addEventListener("mouseleave", () => { tip.style.display = "none"; svg.querySelectorAll(".lit").forEach((p) => p.classList.remove("lit")); });
    node.addEventListener("click", () => opts.onOpen && opts.onOpen(n));
    svg.appendChild(node);
  }
  // unit legend
  units.forEach((u, i) => {
    if (!u) return;
    const y = 26 + i * 22;
    svg.appendChild(el("path", { d: inkCircle(26, y, 7, u), fill: "#fbfaf6", stroke: UNIT_INK[i % UNIT_INK.length], "stroke-width": "2" }));
    svg.appendChild(el("text", { x: 42, y: y + 5, class: "cm-unit" }, u));
  });
  return { nodes, W, H };
}
