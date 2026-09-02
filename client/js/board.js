/**
 * Whiteboard canvas: column-packed cards with reveal gates.
 *
 * Items are absolutely positioned inside `.wb-canvas`. Each column packs
 * cards top-down by measured height; `layout: "newcol"` (or overflowing the
 * viewport height) opens a new column. Gated cards are placed immediately
 * but hidden until `openGate(stepId)`; the viewport then scrolls to keep the
 * newest card visible ("fit-visible").
 */
import { renderMarkdownInto } from "./markdown.js";
import { createWidgetFrame } from "./widgets.js";
import { drawDecoration } from "./decorations.js";

const COL_W = 460;
const GAP_X = 28;
const GAP_Y = 18;
const PAD = 32;

export class Whiteboard {
  constructor(viewportEl, canvasEl) {
    this.viewport = viewportEl;
    this.canvas = canvasEl;
    this.items = new Map(); // uid -> { el, column, y, h, gate, kind }
    this.columns = [];      // [{ height }]
    this.active = -1;
    this.pageTitle = "";
    this._pageEl = null;
  }

  clear() {
    this.canvas.innerHTML = "";
    this.items.clear();
    this.columns = [];
    this.active = -1;
    this._pageEl = null;
    this._resize();
  }

  newPage(title) {
    this.clear();
    if (title) {
      const el = document.createElement("div");
      el.className = "wb-page-title";
      el.textContent = title;
      this.canvas.appendChild(el);
      this._pageEl = el;
    }
  }

  newColumn() {
    this.columns.push({ height: PAD + (this._pageEl ? 48 : 0) });
    this.active = this.columns.length - 1;
    return this.active;
  }

  _ensureColumn(layout, needH) {
    if (this.active < 0 || layout === "newcol") return this.newColumn();
    const limit = Math.max(480, this.viewport.clientHeight - PAD);
    const col = this.columns[this.active];
    if (col.height > PAD + 8 && col.height + needH > limit) return this.newColumn();
    return this.active;
  }

  _place(el, layout, gate, kind) {
    el.style.position = "absolute";
    el.style.width = `${COL_W}px`;
    el.style.visibility = "hidden";
    this.canvas.appendChild(el);
    const h = el.offsetHeight;
    const c = this._ensureColumn(layout, h + GAP_Y);
    const col = this.columns[c];
    const x = PAD + c * (COL_W + GAP_X), y = col.height;
    el.style.left = `${x}px`;
    el.style.top = `${y}px`;
    col.height += h + GAP_Y;
    el.style.visibility = "";
    if (gate != null) el.classList.add("gated");
    this._resize();
    return { el, column: c, x, y, h, gate, kind };
  }

  _resize() {
    const w = PAD * 2 + Math.max(1, this.columns.length) * (COL_W + GAP_X);
    const h = Math.max(this.viewport.clientHeight, ...this.columns.map((c) => c.height + PAD));
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
  }

  addBoard({ uid, title, markdown, layout, gate }) {
    const el = document.createElement("div");
    el.className = "wb-card";
    el.dataset.uid = uid;
    el.innerHTML = `<div class="wb-card-title"></div><div class="wb-card-content"></div>`;
    const t = el.querySelector(".wb-card-title");
    if (title) t.textContent = title; else t.remove();
    renderMarkdownInto(el.querySelector(".wb-card-content"), markdown);
    el.querySelectorAll(".wb-card-content > *").forEach((b, i) => b.style.setProperty("--i", i));
    const item = this._place(el, layout, gate, "board");
    this.items.set(uid, item);
    if (gate == null) this._reveal(item, 0);
    return item;
  }

  addWidget({ uid, title, html, layout, gate }) {
    const el = document.createElement("div");
    el.className = "wb-card wb-widget";
    el.dataset.uid = uid;
    const head = document.createElement("div");
    head.className = "wb-card-title";
    head.textContent = title || "交互教具";
    el.append(head, createWidgetFrame({ html, title }));
    const item = this._place(el, layout, gate, "widget");
    this.items.set(uid, item);
    if (gate == null) this._reveal(item, 0);
    return item;
  }

  async addGraph({ uid, title, mermaid, layout, gate }) {
    const el = document.createElement("div");
    el.className = "wb-card wb-graph";
    el.dataset.uid = uid;
    const head = document.createElement("div");
    head.className = "wb-card-title";
    head.textContent = title || "关系图";
    const body = document.createElement("div");
    body.className = "wb-card-content";
    el.append(head, body);
    try {
      if (window.mermaid) {
        const { svg } = await window.mermaid.render(`m_${uid}_${Date.now()}`, mermaid);
        body.innerHTML = svg;
      } else {
        body.textContent = mermaid;
      }
    } catch (e) {
      body.textContent = `图表渲染失败：${e.message}`;
    }
    const item = this._place(el, layout, gate, "graph");
    this.items.set(uid, item);
    if (gate == null) this._reveal(item, 0);
    return item;
  }

  addPlaceholder({ uid, text, layout }) {
    const el = document.createElement("div");
    el.className = "wb-card wb-placeholder";
    el.dataset.uid = uid;
    el.textContent = text || "教具生成中…";
    const item = this._place(el, layout, null, "placeholder");
    this.items.set(uid, item);
    this._reveal(item, 0);
    return item;
  }

  addReward({ title, description }) {
    const el = document.createElement("div");
    el.className = "wb-card wb-reward";
    el.innerHTML = `<div class="wb-reward-badge">✦ 掌握了</div><div class="wb-card-title"></div><div class="wb-card-content"></div>`;
    el.querySelector(".wb-card-title").textContent = title;
    renderMarkdownInto(el.querySelector(".wb-card-content"), description);
    const item = this._place(el, "follow", null, "reward");
    this._reveal(item, 0);
    return item;
  }

  /** Reveal every item gated on `stepId`, pacing block reveals across `durationMs`. */
  openGate(stepId, durationMs = 0) {
    const opened = [];
    for (const item of this.items.values()) {
      if (item.gate === stepId) {
        this._reveal(item, durationMs);
        item.gate = null;
        opened.push(item);
      }
    }
    return opened;
  }

  _reveal(item, durationMs) {
    const el = item.el;
    const blocks = el.querySelectorAll(".wb-card-content > *");
    const n = blocks.length || 1;
    const span = Math.min(durationMs * 0.6, n * 700);
    blocks.forEach((b, i) => b.style.setProperty("--delay", `${Math.round((span / n) * i)}ms`));
    el.classList.remove("gated");
    el.classList.add("revealed");
    requestAnimationFrame(() => this._fitVisible(item));
  }

  /** Scroll only the canvas viewport (never ancestors) so the item is visible above the dock. */
  _fitVisible(item) {
    const vp = this.viewport;
    const dock = 170;
    let top = vp.scrollTop, left = vp.scrollLeft;
    if (item.y + item.h + dock > top + vp.clientHeight) top = item.y + item.h + dock - vp.clientHeight;
    if (item.y - PAD < top) top = Math.max(0, item.y - PAD);
    if (item.x + COL_W + PAD > left + vp.clientWidth) left = item.x + COL_W + PAD - vp.clientWidth;
    if (item.x - PAD < left) left = Math.max(0, item.x - PAD);
    vp.scrollTo({ top, left, behavior: "smooth" });
  }

  decorate(uid, deco) {
    const item = this.items.get(uid);
    if (!item) return;
    drawDecoration(item.el, deco, 850);
  }
}
