/**
 * Trusted gadget renderers (INV-509): function_plot / vector_2d / matrix_shape.
 *
 * Each renderer owns a sandboxed canvas, supports pointer AND keyboard
 * operation, reports its full observable state as a snapshot, and emits it in
 * the operation-evidence envelope:
 *   {v:1, token, actor, seq, ts, snapshot}
 * The parent (widgets.js host) validates and forwards; the SERVER decides
 * correctness from the predicate — the renderer never declares success.
 */
import { GADGET_CATALOG } from "./gadget-catalog.json";

const W = 560, H = 340;

function envelope(state) {
  return {
    v: 1,
    token: state.token,
    actor: state.actor || "learner",
    seq: ++state.seq,
    ts: Date.now(),
    snapshot: state.snapshot,
  };
}

class BaseGadget {
  constructor(canvas, cfg) {
    this.canvas = canvas;
    this.cfg = cfg;
    this.seq = 0;
    this.listeners = [];
    canvas.width = W; canvas.height = H;
    this.ctx = canvas.getContext("2d");
    this.hint = document.createElement("p");
    this.hint.className = "gadget-kbd-hint";
    this.hint.textContent = "键盘可用：Tab 聚焦后用方向键操作，Enter 提交当前状态";
    canvas.tabIndex = 0;
  }
  on(fn) { this.listeners.push(fn); }
  emit(actor) {
    const ev = { v: 1, token: this.cfg.token, actor, seq: ++this.seq, ts: Date.now(), snapshot: this.snapshot() };
    this.listeners.forEach((fn) => fn(ev));
    return ev;
  }
  submit() { this.emit("learner"); }                  // Enter = explicit learner submission
  demo() { this.emit("teacher"); }                    // teacher/演示 events are excluded server-side
  draw() {}
}

class FunctionPlot extends BaseGadget {
  constructor(canvas, cfg) {
    super(canvas, cfg);
    const p = cfg.params || {};
    this.x_min = p.x_min ?? -5; this.x_max = p.x_max ?? 5;
    this.f = compileExpr(p.expr || "x");
    this.px = (this.x_min + this.x_max) / 2;
    const probe = () => { this.px = clamp(this.px, this.x_min, this.x_max); this.draw(); };
    canvas.addEventListener("pointermove", (e) => {
      if (e.buttons !== 1 && e.pointerType === "mouse") return;
      this.px = xFromPixel(e, canvas, this.x_min, this.x_max); probe(); this.emit("learner");
    });
    canvas.addEventListener("keydown", (e) => {
      const step = (this.x_max - this.x_min) / 100;
      if (e.key === "ArrowLeft") { this.px -= step; probe(); this.emit("learner"); }
      if (e.key === "ArrowRight") { this.px += step; probe(); this.emit("learner"); }
      if (e.key === "Enter") this.submit();
    });
    this.draw();
  }
  snapshot() { return { x: round6(this.px), y: round6(this.f(this.px)) }; }
  draw() {
    const c = this.ctx, X = (v) => (v - this.x_min) / (this.x_max - this.x_min) * W;
    c.fillStyle = "#fbfaf7"; c.fillRect(0, 0, W, H);
    c.strokeStyle = "#4b4b4b"; c.beginPath(); c.moveTo(0, H / 2); c.lineTo(W, H / 2); c.stroke();
    c.strokeStyle = "#8a4b2b"; c.lineWidth = 2; c.beginPath();
    for (let i = 0; i <= W; i++) {
      const x = this.x_min + i / W * (this.x_max - this.x_min), y = this.f(x);
      const Y = H / 2 - clamp(y, -H, H) * (H / 8);
      i ? c.lineTo(i, Y) : c.moveTo(i, Y);
    }
    c.stroke(); c.lineWidth = 1;
    const y = this.f(this.px);
    c.fillStyle = "#b0553a"; c.beginPath(); c.arc(X(this.px), H / 2 - clamp(y, -H, H) * (H / 8), 6, 0, 7); c.fill();
  }
}

class Vector2D extends BaseGadget {
  constructor(canvas, cfg) {
    super(canvas, cfg);
    this.vecs = String(cfg.params?.vectors || "1,0;0,1").split(";").map((s) => s.split(",").map(Number));
    this.active = this.vecs.length - 1;                 // keyboard operates the last vector
    canvas.addEventListener("pointerdown", (e) => {
      const r = this.canvas.getBoundingClientRect();
      const x = (e.clientX - r.left - W / 2) / 40, y = (H / 2 - (e.clientY - r.top)) / 40;
      // grab whichever vector tip is nearest — keyboard operates this.active afterwards
      this.active = this.nearest(x, y);
      this.drag(e); this.emit("learner");
      const move = (ev) => { this.drag(ev); this.emit("learner"); };
      const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
      window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
    });
    canvas.addEventListener("keydown", (e) => {
      const d = 0.25, v = this.vecs[this.active];
      const moves = { ArrowLeft: [-d, 0], ArrowRight: [d, 0], ArrowUp: [0, d], ArrowDown: [0, -d] };
      if (moves[e.key]) { v[0] += moves[e.key][0]; v[1] += moves[e.key][1]; this.draw(); this.emit("learner"); }
      if (e.key === "Enter") this.submit();
    });
    this.draw();
  }
  drag(e) {
    const r = this.canvas.getBoundingClientRect();
    this.vecs[this.active] = [(e.clientX - r.left - W / 2) / 40, (H / 2 - (e.clientY - r.top)) / 40];
    this.draw();
  }
  nearest(x, y) {
    let best = 0, bd = Infinity;
    this.vecs.forEach((v, i) => {
      const d = (v[0] - x) ** 2 + (v[1] - y) ** 2;
      if (d < bd) { bd = d; best = i; }
    });
    return best;
  }
  snapshot() { return { vectors: this.vecs.map((v) => [round6(v[0]), round6(v[1])]) }; }
  draw() {
    const c = this.ctx;
    c.fillStyle = "#fbfaf7"; c.fillRect(0, 0, W, H);
    c.strokeStyle = "#d8d2c4";
    c.beginPath(); c.moveTo(W / 2, 0); c.lineTo(W / 2, H); c.moveTo(0, H / 2); c.lineTo(W, H / 2); c.stroke();
    this.vecs.forEach((v, i) => {
      c.strokeStyle = i === this.active ? "#b0553a" : "#4b4b4b"; c.lineWidth = 2.5;
      c.beginPath(); c.moveTo(W / 2, H / 2); c.lineTo(W / 2 + v[0] * 40, H / 2 - v[1] * 40); c.stroke();
    });
    c.lineWidth = 1;
  }
}

class MatrixShape extends BaseGadget {
  constructor(canvas, cfg) {
    super(canvas, cfg);
    const p = cfg.params || {};
    this.rows = clamp(Math.round(p.rows_min ?? 2), 1, 8);
    this.cols = clamp(Math.round(p.cols_min ?? 2), 1, 8);
    this.active = "rows";
    canvas.addEventListener("pointerdown", (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.active = e.offsetY < H / 2 ? "rows" : "cols";
      this.drag(e.offsetX, r); this.emit("learner");
    });
    canvas.addEventListener("keydown", (e) => {
      // keyboard model: ArrowUp/Down change rows, ArrowLeft/Right change cols
      const moves = { ArrowUp: ["rows", 1], ArrowDown: ["rows", -1], ArrowRight: ["cols", 1], ArrowLeft: ["cols", -1] };
      if (moves[e.key]) {
        const [axis, d] = moves[e.key];
        this[axis] = clamp(this[axis] + d, 1, 8);
        this.draw(); this.emit("learner");
      }
      if (e.key === "Enter") this.submit();
    });
    this.draw();
  }
  drag(x) { this[this.active] = clamp(Math.round(x / W * 8), 1, 8); this.draw(); }
  snapshot() { return { rows: this.rows, cols: this.cols }; }
  draw() {
    const c = this.ctx, cell = Math.min(28, (H - 40) / Math.max(this.rows, 1));
    c.fillStyle = "#fbfaf7"; c.fillRect(0, 0, W, H);
    const ox = (W - this.cols * cell) / 2, oy = (H - this.rows * cell) / 2;
    c.strokeStyle = "#4b4b4b";
    for (let i = 0; i < this.rows; i++) for (let j = 0; j < this.cols; j++)
      c.strokeRect(ox + j * cell, oy + i * cell, cell, cell);
    c.fillStyle = this.active === "rows" ? "#b0553a" : "#4b4b4b";
    c.font = "16px sans-serif"; c.fillText(`行 ${this.rows}`, ox - 46, oy + this.rows * cell / 2);
    c.fillStyle = this.active === "cols" ? "#b0553a" : "#4b4b4b";
    c.fillText(`列 ${this.cols}`, ox, oy + this.rows * cell + 22);
  }
}

function compileExpr(src) {
  const safe = String(src).replace(/[^0-9x+\-*/(). ^]/g, "").replace(/\^/g, "**");
  const f = new Function("x", `"use strict";return (${safe || "x"});`);
  return (x) => { try { const y = f(x); return Number.isFinite(y) ? y : 0; } catch { return 0; } };
}
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const round6 = (v) => Math.round(v * 1e6) / 1e6;
function xFromPixel(e, canvas, lo, hi) {
  const r = canvas.getBoundingClientRect();
  return lo + (e.clientX - r.left) / r.width * (hi - lo);
}

const RENDERERS = { function_plot: FunctionPlot, vector_2d: Vector2D, matrix_shape: MatrixShape };

/** Mount a task into a host element; returns the unmount function. */
export function mountGadgetTask(host, { gadget, params, token }) {
  const Renderer = RENDERERS[gadget];
  if (!Renderer) throw new Error(`unknown gadget: ${gadget}`);
  host.innerHTML = "";
  const canvas = document.createElement("canvas");
  host.appendChild(canvas);
  const g = new Renderer(canvas, { params, token });
  g.draw();
  return {
    on: g.on.bind(g),
    submit: g.submit.bind(g),
    demo: g.demo.bind(g),
    snapshot: g.snapshot.bind(g),
    unmount: () => { host.innerHTML = ""; },
  };
}
