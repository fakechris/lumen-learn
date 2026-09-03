/**
 * HandChart — zero-dependency hand-drawn chart base for explorable widgets.
 *
 * Injected into every generated 2D explorable (widget_generator.py) so the
 * model supplies data + overlays instead of hand-writing axes. The look matches
 * the handwritten page: paper panel, slightly wobbly ink axes, main red /
 * aux green, handwriting labels.
 *
 * API (all coordinates in WORLD units unless noted):
 *   const ch = HandChart.render(canvas, opts)
 *     opts = { type: "function"|"points"|"bars"|"lines",
 *              xlim:[x0,x1], ylim:[y0,y1],                  (all but bars)
 *              fn: x=>y | fns:[{fn, color?, label?}],        (function)
 *              points:[[x,y],...],                          (points)
 *              series:[{name, points, color?, dash?}],      (lines; legend drawn)
 *              bars:{labels:[...], values:[...], highlight?} (bars)
 *              xlabel, ylabel, grid:true|false, title }
 *     returns { ctx, sx, sy, box, W, H, opts }  — keep it to draw overlays
 *   HandChart.marker(ch, x, y, {color, label, r})       dot + optional label
 *   HandChart.vline(ch, x, {color, dash, label})         vertical guide
 *   HandChart.hline(ch, y, {color, dash, label})         horizontal guide
 *   HandChart.segment(ch, [x1,y1], [x2,y2], {color, width, dash, arrow})
 *   HandChart.note(ch, x, y, text, {color, align, size}) handwriting annotation
 *   HandChart.attachProbe(canvas, {xlim, ylim}, onProbe, onLeave?)
 *     pointer follow; onProbe({x, y}) in world coords — redraw + readouts there.
 *   HandChart.style = { INK, AXIS, MAIN, AUX, GRID, PAPER, MUTED }
 */
(function () {
  const style = {
    INK: "#2b2b2b", AXIS: "#8a8377", MAIN: "#c0392b", AUX: "#2e8b6f",
    GRID: "#ece5d4", PAPER: "#fbfaf6", MUTED: "#8a8377", PURPLE: "#7c3aed", BLUE: "#2563eb",
  };
  const PALETTE = [style.MAIN, style.AUX, style.BLUE, style.PURPLE, "#d97706"];
  const PAD = { l: 48, r: 18, t: 18, b: 36 };
  const HAND = "'Patrick Hand', 'LXGW WenKai Lite', 'PingFang SC', sans-serif";

  function font(size, italic) { return (italic ? "italic " : "") + size + "px " + HAND; }

  // deterministic wobble so the "hand" does not shake on every redraw
  function wob(seed, amp) { return Math.sin(seed * 12.9898) * amp; }

  function tickVals(min, max, n) {
    const span = max - min || 1;
    const raw = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || mag * 10;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  function fmt(v) {
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 10000 || a < 0.01) return v.toExponential(1);
    return String(+v.toFixed(Math.max(0, 3 - Math.floor(Math.log10(a)))));
  }

  /** a hand-drawn line: two passes with tiny deterministic wobble */
  function inkLine(ctx, x1, y1, x2, y2, color, width, seed) {
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = "round";
    const n = 6, s = seed || (x1 + y2);
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const t = i / n, x = x1 + (x2 - x1) * t, y = y1 + (y2 - y1) * t;
      const w = i === 0 || i === n ? 0 : wob(s + i, 0.7);
      i ? ctx.lineTo(x + w, y + w) : ctx.moveTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  function frame(ctx, W, H) {
    ctx.clearRect(0, 0, W, H);
    ctx.save();
    ctx.fillStyle = style.PAPER;
    ctx.fillRect(0, 0, W, H);
    ctx.restore();
  }

  function plotBox(W, H, opts) {
    const t = PAD.t + (opts && opts.title ? 18 : 0);
    return { x0: PAD.l, y0: t, x1: W - PAD.r, y1: H - PAD.b };
  }

  function mapping(box, xlim, ylim) {
    const [x0, x1] = xlim, [y0, y1] = ylim;
    return {
      sx: (v) => box.x0 + (v - x0) / (x1 - x0) * (box.x1 - box.x0),
      sy: (v) => box.y1 - (v - y0) / (y1 - y0) * (box.y1 - box.y0),
    };
  }

  function axes(ctx, box, opts) {
    const [x0, x1] = opts.xlim, [y0, y1] = opts.ylim;
    const { sx, sy } = mapping(box, opts.xlim, opts.ylim);
    const xt = tickVals(x0, x1, 6).filter((v) => v >= x0 && v <= x1);
    const yt = tickVals(y0, y1, 4).filter((v) => v >= y0 && v <= y1);
    if (opts.grid !== false) {
      ctx.save(); ctx.strokeStyle = style.GRID; ctx.lineWidth = 1;
      for (const v of xt) { ctx.beginPath(); ctx.moveTo(sx(v), box.y0); ctx.lineTo(sx(v), box.y1); ctx.stroke(); }
      for (const v of yt) { ctx.beginPath(); ctx.moveTo(box.x0, sy(v)); ctx.lineTo(box.x1, sy(v)); ctx.stroke(); }
      ctx.restore();
    }
    // axes sit on zero when zero is inside the range, else on the box edge
    const ax = y0 <= 0 && y1 >= 0 ? sy(0) : box.y1;
    const ay = x0 <= 0 && x1 >= 0 ? sx(0) : box.x0;
    inkLine(ctx, box.x0 - 4, ax, box.x1 + 6, ax, style.AXIS, 1.4, 1);
    inkLine(ctx, ay, box.y1 + 4, ay, box.y0 - 6, style.AXIS, 1.4, 2);
    // arrow heads
    ctx.save(); ctx.fillStyle = style.AXIS;
    ctx.beginPath(); ctx.moveTo(box.x1 + 8, ax); ctx.lineTo(box.x1 + 1, ax - 3.5); ctx.lineTo(box.x1 + 1, ax + 3.5); ctx.fill();
    ctx.beginPath(); ctx.moveTo(ay, box.y0 - 8); ctx.lineTo(ay - 3.5, box.y0 - 1); ctx.lineTo(ay + 3.5, box.y0 - 1); ctx.fill();
    ctx.restore();
    ctx.save();
    ctx.fillStyle = style.MUTED; ctx.font = font(12);
    ctx.textAlign = "center";
    for (const v of xt) {
      if (Math.abs(sx(v) - ay) < 1 && ay !== box.x0) continue; // skip the label under the y axis
      inkLine(ctx, sx(v), ax - 3, sx(v), ax + 3, style.AXIS, 1, v);
      ctx.fillText(fmt(v), sx(v), ax + 16);
    }
    ctx.textAlign = "right";
    for (const v of yt) {
      if (Math.abs(sy(v) - ax) < 1 && ax !== box.y1) continue;
      inkLine(ctx, ay - 3, sy(v), ay + 3, sy(v), style.AXIS, 1, v + 7);
      ctx.fillText(fmt(v), ay - 7, sy(v) + 4);
    }
    ctx.font = font(14, true); ctx.fillStyle = style.INK;
    if (opts.xlabel) { ctx.textAlign = "right"; ctx.fillText(opts.xlabel, box.x1 + 4, ax - 8); }
    if (opts.ylabel) { ctx.textAlign = "left"; ctx.fillText(opts.ylabel, ay + 8, box.y0 + 4); }
    ctx.restore();
    return { sx, sy };
  }

  function stroke(ctx, pts, color, width, dash) {
    if (!pts.length) return;
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = width || 2.2;
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    let pen = false;
    for (const p of pts) {
      if (!p) { pen = false; continue; }
      pen ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
      pen = true;
    }
    ctx.stroke();
    ctx.restore();
  }

  function fnPoints(fn, opts, m) {
    const [x0, x1] = opts.xlim, [y0, y1] = opts.ylim;
    const n = 400, pts = [];
    const margin = (y1 - y0) * 0.02;
    for (let i = 0; i <= n; i++) {
      const x = x0 + (x1 - x0) * i / n;
      let y;
      try { y = fn(x); } catch (e) { y = NaN; }
      if (!isFinite(y) || y < y0 - margin || y > y1 + margin) { pts.push(null); continue; }
      pts.push([m.sx(x), m.sy(y)]);
    }
    return pts;
  }

  function legend(ctx, box, items) {
    if (!items.length) return;
    ctx.save();
    ctx.font = font(13); ctx.textAlign = "left"; ctx.textBaseline = "middle";
    let x = box.x0 + 10, y = box.y0 + 12;
    for (const it of items) {
      inkLine(ctx, x, y, x + 22, y, it.color, 2.4, x);
      ctx.fillStyle = style.INK;
      ctx.fillText(it.label, x + 28, y + 1);
      x += 28 + ctx.measureText(it.label).width + 18;
    }
    ctx.restore();
  }

  function title(ctx, W, text) {
    ctx.save();
    ctx.font = font(15); ctx.fillStyle = style.INK; ctx.textAlign = "center";
    ctx.fillText(text, W / 2, 16);
    ctx.restore();
  }

  function render(canvas, opts) {
    const ctx = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    const W = Math.max(60, Math.round(rect.width)), H = Math.max(60, Math.round(rect.height));
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    frame(ctx, W, H);
    if (opts.title) title(ctx, W, opts.title);
    const box = plotBox(W, H, opts);
    const type = opts.type || "function";

    if (type === "bars") {
      const labels = opts.bars.labels || [], values = opts.bars.values || [];
      const n = Math.max(1, values.length);
      const bw = (box.x1 - box.x0) / n;
      const vmax = Math.max(...values.map((v) => Math.abs(v)), 0) * 1.15 || 1;
      const base = box.y1;
      inkLine(ctx, box.x0 - 4, base, box.x1 + 4, base, style.AXIS, 1.4, 3);
      ctx.save();
      values.forEach((v, i) => {
        const h = (Math.abs(v) / vmax) * (box.y1 - box.y0);
        const x = box.x0 + i * bw + bw * 0.2, w = bw * 0.6, y = base - h;
        const hot = opts.bars.highlight != null && i === opts.bars.highlight;
        ctx.fillStyle = hot ? style.MAIN : "#e6dfcf";
        ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = hot ? style.MAIN : style.AXIS; ctx.lineWidth = 1.2;
        ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
        ctx.font = font(13); ctx.textAlign = "center"; ctx.fillStyle = style.INK;
        ctx.fillText(String(labels[i] == null ? i + 1 : labels[i]), x + w / 2, base + 17);
        ctx.fillStyle = hot ? style.MAIN : style.MUTED;
        ctx.fillText(fmt(v), x + w / 2, y - 6);
      });
      ctx.restore();
      const sx = (i) => box.x0 + (i + 0.5) * bw, sy = (v) => base - (Math.abs(v) / vmax) * (box.y1 - box.y0);
      return { ctx, sx, sy, box, W, H, opts };
    }

    const m = axes(ctx, box, opts);
    const items = [];
    if (type === "lines") {
      (opts.series || []).forEach((s, i) => {
        const color = s.color || PALETTE[i % PALETTE.length];
        stroke(ctx, (s.points || []).map((p) => [m.sx(p[0]), m.sy(p[1])]), color, 2.2, s.dash);
        if (s.name) items.push({ label: s.name, color });
      });
    } else if (type === "points") {
      ctx.save(); ctx.fillStyle = opts.color || style.MAIN;
      (opts.points || []).forEach((p) => { ctx.beginPath(); ctx.arc(m.sx(p[0]), m.sy(p[1]), opts.r || 3.6, 0, Math.PI * 2); ctx.fill(); });
      ctx.restore();
    } else {
      const fns = opts.fns || (typeof opts.fn === "function" ? [{ fn: opts.fn, color: opts.color, label: opts.label }] : []);
      fns.forEach((f, i) => {
        const color = f.color || PALETTE[i % PALETTE.length];
        stroke(ctx, fnPoints(f.fn, opts, m), color, f.width || 2.4, f.dash);
        if (f.label) items.push({ label: f.label, color });
      });
    }
    legend(ctx, box, items);
    return { ctx, sx: m.sx, sy: m.sy, box, W, H, opts };
  }

  // ---- overlays (world coordinates) ----
  function marker(ch, x, y, o) {
    o = o || {};
    const px = ch.sx(x), py = ch.sy(y);
    ch.ctx.save();
    ch.ctx.fillStyle = o.color || style.MAIN;
    ch.ctx.beginPath(); ch.ctx.arc(px, py, o.r || 5, 0, Math.PI * 2); ch.ctx.fill();
    ch.ctx.strokeStyle = style.PAPER; ch.ctx.lineWidth = 1.5; ch.ctx.stroke();
    if (o.label) {
      ch.ctx.font = font(o.size || 13); ch.ctx.fillStyle = o.color || style.MAIN; ch.ctx.textAlign = "left";
      ch.ctx.fillText(o.label, px + 8, py - 8);
    }
    ch.ctx.restore();
  }

  function vline(ch, x, o) {
    o = o || {};
    const px = ch.sx(x);
    stroke(ch.ctx, [[px, ch.box.y0], [px, ch.box.y1]], o.color || style.MUTED, o.width || 1.2, o.dash || [4, 4]);
    if (o.label) note(ch, x, null, o.label, { color: o.color || style.MUTED, px: px + 4, py: ch.box.y0 + 12, align: "left", size: o.size });
  }

  function hline(ch, y, o) {
    o = o || {};
    const py = ch.sy(y);
    stroke(ch.ctx, [[ch.box.x0, py], [ch.box.x1, py]], o.color || style.MUTED, o.width || 1.2, o.dash || [4, 4]);
    if (o.label) note(ch, null, y, o.label, { color: o.color || style.MUTED, px: ch.box.x1 - 4, py: py - 5, align: "right", size: o.size });
  }

  function segment(ch, a, b, o) {
    o = o || {};
    const x1 = ch.sx(a[0]), y1 = ch.sy(a[1]), x2 = ch.sx(b[0]), y2 = ch.sy(b[1]);
    const color = o.color || style.AUX;
    stroke(ch.ctx, [[x1, y1], [x2, y2]], color, o.width || 2, o.dash);
    if (o.arrow) {
      const ang = Math.atan2(y2 - y1, x2 - x1), L = 9;
      ch.ctx.save(); ch.ctx.fillStyle = color; ch.ctx.beginPath();
      ch.ctx.moveTo(x2, y2);
      ch.ctx.lineTo(x2 - L * Math.cos(ang - 0.4), y2 - L * Math.sin(ang - 0.4));
      ch.ctx.lineTo(x2 - L * Math.cos(ang + 0.4), y2 - L * Math.sin(ang + 0.4));
      ch.ctx.fill(); ch.ctx.restore();
    }
  }

  function note(ch, x, y, text, o) {
    o = o || {};
    const px = o.px != null ? o.px : ch.sx(x), py = o.py != null ? o.py : ch.sy(y);
    ch.ctx.save();
    ch.ctx.font = font(o.size || 14, o.italic);
    ch.ctx.fillStyle = o.color || style.INK;
    ch.ctx.textAlign = o.align || "left";
    ch.ctx.fillText(text, px, py);
    ch.ctx.restore();
  }

  /** Pointer probe in world coords; the caller redraws and updates readouts. */
  function attachProbe(canvas, opts, onProbe, onLeave) {
    const [x0, x1] = opts.xlim;
    const [y0, y1] = opts.ylim;
    const toWorld = (px, py) => {
      const r = canvas.getBoundingClientRect();
      const box = plotBox(r.width, r.height, opts);
      const t = (px - r.left - box.x0) / (box.x1 - box.x0);
      const u = 1 - (py - r.top - box.y0) / (box.y1 - box.y0);
      return { x: x0 + t * (x1 - x0), y: y0 + u * (y1 - y0), inside: t >= 0 && t <= 1 };
    };
    const handler = (e) => {
      const p = toWorld(e.clientX, e.clientY);
      if (p.inside) onProbe(p);
    };
    canvas.addEventListener("pointermove", handler);
    canvas.addEventListener("pointerdown", handler);
    if (onLeave) canvas.addEventListener("pointerleave", onLeave);
    canvas.style.touchAction = "none";
    return toWorld;
  }

  window.HandChart = { render, marker, vline, hline, segment, note, attachProbe, style, font: HAND };
})();
