// Headless render check for a widget HTML file.
// Usage: NODE_PATH=$(npm root -g) node tools/render_widget.mjs <html-file> <png-out>
// Prints one JSON line: { errors: [...], blank: bool, canvases: N, ink: 0..1 }
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const [htmlPath, pngOut] = process.argv.slice(2);
const html = readFileSync(htmlPath, "utf8");
const errors = [];
const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 560, height: 340 } });
  page.on("pageerror", (e) => errors.push(String(e.message || e)));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  await page.setContent(html, { waitUntil: "load" });
  await page.waitForTimeout(1200);
  await page.mouse.move(380, 150);
  await page.waitForTimeout(150);

  // "Ink" = fraction of sampled pixels that differ from the dominant colour.
  // For 2D canvases sample the bitmap directly; a drawn plot has ink > ~2%.
  // WebGL canvases cannot be read back without preserveDrawingBuffer, so they
  // fall back to a screenshot-size heuristic below.
  const probe = await page.evaluate(() => {
    const out = { canvases: 0, ink: null, webgl: false };
    const canvases = Array.from(document.querySelectorAll("canvas"));
    out.canvases = canvases.length;
    let inkTotal = 0, sampled = 0;
    for (const c of canvases) {
      const ctx = c.getContext("2d");
      if (!ctx) { out.webgl = true; continue; }
      const w = c.width, h = c.height;
      if (!w || !h) continue;
      const data = ctx.getImageData(0, 0, w, h).data;
      const counts = new Map();
      const step = Math.max(1, Math.floor((w * h) / 4000));
      let n = 0;
      for (let i = 0; i < w * h; i += step) {
        const k = (data[i * 4] >> 4) + "," + (data[i * 4 + 1] >> 4) + "," + (data[i * 4 + 2] >> 4) + "," + (data[i * 4 + 3] >> 6);
        counts.set(k, (counts.get(k) || 0) + 1);
        n++;
      }
      const dominant = Math.max(...counts.values());
      inkTotal += (n - dominant);
      sampled += n;
    }
    if (sampled > 0) out.ink = inkTotal / sampled;
    // SVG-based explorables count as drawn if they contain shapes.
    out.svgShapes = document.querySelectorAll("svg path, svg line, svg circle, svg rect, svg polyline").length;
    return out;
  });
  // Three.js widgets expose window.__hkScene (required by THREE_SYSTEM);
  // count drawables after 900ms so a silent-failure scene (0 objects) is caught.
  await page.waitForTimeout(900);
  const sceneObjects = await page.evaluate(() => {
    if (!window.__hkScene) return null;
    let n = 0;
    window.__hkScene.traverse((o) => { if (o.isMesh || o.isLine || o.isPoints) n++; });
    return n;
  }).catch(() => null);

  const jpeg = await page.screenshot({ type: "jpeg", quality: 60 });
  await page.screenshot({ path: pngOut, type: "png" });
  let blank;
  if (probe.ink !== null) blank = probe.ink < 0.02 && probe.svgShapes < 3;
  else if (probe.webgl) blank = (sceneObjects !== null ? sceneObjects === 0 : jpeg.length < 6000);
  else blank = probe.svgShapes < 3 && jpeg.length < 6000;
  console.log(JSON.stringify({ errors: errors.slice(0, 5), blank, canvases: probe.canvases, ink: probe.ink, svgShapes: probe.svgShapes, bytes: jpeg.length }));
} finally {
  await browser.close();
}
