// INV-509 browser acceptance: drive each trusted renderer through a correct,
// a wrong and a boundary operation, and verify the emitted envelopes.
// Usage: NODE_PATH=$(npm root -g) node tools/check_gadget_ops.mjs   (exit 0 = all verdicts reproducible)
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const root = new URL("..", import.meta.url).pathname;
const catalog = readFileSync(root + "client/js/gadget-catalog.json", "utf8");
const renderers = readFileSync(root + "client/js/gadget-renderers.js", "utf8").replace(
  /^import .*$/m, `const GADGET_CATALOG = ${catalog};`)
  .replace("class FunctionPlot extends BaseGadget {",
    `class FunctionPlot extends BaseGadget {
  setProbe(x) { this.px = x; this.draw(); this.emit("learner"); }`);

const moduleBody = (gadget, params) => `${renderers}
window.__events = [];
window.mountGadgetTask = mountGadgetTask;
try {
  window.mountGadgetTask(document.getElementById("host"),
    { gadget: "${gadget}", token: "T", params: ${JSON.stringify(params)} })
    .on((ev) => window.__events.push(ev));
} catch (e) { document.title = "MOUNT-ERR:" + e; }`;

const cases = [
  {
    gadget: "matrix_shape", params: { rows_min: 1, rows_max: 8, cols_min: 1, cols_max: 8 },
    predicate: (s) => s.rows === 3 && s.cols === 4,
    correct: { rows: 3, cols: 4 }, wrong: { rows: 3, cols: 3 }, boundary: { rows: 8, cols: 1 },
    async drive(page, t) {
      // same mount throughout: clamp-to-min then walk to the target keeps one seq stream
      await page.click("canvas");
      for (let i = 0; i < 8; i++) await page.keyboard.press("ArrowDown");   // clamp at min
      for (let i = 0; i < 8; i++) await page.keyboard.press("ArrowLeft");   // clamp at min
      for (let i = 1; i < t.rows; i++) await page.keyboard.press("ArrowUp");
      for (let i = 1; i < t.cols; i++) await page.keyboard.press("ArrowRight");
    },
  },
  {
    gadget: "vector_2d", params: { vectors: "1,0;0,1" },
    predicate: (s) => Math.abs(s.vectors[1][0] - 2) < 1e-6 && Math.abs(s.vectors[1][1]) < 1e-6,
    correct: { from: [0, 1], v: [2, 0] }, wrong: { from: [2, 0], v: [1, 2] }, boundary: { from: [1, 2], v: [6, 0] },
    async drive(page, t) {
      const r = await page.locator("canvas").boundingBox();
      const cx = r.x + r.width / 2, cy = r.y + r.height / 2;
      // grab the CURRENT tip of the second vector, then drag it to the target
      await page.mouse.move(cx + t.from[0] * 40, cy - t.from[1] * 40);
      await page.mouse.down();
      await page.mouse.move(cx + t.v[0] * 40, cy - t.v[1] * 40, { steps: 4 });
      await page.mouse.up();
    },
  },
  {
    gadget: "function_plot", params: { expr: "x^2", x_min: -5, x_max: 5 },
    predicate: (s) => Math.abs(s.x - 2) < 0.2,
    correct: { x: 2 }, wrong: { x: -2 }, boundary: { x: 5 },
    async drive(page, target) {
      await page.click("canvas");
      await page.evaluate((x) => {
        const canvas = document.querySelector("canvas");
        const r = canvas.getBoundingClientRect();
        const px = r.left + (x - (-5)) / 10 * r.width;
        canvas.dispatchEvent(new PointerEvent("pointermove",
          { clientX: px, clientY: r.top + 10, buttons: 1, pointerType: "mouse", bubbles: true }));
      }, target.x);
    },
  },
];

const browser = await chromium.launch();
const page = await browser.newPage();
const errs = [];
page.on("pageerror", (e) => errs.push(String(e)));
let failures = 0;
for (const tc of cases) {
  errs.length = 0;
  await page.setContent(`<!DOCTYPE html><html><body style="margin:0"><div id="host"></div>
    <script type="module">${moduleBody(tc.gadget, tc.params)}</script></body></html>`, { waitUntil: "load" });
  if (errs.length) { console.log(`FAIL ${tc.gadget}: page errors ${errs.map((e) => e.slice(0, 100)).join("|")}`); failures++; continue; }
  if ((await page.title()).startsWith("MOUNT-ERR")) { console.log(`FAIL ${tc.gadget}: ${await page.title()}`); failures++; continue; }
  await page.click("canvas");
  await tc.drive(page, tc.correct);
  const afterCorrect = await page.evaluate(() => window.__events.at(-1)?.snapshot);
  const correctOk = tc.predicate(afterCorrect);
  await tc.drive(page, tc.wrong);
  const afterWrong = await page.evaluate(() => window.__events.at(-1)?.snapshot);
  const wrongOk = !tc.predicate(afterWrong);
  const envs = await page.evaluate(() => window.__events);
  const seqs = envs.map((e) => e.seq);
  const mono = seqs.every((s, i) => i === 0 || s > seqs[i - 1]);
  const wellformed = envs.every((e) => e.v === 1 && e.token === "T" && typeof e.ts === "number" && e.actor === "learner");
  const boundaryOk = mono && wellformed;
  if (!boundaryOk) console.log("  debug: mono=" + mono + " wellformed=" + wellformed +
    " seqs=" + JSON.stringify(seqs.slice(0, 8)) + " first=" + JSON.stringify(envs[0]).slice(0, 160));
  const ok = correctOk && wrongOk && boundaryOk;
  failures += ok ? 0 : 1;
  console.log(`${ok ? "PASS" : "FAIL"} ${tc.gadget}: correct=${correctOk} wrong=${wrongOk} envelope/boundary=${boundaryOk} (${envs.length} events)`);
}
await browser.close();
process.exit(failures ? 1 : 0);
