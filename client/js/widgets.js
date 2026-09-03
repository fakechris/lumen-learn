/**
 * Sandboxed HTML widgets (Three.js manipulatives etc.).
 *
 * The iframe runs with `sandbox="allow-scripts"` only: no same-origin access,
 * so generated code cannot touch our DOM, storage or cookies. A small shim is
 * injected so runtime errors inside the widget are reported back via
 * postMessage and shown as an overlay instead of a silent blank box.
 */
const SHIM_STORAGE = `<script>
// null-origin iframe: touching localStorage throws SecurityError and can blank the page.
try { window.localStorage.getItem("x"); } catch (e) {
  var mem = {};
  var shim = { getItem: function(k){ return Object.prototype.hasOwnProperty.call(mem, k) ? mem[k] : null; },
    setItem: function(k, v){ mem[k] = String(v); }, removeItem: function(k){ delete mem[k]; },
    clear: function(){ mem = {}; }, key: function(i){ return Object.keys(mem)[i] || null; } };
  Object.defineProperty(shim, "length", { get: function(){ return Object.keys(mem).length; } });
  Object.defineProperty(window, "localStorage", { value: shim, configurable: true });
  Object.defineProperty(window, "sessionStorage", { value: shim, configurable: true });
}
</script>`;

const shimFor = (token) => SHIM_STORAGE + `<script>
(function(){
  var T = ${JSON.stringify(token)};
  function post(m){ try { parent.postMessage(Object.assign({ source: "hk-widget", token: T }, m), "*"); } catch(e){} }
  window.addEventListener("error", function(e){ post({ type: "error", message: e.message || "script error" }); });
  window.addEventListener("unhandledrejection", function(e){ post({ type: "error", message: (e.reason && e.reason.message) || String(e.reason) }); });
  window.addEventListener("message", function(e){
    var d = e.data;
    if (!d || d.source !== "hk-host-control" || !window.hkControl) return;
    var fn = window.hkControl[d.op] || (d.op === "set" ? window.hkControl.setState : null);
    try { if (fn) { fn.call(window.hkControl, d.payload); post({ type: "control", op: d.op, ok: true }); } else { post({ type: "control", op: d.op, ok: false }); } } catch (err) { post({ type: "error", message: "control " + d.op + ": " + (err.message || err) }); }
  });
  window.addEventListener("load", function(){
    post({ type: "ready" });
    // The iframe may be laid out after the guest script measured it (widgets draw once on
    // resize()); re-dispatch once the frame has a real size so first paint is not blank.
    requestAnimationFrame(function(){ requestAnimationFrame(function(){ window.dispatchEvent(new Event("resize")); }); });
  });
})();
</script>`;

const frames = new Map(); // token -> { overlay, status }
let nextToken = 1;

window.addEventListener("message", (ev) => {
  const data = ev.data;
  if (!data || data.source !== "hk-widget") return;
  const entry = frames.get(data.token);
  if (!entry) return;
  if (data.type === "error") {
    entry.overlay.textContent = `教具运行出错：${data.message}`;
    entry.overlay.classList.add("visible");
  } else if (data.type === "ready") {
    entry.status.textContent = "可拖拽旋转";
  }
});

// The page's handwriting fonts, so HandChart labels and widget UI match the board.
const FONT_LINKS = `<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/lxgw-wenkai-lite-webfont@1.7.0/style.css"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Patrick+Hand&display=swap"><style>:root{--hand:"Patrick Hand","LXGW WenKai Lite","PingFang SC",sans-serif}</style>`;

function injectShim(html, token) {
  const shim = FONT_LINKS + shimFor(token);
  const idx = html.search(/<head[^>]*>/i);
  if (idx >= 0) {
    const end = html.indexOf(">", idx) + 1;
    return html.slice(0, end) + shim + html.slice(end);
  }
  return shim + html;
}

export function createWidgetFrame({ html, title, height = 360 }) {
  const wrap = document.createElement("div");
  wrap.className = "widget-frame";
  wrap.style.height = `${height}px`;

  const iframe = document.createElement("iframe");
  iframe.setAttribute("sandbox", "allow-scripts");
  iframe.setAttribute("title", title || "interactive widget");
  const token = `w${nextToken++}`;
  iframe.srcdoc = injectShim(html, token);

  const overlay = document.createElement("div");
  overlay.className = "widget-error";

  const status = document.createElement("div");
  status.className = "widget-status";
  status.textContent = "加载中…";

  const fullscreen = document.createElement("button");
  fullscreen.className = "widget-fullscreen";
  fullscreen.textContent = "⛶";
  fullscreen.title = "全屏";
  fullscreen.addEventListener("click", () => wrap.classList.toggle("fullscreen"));

  wrap.append(iframe, status, overlay, fullscreen);
  frames.set(token, { overlay, status });
  // Teacher-driven control channel (see WidgetControl in the protocol): the
  // audio clock fires op/payload pairs; the guest implements window.hkControl.
  wrap.hkPost = (op, payload) => {
    try { iframe.contentWindow.postMessage({ source: "hk-host-control", op, payload }, "*"); } catch (e) {}
  };
  return wrap;
}
