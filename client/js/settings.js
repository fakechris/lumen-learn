/**
 * BYOK settings dialog (INV-570): fetch masked settings, PUT changes (keys stay
 * on the server machine, never in git), probe connectivity, and refresh the
 * capabilities pill so the generate flow reflects the new client immediately.
 */
const $ = (id) => document.getElementById(id);

function fillFrom(view) {
  const s = view.settings || {};
  $("setProvider").value = s.provider || "";
  $("setTts").value = s.tts_engine || "";
  $("setBaseUrl").value = s.base_url || "";
  $("setModel").value = s.model || "";
  $("setModelPro").value = s.model_pro || "";
  $("setApiKey").value = "";
  $("setApiKey").placeholder = s.api_key ? `已保存 ${s.api_key}` : "留空 = 保持现状";
}

function payload(includeKey) {
  const body = {
    provider: $("setProvider").value,
    base_url: $("setBaseUrl").value.trim(),
    model: $("setModel").value.trim(),
    model_pro: $("setModelPro").value.trim(),
    tts_engine: $("setTts").value,
  };
  if (includeKey) body.api_key = $("setApiKey").value.trim();
  return body;
}

export async function refreshCapsPill() {
  try {
    const caps = await (await fetch("/api/v1/capabilities")).json();
    const el = $("capsPill");
    if (!el) return;
    el.textContent = caps.llm.configured ? `LLM ${caps.llm.model} · TTS ${caps.tts.engine}` : `无 LLM · TTS ${caps.tts.engine}`;
    el.classList.toggle("warn", !caps.llm.configured);
  } catch { /* the pill stays as-is when the probe fails */ }
}

function status(html) { $("settingsStatus").innerHTML = html; }

export function initSettings() {
  $("openSettings").addEventListener("click", async () => {
    $("settingsModal").classList.add("open");
    try { fillFrom(await (await fetch("/api/v1/settings")).json()); }
    catch { status("<p class='hint'>读取设置失败</p>"); }
  });
  $("closeSettings").addEventListener("click", () => $("settingsModal").classList.remove("open"));
  $("settingsModal").addEventListener("click", (e) => {
    if (e.target === $("settingsModal")) $("settingsModal").classList.remove("open");
  });

  $("settingsSave").addEventListener("click", async () => {
    status("<p class='hint'>保存中…</p>");
    try {
      const res = await fetch("/api/v1/settings", {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload(true)),
      });
      if (!res.ok) throw new Error(await res.text());
      const view = await res.json();
      fillFrom(view);
      const llm = view.llm.configured ? `模型就绪（${view.llm.provider} / ${view.llm.model}）`
                                       : "未配置模型（大模型生成不可用，逐段讲读不受影响）";
      status(`<p class='hint'>已保存，立即生效。${llm} · 语音 ${view.tts.engine}</p>`);
      refreshCapsPill();
    } catch (err) {
      status(`<p class='hint'>保存失败：${String(err).slice(0, 140)}</p>`);
    }
  });

  $("settingsTest").addEventListener("click", async () => {
    status("<p class='hint'>测试中：发一次最小补全 + 一段短语音…</p>");
    try {
      const res = await fetch("/api/v1/settings/test", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload(true)),
      });
      const r = await res.json();
      const fmt = (x, name) => x && x.ok
        ? `✅ ${name} 通（${x.latency_ms != null ? `${x.latency_ms} ms` : ""}${x.model ? ` · ${x.model}` : ""}${x.engine ? ` · ${x.engine}` : ""}）`
        : `❌ ${name} 不通：${x ? x.error : "无结果"}`;
      status(`<p class='hint'>${fmt(r.llm, "模型")}<br>${fmt(r.tts, "语音")}</p>`);
    } catch (err) {
      status(`<p class='hint'>测试失败：${String(err).slice(0, 140)}</p>`);
    }
  });
}
