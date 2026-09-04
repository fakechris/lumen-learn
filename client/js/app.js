/**
 * Socratic Whiteboard client controller.
 *
 * Consumes the action stream over WebSocket, drives the board and the
 * audio clock, and sends acks / answers / interjections back.
 */
import { WhiteboardSocket } from "./ws.js";
import { AudioClock, charsAtMs } from "./audio-clock.js";
import { Whiteboard } from "./board.js";
import { renderMarkdownInto, escapeHtml } from "./markdown.js";
import { ExerciseView } from "./exercises.js";

const $ = (id) => document.getElementById(id);
const SPEEDS = [1.0, 1.25, 1.5, 2.0];
const ASK_HINT = "点击一个选项，或直接输入你的答案 / 提问";

const BEAT_LABEL = { hook: "钩子", analogy: "类比", poe: "预测", define: "定义", derive: "推导", worked_example: "例题", contrast: "对比", counterexample: "反例", apply: "应用", recap: "回顾" };
const AXES = [["memory", "记忆"], ["comprehension", "理解"], ["structure", "结构"], ["application", "应用"]];
/** four short ink bars, one per axis; labelled when `wide` */
function masteryBars(scores, wide = false) {
  return `<span class="mastery${wide ? " wide" : ""}" title="${AXES.map(([k, l]) => `${l} ${Math.round(scores[k] || 0)}`).join(" · ")}">${AXES.map(([k, l]) =>
    `<i class="${k}" style="--v:${Math.round(scores[k] || 0)}%">${wide ? `<b>${l}</b>` : ""}</i>`).join("")}</span>`;
}

class App {
  constructor() {
    this.ws = new WhiteboardSocket();
    this.clock = new AudioClock();
    this.board = new Whiteboard($("whiteboardViewport"), $("whiteboardCanvas"));
    this.exercises = new ExerciseView();

    this.courses = [];
    this.course = null;
    this.sessions = [];
    this.sessionId = null;
    this.state = "idle";
    this.speed = 1.0;
    this.transcript = [];
    this.widgetControls = new Map();
    this.rawLog = [];
    this.activeTab = "transcript";
    this.firstBoardPending = false;
    this.zoom = 1;
    this.progress = JSON.parse(localStorage.getItem("hk_progress") || "{}"); // courseId/sessionId -> {started, finished, score}

    this.speakText = new Map();       // step_id -> text
    this.stepKinds = new Map();       // step_id -> Set of media kinds seen before its speak
    this.keypoints = [];
    this.pendingKinds = new Set();
    this.pendingDecos = new Map();    // during_step -> [decoration]
    this.currentStep = null;          // step_id of playing tts
    this.interject = null;            // { id, bubble, text, audioPlayed }

    this.initFeynman();
    this.bindUi();
    this.ws.onMessage = (m) => this.onMessage(m);
    this.ws.onOpen = () => this.setConn(true);
    this.ws.onClose = () => this.setConn(false);
    this.boot();
  }

  // ------------------------------------------------------------------ boot

  async boot() {
    await this.ws.connect();
    const caps = await (await fetch("/api/v1/capabilities")).json();
    $("capsPill").textContent = caps.llm.configured ? `LLM ${caps.llm.model} · TTS ${caps.tts.engine}` : `无 LLM · TTS ${caps.tts.engine}`;
    $("capsPill").classList.toggle("warn", !caps.llm.configured);
    await this.refreshCourses();
    if (this.courses.length) await this.loadCourse(this.courses[0].course_id);
  }

  async refreshCourses() {
    const data = await (await fetch("/api/v1/courses")).json();
    this.courses = data.courses;
    const sel = $("courseSelect");
    sel.innerHTML = "";
    for (const c of this.courses) {
      const o = document.createElement("option");
      o.value = c.course_id;
      o.textContent = `${c.title} (${c.total_sessions} 节 · ${c.generation_mode})`;
      sel.appendChild(o);
    }
  }

  async loadCourse(courseId, { home = true } = {}) {
    this.course = await (await fetch(`/api/v1/courses/${courseId}`)).json();
    $("courseSelect").value = courseId;
    this.sessions = this.course.chapters.flatMap((ch) => ch.sessions.map((s) => ({ ...s, chapter: ch.title, unit: ch.unit || "" })));
    this.renderSidebar();
    if (home) this.openHome();
    else if (this.sessions.length) this.startSession(this.sessions[0].session_id);
  }

  setZoom(z) {
    this.zoom = Math.max(0.6, Math.min(2.0, Math.round(z * 10) / 10));
    $("whiteboardCanvas").style.zoom = this.zoom;
    $("zoomLabel").textContent = `${Math.round(this.zoom * 100)}%`;
  }

  // ------------------------------------------------------------------ course home

  progressOf(sessionId) {
    return this.progress[`${this.course.course_id}/${sessionId}`] || {};
  }

  markProgress(sessionId, patch) {
    const key = `${this.course.course_id}/${sessionId}`;
    this.progress[key] = { ...(this.progress[key] || {}), ...patch };
    localStorage.setItem("hk_progress", JSON.stringify(this.progress));
  }

  statusClass(p) {
    if (p.score != null && p.score >= 0.8) return "done";
    if (p.score != null) return "good";
    if (p.started) return "tried";
    return "todo";
  }

  async openHome() {
    const c = this.course;
    if (!c) return;
    $("homeKicker").textContent = `${c.chapters.length} 讲 · ${this.sessions.length} 节 · ${c.generation_mode}`;
    $("homeTitle").textContent = c.title;
    $("homeOverview").textContent = c.overview || "";
    // four-axis mastery from the server (evidence: 提问回答 / 练习 / 讲给我听)
    this.mastery = {};
    try {
      const m = await (await fetch(`/api/v1/courses/${c.course_id}/mastery`)).json();
      (m.mastery || []).forEach((r) => { this.mastery[r.session_id] = r; });
      $("homeMastery").innerHTML = m.course ? `<span class="k">掌握度</span>${masteryBars(m.course, true)}<span class="n">综合 ${Math.round(m.composite)}</span>` : "";
    } catch (e) { $("homeMastery").innerHTML = ""; }
    const box = $("homeUnits");
    box.innerHTML = "";
    const groups = [];
    c.chapters.forEach((ch, ci) => {
      const unit = ch.unit || "";
      let g = groups[groups.length - 1];
      if (!g || g.unit !== unit) { g = { unit, chapters: [] }; groups.push(g); }
      g.chapters.push({ ...ch, index: ci });
    });
    groups.forEach((g, gi) => {
      const u = document.createElement("div");
      u.className = "home-unit";
      if (g.unit) u.innerHTML = `<div class="u">UNIT ${String(gi + 1).padStart(2, "0")} · ${escapeHtml(g.unit)}</div>`;
      g.chapters.forEach((ch) => {
        const lec = document.createElement("div");
        lec.className = "home-lecture";
        lec.innerHTML = `<h3>讲次 ${ch.index + 1}：${escapeHtml(ch.title)}</h3><div class="d">${escapeHtml(ch.description || "")}</div>`;
        ch.sessions.forEach((s) => {
          const p = this.progressOf(s.session_id);
          const row = document.createElement("div");
          row.className = "home-session";
          const tags = (s.tags || []).map((t) => `<span class="tag tag-${t}">${t}</span>`).join("");
          const m = this.mastery[s.session_id];
          row.innerHTML = `<div>${escapeHtml(s.title)}<span class="tags">${tags}</span>${m ? masteryBars(m.scores) : ""}<div class="meta" style="font-size:.76rem;color:#64748b">${escapeHtml(m && m.note ? m.note : s.learning_goal)}</div></div>`;
          const learn = document.createElement("button");
          learn.className = "pill btn learn" + (p.finished ? " secondary" : "");
          learn.textContent = p.finished ? "▶ 再学一遍" : "▶ 学习";
          learn.addEventListener("click", () => { $("courseHome").classList.remove("open"); this.startSession(s.session_id); });
          const practice = document.createElement("button");
          practice.className = "pill btn";
          practice.textContent = p.score != null ? `↻ 练习 ${Math.round(p.score * 100)}%` : "↻ 练习";
          practice.addEventListener("click", async () => {
            const ok = await this.exercises.open(c.course_id, s.session_id, { onDone: (score) => { this.markProgress(s.session_id, { score }); this.openHome(); } });
            if (!ok) this.toast("这一节还没有课后练习");
          });
          const feynman = document.createElement("button");
          feynman.className = "pill btn";
          feynman.textContent = "🎤 讲给我听";
          feynman.title = "费曼回合：用自己的话讲，同学只追问不纠错";
          feynman.addEventListener("click", () => this.feynmanOpen(c.course_id, s.session_id));
          const st = document.createElement("i");
          st.className = `st ${this.statusClass(p)}`;
          row.append(learn, practice, feynman, st);
          lec.appendChild(row);
        });
        u.appendChild(lec);
      });
      const quiz = document.createElement("div");
      quiz.className = "home-quiz";
      const label = g.unit ? "单元测验" : "综合测验";
      quiz.innerHTML = `<div><div class="t">${label}：${escapeHtml(g.unit || c.title)}</div><div class="d">把这一部分所有小节的课后题连起来做一遍</div></div>`;
      const btn = document.createElement("button");
      btn.className = "pill btn primary";
      btn.textContent = "开始测验";
      btn.addEventListener("click", async () => {
        const ids = g.chapters.flatMap((ch) => ch.sessions.map((s) => s.session_id));
        const ok = await this.exercises.openMany(c.course_id, ids, { onDone: (score) => { ids.forEach((id) => this.markProgress(id, { quiz: score })); this.openHome(); } });
        if (!ok) this.toast("这一部分还没有习题");
      });
      quiz.appendChild(btn);
      u.appendChild(quiz);
      box.appendChild(u);
    });
    $("courseHome").classList.add("open");
  }

  async startSession(sessionId, fromStep = null, { skipEntry = false, level = null } = {}) {
    // a fresh start asks the server whether it needs a prerequisite check first; the level itself is
    // decided server-side from evidence unless the learner picked one (档位 button)
    if (fromStep == null && !skipEntry) {
      const entry = await this.fetchEntry(sessionId);
      if (entry && entry.needs_diagnosis && entry.questions.length) { this.openEntry(sessionId, entry); return; }
    }
    level = level || this.profileLevel || null;
    this.sessionId = sessionId;
    this.markProgress(sessionId, { started: true });
    $("courseHome").classList.remove("open");
    this.exercises.close();
    this.suspended = null;
    this.interject = null;
    this.clock.stop();
    this.board.clear();
    this.speakText.clear();
    this.pendingDecos.clear();
    this.widgetControls.clear();
    this.transcript = [];
    this.interject = null;
    this.currentStep = null;
    this.setSubtitle("");
    $("askRow").innerHTML = "";
    $("nextSessionHint").style.display = "none";
    $("exerciseHint").style.display = "none";
    const idx = this.sessions.findIndex((s) => s.session_id === sessionId);
    $("sessionCounter").textContent = `${idx + 1} / ${this.sessions.length}`;
    $("sessionTitle").textContent = this.sessions[idx]?.title || "";
    this.ws.send({ type: "start_session", course_id: this.course.course_id, session_id: sessionId, from_step_id: fromStep, tts_speed: this.speed, level: level || null });
    this.renderSidebar();
  }

  // ------------------------------------------------------------------ adaptive entry (SYSTEM_DESIGN §10.2)
  async fetchEntry(sessionId) {
    try {
      const r = await fetch(`/api/v1/courses/${this.course.course_id}/sessions/${sessionId}/entry`);
      return r.ok ? await r.json() : null;
    } catch (e) { return null; }
  }

  openEntry(sessionId, entry) {
    const view = $("entryView");
    view.classList.add("open");
    $("entryNote").textContent = `这节课建立在「${entry.prereqs.map((p) => p.title).join("、")}」上。先做 ${entry.questions.length} 道小题，我好知道从哪讲起。`;
    const body = $("entryBody");
    body.innerHTML = "";
    let answered = 0;
    const done = () => { answered += 1; $("entryCounter").textContent = `${answered} / ${entry.questions.length}`; if (answered >= entry.questions.length) { $("entryDone").style.display = ""; $("entrySkip").style.display = "none"; } };
    entry.questions.forEach((q, i) => {
      const box = document.createElement("div");
      box.className = "entry-q";
      const stem = document.createElement("div"); stem.className = "stem";
      renderMarkdownInto(stem, `${i + 1}. ${q.stem}`);
      box.appendChild(stem);
      const fb = document.createElement("div"); fb.className = "fb";
      const grade = async (body) => {
        const r = await fetch("/api/v1/grade", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ course_id: q.course_id, session_id: q.session_id, exercise_id: q.exercise_id, ...body }) });
        return r.ok ? await r.json() : { correct: false, feedback: "" };
      };
      if (q.kind === "single_choice") {
        const opts = document.createElement("div"); opts.className = "opts";
        (q.options || []).forEach((o, j) => {
          const b = document.createElement("button"); renderMarkdownInto(b, o);
          b.addEventListener("click", async () => {
            if (opts.dataset.done) return; opts.dataset.done = "1";
            const res = await grade({ answer_index: j });
            b.classList.add(res.correct ? "correct" : "wrong");
            fb.textContent = res.correct ? "✓ 对" : `✗ 答案是「${res.answer || ""}」`;
            done();
          });
          opts.appendChild(b);
        });
        box.appendChild(opts);
      } else {
        const input = document.createElement("input"); input.placeholder = "填空，回车提交";
        input.addEventListener("keydown", async (ev) => {
          if (ev.key !== "Enter" || input.disabled) return;
          input.disabled = true;
          const res = await grade({ answer_text: input.value });
          fb.textContent = res.correct ? "✓ 对" : `✗ 答案是「${res.answer || ""}」`;
          done();
        });
        box.appendChild(input);
      }
      box.appendChild(fb);
      body.appendChild(box);
    });
    $("entryCounter").textContent = `0 / ${entry.questions.length}`;
    $("entryDone").style.display = "none";
    $("entrySkip").style.display = "";
    const close = () => { view.classList.remove("open"); $("entryDone").onclick = $("entrySkip").onclick = $("entryClose").onclick = null; };
    $("entryDone").onclick = () => { close(); this.startSession(sessionId, null, { skipEntry: true }); };
    $("entrySkip").onclick = $("entryClose").onclick = () => { close(); this.startSession(sessionId, null, { skipEntry: true }); };
  }

  on_level_update(m) {
    this.level = m.level;
    const btn = $("levelBtn");
    const label = { novice: "打基础", standard: "标准", fast: "快进" }[m.level] || m.level;
    btn.textContent = `档位 · ${label}`;
    btn.dataset.level = m.level;
    btn.title = m.reason ? `${m.reason}（点击切换：慢一点 / 标准 / 快一点 / 自动）` : "讲解档位";
    (m.skipped_steps || []).forEach((id) => { const k = this.keypoints.find((x) => x.step_id === id); if (k) k.skipped = true; });
    this.renderKeypoints(this.currentStep);
    if (m.reason) this.toast(`${label}：${m.reason}`);
  }

  async cycleLevel() {
    const order = [null, "novice", "standard", "fast"];
    const cur = this.profileLevel === undefined ? null : this.profileLevel;
    const next = order[(order.indexOf(cur) + 1) % order.length];
    this.profileLevel = next;
    await fetch(`/api/v1/courses/${this.course.course_id}/profile`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ level: next }) });
    const label = next ? { novice: "打基础", standard: "标准", fast: "快进" }[next] : "自动";
    $("levelBtn").textContent = `档位 · ${label}`;
    $("levelBtn").dataset.level = next || "";
    if (this.sessionId && this.state !== "idle") this.startSession(this.sessionId, null, { skipEntry: true, level: next || null });
  }

  skipStep() {
    const play = this.currentPlay;
    if (!play || !play.finish || this.state !== "teaching") return;
    this.clock.stop();
    play.finish(true);
  }

  // ------------------------------------------------------------------ feynman round

  initFeynman() {
    this.fm = { courseId: null, sessionId: null, round: 0, max: 4 };
    $("fmClose").addEventListener("click", () => $("feynmanView").classList.remove("open"));
    $("mapClose").addEventListener("click", () => $("mapView").classList.remove("open"));
    $("levelBtn").addEventListener("click", () => this.cycleLevel());
    $("skipBtn").addEventListener("click", () => this.skipStep());
    $("openMap").addEventListener("click", () => this.openConceptMap());
    $("fmSend").addEventListener("click", () => this.feynmanSend());
    $("fmDone").addEventListener("click", () => this.feynmanSummary());
  }

  async openConceptMap() {
    const c = this.course;
    if (!c) return;
    $("mapView").classList.add("open");
    $("mapMeta").textContent = "生成中…";
    $("mapCanvas").innerHTML = "";
    let map;
    try {
      const res = await fetch(`/api/v1/courses/${c.course_id}/concept_map`);
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
      map = await res.json();
    } catch (e) { $("mapMeta").textContent = ""; $("mapNote").textContent = `概念地图生成失败：${e.message}`; return; }
    const { renderConceptMap } = await import("./concept-map.js");
    const titles = Object.fromEntries(this.sessions.map((s) => [s.session_id, s.title]));
    $("mapMeta").textContent = `${map.nodes.length} 个概念 · ${map.edges.length} 条关系${map.source === "structure" ? " · 结构图（未配置模型）" : ""}`;
    $("mapNote").textContent = map.note || "";
    renderConceptMap($("mapCanvas"), map, {
      mastery: this.mastery || {},
      sessionTitle: (id) => titles[id],
      onOpen: (n) => {
        const sid = (n.sessions || [])[0];
        if (!sid) return;
        $("mapView").classList.remove("open");
        $("courseHome").classList.remove("open");
        this.startSession(sid);
      },
    });
  }

  async feynmanOpen(courseId, sessionId) {
    const res = await fetch(`/api/v1/courses/${courseId}/sessions/${sessionId}/feynman/start`, { method: "POST" });
    if (!res.ok) { this.toast((await res.json().catch(() => ({}))).detail || "无法开始费曼回合"); return; }
    const data = await res.json();
    if (data.llm === false) { this.toast("费曼回合需要配置 LLM；当前服务端没有模型 key"); return; }
    this.fm = { courseId, sessionId, round: 0, max: data.max_rounds };
    $("feynmanView").classList.add("open");
    $("fmLog").innerHTML = "";
    $("fmVerdict").innerHTML = "";
    $("fmInput").value = "";
    $("fmRound").textContent = `0 / ${data.max_rounds}`;
    $("fmPrompt").innerHTML = `<b>${escapeHtml(this.course.title)}</b>`;
    this.fmAdd("buddy", data.prompt);
    $("fmInput").focus();
  }

  fmAdd(who, text) {
    const el = document.createElement("div");
    el.className = `turn ${who}`;
    el.textContent = text;
    $("fmLog").appendChild(el);
    el.scrollIntoView({ block: "nearest" });
  }

  async feynmanSend() {
    const text = $("fmInput").value.trim();
    if (!text) return;
    $("fmInput").value = "";
    this.fmAdd("me", text);
    const { courseId, sessionId } = this.fm;
    const res = await fetch(`/api/v1/courses/${courseId}/sessions/${sessionId}/feynman/turn`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ explanation: text }) });
    if (!res.ok) { this.fmAdd("buddy", "（费曼回合需要配置 LLM，当前无法追问）"); return; }
    const data = await res.json();
    this.fm.round = data.round;
    $("fmRound").textContent = `${data.round} / ${data.max_rounds}`;
    if (data.question) this.fmAdd("buddy", data.question);
    if (data.done) { $("fmSend").style.display = "none"; $("fmDone").style.display = ""; }
  }

  async feynmanSummary() {
    const { courseId, sessionId } = this.fm;
    const res = await fetch(`/api/v1/courses/${courseId}/sessions/${sessionId}/feynman/summary`, { method: "POST" });
    if (!res.ok) { this.toast("总结失败"); return; }
    const data = await res.json();
    $("fmVerdict").innerHTML = `<div class="fm-summary">${escapeHtml(data.summary || "")}</div>`;
    this.markProgress(sessionId, { feynman: true });
    $("fmDone").style.display = "none";
    $("fmSend").style.display = "";
  }

  // ------------------------------------------------------------------ ui

  bindUi() {
    $("courseSelect").addEventListener("change", (e) => this.loadCourse(e.target.value));
    $("prevSessionBtn").addEventListener("click", () => this.navigate(-1));
    $("nextSessionBtn").addEventListener("click", () => this.navigate(1));
    $("nextSessionHint").addEventListener("click", () => this.navigate(1));
    $("exerciseHint").addEventListener("click", () => this.exercises.open(this.course.course_id, this.sessionId));
    $("playPauseBtn").addEventListener("click", () => this.togglePause());
    $("speedBtn").addEventListener("click", () => this.cycleSpeed());
    $("interjectBtn").addEventListener("click", () => this.beginInterject());
    $("interjectInput").addEventListener("keydown", (e) => {
      if (e.key === "Enter") this.sendInterject();
      if (e.key === "Escape") this.cancelInterject();
    });
    $("interjectSend").addEventListener("click", () => this.sendInterject());
    $("interjectCancel").addEventListener("click", () => this.cancelInterject());
    for (const tab of ["transcript", "outline", "inspector"]) {
      $(`tab-${tab}`).addEventListener("click", () => { this.activeTab = tab; this.renderSidebar(); });
    }
    $("toggleSidebar").addEventListener("click", () => $("sidebar").classList.toggle("open"));
    $("openHome").addEventListener("click", () => this.openHome());
    $("closeHome").addEventListener("click", () => $("courseHome").classList.remove("open"));
    $("zoomIn").addEventListener("click", () => this.setZoom(this.zoom + 0.1));
    $("zoomOut").addEventListener("click", () => this.setZoom(this.zoom - 0.1));
    $("openGenerate").addEventListener("click", () => $("generateModal").classList.add("open"));
    $("closeGenerate").addEventListener("click", () => $("generateModal").classList.remove("open"));
    $("ingestBtn").addEventListener("click", () => this.ingestAndPlan());
    $("buildBtn").addEventListener("click", () => this.buildFromPlan());
    $("backToImport").addEventListener("click", () => this.showGenStep("import"));
    $("loadExample").addEventListener("click", async () => {
      $("genContent").value = await (await fetch("/examples/linear_algebra_basis.md")).text().catch(() => "");
    });
    document.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      if (e.code === "Space") { e.preventDefault(); this.togglePause(); }
    });
  }

  setConn(ok) {
    $("connDot").classList.toggle("ok", ok);
    $("connDot").title = ok ? "已连接" : "连接断开，重连中…";
  }

  setState(state) {
    this.state = state;
    const labels = { idle: "空闲", teaching: "讲解中", awaiting_answer: "等待你的回答", interjecting: "答疑中", paused: "已暂停", finished: "本节完成" };
    $("stateBadge").textContent = labels[state] || state;
    $("stateBadge").dataset.state = state;
    $("playIcon").textContent = state === "paused" ? "▶" : "❚❚";
  }

  setSubtitle(text, cursor = false) {
    $("subtitleText").textContent = text;
    $("typingCursor").style.visibility = cursor ? "visible" : "hidden";
  }

  navigate(delta) {
    const idx = this.sessions.findIndex((s) => s.session_id === this.sessionId);
    const next = this.sessions[idx + delta];
    if (next) this.startSession(next.session_id);
  }

  togglePause() {
    if (this.state === "paused") {
      this.clock.resume();
      this.ws.send({ type: "resume_session" });
      this.setState(this.currentStep ? "teaching" : "teaching");
    } else if (this.state === "teaching" || this.state === "awaiting_answer") {
      this.clock.pause();
      this.ws.send({ type: "pause_session" });
      this.setState("paused");
    }
  }

  cycleSpeed() {
    this.speed = SPEEDS[(SPEEDS.indexOf(this.speed) + 1) % SPEEDS.length];
    $("speedBtn").textContent = `沉稳 · ${this.speed}×`;
    this.clock.setRate(this.speed);
    this.ws.send({ type: "set_tts_config", speed: this.speed });
  }

  addBubble(role, text, opts = {}) {
    const entry = { role, text, streaming: !!opts.streaming, kind: opts.kind || null };
    this.transcript.push(entry);
    if (this.activeTab === "transcript") this.renderSidebar();
    return entry;
  }

  renderSidebar() {
    for (const tab of ["transcript", "outline", "inspector"]) $(`tab-${tab}`).classList.toggle("active", tab === this.activeTab);
    const box = $("sidebarContent");
    box.innerHTML = "";
    if (this.activeTab === "transcript") {
      for (const m of this.transcript) {
        const b = document.createElement("div");
        b.className = `bubble ${m.role}${m.streaming ? " streaming" : ""}`;
        b.innerHTML = `<span class="who">${m.kind ? `<span class="kind">${escapeHtml(m.kind)}</span>` : ""}${m.role === "tutor" ? "导师" : "你"}</span><div class="body">${escapeHtml(m.text)}</div>`;
        box.appendChild(b);
      }
      box.scrollTop = box.scrollHeight;
    } else if (this.activeTab === "outline") {
      let lastChapter = null;
      this.sessions.forEach((s, i) => {
        if (s.chapter !== lastChapter) {
          const h = document.createElement("div");
          h.className = "outline-chapter";
          h.textContent = s.chapter;
          box.appendChild(h);
          lastChapter = s.chapter;
        }
        const item = document.createElement("div");
        item.className = `outline-item${s.session_id === this.sessionId ? " active" : ""}`;
        item.innerHTML = `<div class="t">${i + 1}. ${escapeHtml(s.title)}</div><div class="g">${escapeHtml(s.learning_goal)}</div>${s.cognitive_hurdle ? `<div class="h">误区：${escapeHtml(s.cognitive_hurdle)}</div>` : ""}`;
        item.addEventListener("click", () => this.startSession(s.session_id));
        box.appendChild(item);
      });
    } else {
      const pre = document.createElement("pre");
      pre.textContent = this.rawLog.slice(-25).map((m) => JSON.stringify(m)).join("\n");
      box.appendChild(pre);
    }
  }

  // ------------------------------------------------------------------ messages

  onMessage(m) {
    const compact = m.type === "generated_animation" ? { ...m, html: `<${m.html.length} bytes>` }
      : m.type === "illustration" && m.svg ? { ...m, svg: `<${m.svg.length} bytes>` } : m;
    this.rawLog.push(compact);
    if (this.activeTab === "inspector") this.renderSidebar();
    const h = this[`on_${m.type}`];
    if (h) h.call(this, m);
  }

  ack(stepId) { this.ws.send({ type: "action_step_complete", step_id: stepId }); }

  on_connection_established() {}
  on_pong() {}
  on_error(m) { this.toast(m.message, true); }
  on_status(m) { this.setState(m.state); }

  on_session_ready(m) {
    $("sessionTitle").textContent = m.title;
    this.board.newPage(m.title);
    this.firstBoardPending = true;
    this.keypoints = m.keypoints || [];
    this.pendingKinds = new Set();
    this.stepKinds.clear();
    this.renderKeypoints(null);
    this.setState("teaching");
    this.addBubble("tutor", m.title, { kind: "课程开场" });
    if (m.learning_goal) this.addBubble("tutor", m.learning_goal, { kind: "学习目标" });
    this.stepTitles = new Map();
  }

  renderKeypoints(currentStepId) {
    const box = $("keypoints");
    box.innerHTML = "";
    if (!this.keypoints.length) return;
    const t = document.createElement("div");
    t.className = "kp-title";
    t.textContent = "课堂要点";
    box.appendChild(t);
    const idx = this.keypoints.findIndex((k) => k.step_id === currentStepId);
    this.keypoints.forEach((k, i) => {
      const el = document.createElement("div");
      el.className = "kp" + (k.detour ? " detour" : "") + (k.skipped ? " skipped" : "") + (i === idx ? " current" : i < idx || this.state === "finished" ? " done" : "");
      const beat = k.beat ? `<span class="beat">${escapeHtml(BEAT_LABEL[k.beat] || k.beat)}</span>` : "";
      el.innerHTML = `${beat}<span>${escapeHtml(k.title)}</span><span class="dot"></span>`;
      box.appendChild(el);
    });
  }

  noteKind(kind) { this.pendingKinds.add(kind); }

  on_new_page(m) { this.board.newPage(m.title); this.ack(m.step_id); }
  on_new_column() { this.board.newColumn(); }

  on_board(m) {
    this.noteKind(m.title && !this.firstBoardPending ? m.title : "板书");
    this.board.addBoard({ uid: m.board_uid, title: m.title, markdown: m.board_content, layout: m.layout, gate: m.reveal_gate_step, hook: this.firstBoardPending });
    this.firstBoardPending = false;
    this.zoom = 1;
    this.progress = JSON.parse(localStorage.getItem("hk_progress") || "{}"); // courseId/sessionId -> {started, finished, score}
    this.ack(m.step_id);
  }

  on_illustration(m) {
    this.noteKind("示意图");
    this.board.addIllustration({ uid: m.board_uid, caption: m.caption, svg: m.svg, imageUrl: m.image_url, layout: m.layout, gate: m.reveal_gate_step });
    this.ack(m.step_id);
  }

  async on_graph(m) {
    this.noteKind("流程图");
    await this.board.addGraph({ uid: m.board_uid, title: m.title, mermaid: m.mermaid, layout: m.layout, gate: m.reveal_gate_step });
    this.ack(m.step_id);
  }

  on_generated_animation(m) {
    // teacher controls fire on the audio clock of the speak step that reveals this widget
    if (m.controls && m.controls.length) this.widgetControls.set(m.reveal_gate_step ?? m.step_id, m.board_uid);
    this.noteKind("互动动画");
    this.board.addWidget({ uid: m.board_uid, title: m.title, html: m.html, layout: m.layout, gate: m.reveal_gate_step, controls: m.controls });
    this.ack(m.step_id);
  }

  on_animation_pending(m) { this.board.addPlaceholder({ uid: m.board_uid, text: `教具生成中：${m.task_preview}`, layout: m.layout }); }
  on_animation_failed(m) { this.toast(`教具生成失败，已跳过（${m.reason}）`); }

  on_speak(m) {
    this.speakText.set(m.step_id, m.spoken_text);
    const kinds = new Set(this.pendingKinds);
    if (this.speakText.size === 1 && !kinds.size) kinds.add("导引");
    this.stepKinds.set(m.step_id, kinds);
    this.pendingKinds = new Set();
  }

  on_highlight(m) { this.queueDecoration(m); }
  on_circle(m) { this.queueDecoration(m); }
  on_spotlight(m) { this.queueDecoration(m); }
  queueDecoration(m) {
    const key = m.during_step ?? this.currentStep ?? -1;
    if (this.stepKinds.has(key)) this.stepKinds.get(key).add("圈画");
    if (!this.pendingDecos.has(key)) this.pendingDecos.set(key, []);
    this.pendingDecos.get(key).push({ ...m, drawn: false });
    if (key === this.currentStep && this.clock.currentMs >= m.at_ms) this.drawNow(m);
  }
  drawNow(d) {
    d.drawn = true;
    this.board.decorate(d.target_board_uid, { kind: d.type, snippet: d.snippet, color: d.color });
  }

  on_tts_segment(m) {
    const text = this.speakText.get(m.step_id) || "";
    this.currentStep = m.step_id;
    const kinds = [...(this.stepKinds.get(m.step_id) || [])];
    this.addBubble("tutor", text, { kind: kinds.length ? kinds[0] : (m.step_id >= 100000 ? "点评" : "讲解") });
    this.renderKeypoints(m.step_id);
    this.board.openGate(m.step_id, m.duration_ms);
    const decos = this.pendingDecos.get(m.step_id) || [];
    const controlUid = this.widgetControls.get(m.step_id);
    const chars = Array.from(text);
    // marks index the JS string by UTF-16 code units; map to code points for slicing
    const start = (startMs = 0) => this.clock.play({
      url: m.audio_url,
      durationMs: m.duration_ms,
      startMs,
      onTick: (progress, ms) => {
        const n = m.marks ? Math.min(chars.length, charsAtMs(m.marks, ms, text.length, m.duration_ms))
                          : Math.floor(progress * chars.length);
        this.setSubtitle(chars.slice(0, n).join(""), progress < 1);
        for (const d of decos) if (!d.drawn && ms >= d.at_ms) this.drawNow(d);
        if (controlUid != null) this.board.tickControls(controlUid, ms);
      },
      onEnded: () => finish(false),
    });
    const finish = (skipped) => {
      for (const d of decos) if (!d.drawn) this.drawNow(d);
      if (controlUid != null) this.board.tickControls(controlUid, Infinity);
      this.setSubtitle(text, false);
      if (this.currentPlay?.stepId === m.step_id) this.currentPlay = null;
      this.currentStep = null;
      if (skipped) this.ws.send({ type: "skip_step", step_id: m.step_id });
      else this.ack(m.step_id);
    };
    if (m.step_id < 100000 || !this.interject) this.currentPlay = { stepId: m.step_id, start, finish };
    start(0);
    if (this.state === "paused") this.clock.pause();
  }

  on_ask(m) {
    const row = $("askRow");
    row.innerHTML = "";
    const q = document.createElement("div");
    q.className = "ask-question";
    q.textContent = m.question;
    row.appendChild(q);
    this.addBubble("tutor", `❓ ${m.question}`, { kind: "小测" });
    const finish = (payload, label) => {
      row.querySelectorAll("button, input").forEach((b) => (b.disabled = true));
      this.addBubble("student", label);
      this.ws.send({ type: "question_answers", step_id: m.step_id, ...payload });
      setTimeout(() => { if ($("askRow").contains(q)) row.innerHTML = ""; }, 1200);
    };
    if (m.mode === "choice") {
      const hint = document.createElement("div");
      hint.className = "ask-hint";
      hint.textContent = ASK_HINT;
      m.options.forEach((opt, i) => {
        const chip = document.createElement("button");
        chip.className = "chip";
        chip.textContent = opt.text;
        chip.addEventListener("click", () => {
          chip.classList.add(i === m.correct_index ? "correct" : "incorrect");
          finish({ answer_index: i }, opt.text);
        });
        row.appendChild(chip);
      });
      row.appendChild(hint);
    } else {
      const input = document.createElement("input");
      input.className = "ask-input";
      input.placeholder = "写下你的想法，回车提交";
      input.addEventListener("keydown", (e) => { if (e.key === "Enter" && input.value.trim()) finish({ answer_text: input.value.trim() }, input.value.trim()); });
      row.appendChild(input);
      input.focus();
    }
  }

  on_reward_user(m) {
    this.board.addReward({ title: m.master_concept_title, description: m.master_concept_description });
  }

  on_done() {}
  async on_response_complete() {
    this.setState("finished");
    this.markProgress(this.sessionId, { finished: true });
    this.renderKeypoints(null);
    this.addBubble("tutor", "本节课程内容已讲完。你可以做几道课后练习，进入下一课，或者退出当前课程。", { kind: "结课" });
    this.setSubtitle("本节课程内容已讲完。", false);
    const ex = await (await fetch(`/api/v1/courses/${this.course.course_id}/sessions/${this.sessionId}/exercises`)).json().catch(() => ({ exercises: [] }));
    if (ex.exercises?.length) {
      $("exerciseHint").textContent = `📝 课后练习（${ex.exercises.length} 题）`;
      $("exerciseHint").style.display = "";
    }
    const idx = this.sessions.findIndex((s) => s.session_id === this.sessionId);
    if (this.sessions[idx + 1]) {
      $("nextSessionHint").textContent = `▶ 下一节：${this.sessions[idx + 1].title}`;
      $("nextSessionHint").style.display = "";
    }
  }

  // ------------------------------------------------------------------ interjection

  beginInterject() {
    if (this.interject || this.state === "idle") return;
    const offset = Math.round(this.clock.currentMs);
    // Suspend the main narration; the detour is a mini lesson that reuses the same action handlers.
    this.suspended = this.currentPlay ? { start: this.currentPlay.start, ms: offset, stepId: this.currentPlay.stepId } : null;
    this.clock.stop();
    this.ws.send({ type: "interject_start", step_id: this.currentStep, offset_ms: offset });
    this.interject = { id: null, bubble: null, text: "", done: false };
    $("interjectBox").classList.add("open");
    $("interjectInput").value = "";
    $("interjectInput").focus();
  }

  sendInterject() {
    const text = $("interjectInput").value.trim();
    if (!text || !this.interject) return;
    $("interjectBox").classList.remove("open");
    this.addBubble("student", text);
    this.interject.bubble = this.addBubble("tutor", "正在准备岔路讲解…", { streaming: true, kind: "岔路" });
    this.interject.question = text;
    // show the detour in the keypoints list as a sub-item of the current point
    const idx = this.keypoints.findIndex((k) => k.step_id === this.suspended?.stepId);
    this.keypoints.splice(idx + 1, 0, { step_id: -1, title: `岔路：${text.slice(0, 18)}`, detour: true });
    this.renderKeypoints(this.suspended?.stepId ?? null);
    this.setSubtitle("", false);
    this.ws.send({ type: "interject_question", text });
  }

  cancelInterject() {
    $("interjectBox").classList.remove("open");
    if (!this.interject) return;
    this.interject = null;
    this.ws.send({ type: "interject_resume" });
    this.resumeSuspended();
  }

  on_interject_ready(m) { if (this.interject) this.interject.id = m.interject_id; }

  on_interject_text() {}
  on_interject_audio() {}

  on_interject_done(m) {
    if (!this.interject) return;
    const b = this.interject.bubble;
    if (b) {
      b.streaming = false;
      b.text = `岔路讲解结束，回到主线。`;
      if (m.cost_usd != null) b.kind = `岔路 · ${m.seconds}s · $${Number(m.cost_usd).toFixed(4)} · ${m.tokens} tokens`;
    }
    this.interject = null;
    this.ws.send({ type: "interject_resume" });
    this.resumeSuspended();
    this.renderSidebar();
  }

  resumeSuspended() {
    const s = this.suspended;
    this.suspended = null;
    if (s) { this.currentStep = s.stepId; this.renderKeypoints(s.stepId); s.start(s.ms); }
    else if (this.state === "paused") this.clock.resume();
  }

  // ------------------------------------------------------------------ generation (ingest -> plan -> build)

  showGenStep(step) {
    $("panelImport").style.display = step === "import" ? "" : "none";
    $("panelPlan").style.display = step === "import" ? "none" : "";
    for (const [id, key] of [["stepImport", "import"], ["stepPlan", "plan"], ["stepBuild", "build"]]) $(id).classList.toggle("active", key === step);
  }

  async pollJob(jobId, log) {
    let job;
    do {
      await new Promise((r) => setTimeout(r, 1500));
      job = await (await fetch(`/api/v1/jobs/${jobId}`)).json();
      log.textContent = job.events.map((e) => `[${e.stage}] ${e.detail}`).join("\n");
      log.scrollTop = log.scrollHeight;
    } while (job.status === "running");
    if (job.status === "error") throw new Error(job.error);
    return job;
  }

  async ingestAndPlan() {
    const log = $("genLog");
    const file = $("genFile").files[0];
    const content = $("genContent").value;
    const title = $("genTitle").value.trim();
    const mode = $("genMode").value;
    if (!file && !content.trim()) { log.textContent = "请上传 PDF / Markdown，或粘贴讲义内容"; return; }
    $("ingestBtn").disabled = true;
    log.textContent = "正在解析讲义…";
    try {
      const form = new FormData();
      if (file) form.append("file", file); else form.append("content", content);
      if (title) form.append("title", title);
      const res = await fetch("/api/v1/ingest", { method: "POST", body: form });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const doc = await res.json();
      this.genDoc = doc;
      $("docSummary").textContent = `《${doc.title}》：${doc.sections} 个小节${doc.pages ? `，${doc.pages} 页` : ""}，${doc.figures} 张教材图，约 ${doc.chars} 字`;
      log.textContent += "\n正在生成教案…";
      const pres = await fetch("/api/v1/plan", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ doc_key: doc.doc_key, mode }) });
      if (!pres.ok) throw new Error((await pres.json()).detail || pres.statusText);
      const job = await this.pollJob((await pres.json()).job_id, log);
      this.genPlan = job.plan;
      if (job.estimate) log.textContent += `\n💰 生成本课粗估 ≈ $${job.estimate.usd}（${job.estimate.segments} 段，${job.estimate.widgets} 教具，${job.estimate.figures} 图）；教案本身花费 $${Number(job.cost?.cost_usd || 0).toFixed(4)}`;
      this.renderPlanTree();
      this.showGenStep("plan");
    } catch (e) {
      log.textContent += `\n失败：${e.message}`;
    } finally {
      $("ingestBtn").disabled = false;
    }
  }

  renderPlanTree() {
    const tree = $("planTree");
    tree.innerHTML = "";
    const MEDIA = { board: "板书", illustration: "示意图", explorable: "探针图", threejs: "3D", reference_figure: "教材原图", mermaid: "流程图" };
    const plan = this.genPlan;
    let lastUnit = null;
    plan.chapters.forEach((ch, ci) => {
      if (ch.unit && ch.unit !== lastUnit) {
        const u = document.createElement("div");
        u.className = "plan-unit";
        u.textContent = `UNIT · ${ch.unit}`;
        tree.appendChild(u);
        lastUnit = ch.unit;
      }
      const h = document.createElement("div");
      h.className = "plan-chapter";
      h.innerHTML = `<span class="lec">LECTURE ${String(ci + 1).padStart(2, "0")}</span> ${escapeHtml(ch.title)}${ch.description ? ` <span class="meta">· ${escapeHtml(ch.description)}</span>` : ""}`;
      tree.appendChild(h);
      ch.sessions.forEach((s) => {
        const box = document.createElement("div");
        box.className = "plan-session";
        const tags = (s.tags || []).map((t) => `<span class="tag tag-${t}">${t}</span>`).join("");
        box.innerHTML = `<div class="t">${escapeHtml(s.title)} <span class="meta">${s.estimated_duration_min} 分钟</span> ${tags}</div>
          <div class="meta">目标：${escapeHtml(s.learning_goal)}</div>
          <div class="meta"><b>误区：</b>${escapeHtml(s.cognitive_hurdle || "—")}</div>`;
        s.segments.forEach((seg, i) => {
          const row = document.createElement("div");
          row.className = "plan-seg";
          const sel = document.createElement("select");
          for (const [k, label] of Object.entries(MEDIA)) {
            const o = document.createElement("option");
            o.value = k; o.textContent = label; o.selected = seg.media === k;
            sel.appendChild(o);
          }
          sel.className = `media-${seg.media}`;
          sel.addEventListener("change", () => { seg.media = sel.value; sel.className = `media-${seg.media}`; });
          const ask = document.createElement("label");
          const cb = document.createElement("input");
          cb.type = "checkbox"; cb.checked = !!seg.ask;
          cb.addEventListener("change", () => { seg.ask = cb.checked; });
          ask.append(cb, " 提问");
          const main = document.createElement("div");
          main.innerHTML = `<div>${escapeHtml(seg.title)}</div><div class="intent">${escapeHtml(seg.intent)}${seg.media_brief ? " · " + escapeHtml(seg.media_brief) : ""}</div>`;
          const n = document.createElement("span");
          n.className = "n"; n.textContent = `${i + 1}.`;
          row.append(n, main, sel, ask);
          box.appendChild(row);
        });
        tree.appendChild(box);
      });
    });
  }

  async buildFromPlan() {
    const log = $("genLog");
    const mode = $("genMode").value;
    $("buildBtn").disabled = true;
    this.showGenStep("build");
    log.textContent = "正在按教案生成课程（写稿、画图、教具、配音）…";
    try {
      const res = await fetch("/api/v1/build", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ doc_key: this.genDoc.doc_key, plan: this.genPlan, mode }) });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const job = await this.pollJob((await res.json()).job_id, log);
      if (job.cost) this.toast(`本课生成花费 ≈ $${Number(job.cost.cost_usd).toFixed(3)}，${job.cost.calls} 次调用`);
      await this.refreshCourses();
      $("generateModal").classList.remove("open");
      this.showGenStep("import");
      await this.loadCourse(job.course_id);
    } catch (e) {
      log.textContent += `\n生成失败：${e.message}`;
      this.showGenStep("plan");
    } finally {
      $("buildBtn").disabled = false;
    }
  }

  toast(text, isError = false) {
    const t = document.createElement("div");
    t.className = `toast${isError ? " error" : ""}`;
    t.textContent = text;
    $("toasts").appendChild(t);
    setTimeout(() => t.remove(), 5000);
  }
}

window.app = new App();
