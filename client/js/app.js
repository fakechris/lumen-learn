/**
 * Socratic Whiteboard client controller.
 *
 * Consumes the action stream over WebSocket, drives the board and the
 * audio clock, and sends acks / answers / interjections back.
 */
import { WhiteboardSocket } from "./ws.js";
import { AudioClock, playClip } from "./audio-clock.js";
import { Whiteboard } from "./board.js";
import { escapeHtml } from "./markdown.js";
import { ExerciseView } from "./exercises.js";

const $ = (id) => document.getElementById(id);
const SPEEDS = [1.0, 1.25, 1.5, 2.0];
const ASK_HINT = "点击一个选项，或直接输入你的答案 / 提问";

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
    this.rawLog = [];
    this.activeTab = "transcript";
    this.firstBoardPending = false;

    this.speakText = new Map();       // step_id -> text
    this.stepKinds = new Map();       // step_id -> Set of media kinds seen before its speak
    this.keypoints = [];
    this.pendingKinds = new Set();
    this.pendingDecos = new Map();    // during_step -> [decoration]
    this.currentStep = null;          // step_id of playing tts
    this.interject = null;            // { id, bubble, text, audioPlayed }

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

  async loadCourse(courseId) {
    this.course = await (await fetch(`/api/v1/courses/${courseId}`)).json();
    $("courseSelect").value = courseId;
    this.sessions = this.course.chapters.flatMap((ch) => ch.sessions.map((s) => ({ ...s, chapter: ch.title })));
    this.renderSidebar();
    if (this.sessions.length) this.startSession(this.sessions[0].session_id);
  }

  startSession(sessionId, fromStep = null) {
    this.sessionId = sessionId;
    this.clock.stop();
    this.board.clear();
    this.speakText.clear();
    this.pendingDecos.clear();
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
    this.ws.send({ type: "start_session", course_id: this.course.course_id, session_id: sessionId, from_step_id: fromStep, tts_speed: this.speed });
    this.renderSidebar();
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
    this.addBubble("tutor", `本节：${m.title}。目标：${m.learning_goal}`, { kind: "导引" });
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
      el.className = "kp" + (i === idx ? " current" : i < idx || this.state === "finished" ? " done" : "");
      el.innerHTML = `<span>${escapeHtml(k.title)}</span><span class="dot"></span>`;
      box.appendChild(el);
    });
  }

  noteKind(kind) { this.pendingKinds.add(kind); }

  on_new_page(m) { this.board.newPage(m.title); this.ack(m.step_id); }
  on_new_column() { this.board.newColumn(); }

  on_board(m) {
    this.noteKind("板书");
    this.board.addBoard({ uid: m.board_uid, title: m.title, markdown: m.board_content, layout: m.layout, gate: m.reveal_gate_step, hook: this.firstBoardPending });
    this.firstBoardPending = false;
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
    this.noteKind("互动动画");
    this.board.addWidget({ uid: m.board_uid, title: m.title, html: m.html, layout: m.layout, gate: m.reveal_gate_step });
    this.ack(m.step_id);
  }

  on_animation_pending(m) { this.board.addPlaceholder({ uid: m.board_uid, text: `教具生成中：${m.task_preview}`, layout: m.layout }); }
  on_animation_failed(m) { this.toast(`教具生成失败，已跳过（${m.reason}）`); }

  on_speak(m) {
    this.speakText.set(m.step_id, m.spoken_text);
    const kinds = new Set(this.pendingKinds);
    if (this.speakText.size === 1) kinds.add("导引");
    this.stepKinds.set(m.step_id, kinds);
    this.pendingKinds = new Set();
  }

  on_highlight(m) { this.queueDecoration(m); }
  on_circle(m) { this.queueDecoration(m); }
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
    this.addBubble("tutor", text, { kind: kinds.length ? kinds.join(" · ") : (m.step_id >= 100000 ? "点评" : "讲解") });
    this.renderKeypoints(m.step_id);
    this.board.openGate(m.step_id, m.duration_ms);
    const decos = this.pendingDecos.get(m.step_id) || [];
    const chars = Array.from(text);
    this.clock.play({
      url: m.audio_url,
      durationMs: m.duration_ms,
      onTick: (progress, ms) => {
        const n = Math.floor(progress * chars.length);
        this.setSubtitle(chars.slice(0, n).join(""), progress < 1);
        for (const d of decos) if (!d.drawn && ms >= d.at_ms) this.drawNow(d);
      },
      onEnded: () => {
        for (const d of decos) if (!d.drawn) this.drawNow(d);
        this.setSubtitle(text, false);
        this.currentStep = null;
        this.ack(m.step_id);
      },
    });
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
    this.clock.pause();
    this.ws.send({ type: "interject_start", step_id: this.currentStep, offset_ms: Math.round(this.clock.currentMs) });
    this.interject = { id: null, bubble: null, text: "", audioPlayed: false, done: false };
    $("interjectBox").classList.add("open");
    $("interjectInput").value = "";
    $("interjectInput").focus();
  }

  sendInterject() {
    const text = $("interjectInput").value.trim();
    if (!text || !this.interject) return;
    $("interjectBox").classList.remove("open");
    this.addBubble("student", text);
    this.interject.bubble = this.addBubble("tutor", "", { streaming: true });
    this.ws.send({ type: "interject_question", text });
  }

  cancelInterject() {
    $("interjectBox").classList.remove("open");
    if (!this.interject) return;
    this.interject = null;
    this.ws.send({ type: "interject_resume" });
    this.clock.resume();
  }

  on_interject_ready(m) { if (this.interject) this.interject.id = m.interject_id; }

  on_interject_text(m) {
    if (!this.interject?.bubble) return;
    this.interject.text += m.delta;
    this.interject.bubble.text = this.interject.text;
    if (this.activeTab === "transcript") this.renderSidebar();
  }

  async on_interject_audio(m) {
    if (!this.interject) return;
    this.interject.bubble.text = m.text;
    this.interject.bubble.streaming = false;
    this.renderSidebar();
    this.setSubtitle(m.text, false);
    await playClip(m.audio_url, m.duration_ms);
    this.interject.audioPlayed = true;
    this.finishInterject();
  }

  on_interject_done() {
    if (!this.interject) return;
    this.interject.done = true;
    if (!this.interject.audioPlayed) {
      // No audio arrived (error path): give the reader a moment, then resume.
      setTimeout(() => this.finishInterject(), 1500);
    }
  }

  finishInterject() {
    if (!this.interject) return;
    if (this.interject.bubble) this.interject.bubble.streaming = false;
    this.interject = null;
    this.ws.send({ type: "interject_resume" });
    this.clock.resume();
    this.renderSidebar();
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
