/**
 * Post-session exercises view (课后习题).
 *
 * fill_blank    -> inline input inside the stem, graded semantically server-side
 * single_choice -> 2x2 option cards with shape icons, local compare + explanation
 * interactive   -> sandboxed widget above the options, then choose
 */
import { renderMarkdownInto, escapeHtml } from "./markdown.js";
import { createWidgetFrame } from "./widgets.js";
import { playClip } from "./audio-clock.js";

const $ = (id) => document.getElementById(id);
const SHAPES = ["▲", "◆", "●", "■"];
const KIND_LABEL = { fill_blank: "填空题", single_choice: "单选题", interactive: "互动" };

export class ExerciseView {
  constructor() {
    this.root = $("exerciseView");
    this.items = [];
    this.index = 0;
    this.courseId = null;
    this.sessionId = null;
    this.results = new Map(); // exercise_id -> { correct, feedback }
    $("exClose").addEventListener("click", () => this.close());
    $("exPrev").addEventListener("click", () => this.go(-1));
    $("exNext").addEventListener("click", () => this.go(1));
    $("exCheck").addEventListener("click", () => this.check());
    $("exRead").addEventListener("click", () => this.readAloud());
  }

  async open(courseId, sessionId) {
    const res = await fetch(`/api/v1/courses/${courseId}/sessions/${sessionId}/exercises`);
    const data = res.ok ? await res.json() : { exercises: [] };
    this.items = data.exercises || [];
    this.courseId = courseId;
    this.sessionId = sessionId;
    this.results.clear();
    this.index = 0;
    if (!this.items.length) return false;
    this.root.classList.add("open");
    this.render();
    return true;
  }

  close() { this.root.classList.remove("open"); }

  go(delta) {
    const next = this.index + delta;
    if (next < 0 || next >= this.items.length) { if (next >= this.items.length) this.close(); return; }
    this.index = next;
    this.render();
  }

  current() { return this.items[this.index]; }

  render() {
    const ex = this.current();
    const done = this.results.get(ex.exercise_id);
    $("exKind").textContent = KIND_LABEL[ex.kind] || ex.kind;
    $("exCounter").textContent = `${this.index + 1} / ${this.items.length}`;

    // stem (with inline blank for fill_blank)
    const stem = $("exStem");
    if (ex.kind === "fill_blank") {
      const [before, after = ""] = ex.stem.split("____");
      stem.innerHTML = "";
      const b = document.createElement("span"); renderMarkdownInto(b, before); b.classList.add("inline-md");
      const input = document.createElement("input");
      input.className = "blank"; input.id = "exBlank"; input.placeholder = "填写答案";
      input.addEventListener("keydown", (e) => { if (e.key === "Enter") this.check(); });
      const a = document.createElement("span"); renderMarkdownInto(a, after); a.classList.add("inline-md");
      stem.append(b, input, a);
      if (done) { input.value = done.answer_text || ""; input.disabled = true; input.classList.add(done.correct ? "ok" : "bad"); }
    } else {
      renderMarkdownInto(stem, ex.stem);
    }

    // widget
    const wbox = $("exWidget");
    wbox.innerHTML = "";
    wbox.style.display = "none";
    if (ex.kind === "interactive" && ex.widget?.html) {
      wbox.style.display = "";
      const frame = createWidgetFrame({ html: ex.widget.html, title: ex.widget.title, height: 340 });
      wbox.appendChild(frame);
      if (ex.widget_hint) {
        const hint = document.createElement("div"); hint.className = "ex-hint"; hint.textContent = ex.widget_hint; wbox.appendChild(hint);
      }
    }

    // options
    const opts = $("exOptions");
    opts.innerHTML = "";
    this.selected = done ? done.answer_index : null;
    if (ex.kind !== "fill_blank") {
      ex.options.forEach((text, i) => {
        const card = document.createElement("button");
        card.className = "ex-option";
        card.innerHTML = `<span class="shape s${i}">${SHAPES[i] || "•"}</span><span class="txt">${escapeHtml(text)}</span><span class="num">${i + 1}</span>`;
        if (done) {
          if (i === ex.correct_index) card.classList.add("correct");
          if (i === done.answer_index && !done.correct) card.classList.add("wrong");
          card.disabled = true;
        } else {
          card.addEventListener("click", () => {
            this.selected = i;
            opts.querySelectorAll(".ex-option").forEach((c) => c.classList.remove("selected"));
            card.classList.add("selected");
            $("exCheck").disabled = false;
          });
        }
        opts.appendChild(card);
      });
    }

    // verdict panel
    const panel = $("exVerdict");
    panel.innerHTML = "";
    if (done) this.renderVerdict(done, ex);

    $("exCheck").style.display = done ? "none" : "";
    $("exCheck").disabled = ex.kind !== "fill_blank" && this.selected == null;
    $("exNext").textContent = this.index + 1 < this.items.length ? "下一题" : "完成";
    $("exNext").style.display = done ? "" : "none";
    $("exPrev").disabled = this.index === 0;
  }

  renderVerdict(done, ex) {
    const panel = $("exVerdict");
    const pill = document.createElement("div");
    pill.className = `verdict ${done.correct ? "ok" : "bad"}`;
    pill.textContent = done.correct ? "✓ 正确" : "✕ 错误";
    panel.appendChild(pill);
    if (!done.correct || ex.kind !== "fill_blank") {
      const ans = document.createElement("div");
      ans.className = "answer";
      ans.innerHTML = `答案：<b>${escapeHtml(done.answer ?? "")}</b>`;
      panel.appendChild(ans);
    }
    const fb = document.createElement("div");
    fb.className = "feedback";
    renderMarkdownInto(fb, done.feedback || "");
    panel.appendChild(fb);
    if (done.explanation && done.feedback !== done.explanation) {
      const ex2 = document.createElement("div");
      ex2.className = "explanation";
      renderMarkdownInto(ex2, done.explanation);
      panel.appendChild(ex2);
    }
  }

  async check() {
    const ex = this.current();
    const body = { course_id: this.courseId, session_id: this.sessionId, exercise_id: ex.exercise_id };
    if (ex.kind === "fill_blank") body.answer_text = $("exBlank").value.trim();
    else body.answer_index = this.selected;
    $("exCheck").disabled = true;
    $("exCheck").textContent = "判卷中…";
    try {
      const res = await fetch("/api/v1/grade", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const data = await res.json();
      this.results.set(ex.exercise_id, { ...data, answer_text: body.answer_text, answer_index: body.answer_index });
    } finally {
      $("exCheck").textContent = "检查";
    }
    this.render();
  }

  async readAloud() {
    const ex = this.current();
    const text = ex.stem.replace(/____/g, "空格").replace(/\$[^$]*\$/g, (m) => m.slice(1, -1));
    const res = await fetch("/api/v1/tts", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
    if (!res.ok) return;
    const { audio_url, duration_ms } = await res.json();
    await playClip(audio_url, duration_ms);
  }
}
