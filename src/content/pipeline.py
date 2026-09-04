"""
End-to-end content pipeline, in three explicit stages so a lesson plan can be
reviewed (and edited) before anything expensive is generated:

  ingest(path|text)  -> ParsedDocument   saved to <output>/_docs/<doc_key>/doc.json (+ figures/)
  plan(doc)          -> CourseStructure  (教案: chapters -> sessions -> segments with media decisions)
  build(doc, plan)   -> course package   (scripts, widgets, figures, audio, compiled sessions)

Usage:
  python -m src.content.pipeline --input book.pdf                   # ingest + plan + build
  python -m src.content.pipeline --input book.pdf --plan-only       # writes plan.json, stops
  python -m src.content.pipeline --doc <doc_key> --from-plan plan.json
  python -m src.content.pipeline --script examples/authored/x.json  # authored session
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
from typing import Callable, Dict, List, Optional

from src.content.compiler import StepAudio, compile_session
from src.content.curriculum_planner import plan_course_heuristic, plan_course_llm
from src.content.document_parser import ParsedDocument, parse_file, parse_markdown
from src.content.exercise_generator import generate_exercises
from src.content.illustration_generator import fill_illustration
from src.content.qa import qa_session
from src.content.session_synthesizer import synthesize_session_heuristic, synthesize_session_llm
from src.content.store import CourseStore, write_package
from src.content.validators import sanitize_script
from src.content.widget_check import available as widget_check_available, render_check, vision_review
from src.content.widget_generator import generate_widget_html
from src.llm.client import LLMClient, make_client
from src.protocol.session import (
    ChapterOutline, CompiledSession, CourseStructure, SessionOutline, SessionScript, stable_id,
)
from src.llm.usage import GLOBAL_LEDGER
from src.obs.db import get_db
from src.obs.log import Run, start_run
from src.tts.align import synthesize_aligned
from src.tts.engine import TtsEngine, choose_engine

ProgressFn = Callable[[str, str], None]


def _noop_progress(stage: str, detail: str) -> None:
    pass


class DocStore:
    """Parsed documents (with extracted figures) persisted under <output>/_docs/<key>/."""

    def __init__(self, output_root: str):
        self.root = os.path.join(output_root, "_docs")

    def dir_for(self, key: str) -> str:
        return os.path.join(self.root, key)

    @staticmethod
    def key_for_file(path: str) -> str:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return "doc_" + h.hexdigest()[:12]

    @staticmethod
    def key_for_text(text: str) -> str:
        return "doc_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    def save(self, key: str, doc: ParsedDocument) -> None:
        doc.save(os.path.join(self.dir_for(key), "doc.json"))

    def load(self, key: str) -> Optional[ParsedDocument]:
        path = os.path.join(self.dir_for(key), "doc.json")
        return ParsedDocument.load(path) if os.path.isfile(path) else None

    def save_plan(self, key: str, plan: CourseStructure) -> str:
        path = os.path.join(self.dir_for(key), "plan.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(plan.model_dump_json(indent=2))
        return path

    def load_plan(self, key: str) -> Optional[CourseStructure]:
        path = os.path.join(self.dir_for(key), "plan.json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            return CourseStructure.model_validate_json(f.read())


class ContentPipeline:
    def __init__(self, output_root: str = "output", llm: Optional[LLMClient] = None,
                 tts: Optional[TtsEngine] = None, mode: str = "auto", tts_concurrency: int = 3,
                 progress: ProgressFn = _noop_progress):
        if mode not in ("auto", "llm", "heuristic"):
            raise ValueError("mode must be auto|llm|heuristic")
        if mode == "llm" and llm is None:
            raise RuntimeError("mode=llm but no LLM is configured (set DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)")
        self.output_root = os.path.abspath(output_root)
        self.docs = DocStore(self.output_root)
        self.llm = llm if mode != "heuristic" else None
        self.tts = tts or choose_engine()
        self._progress = progress
        self.run: Optional[Run] = None
        self._tts_sem = asyncio.Semaphore(tts_concurrency)

    @property
    def run_id(self) -> Optional[str]:
        return self.run.run_id if self.run else None

    def progress(self, stage: str, detail: str, session_id: Optional[str] = None) -> None:
        if self.run:
            self.run.progress(stage, detail, session_id=session_id)
        else:
            self._progress(stage, detail)

    def begin_run(self, kind: str, course_id: Optional[str] = None, doc_key: Optional[str] = None, scope: str = "",
                  echo: bool = True) -> Run:
        # Console echo comes from the Run; only non-console sinks (server jobs) are forwarded.
        sink = self._progress if (self._progress is not _noop_progress and self._progress is not _print_progress) else None
        self.run = start_run(kind, self.output_root, course_id=course_id, doc_key=doc_key, scope=scope, echo=echo,
                             extra_sink=sink)
        self.run._ledger_mark = GLOBAL_LEDGER.mark()
        return self.run

    def end_run(self, status: str, error: Optional[str] = None, course_id: Optional[str] = None) -> None:
        if self.run:
            self.run.finish(status, GLOBAL_LEDGER.summary(since=getattr(self.run, "_ledger_mark", 0))["total"], error, course_id)
            self.run = None

    @property
    def mode(self) -> str:
        return "llm" if self.llm else "heuristic"

    # ------------------------------------------------------------------ stages

    def ingest_file(self, path: str, title: Optional[str] = None) -> tuple[str, ParsedDocument]:
        key = self.docs.key_for_file(path)
        doc = parse_file(path, title=title, assets_dir=os.path.join(self.docs.dir_for(key), "figures"))
        self.docs.save(key, doc)
        get_db(self.output_root).upsert_document(key, doc.title, doc.source_path, doc.total_pages, len(doc.sections),
                                                 len(doc.figures), doc.char_count())
        self.progress("parse", f"{doc.title}: {len(doc.sections)} sections, {doc.total_pages or '-'} pages, {len(doc.figures)} figures")
        return key, doc

    async def describe_figures(self, key: str, doc: ParsedDocument) -> ParsedDocument:
        """Caption textbook figures that have no caption, using the vision model, so the
        planner can decide when to reuse them."""
        if not self.llm or not self.llm.has_vision:
            return doc
        todo = [f for f in doc.figures if not f.caption and os.path.isfile(f.path)]
        for fig in todo:
            try:
                with open(fig.path, "rb") as fh:
                    png = fh.read()
                text = await self.llm.complete("你是教材图注撰写者。", "用一句话（≤30 字）说明这张教材图画的是什么，直接输出题注文字。",
                                               temperature=0.2, images=[png], purpose="caption")
                fig.caption = text.strip().strip("。").splitlines()[0][:60]
                self.progress("figure", f"{fig.figure_id}: captioned by vision model → {fig.caption}")
            except Exception as e:
                self.progress("warn", f"{fig.figure_id}: vision caption failed: {str(e)[:100]}")
        if todo:
            self.docs.save(key, doc)
        return doc

    def ingest_text(self, content: str, title: Optional[str] = None) -> tuple[str, ParsedDocument]:
        key = self.docs.key_for_text(content)
        doc = parse_markdown(content, title=title)
        self.docs.save(key, doc)
        get_db(self.output_root).upsert_document(key, doc.title, None, 0, len(doc.sections), 0, doc.char_count())
        self.progress("parse", f"{doc.title}: {len(doc.sections)} sections")
        return key, doc

    async def plan(self, key: str, doc: ParsedDocument) -> CourseStructure:
        doc = await self.describe_figures(key, doc)
        course = await plan_course_llm(doc, self.llm, progress=self.progress) if self.llm else plan_course_heuristic(doc)
        outlines = course.all_sessions()
        n_seg = sum(len(s.segments) for s in outlines)
        media: Dict[str, int] = {}
        for s in outlines:
            for seg in s.segments:
                media[seg.media] = media.get(seg.media, 0) + 1
        self.progress("plan", f"{len(course.chapters)} chapters, {len(outlines)} sessions, {n_seg} segments "
                              f"[{course.generation_mode}] media={media}")
        if self.llm:
            est = self.estimate_build_cost(course)
            self.progress("estimate", f"生成本课粗估 ≈ ${est['usd']} · {est['segments']} 段 · {est['widgets']} 教具 · {est['figures']} 图")
        self.docs.save_plan(key, course)
        get_db(self.output_root).upsert_plan(course.course_id, key, course.title, len(course.chapters), len(outlines), n_seg,
                                             course.model_dump_json())
        return course

    @staticmethod
    def estimate_build_cost(course: CourseStructure) -> dict:
        """Rough pre-build estimate from the plan (labelled as such in the UI)."""
        segs = sum(len(s.segments) for s in course.all_sessions())
        n_sessions = len(course.all_sessions())
        widgets = sum(1 for s in course.all_sessions() for g in s.segments if g.media in ("explorable", "threejs"))
        figs = sum(1 for s in course.all_sessions() for g in s.segments if g.media in ("illustration", "mermaid"))
        prices = GLOBAL_LEDGER.llm_prices.get(os.getenv("LLM_MODEL", "deepseek-v4-flash"), [0.27, 1.10])
        tokens_in = segs * 2500 + widgets * 4000 + figs * 1500 + n_sessions * 3500
        tokens_out = segs * 700 + widgets * 3000 + figs * 1500 + n_sessions * 1500
        usd = (tokens_in * prices[0] + tokens_out * prices[1]) / 1_000_000
        return {"segments": segs, "widgets": widgets, "figures": figs, "tokens_in": tokens_in, "tokens_out": tokens_out,
                "usd": round(usd, 4)}

    async def build(self, doc: ParsedDocument, course: CourseStructure, only: Optional[set] = None,
                    chapters: Optional[set] = None, qa: bool = True, resume: bool = True,
                    max_attempts: int = 2) -> str:
        """Build the package. `only` = session ids, `chapters` = chapter ids to (re)generate;
        everything else is kept from the existing package. With `resume`, sessions that already
        passed QA in the DB are kept too. Each generated session is QA-checked and regenerated
        once with the issues as feedback if it fails."""
        course_dir = os.path.join(self.output_root, course.course_id)
        ledger_mark = GLOBAL_LEDGER.mark()
        db = get_db(self.output_root)
        existing = CourseStore([self.output_root])
        scripts: List[SessionScript] = []
        compiled: List[CompiledSession] = []
        chapter_of = {s.session_id: ch.chapter_id for ch in course.chapters for s in ch.sessions}
        run_id = getattr(self, "run_id", None)
        for outline in course.all_sessions():
            sid = outline.session_id
            selected = (only is None or sid in only) and (chapters is None or chapter_of.get(sid) in chapters)
            prior = db.session(course.course_id, sid) if resume else None
            keep = (not selected) or (resume and only is None and prior and prior.get("qa_pass"))
            if keep:
                old_script = existing.get_script(course.course_id, sid)
                old_session = existing.get_session(course.course_id, sid)
                if old_script and old_session:
                    scripts.append(old_script)
                    compiled.append(old_session)
                    if selected:
                        self.progress("skip", f"{sid}: already passed QA; kept", session_id=sid)
                    continue
                if not selected:
                    continue  # never generated and not selected: leave out of this build
            sess_mark = GLOBAL_LEDGER.mark()
            script, report, attempts = None, None, 0
            feedback: Optional[str] = None
            while attempts < max_attempts:
                attempts += 1
                script = await self._script_for(outline, course, doc, feedback=feedback)
                self.progress("script", f"{sid} {outline.title}: {len(script.steps)} steps (attempt {attempts})", session_id=sid)
                script = await self._fill_widgets(script, course_dir)
                script = await self._fill_illustrations(script, course_dir, doc)
                script = await self._fill_exercises(script, course_dir)
                if not qa:
                    break
                report = await qa_session(script, outline, self.llm)
                self.progress("qa", f"{sid}: score {report.score} {'PASS' if report.passed else 'FAIL'}"
                                    + (f" · {len(report.issues)} issues: " + " | ".join(report.issues[:4]) if report.issues else ""),
                              session_id=sid)
                if report.passed or attempts >= max_attempts:
                    break
                feedback = "上一次生成经审核发现以下问题，请在这次生成中修正：\n- " + "\n- ".join(report.issues[:8])
                self.progress("retry", f"{sid}: regenerating with QA feedback", session_id=sid)
            audio = await self._synthesize_audio(script, course_dir)
            session = compile_session(script, audio, generation_mode=course.generation_mode)
            scripts.append(script)
            compiled.append(session)
            sess_cost = GLOBAL_LEDGER.summary(since=sess_mark)["total"]["cost_usd"]
            db.upsert_session(course.course_id, sid, chapter_id=chapter_of.get(sid), title=outline.title,
                              steps=len(script.steps),
                              widgets=sum(1 for st in script.steps if st.widget and (st.widget.html or st.widget.mermaid)),
                              figures=sum(1 for st in script.steps if st.illustration and (st.illustration.svg or st.illustration.image_url)),
                              exercises=len(script.exercises), warnings=len(report.issues) if report else 0,
                              duration_ms=session.total_duration_ms, cost_usd=sess_cost,
                              qa_score=report.score if report else None, qa_pass=int(report.passed) if report else 1,
                              qa_json=report.to_dict() if report else None, attempts=attempts, run_id=run_id)
            self.progress("compile", f"{sid}: {len(session.actions)} actions, {session.total_duration_ms / 1000:.0f}s audio, "
                                     f"${sess_cost:.3f}", session_id=sid)
        write_package(course_dir, course, scripts, compiled)
        usage = GLOBAL_LEDGER.summary(since=ledger_mark)
        with open(os.path.join(course_dir, "cost.json"), "w", encoding="utf-8") as f:
            json.dump(usage, f, ensure_ascii=False, indent=1)
        t = usage["total"]
        self.progress("cost", f"{GLOBAL_LEDGER.format_usd(t['cost_usd'])} ({'按估算单价' if t['estimated_price'] else '按配置单价'}) · "
                              f"{t['calls']} 次调用 · {t['prompt_tokens'] + t['completion_tokens'] + t['reasoning_tokens']} tokens · "
                              f"TTS {t['tts_chars']} 字 · {t['seconds']}s 模型耗时")
        self.progress("done", course_dir)
        return course_dir

    # ------------------------------------------------------------------ convenience

    async def run_file(self, input_path: str, title: Optional[str] = None) -> str:
        key, doc = self.ingest_file(input_path, title=title)
        return await self.build(doc, await self.plan(key, doc))

    async def run_text(self, content: str, title: Optional[str] = None) -> str:
        key, doc = self.ingest_text(content, title=title)
        return await self.build(doc, await self.plan(key, doc))

    async def run_document(self, doc: ParsedDocument) -> str:
        key = self.docs.key_for_text(doc.raw_markdown)
        self.docs.save(key, doc)
        return await self.build(doc, await self.plan(key, doc))

    async def run_script(self, script_path: str) -> str:
        """Compile a hand-authored SessionScript (mode "authored") into a package."""
        with open(script_path, encoding="utf-8") as f:
            raw = f.read()
        script = SessionScript.model_validate_json(raw)
        base = os.path.dirname(os.path.abspath(script_path))
        for step in script.steps:
            w = step.widget
            if w and w.html_path and not w.html:
                with open(os.path.join(base, w.html_path), encoding="utf-8") as f:
                    step.widget = w.model_copy(update={"html": f.read()})
            il = step.illustration
            if il and il.svg_path and not il.svg:
                with open(os.path.join(base, il.svg_path), encoding="utf-8") as f:
                    step.illustration = il.model_copy(update={"svg": f.read()})
        course_id = stable_id("course", "authored", script.session_id, raw)
        script = script.model_copy(update={"course_id": course_id})
        course_dir = os.path.join(self.output_root, course_id)
        for i, step in enumerate(script.steps):
            il = step.illustration
            if il and il.image_path and not il.image_url:
                step.illustration = il.model_copy(update={"image_url": self._copy_image(
                    os.path.join(base, il.image_path), course_dir, course_id, script.session_id, i + 1)})
        script, warnings = sanitize_script(script)
        for w in warnings:
            self.progress("warn", w)
        outline = SessionOutline(session_id=script.session_id, title=script.title, learning_goal=script.learning_goal,
                                 core_concept=script.title)
        course = CourseStructure(course_id=course_id, title=script.title, overview=script.learning_goal,
                                 generation_mode="authored",
                                 chapters=[ChapterOutline(chapter_id="ch_1", title=script.title, sessions=[outline])])
        script = await self._fill_illustrations(script, course_dir, None)
        audio = await self._synthesize_audio(script, course_dir)
        session = compile_session(script, audio, generation_mode="authored")
        self.progress("compile", f"{script.session_id}: {len(session.actions)} actions, {session.total_duration_ms / 1000:.0f}s audio")
        write_package(course_dir, course, [script], [session])
        self.progress("done", course_dir)
        return course_dir

    async def add_variants(self, course_id: str, roots: Optional[List[str]] = None, only: Optional[set] = None,
                           kind: str = "compressed", force: bool = False) -> str:
        """Precompute step variants for an existing package (compressed = for fast learners) so no
        learner waits on generation. Steps whose beat is define/derive/worked_example/contrast/apply."""
        from src.content.adaptive import fill_beats
        from src.content.variants import COMPRESS_BEATS, build_variant, step_context, variant_path
        from src.runtime.tutor import LiveTutor
        if self.llm is None:
            raise SystemExit("variants need an LLM")
        store = CourseStore(roots or [self.output_root])
        course = store.get_course(course_id)
        if course is None:
            raise SystemExit(f"course {course_id} not found")
        course_dir = store._course_dir(course_id)
        tutor = LiveTutor(self.llm)
        made = skipped = 0
        for outline in course.all_sessions():
            if only is not None and outline.session_id not in only:
                continue
            script = store.get_script(course_id, outline.session_id)
            session = store.get_session(course_id, outline.session_id)
            if script is None or session is None:
                continue
            kps = fill_beats(session, script)
            for i, (st, kp) in enumerate(zip(script.steps, kps)):
                if kp.beat not in COMPRESS_BEATS:
                    continue
                if not force and os.path.isfile(variant_path(course_dir, outline.session_id, kp.step_id, kind)):
                    skipped += 1
                    continue
                try:
                    await build_variant(tutor, self.tts, step_context(script, i), kind, course_id, course_dir,
                                        outline.session_id, kp.step_id)
                    made += 1
                    self.progress("variant", f"{outline.session_id} step {kp.step_id} ({kp.beat}) {kind}", outline.session_id)
                except Exception as exc:  # noqa: BLE001
                    self.progress("warn", f"{outline.session_id} step {kp.step_id}: variant failed ({exc})", outline.session_id)
        self.progress("done", f"{made} variants made, {skipped} cached, {course_dir}")
        return course_dir

    async def add_exercises(self, course_id: str, roots: Optional[List[str]] = None, only: Optional[set] = None,
                            missing_only: bool = False) -> str:
        """Generate exercises for an existing package and rewrite its sessions in place.
        `only` limits to session ids; `missing_only` skips sessions that already have exercises."""
        store = CourseStore(roots or [self.output_root])
        course = store.get_course(course_id)
        if course is None:
            raise SystemExit(f"course {course_id} not found")
        course_dir = store._course_dir(course_id)
        scripts, compiled = [], []
        for outline in course.all_sessions():
            script = store.get_script(course_id, outline.session_id)
            session = store.get_session(course_id, outline.session_id)
            if script is None or session is None:
                continue
            selected = (only is None or outline.session_id in only) and not (missing_only and script.exercises)
            if selected:
                script = await self._fill_exercises(script, course_dir)
            scripts.append(script)
            compiled.append(session.model_copy(update={"exercises": list(script.exercises)}))
        write_package(course_dir, course, scripts, compiled)
        self.progress("done", course_dir)
        return course_dir

    # ------------------------------------------------------------------ internals

    async def _fill_exercises(self, script: SessionScript, course_dir: str) -> SessionScript:
        if not self.llm:
            return script
        try:
            exercises, warnings = await generate_exercises(script, self.llm)
        except Exception as e:  # exercises are optional; never fail the build
            self.progress("warn", f"{script.session_id}: exercise generation failed: {e}")
            return script
        for w in warnings:
            self.progress("warn", w)
        check = widget_check_available()
        for ex in exercises:
            if ex.kind == "interactive" and ex.widget and not ex.widget.html:
                problems: List[str] = []
                html = await generate_widget_html(ex.widget, self.llm, problems=problems)
                png = os.path.join(course_dir, "widgets", f"{script.session_id}_{ex.exercise_id}.png")
                if html and check:
                    res = await self._check_widget(html, png, ex.widget.task)
                    if not res.ok:
                        html = await generate_widget_html(ex.widget, self.llm, feedback=res.problem)
                        res = await self._check_widget(html, png, ex.widget.task) if html else res
                        if not res.ok:
                            html = None
                if html:
                    ex.widget = ex.widget.model_copy(update={"html": html})
                else:
                    ex.kind = "single_choice"
                    ex.widget = None
                    self.progress("warn", f"{script.session_id} {ex.exercise_id}: interactive widget failed; downgraded to choice. reasons: "
                                          + " | ".join(p[:160] for p in problems))
        kinds = {}
        for ex in exercises:
            kinds[ex.kind] = kinds.get(ex.kind, 0) + 1
        self.progress("exercise", f"{script.session_id}: {len(exercises)} exercises {kinds}")
        return script.model_copy(update={"exercises": exercises})

    async def _script_for(self, outline: SessionOutline, course: CourseStructure, doc: ParsedDocument,
                          feedback: Optional[str] = None) -> SessionScript:
        ids = list(dict.fromkeys(outline.source_sections + [s for seg in outline.segments for s in seg.source_sections]))
        source = doc.section_text(ids)
        if self.llm:
            last = None
            for attempt in range(2):
                try:
                    script, warnings = await synthesize_session_llm(outline, course, source, self.llm, feedback=feedback)
                    for w in warnings:
                        self.progress("warn", w)
                    return script
                except Exception as e:  # never let one session kill a long build
                    last = e
                    self.progress("warn", f"{outline.session_id}: synthesis attempt {attempt + 1} failed: {str(e)[:200]}")
            self.progress("warn", f"{outline.session_id}: falling back to read-through mode ({last})")
        return synthesize_session_heuristic(outline, course, source)

    async def _fill_widgets(self, script: SessionScript, course_dir: str) -> SessionScript:
        if not self.llm:
            return script
        check = widget_check_available()
        if not check:
            self.progress("warn", "widget render check unavailable (needs node + global playwright); skipping")

        async def fill(i: int):
            w = script.steps[i].widget
            if not w or w.kind == "mermaid" or w.html:
                return
            problems: List[str] = []
            html = await generate_widget_html(w, self.llm, problems=problems)
            png = os.path.join(course_dir, "widgets", f"{script.session_id}_step_{i + 1}.png")
            if html and check:
                res = await self._check_widget(html, png, w.task)
                if not res.ok:
                    self.progress("warn", f"{script.session_id} step {i + 1}: widget {res.problem}; regenerating")
                    html = await generate_widget_html(w, self.llm, feedback=res.problem, problems=problems)
                    res = await self._check_widget(html, png, w.task) if html else res
                    if not res.ok:
                        problems.append(str(res.problem))
                        html = None
            if html:
                script.steps[i].widget = w.model_copy(update={"html": html})
                self.progress("widget", f"{script.session_id} step {i + 1}: {w.kind} ok ({len(html)} bytes)" +
                              (f" after: {problems[-1][:120]}" if problems else ""))
            else:
                script.steps[i].widget = None
                self.progress("warn", f"{script.session_id} step {i + 1}: widget generation failed; dropped. reasons: "
                                      + " | ".join(p[:160] for p in problems))

        await asyncio.gather(*(fill(i) for i in range(len(script.steps))))
        return script

    async def _check_widget(self, html: str, png: str, task: str):
        """Headless render check, then (if a vision model is configured) a visual review
        against the widget's task. Either failure carries a concrete problem back to the generator."""
        res = await render_check(html, png)
        if not res.ok or not self.llm or not self.llm.has_vision or not res.png_path:
            return res
        try:
            passed, notes = await vision_review(self.llm, task, res.png_path)
        except Exception as e:  # vision review is best-effort
            self.progress("warn", f"vision review skipped: {str(e)[:120]}")
            return res
        if not passed:
            res.ok = False
            res.problem = f"visual review: {notes}"
        res.review = notes
        return res

    @staticmethod
    def _copy_image(src: str, course_dir: str, course_id: str, session_id: str, step_no: int) -> str:
        dst_dir = os.path.join(course_dir, "images", session_id)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f"step_{step_no}{os.path.splitext(src)[1] or '.png'}")
        shutil.copyfile(src, dst)
        return f"/courses/{course_id}/images/{session_id}/{os.path.basename(dst)}"

    async def _fill_illustrations(self, script: SessionScript, course_dir: str,
                                  doc: Optional[ParsedDocument]) -> SessionScript:
        async def fill(i: int):
            il = script.steps[i].illustration
            if not il or il.svg or il.image_url:
                return
            if il.kind == "reference":
                fig = doc.figure(il.figure_id) if (doc and il.figure_id) else None
                if fig and os.path.isfile(fig.path):
                    url = self._copy_image(fig.path, course_dir, script.course_id, script.session_id, i + 1)
                    script.steps[i].illustration = il.model_copy(update={"image_url": url, "caption": il.caption or fig.caption})
                    self.progress("figure", f"{script.session_id} step {i + 1}: textbook figure {il.figure_id}")
                else:
                    script.steps[i].illustration = None
                    self.progress("warn", f"{script.session_id} step {i + 1}: textbook figure {il.figure_id!r} not found; dropped")
                return
            out = os.path.join(course_dir, "images", script.session_id, f"step_{i + 1}.png")
            url = f"/courses/{script.course_id}/images/{script.session_id}/step_{i + 1}.png"
            filled = await fill_illustration(il, self.llm, out, url)
            if filled.svg or filled.image_url:
                script.steps[i].illustration = filled
                self.progress("figure", f"{script.session_id} step {i + 1}: {il.kind} ok")
            else:
                script.steps[i].illustration = None
                self.progress("warn", f"{script.session_id} step {i + 1}: illustration ({il.kind}) failed; dropped")

        await asyncio.gather(*(fill(i) for i in range(len(script.steps))))
        return script

    async def _synthesize_audio(self, script: SessionScript, course_dir: str) -> Dict[int, StepAudio]:
        audio_dir = os.path.join(course_dir, "audio", script.session_id)
        os.makedirs(audio_dir, exist_ok=True)

        async def one(i: int) -> StepAudio:
            text = script.steps[i].spoken_text
            stem = os.path.join(audio_dir, f"step_{i + 1}")
            async with self._tts_sem:
                import time
                t0 = time.time()
                try:
                    res = await synthesize_aligned(self.tts, text, stem)
                except Exception as e:  # never let audio kill a build: plain synthesis, then a virtual clock
                    self.progress("warn", f"{script.session_id} step {i + 1}: aligned TTS failed ({str(e)[:120]}); plain synthesis",
                                  session_id=script.session_id)
                    try:
                        res = await self.tts.synthesize(text, stem)
                    except Exception as e2:
                        self.progress("warn", f"{script.session_id} step {i + 1}: TTS failed ({str(e2)[:120]}); silent",
                                      session_id=script.session_id)
                        from src.tts.engine import SilentEngine
                        res = await SilentEngine().synthesize(text, stem)
                GLOBAL_LEDGER.add_tts(self.tts.name, "tts", len(text), time.time() - t0)
            url = None
            if res.audio_path:
                url = f"/courses/{script.course_id}/audio/{script.session_id}/{os.path.basename(res.audio_path)}"
            return StepAudio(url, res.duration_ms, res.cjk, res.latin, res.marks)

        results = await asyncio.gather(*(one(i) for i in range(len(script.steps))))
        return dict(enumerate(results))


def _print_progress(stage: str, detail: str) -> None:
    icons = {"parse": "📄", "plan": "🧠", "estimate": "💰", "script": "✍️", "widget": "🎲", "figure": "🖼️", "exercise": "📝", "compile": "🎬", "skip": "⏭️", "cost": "💰", "warn": "⚠️", "done": "✅"}
    print(f"{icons.get(stage, '•')} [{stage}] {detail}", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Socratic whiteboard content pipeline")
    p.add_argument("--input", help="Lecture file (md/txt/pdf)")
    p.add_argument("--script", help="Hand-authored SessionScript JSON to compile instead of --input")
    p.add_argument("--doc", help="Existing doc key under <output>/_docs (use with --from-plan)")
    p.add_argument("--plan-only", action="store_true", help="Ingest and plan, write plan.json, stop")
    p.add_argument("--from-plan", help="Build from an (edited) plan.json")
    p.add_argument("--exercises-for", help="Generate exercises for an existing course id (in --output or examples/courses)")
    p.add_argument("--variants-for", help="Precompute compressed step variants (for fast learners) for an existing course id")
    p.add_argument("--only", help="Comma-separated session ids to (re)build; others are kept from the existing package")
    p.add_argument("--chapters", help="Comma-separated chapter ids to build (e.g. ch_1,ch_2); use with --from-plan")
    p.add_argument("--all-chapters", action="store_true", help="Build every chapter in order, one run per chapter, resumable")
    p.add_argument("--no-qa", action="store_true", help="Skip per-session QA/retry")
    p.add_argument("--force", action="store_true", help="Regenerate even sessions that already passed QA")
    p.add_argument("--quiet", action="store_true", help="No console progress (events still go to DB/JSONL)")
    p.add_argument("--title", default=None)
    p.add_argument("--output", default="output")
    p.add_argument("--mode", default="auto", choices=["auto", "llm", "heuristic"])
    p.add_argument("--tts", default=None, help="say|edge|silent|auto (default: TTS_ENGINE env or auto)")
    args = p.parse_args(argv)

    if not (args.input or args.script or (args.doc and args.from_plan) or args.exercises_for or args.variants_for):
        p.error("need --input, --script, --doc with --from-plan, --exercises-for, or --variants-for")
    llm = make_client() if args.mode != "heuristic" else None
    if args.mode == "auto" and llm is None:
        print("ℹ️  no LLM key found; running heuristic walk-through mode", flush=True)
    pipeline = ContentPipeline(args.output, llm=llm, tts=choose_engine(args.tts), mode=args.mode,
                               progress=_noop_progress if args.quiet else _print_progress)
    if llm:
        c = llm.config
        print(f"ℹ️  LLM: {c.provider} fast={c.model} pro={c.model_pro or c.model} vision={c.model_vision or '-'}   TTS: {pipeline.tts.name}", flush=True)
    else:
        print(f"ℹ️  LLM: none   TTS: {pipeline.tts.name}", flush=True)

    async def build_with_run(doc, plan, key, **kw):
        scope = ",".join(sorted(kw.get("chapters") or [])) or ",".join(sorted(kw.get("only") or [])) or "all"
        pipeline.begin_run("build", course_id=plan.course_id, doc_key=key, scope=scope, echo=not args.quiet)
        try:
            out = await pipeline.build(doc, plan, **kw)
            pipeline.end_run("done", course_id=plan.course_id)
            return out
        except Exception as e:
            pipeline.end_run("error", error=str(e)[:500], course_id=plan.course_id)
            raise

    get_db(pipeline.output_root).abort_stale_runs()  # a fresh CLI run means earlier CLI runs are dead

    async def run():
        if args.variants_for:
            only = set(args.only.split(",")) if args.only else None
            return await pipeline.add_variants(args.variants_for, roots=[args.output, "examples/courses"], only=only,
                                              force=args.force)
        if args.exercises_for:
            only = set(x.strip() for x in args.only.split(",")) if args.only else None
            return await pipeline.add_exercises(args.exercises_for, roots=[args.output, "examples/courses"], only=only,
                                                missing_only=not args.force)
        if args.script:
            return await pipeline.run_script(args.script)
        if args.from_plan:
            if args.input:
                key, doc = pipeline.ingest_file(args.input, title=args.title)
            else:
                key, doc = args.doc, pipeline.docs.load(args.doc)
                if doc is None:
                    raise SystemExit(f"no ingested document under {pipeline.docs.dir_for(args.doc)}")
            with open(args.from_plan, encoding="utf-8") as f:
                plan = CourseStructure.model_validate_json(f.read())
            only = set(x.strip() for x in args.only.split(",")) if args.only else None
            kw = dict(only=only, qa=not args.no_qa, resume=not args.force)
            if args.all_chapters:
                out = None
                for ch in plan.chapters:
                    print(f"📚 [chapter] {ch.chapter_id} {ch.title} ({len(ch.sessions)} sessions)", flush=True)
                    out = await build_with_run(doc, plan, key, chapters={ch.chapter_id}, **kw)
                return out
            chapters = set(x.strip() for x in args.chapters.split(",")) if args.chapters else None
            return await build_with_run(doc, plan, key, chapters=chapters, **kw)
        key, doc = pipeline.ingest_file(args.input, title=args.title)
        pipeline.begin_run("plan", doc_key=key, scope="plan", echo=not args.quiet)
        try:
            plan = await pipeline.plan(key, doc)
            pipeline.end_run("done", course_id=plan.course_id)
        except Exception as e:
            pipeline.end_run("error", error=str(e)[:500])
            raise
        print(f"📝 plan written to {os.path.join(pipeline.docs.dir_for(key), 'plan.json')}  (doc key: {key})", flush=True)
        if args.plan_only:
            return None
        return await build_with_run(doc, plan, key, qa=not args.no_qa, resume=not args.force)

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
