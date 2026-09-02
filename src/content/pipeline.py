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
import os
import shutil
import sys
from typing import Callable, Dict, List, Optional

from src.content.compiler import StepAudio, compile_session
from src.content.curriculum_planner import plan_course_heuristic, plan_course_llm
from src.content.document_parser import ParsedDocument, parse_file, parse_markdown
from src.content.illustration_generator import fill_illustration
from src.content.session_synthesizer import synthesize_session_heuristic, synthesize_session_llm
from src.content.store import write_package
from src.content.validators import sanitize_script
from src.content.widget_check import available as widget_check_available, render_check
from src.content.widget_generator import generate_widget_html
from src.llm.client import LLMClient, make_client
from src.protocol.session import (
    ChapterOutline, CompiledSession, CourseStructure, SessionOutline, SessionScript, stable_id,
)
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
        self.progress = progress
        self._tts_sem = asyncio.Semaphore(tts_concurrency)

    @property
    def mode(self) -> str:
        return "llm" if self.llm else "heuristic"

    # ------------------------------------------------------------------ stages

    def ingest_file(self, path: str, title: Optional[str] = None) -> tuple[str, ParsedDocument]:
        key = self.docs.key_for_file(path)
        doc = parse_file(path, title=title, assets_dir=os.path.join(self.docs.dir_for(key), "figures"))
        self.docs.save(key, doc)
        self.progress("parse", f"{doc.title}: {len(doc.sections)} sections, {doc.total_pages or '-'} pages, {len(doc.figures)} figures")
        return key, doc

    def ingest_text(self, content: str, title: Optional[str] = None) -> tuple[str, ParsedDocument]:
        key = self.docs.key_for_text(content)
        doc = parse_markdown(content, title=title)
        self.docs.save(key, doc)
        self.progress("parse", f"{doc.title}: {len(doc.sections)} sections")
        return key, doc

    async def plan(self, key: str, doc: ParsedDocument) -> CourseStructure:
        course = await plan_course_llm(doc, self.llm, progress=self.progress) if self.llm else plan_course_heuristic(doc)
        outlines = course.all_sessions()
        n_seg = sum(len(s.segments) for s in outlines)
        media: Dict[str, int] = {}
        for s in outlines:
            for seg in s.segments:
                media[seg.media] = media.get(seg.media, 0) + 1
        self.progress("plan", f"{len(course.chapters)} chapters, {len(outlines)} sessions, {n_seg} segments "
                              f"[{course.generation_mode}] media={media}")
        self.docs.save_plan(key, course)
        return course

    async def build(self, doc: ParsedDocument, course: CourseStructure) -> str:
        course_dir = os.path.join(self.output_root, course.course_id)
        scripts: List[SessionScript] = []
        compiled: List[CompiledSession] = []
        for outline in course.all_sessions():
            script = await self._script_for(outline, course, doc)
            self.progress("script", f"{outline.session_id} {outline.title}: {len(script.steps)} steps")
            script = await self._fill_widgets(script, course_dir)
            script = await self._fill_illustrations(script, course_dir, doc)
            audio = await self._synthesize_audio(script, course_dir)
            session = compile_session(script, audio, generation_mode=course.generation_mode)
            scripts.append(script)
            compiled.append(session)
            self.progress("compile", f"{outline.session_id}: {len(session.actions)} actions, {session.total_duration_ms / 1000:.0f}s audio")
        write_package(course_dir, course, scripts, compiled)
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

    # ------------------------------------------------------------------ internals

    async def _script_for(self, outline: SessionOutline, course: CourseStructure, doc: ParsedDocument) -> SessionScript:
        ids = list(dict.fromkeys(outline.source_sections + [s for seg in outline.segments for s in seg.source_sections]))
        source = doc.section_text(ids)
        if self.llm:
            script, warnings = await synthesize_session_llm(outline, course, source, self.llm)
            for w in warnings:
                self.progress("warn", w)
            return script
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
            html = await generate_widget_html(w, self.llm)
            png = os.path.join(course_dir, "widgets", f"{script.session_id}_step_{i + 1}.png")
            if html and check:
                res = await render_check(html, png)
                if not res.ok:
                    self.progress("warn", f"{script.session_id} step {i + 1}: widget {res.problem}; regenerating")
                    html = await generate_widget_html(w, self.llm, feedback=res.problem)
                    res = await render_check(html, png) if html else res
                    if not res.ok:
                        html = None
            if html:
                script.steps[i].widget = w.model_copy(update={"html": html})
                self.progress("widget", f"{script.session_id} step {i + 1}: {w.kind} ok ({len(html)} bytes)")
            else:
                script.steps[i].widget = None
                self.progress("warn", f"{script.session_id} step {i + 1}: widget generation failed; dropped")

        await asyncio.gather(*(fill(i) for i in range(len(script.steps))))
        return script

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
            async with self._tts_sem:
                res = await self.tts.synthesize(script.steps[i].spoken_text, os.path.join(audio_dir, f"step_{i + 1}"))
            url = None
            if res.audio_path:
                url = f"/courses/{script.course_id}/audio/{script.session_id}/{os.path.basename(res.audio_path)}"
            return StepAudio(url, res.duration_ms, res.cjk, res.latin)

        results = await asyncio.gather(*(one(i) for i in range(len(script.steps))))
        return dict(enumerate(results))


def _print_progress(stage: str, detail: str) -> None:
    icons = {"parse": "📄", "plan": "🧠", "script": "✍️", "widget": "🎲", "figure": "🖼️", "compile": "🎬", "warn": "⚠️", "done": "✅"}
    print(f"{icons.get(stage, '•')} [{stage}] {detail}", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Socratic whiteboard content pipeline")
    p.add_argument("--input", help="Lecture file (md/txt/pdf)")
    p.add_argument("--script", help="Hand-authored SessionScript JSON to compile instead of --input")
    p.add_argument("--doc", help="Existing doc key under <output>/_docs (use with --from-plan)")
    p.add_argument("--plan-only", action="store_true", help="Ingest and plan, write plan.json, stop")
    p.add_argument("--from-plan", help="Build from an (edited) plan.json")
    p.add_argument("--title", default=None)
    p.add_argument("--output", default="output")
    p.add_argument("--mode", default="auto", choices=["auto", "llm", "heuristic"])
    p.add_argument("--tts", default=None, help="say|edge|silent|auto (default: TTS_ENGINE env or auto)")
    args = p.parse_args(argv)

    if not (args.input or args.script or (args.doc and args.from_plan)):
        p.error("need --input, --script, or --doc with --from-plan")
    llm = make_client() if args.mode != "heuristic" else None
    if args.mode == "auto" and llm is None:
        print("ℹ️  no LLM key found; running heuristic walk-through mode", flush=True)
    pipeline = ContentPipeline(args.output, llm=llm, tts=choose_engine(args.tts), mode=args.mode,
                               progress=_print_progress)
    print(f"ℹ️  LLM: {llm.config.provider + '/' + llm.model if llm else 'none'}   TTS: {pipeline.tts.name}", flush=True)

    async def run():
        if args.script:
            return await pipeline.run_script(args.script)
        if args.from_plan:
            if args.input:
                _, doc = pipeline.ingest_file(args.input, title=args.title)
            else:
                doc = pipeline.docs.load(args.doc)
                if doc is None:
                    raise SystemExit(f"no ingested document under {pipeline.docs.dir_for(args.doc)}")
            with open(args.from_plan, encoding="utf-8") as f:
                plan = CourseStructure.model_validate_json(f.read())
            return await pipeline.build(doc, plan)
        key, doc = pipeline.ingest_file(args.input, title=args.title)
        plan = await pipeline.plan(key, doc)
        print(f"📝 plan written to {os.path.join(pipeline.docs.dir_for(key), 'plan.json')}  (doc key: {key})", flush=True)
        if args.plan_only:
            return None
        return await pipeline.build(doc, plan)

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
