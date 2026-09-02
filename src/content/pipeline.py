"""
End-to-end content pipeline:
  parse -> plan course -> per session: script -> widgets -> TTS -> compile -> package.

Usage:
  python -m src.content.pipeline --input examples/linear_algebra_basis.md [--mode auto|llm|heuristic] [--tts auto|say|edge|silent]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Callable, Dict, List, Optional

from src.content.compiler import StepAudio, compile_session
from src.content.curriculum_planner import plan_course_heuristic, plan_course_llm
from src.content.document_parser import ParsedDocument, parse_file, parse_markdown
from src.content.session_synthesizer import synthesize_session_heuristic, synthesize_session_llm
from src.content.store import write_package
from src.content.widget_generator import generate_widget_html
from src.llm.client import LLMClient, make_client
from src.protocol.session import CompiledSession, CourseStructure, SessionOutline, SessionScript
from src.tts.engine import TtsEngine, choose_engine

ProgressFn = Callable[[str, str], None]


def _noop_progress(stage: str, detail: str) -> None:
    pass


class ContentPipeline:
    def __init__(self, output_root: str = "output", llm: Optional[LLMClient] = None,
                 tts: Optional[TtsEngine] = None, mode: str = "auto", tts_concurrency: int = 3,
                 progress: ProgressFn = _noop_progress):
        if mode not in ("auto", "llm", "heuristic"):
            raise ValueError("mode must be auto|llm|heuristic")
        if mode == "llm" and llm is None:
            raise RuntimeError("mode=llm but no LLM is configured (set DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)")
        self.output_root = os.path.abspath(output_root)
        self.llm = llm if mode != "heuristic" else None
        self.tts = tts or choose_engine()
        self.progress = progress
        self._tts_sem = asyncio.Semaphore(tts_concurrency)

    @property
    def mode(self) -> str:
        return "llm" if self.llm else "heuristic"

    async def run_file(self, input_path: str, title: Optional[str] = None) -> str:
        return await self.run_document(parse_file(input_path, title=title))

    async def run_text(self, content: str, title: Optional[str] = None) -> str:
        return await self.run_document(parse_markdown(content, title=title))

    async def run_document(self, doc: ParsedDocument) -> str:
        self.progress("parse", f"{doc.title}: {len(doc.sections)} sections")
        course = await plan_course_llm(doc, self.llm) if self.llm else plan_course_heuristic(doc)
        outlines = course.all_sessions()
        self.progress("plan", f"{len(course.chapters)} chapters, {len(outlines)} sessions [{course.generation_mode}]")

        course_dir = os.path.join(self.output_root, course.course_id)
        scripts: List[SessionScript] = []
        compiled: List[CompiledSession] = []
        for outline in outlines:
            script = await self._script_for(outline, course, doc)
            self.progress("script", f"{outline.session_id} {outline.title}: {len(script.steps)} steps")
            script = await self._fill_widgets(script)
            audio = await self._synthesize_audio(script, course_dir)
            session = compile_session(script, audio, generation_mode=course.generation_mode)
            scripts.append(script)
            compiled.append(session)
            self.progress("compile", f"{outline.session_id}: {len(session.actions)} actions, {session.total_duration_ms / 1000:.0f}s audio")

        write_package(course_dir, course, scripts, compiled)
        self.progress("done", course_dir)
        return course_dir

    async def _script_for(self, outline: SessionOutline, course: CourseStructure, doc: ParsedDocument) -> SessionScript:
        source = doc.section_text(outline.source_sections)
        if self.llm:
            script, warnings = await synthesize_session_llm(outline, course, source, self.llm)
            for w in warnings:
                self.progress("warn", w)
            return script
        return synthesize_session_heuristic(outline, course, source)

    async def _fill_widgets(self, script: SessionScript) -> SessionScript:
        if not self.llm:
            return script

        async def fill(i: int):
            w = script.steps[i].widget
            if w and w.kind != "mermaid" and not w.html:
                html = await generate_widget_html(w, self.llm)
                if html:
                    script.steps[i].widget = w.model_copy(update={"html": html})
                    self.progress("widget", f"{script.session_id} step {i + 1}: {w.kind} ok ({len(html)} bytes)")
                else:
                    self.progress("warn", f"{script.session_id} step {i + 1}: widget generation failed; degraded")

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
    icons = {"parse": "📄", "plan": "🧠", "script": "✍️", "widget": "🎲", "compile": "🎬", "warn": "⚠️", "done": "✅"}
    print(f"{icons.get(stage, '•')} [{stage}] {detail}", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Socratic whiteboard content pipeline")
    p.add_argument("--input", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--output", default="output")
    p.add_argument("--mode", default="auto", choices=["auto", "llm", "heuristic"])
    p.add_argument("--tts", default=None, help="say|edge|silent|auto (default: TTS_ENGINE env or auto)")
    args = p.parse_args(argv)

    llm = make_client() if args.mode != "heuristic" else None
    if args.mode == "auto" and llm is None:
        print("ℹ️  no LLM key found; running heuristic walk-through mode", flush=True)
    pipeline = ContentPipeline(args.output, llm=llm, tts=choose_engine(args.tts), mode=args.mode,
                               progress=_print_progress)
    if llm:
        print(f"ℹ️  LLM: {llm.config.provider}/{llm.model}   TTS: {pipeline.tts.name}", flush=True)
    else:
        print(f"ℹ️  TTS: {pipeline.tts.name}", flush=True)
    asyncio.run(pipeline.run_file(args.input, title=args.title))
    return 0


if __name__ == "__main__":
    sys.exit(main())
