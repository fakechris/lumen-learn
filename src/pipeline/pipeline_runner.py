"""
End-to-End Content Production Pipeline Runner.
Transforms raw lecture notes/PDFs into a full Socratic Whiteboard Course Package.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import argparse
from src.pipeline.document_parser import DocumentParser
from src.pipeline.curriculum_planner import CurriculumPlanner
from src.pipeline.session_synthesizer import SessionSynthesizer
from src.tts.tts_synthesizer import TTSSynthesizer


class ContentPipelineRunner:
    def __init__(self, output_root: str = "output"):
        self.output_root = output_root
        self.doc_parser = DocumentParser()
        self.curriculum_planner = CurriculumPlanner()
        self.session_synthesizer = SessionSynthesizer()
        self.tts_synthesizer = TTSSynthesizer(output_dir=os.path.join(output_root, "audio"))

    def run(self, input_file_path: str) -> str:
        print(f"🚀 [Stage 1/4] 解析讲义文档: {input_file_path}...")
        parsed_doc = self.doc_parser.parse_file(input_file_path)
        print(f"   ✓ 提取到 {parsed_doc.total_pages} 个章节/逻辑切片，文档标题: 《{parsed_doc.title}》")

        print(f"🧠 [Stage 2/4] 认知图谱解构 (Curriculum Decomposition)...")
        course_map = self.curriculum_planner.plan_course(parsed_doc)
        total_sessions = sum(len(ch.sessions) for ch in course_map.chapters)
        print(f"   ✓ 成功生成课程图谱: {len(course_map.chapters)} 个章节, 共 {total_sessions} 个原子互动会话")

        course_dir = os.path.join(self.output_root, course_map.course_id)
        sessions_dir = os.path.join(course_dir, "sessions")
        os.makedirs(sessions_dir, exist_ok=True)

        # Save Course Structure Map
        with open(os.path.join(course_dir, "course_structure.json"), "w", encoding="utf-8") as f:
            f.write(course_map.model_dump_json(indent=2))

        print(f"🎨 [Stage 3/4 & 4/4] 合成苏格拉底多模态板书、3D教具与 TTS 音画时间轴...")
        session_manifests = []
        for ch in course_map.chapters:
            for s_outline in ch.sessions:
                # Find matching chunk context
                page_idx = s_outline.pdf_page_references[0] - 1 if s_outline.pdf_page_references else 0
                chunk_ctx = parsed_doc.chunks[page_idx].content if page_idx < len(parsed_doc.chunks) else ""

                # Synthesize interactive session
                manifest = self.session_synthesizer.synthesize_session(
                    session_outline=s_outline,
                    course_id=course_map.course_id,
                    source_context=chunk_ctx,
                )

                # Process TTS and Audio-visual timing metadata
                manifest = self.tts_synthesizer.process_manifest(manifest)
                session_manifests.append(manifest)

                # Export session manifest
                s_path = os.path.join(sessions_dir, f"{manifest.session_id}.json")
                with open(s_path, "w", encoding="utf-8") as f:
                    f.write(manifest.model_dump_json(indent=2))
                print(f"   ✓ 已生成交互会话: [{manifest.session_id}] {manifest.title} ({len(manifest.steps)} 步骤)")

        print(f"\n✨ [完成] 完整课程产物包已导出至: {course_dir}")
        return course_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lumen Learn Socratic Content Production Pipeline")
    parser.add_argument("--input", default="examples/linear_algebra_basis.md", help="Path to input lecture file")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    runner = ContentPipelineRunner(output_root=args.output)
    runner.run(args.input)
