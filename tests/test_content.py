import os

import pytest

from src.content.compiler import StepAudio, compile_session, decoration_offset_ms
from src.content.curriculum_planner import plan_course_heuristic
from src.content.document_parser import parse_markdown
from src.content.session_synthesizer import synthesize_session_heuristic
from src.content.validators import math_issues, sanitize_script, snippet_in
from src.content.widget_generator import ORBIT_CDN, THREE_CDN, static_check
from src.protocol.actions import parse_action
from src.protocol.session import (
    BoardSpec, DecorationSpec, QuestionSpec, RewardSpec, SessionScript, StepSpec, WidgetSpec, stable_id,
)

LECTURE = """# 线性代数：基与维数

## 第一节：生成
两个向量 $\\vec{v}_1, \\vec{v}_2$ 的组合：
$$c_1 \\vec{v}_1 + c_2 \\vec{v}_2$$

## 第二节：相关
如果 $\\vec{v}_3 = \\vec{v}_1 + \\vec{v}_2$ 则冗余。
"""


def test_parse_markdown_sections_and_stable_ids():
    doc = parse_markdown(LECTURE)
    assert doc.title == "线性代数：基与维数"
    assert [s.heading for s in doc.sections] == ["第一节：生成", "第二节：相关"]
    assert doc.sections[0].has_math()
    assert doc.document_id == parse_markdown(LECTURE).document_id  # deterministic across runs
    assert stable_id("x", "a") != stable_id("x", "b")


def test_heuristic_course_and_session_are_honest():
    doc = parse_markdown(LECTURE)
    course = plan_course_heuristic(doc)
    assert course.generation_mode == "heuristic"
    sessions = course.all_sessions()
    assert len(sessions) == 2
    script = synthesize_session_heuristic(sessions[0], course, doc.section_text(sessions[0].source_sections))
    assert script.steps and all(s.question is None for s in script.steps)
    assert "$" not in script.steps[0].spoken_text  # LaTeX was converted to speech


def test_snippet_and_math_validators():
    md = "组合：$c_1 \\vec{v}_1 + c_2 \\vec{v}_2$"
    assert snippet_in(md, "c_1 \\vec{v}_1 + c_2 \\vec{v}_2")
    assert snippet_in(md, "c_1\\vec{v}_1+c_2\\vec{v}_2")  # whitespace-insensitive
    assert not snippet_in(md, "c_3")
    assert math_issues("$a$ and $b$") == []
    assert "unbalanced $" in math_issues("$a$ and $b")[0]
    assert "braces" in math_issues("$\\frac{a}{b$")[0]


def test_sanitize_drops_bad_decorations_and_trigger_phrases():
    step = StepSpec(
        spoken_text="我们看 c1 乘 v1",
        boards=[BoardSpec(markdown="$c_1 \\vec{v}_1$")],
        decorations=[
            DecorationSpec(snippet="c_1 \\vec{v}_1", trigger_phrase="c1 乘 v1"),
            DecorationSpec(snippet="missing"),
            DecorationSpec(snippet="c_1 \\vec{v}_1", board_index=5),
            DecorationSpec(snippet="c_1 \\vec{v}_1", trigger_phrase="not spoken"),
        ],
    )
    script = SessionScript(session_id="s", course_id="c", title="t", steps=[step])
    clean, warnings = sanitize_script(script)
    kept = clean.steps[0].decorations
    assert len(kept) == 2
    assert kept[0].trigger_phrase == "c1 乘 v1"
    assert kept[1].trigger_phrase is None
    assert len(warnings) == 3


def test_question_spec_requires_valid_correct_index():
    with pytest.raises(ValueError):
        QuestionSpec(question="q", options=["a", "b"], correct_index=2)
    with pytest.raises(ValueError):
        QuestionSpec(question="q", options=["a"], correct_index=0)
    QuestionSpec(mode="open", question="q")


def test_compile_orders_actions_and_gates_boards():
    step = StepSpec(
        title="s1",
        spoken_text="先看公式，然后我们圈出 c1 乘 v1 的位置。",
        boards=[BoardSpec(title="板书", markdown="$c_1 \\vec{v}_1$"), BoardSpec(markdown="第二张", layout="newcol")],
        decorations=[DecorationSpec(snippet="c_1 \\vec{v}_1", trigger_phrase="c1 乘 v1")],
        widget=WidgetSpec(kind="threejs", title="3d", html="<html><script>x</script></html>"),
        question=QuestionSpec(question="q?", options=["错", "对"], correct_index=1, misconceptions=["误区", None]),
        reward=RewardSpec(title="掌握", description="ok"),
    )
    script = SessionScript(session_id="sess_1", course_id="c", title="t", steps=[step])
    audio = {0: StepAudio("/courses/c/audio/sess_1/step_1.wav", 10_000, 20, 4)}
    compiled = compile_session(script, audio, generation_mode="authored")

    types = [a.type for a in compiled.actions]
    assert types == ["board", "board", "generated_animation", "speak", "circle", "tts_segment", "reward_user", "ask", "done"]

    speak = next(a for a in compiled.actions if a.type == "speak")
    boards = [a for a in compiled.actions if a.type == "board"]
    assert all(b.reveal_gate_step == speak.step_id for b in boards)
    assert boards[1].layout == "newcol"
    assert [b.board_uid for b in boards] == [1, 2]

    tts = next(a for a in compiled.actions if a.type == "tts_segment")
    assert tts.step_id == speak.step_id and tts.duration_ms == 10_000 and tts.tts_cjk == 20

    deco = next(a for a in compiled.actions if a.type == "circle")
    assert deco.target_board_uid == 1 and deco.during_step == speak.step_id
    assert 0 < deco.at_ms < 10_000

    ask = next(a for a in compiled.actions if a.type == "ask")
    assert ask.correct_index == 1 and ask.options[0].misconception == "误区" and ask.options[1].misconception is None

    ids = [a.step_id for a in compiled.actions if a.type != "tts_segment"]
    assert len(ids) == len(set(ids))
    assert compiled.total_duration_ms == 10_000

    # round-trips through the wire format
    for a in compiled.actions:
        assert parse_action(a.model_dump(mode="json")).type == a.type


def test_decoration_offset_uses_trigger_phrase_position():
    step = StepSpec(spoken_text="AAAAAAAAAA" + "trigger" + "BBBBB")
    assert decoration_offset_ms(step, "trigger", 10_000) == int(10 / 22 * 10_000)
    assert decoration_offset_ms(step, None, 10_000) == 3_500
    assert decoration_offset_ms(step, "trigger", 500) == 0  # clamped so drawing can finish


def test_explorable_static_check():
    good = '<!DOCTYPE html><html><body><canvas id="p"></canvas><script>addEventListener("pointermove",()=>{})</script></body></html>'
    assert static_check(good, "explorable") is None
    assert "external" in static_check(good.replace("<script>", '<script src="https://x/y.js"></script><script>'), "explorable")
    assert "canvas" in static_check(good.replace("<canvas id=\"p\"></canvas>", "<div></div>"), "explorable")
    assert "interaction" in static_check(good.replace("pointermove", "load"), "explorable")


def test_widget_static_check():
    good = f'<!DOCTYPE html><html><head><script src="{THREE_CDN}"></script><script src="{ORBIT_CDN}"></script></head><body><script>const c=new THREE.OrbitControls();function a(){{requestAnimationFrame(a)}}a();</script></body></html>'
    assert static_check(good) is None
    assert "three.js" in static_check(good.replace(THREE_CDN, "x"))
    assert "modules" in static_check(good.replace("<script>", '<script type="module">'))
    assert "render loop" in static_check(good.replace("requestAnimationFrame", "setTimeout"))


def test_illustration_compiles_and_svg_check():
    from src.content.illustration_generator import svg_check
    from src.protocol.session import IllustrationSpec

    ok = '<svg viewBox="0 0 800 520" xmlns="http://www.w3.org/2000/svg"><text x="10" y="20">四个格子</text></svg>'
    assert svg_check(ok) is None
    assert "script" in svg_check(ok.replace("<text", "<script>alert(1)</script><text"))
    assert "viewBox" in svg_check('<svg><rect/></svg>')
    assert "external" in svg_check('<svg viewBox="0 0 1 1"><image href="https://x/y.png"/></svg>')

    step = StepSpec(spoken_text="看右边这张图。", boards=[BoardSpec(markdown="a")],
                    illustration=IllustrationSpec(caption="对比", brief="2x2 vs 8x8", svg=ok))
    script = SessionScript(session_id="s", course_id="c", title="t", steps=[step])
    compiled = compile_session(script, {0: StepAudio(None, 3000, 5, 0)})
    types = [a.type for a in compiled.actions]
    assert types == ["board", "illustration", "speak", "tts_segment", "done"]
    fig = compiled.actions[1]
    assert fig.reveal_gate_step == compiled.actions[2].step_id and fig.svg == ok and fig.board_uid == 2

    # an unfilled illustration (generation failed) is simply omitted
    step2 = step.model_copy(update={"illustration": IllustrationSpec(caption="x", brief="y")})
    compiled2 = compile_session(SessionScript(session_id="s", course_id="c", title="t", steps=[step2]), {})
    assert "illustration" not in [a.type for a in compiled2.actions]


def test_pdf_parse_sections_figures_and_logo_filter(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    from src.content.document_parser import parse_pdf

    doc = pymupdf.open()
    logo = pymupdf.open()  # a small image used on every page (logo)
    lp = logo.new_page(width=60, height=60)
    lp.draw_rect(pymupdf.Rect(5, 5, 55, 55), color=(0, 0, 1), fill=(0, 0, 1))
    logo_pdf = logo.convert_to_pdf()
    logo_bytes = pymupdf.open("pdf", logo_pdf)[0].get_pixmap().tobytes("png")
    fig = pymupdf.open()
    fp = fig.new_page(width=300, height=200)
    fp.draw_circle(pymupdf.Point(150, 100), 80, color=(1, 0, 0), fill=(1, 0, 0))
    fig_bytes = pymupdf.open("pdf", fig.convert_to_pdf())[0].get_pixmap().tobytes("png")
    for i in range(3):
        page = doc.new_page()
        page.insert_image(pymupdf.Rect(20, 20, 60, 60), stream=logo_bytes)
        page.insert_text((72, 100), f"Chapter {i + 1} heading", fontsize=18)
        page.insert_text((72, 140), "Body text line one of the chapter.", fontsize=11)
        page.insert_text((72, 160), "Body text line two.", fontsize=11)
        if i == 1:
            page.insert_image(pymupdf.Rect(72, 200, 372, 400), stream=fig_bytes)
            page.insert_text((72, 420), "Figure 1 a red circle", fontsize=11)
    doc.set_toc([[1, f"Chapter {i + 1} heading", i + 1] for i in range(3)])
    path = str(tmp_path / "book.pdf")
    doc.save(path)

    parsed = parse_pdf(path, assets_dir=str(tmp_path / "figs"))
    assert [s.heading for s in parsed.sections] == ["Chapter 1 heading", "Chapter 2 heading", "Chapter 3 heading"]
    assert parsed.sections[1].pages == [2] and "line one" in parsed.sections[1].content
    assert len(parsed.figures) == 1, "logo must be filtered, the figure kept"
    f = parsed.figures[0]
    assert f.page == 2 and f.caption.startswith("Figure 1") and os.path.isfile(f.path)
    assert parsed.sections[1].figure_ids == [f.figure_id]


def test_heuristic_plan_has_segments_and_synthesizer_follows_them():
    from src.protocol.session import SegmentPlan
    doc = parse_markdown(LECTURE)
    course = plan_course_heuristic(doc)
    for s in course.all_sessions():
        assert 3 <= len(s.segments) <= 8 and all(seg.media == "board" for seg in s.segments)
    # media decisions from the plan override what the model produced
    from src.content.session_synthesizer import _apply_plan
    from src.protocol.session import IllustrationSpec, SessionOutline, WidgetSpec
    outline = SessionOutline(session_id="s", title="t", learning_goal="g", core_concept="c", segments=[
        SegmentPlan(title="a", intent="i", media="board", ask=False),
        SegmentPlan(title="b", intent="i", media="reference_figure", figure_id="fig_p2_1", ask=True),
        SegmentPlan(title="c", intent="i", media="explorable", ask=False),
    ])
    steps = [
        StepSpec(spoken_text="x", illustration=IllustrationSpec(caption="unwanted", brief="b"),
                 question=QuestionSpec(question="q", options=["a", "b"], correct_index=0)),
        StepSpec(spoken_text="y"),
        StepSpec(spoken_text="z", widget=WidgetSpec(kind="threejs", task="t")),
    ]
    applied = _apply_plan(steps, outline)
    assert applied[0].illustration is None and applied[0].question is None
    assert applied[1].illustration.kind == "reference" and applied[1].illustration.figure_id == "fig_p2_1"
    assert applied[2].widget.kind == "explorable"


def test_exercise_schema_and_exact_grading():
    import asyncio
    from src.content.exercise_generator import grade_fill_blank, normalize_answer
    from src.protocol.session import ExerciseSpec
    with pytest.raises(ValueError):
        ExerciseSpec(kind="fill_blank", stem="没有空", answer="x")
    with pytest.raises(ValueError):
        ExerciseSpec(kind="single_choice", stem="q", options=["a"], correct_index=0)
    ex = ExerciseSpec(exercise_id="e1", kind="fill_blank", stem="函数被称为 ____ 函数", answer="夹逼",
                      accepted=["逼近", "包络"], explanation="因为它们提供了范围。")
    assert normalize_answer(" 逼 近。") == "逼近"
    ok, fb = asyncio.run(grade_fill_blank(ex, "逼近", None))
    assert ok and fb == ex.explanation
    ok, fb = asyncio.run(grade_fill_blank(ex, "发散", None))
    assert not ok and "夹逼" in fb
    ok, _ = asyncio.run(grade_fill_blank(ex, "", None))
    assert not ok
    choice = ExerciseSpec(exercise_id="e2", kind="single_choice", stem="q", options=["a", "b", "c"], correct_index=2)
    assert choice.correct_index == 2


def test_compile_keypoints_follow_speak_steps():
    script = SessionScript(session_id="s", course_id="c", title="t", steps=[
        StepSpec(title="第一段", spoken_text="a", boards=[BoardSpec(markdown="x")]),
        StepSpec(title="", spoken_text="b"),
    ])
    compiled = compile_session(script, {})
    speaks = [a.step_id for a in compiled.actions if a.type == "speak"]
    assert [k.step_id for k in compiled.keypoints] == speaks
    assert [k.title for k in compiled.keypoints] == ["第一段", "第 2 段"]


def test_thin_sessions_are_merged_into_predecessor():
    from src.content.curriculum_planner import PlannedChapter, PlannedCourse, PlannedSegment, PlannedSession, _assign_ids
    doc = parse_markdown(LECTURE)
    seg = lambda t: PlannedSegment(title=t, intent="i")
    planned = PlannedCourse(title="T", chapters=[PlannedChapter(title="c", sessions=[
        PlannedSession(title="a", learning_goal="g", core_concept="c", segments=[seg("1"), seg("2")], estimated_duration_min=4),
        PlannedSession(title="b", learning_goal="g", core_concept="c", segments=[seg("3")], estimated_duration_min=2),
        PlannedSession(title="c", learning_goal="g", core_concept="c", segments=[seg("4"), seg("5"), seg("6")]),
    ])])
    course = _assign_ids(planned, doc, "llm")
    sessions = course.all_sessions()
    assert [s.title for s in sessions] == ["a", "c"]
    assert [seg.title for seg in sessions[0].segments] == ["1", "2", "3"]
    assert sessions[0].estimated_duration_min == 5


def test_extract_json_tolerates_latex_escapes_and_trailing_commas():
    from src.llm.client import LLMError, extract_json
    assert extract_json('{"a": "$x \\le y$", "b": [1, 2,],}')["a"] == "$x \\le y$"
    assert extract_json('```json\n{"k": "\\\\vec{v}"}\n```')["k"] == "\\vec{v}"
    assert extract_json('前言 {"ok": true} 后记')["ok"] is True
    assert extract_json('{"a": "unterminated')["a"] == "unterminated"  # json-repair closes it
    with pytest.raises(LLMError):
        extract_json("no json here at all")


def test_extract_json_repairs_unescaped_quotes_in_code():
    from src.llm.client import extract_json
    raw = '{"steps": [{"markdown": "require(x, "insufficient");\\nfunction f() {}", "n": 1}]}'
    data = extract_json(raw)
    assert data["steps"][0]["n"] == 1 and "require" in data["steps"][0]["markdown"]


def test_extract_json_ignores_inner_code_fences():
    from src.llm.client import extract_json
    raw = '{"steps": [{"markdown": "```solidity\\nfunction f() {\\n  require(a, \\"b\\");\\n}\\n```", "n": 2}]}'
    assert extract_json(raw)["steps"][0]["n"] == 2
    wrapped = "```json\n" + raw + "\n```"
    assert extract_json(wrapped)["steps"][0]["n"] == 2


def test_llm_config_tiers(monkeypatch):
    from src.llm.client import LLMConfig
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    cfg = LLMConfig.from_env()
    assert cfg.model == "deepseek-v4-flash" and cfg.for_tier("pro") == "deepseek-v4-pro"
    assert cfg.for_tier("vision") == "deepseek-v4-flash-vision-exp" and cfg.for_tier("fast") == cfg.model
    monkeypatch.setenv("LLM_MODEL_PRO", "x-pro")
    assert LLMConfig.from_env().for_tier("pro") == "x-pro"
    assert LLMConfig("openai", "k", "m").for_tier("pro") == "m"  # unset tier falls back


def test_decoration_timing_uses_marks():
    from src.content.compiler import ms_at_char
    marks = [[0, 0], [10, 2000], [20, 6000]]
    assert ms_at_char(marks, 5, 6000, 20) == 1000 and ms_at_char(marks, 15, 6000, 20) == 4000
    assert ms_at_char(None, 10, 6000, 20) == 3000
    step = StepSpec(spoken_text="A" * 10 + "trigger" + "B" * 3)
    assert decoration_offset_ms(step, "trigger", 6000, marks) == 2000


def test_thinking_only_for_plan_by_default(monkeypatch):
    from src.llm.client import LLMConfig
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.delenv("LLM_THINK_PURPOSES", raising=False)
    cfg = LLMConfig.from_env()
    assert cfg.is_deepseek and cfg.think_purposes == ("plan",)
    monkeypatch.setenv("LLM_THINK_PURPOSES", "plan,synth")
    assert LLMConfig.from_env().think_purposes == ("plan", "synth")


def test_clean_mermaid_unescapes_and_unfences():
    from src.content.validators import clean_mermaid
    assert clean_mermaid("```mermaid\ngraph LR\\nA --> B\n```") == "graph LR\nA --> B"
    assert clean_mermaid("graph LR\nA --> B") == "graph LR\nA --> B"


def test_degenerate_output_detection_and_budgets():
    from src.llm.client import LLMConfig, looks_degenerate
    assert looks_degenerate("正常开头 " + ("<div class='row'>重复的内容行</div>\n" * 40))
    assert not looks_degenerate("这是一段正常的、不重复的输出。" + "".join(f"第{i}行内容不同。" for i in range(60)))
    cfg = LLMConfig("openai", "k", "m")
    assert cfg.budget("widget") == 6000 and cfg.budget("plan") == cfg.max_tokens and cfg.budget("synth_repair") == 7000
