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
