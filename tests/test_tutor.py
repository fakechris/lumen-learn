from src.protocol.session import BoardSpec, IllustrationSpec, SessionScript, StepSpec
from src.runtime.tutor import TutorContext, ensure_detour_figure, wants_figure


def test_wants_figure_detects_draw_requests():
    assert wants_figure("帮我画个图看看径向分布")
    assert wants_figure("画一下 2p 轨道")
    assert wants_figure("给一张示意图")
    assert not wants_figure("为什么这一行是零")
    assert not wants_figure("节面在哪里")


def test_ensure_detour_figure_injects_when_llm_only_wrote_text():
    script = SessionScript(session_id="s", course_id="c", title="岔路", steps=[
        StepSpec(spoken_text="看这一行，径向分布先升后降。好，我们回到刚才的地方。",
                 boards=[BoardSpec(title="岔路：径向分布", markdown="- 先升后降", layout="newcol")]),
    ])
    ctx = TutorContext(session_title="轨道", learning_goal="形状", current_narration="径向分布函数")
    out = ensure_detour_figure(script, "帮我画个径向分布的图", ctx)
    il = out.steps[0].illustration
    assert il is not None and il.kind == "svg"
    assert "径向分布" in il.brief and "学生要求画图" in il.brief


def test_ensure_detour_figure_keeps_existing_brief_and_ignores_empty_shell():
    with_brief = SessionScript(session_id="s", course_id="c", title="岔路", steps=[
        StepSpec(spoken_text="看这张图。",
                 boards=[BoardSpec(title="岔路：图", markdown="- 图", layout="newcol")],
                 illustration=IllustrationSpec(kind="svg", caption="RDF", brief="画 R(r) 曲线")),
    ])
    kept = ensure_detour_figure(with_brief, "画个图", None)
    assert kept.steps[0].illustration.brief == "画 R(r) 曲线"

    empty = SessionScript(session_id="s", course_id="c", title="岔路", steps=[
        StepSpec(spoken_text="看这一行。",
                 boards=[BoardSpec(title="岔路：图", markdown="- 图", layout="newcol")],
                 illustration=IllustrationSpec(kind="svg", caption="", brief="")),
    ])
    filled = ensure_detour_figure(empty, "画张示意图", None)
    assert filled.steps[0].illustration.brief.startswith("学生要求画图")
