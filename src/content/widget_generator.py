"""
Interactive widget ("教具") generation with static checks.

Kinds (see WidgetSpec.kind):
  explorable : default. A self-contained 2D canvas plot/diagram with a pointer
               probe and a live readout, matching the real product (a function
               plot with dashed envelopes, a probe line, "当前 X / F(X)").
               No external dependencies.
  threejs    : only for genuinely 3D concepts. Three.js r128 + OrbitControls
               from CDN, same muted aesthetic.
  html       : free-form self-contained HTML (authored).
  mermaid    : rendered by the client, no generation here.

Widgets are generated in a separate LLM call; a failure degrades to
`animation_failed` instead of failing the session. `widget_check` can render
the result headlessly to catch runtime errors and blank canvases.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from src.llm.client import LLMClient, LLMError
from src.protocol.session import WidgetSpec

THREE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
ORBIT_CDN = "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"

EXEMPLAR_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "authored", "squeeze_explorable.html")


def _exemplar() -> str:
    try:
        with open(EXEMPLAR_PATH, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


AESTHETIC = """视觉规范（必须遵守，这是产品的统一风格）：
- 背景 #faf8f3，绘图区 #fbfaf6 带 1px #d9d2c3 圆角边框；主曲线/主元素 #c0392b 线宽 2；辅助线/包络/参考线 #2e8b6f 虚线 [5,4]；坐标轴 #b9b3a6；刻度与轴名用 italic 11~12px Georgia；曲线旁的标注用 italic 12px Georgia、与曲线同色。
- 整体安静、留白多，不用渐变、阴影、饱和色，不用 emoji。
- 画布区下方一行读数：小灰字标签 + 红色等宽粗体数值（例如"当前 X 0.224  F(X) -0.049"），右侧一枚公式小标签（白底、细边框、italic Georgia）。"""

EXPLORABLE_SYSTEM = """你是一名教学可视化工程师，为白板课生成一个**单文件、零依赖**的交互教具（HTML + Canvas 2D + 原生 JS）。
它会嵌在手写笔记页里、在 sandbox iframe 中运行，宽约 560px、高约 340px。

交互规范：
1. 学生用指针（pointermove / pointerdown，touch 也要能用）在画布上滑动，出现一条跟随的探针虚线和曲线上的一个点，下方读数实时更新。若任务里有"参数"，用一个原生 <input type=range> 放在读数行里，拖动时重绘。
2. 首帧就要有完整画面（不依赖用户操作）；监听 window resize 并按 devicePixelRatio 重设画布。
3. 数学要正确：函数、包络、渐近线、关键点按任务描述精确绘制；坐标范围要让关键现象一眼可见。
   对象不限于函数曲线：向量（箭头 + 平行四边形）、网格/点阵、几何变换、面积/长度对比都可以，用同一套视觉规范画。
4. 必须在脚本末尾**无条件调用一次完整绘制**（resize() 或 draw()），保证页面加载后画布不是空的；所有绘制函数不能依赖用户先动过滑块或鼠标。
5. 不引入任何外部脚本、字体或图片；不使用 alert；不输出解释文字。

""" + AESTHETIC + """

下面是一个完整范例（夹逼定理：f(x)=x²·sin(1/x) 被 y=±x² 夹住，探针 + 读数），请严格对齐它的结构与风格，只替换"被探索的对象"和标注：

""" + _exemplar() + """

只输出完整 HTML，从 <!DOCTYPE html> 开始到 </html> 结束。"""

THREE_SYSTEM = f"""你是一名 Three.js 教学可视化工程师。根据"教具任务"生成一个**单文件、自包含**的 HTML 页面，
它会在无同源权限的 sandbox iframe 中运行，因此不能访问 localStorage、cookie 或父窗口。

硬性要求：
1. 只允许通过这两个 CDN 引入依赖，顺序固定：
   <script src="{THREE_CDN}"></script>
   <script src="{ORBIT_CDN}"></script>
   不得使用 ES module / import / type="module"。
2. 使用 THREE.OrbitControls(camera, renderer.domElement)，enableDamping = true，在 requestAnimationFrame 循环中调用 controls.update() 与 renderer.render()。
3. 监听 window resize 并更新相机与 renderer 尺寸；renderer.setSize(window.innerWidth, window.innerHeight)。
4. 相机初始位置要让所有关键元素同时可见且不重叠（例如 (4.5, 3.2, 5.5) 看向场景中心），包含 GridHelper 与 AxesHelper；向量用 ArrowHelper；平面用半透明 MeshBasicMaterial（transparent: true, opacity 0.22, DoubleSide）。
5. 文字标注用 Canvas 绘制的 THREE.Sprite；底部放读数行和必要的 <input type=range> 控件，交互时实时更新读数。
6. 所有代码放在一个 <script> 中，不使用任何外部图片或字体；不使用 alert/prompt；不输出解释文字。

{AESTHETIC}
（3D 场景中：背景 #faf8f3，主向量 #c0392b，第二向量 #2f6fb5，平面 #b8dcc6 半透明，参考线 #2e8b6f。）

只输出完整 HTML，从 <!DOCTYPE html> 开始到 </html> 结束。"""


def static_check(html: str, kind: str = "threejs") -> Optional[str]:
    """Return a problem description, or None if the HTML looks runnable."""
    if len(html) > 80_000:
        return "html too large"
    if "<script" not in html:
        return "no <script> block"
    if 'type="module"' in html or re.search(r"^\s*import\s", html, re.M):
        return "ES modules are not allowed in the sandbox"
    if kind == "threejs":
        if "THREE." in html and THREE_CDN not in html:
            return "missing three.js CDN include"
        if "OrbitControls" in html and ORBIT_CDN not in html:
            return "missing OrbitControls CDN include"
        if "requestAnimationFrame" not in html:
            return "no render loop"
        return None
    # explorable / html: zero external dependencies
    if re.search(r"<script[^>]+src=", html, flags=re.I):
        return "external scripts are not allowed for a 2D explorable"
    if re.search(r"<link[^>]+href=\s*[\"']?https?:", html, flags=re.I):
        return "external stylesheets are not allowed"
    if kind == "explorable":
        if "<canvas" not in html and "<svg" not in html:
            return "no canvas or svg"
        if not re.search(r"pointermove|mousemove|touchmove|pointerdown|input", html):
            return "no pointer/input interaction"
    return None


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:html)?\s*(.*?)```", text, flags=re.S)
    html = m.group(1) if m else text
    start = html.find("<!DOCTYPE")
    return html[start:].strip() if start >= 0 else html.strip()


async def generate_widget_html(spec: WidgetSpec, llm: LLMClient, attempts: int = 2,
                               feedback: Optional[str] = None) -> Optional[str]:
    """Generate HTML for an explorable/threejs widget. Returns None on failure.
    `feedback` carries a runtime problem from a previous render check."""
    if spec.kind == "mermaid":
        return None
    if spec.html and not feedback:
        return spec.html
    system = THREE_SYSTEM if spec.kind == "threejs" else EXPLORABLE_SYSTEM
    user = f"教具标题：{spec.title}\n教具任务：{spec.task}"
    problem: Optional[str] = feedback
    for _ in range(attempts):
        try:
            prompt = user if not problem else f"{user}\n\n上一次生成的问题：{problem}。请修正后重新输出完整 HTML。"
            html = _strip_fences(await llm.complete(system, prompt, temperature=0.2, purpose="widget"))
        except LLMError as e:
            problem = str(e)
            continue
        problem = static_check(html, spec.kind)
        if problem is None:
            return html
    return None
