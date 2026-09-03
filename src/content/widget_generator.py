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
HANDCHART_PATH = os.path.join(os.path.dirname(__file__), "handchart.js")


def handchart_source() -> str:
    try:
        with open(HANDCHART_PATH, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _inject_handchart(html: str) -> str:
    """Make the HandChart base library available to every generated explorable."""
    src = handchart_source()
    if not src:
        return html
    m = re.search(r"<head[^>]*>", html, re.I)
    if not m:
        return html
    end = m.end()
    return html[:end] + "<script>" + src + "</script>" + html[end:]


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
6. 命名约定：滑块 id 用 `#{量}-slider`、按钮 `#{动作}-btn`、读数 `#{量}-display`（如 `warmup-slider` / `eta-display`），便于外部定位与驱动。
7. 控制面板（滑块/按钮/读数行）一律放画布下方，不得遮挡画布；启动后必须已有可见画面。
8. 脚本铁律：动画时间单位一律是**秒**（不要把 dt 再乘 0.001）；**禁用模板字符串**（反引号）——本 HTML 会作为 JSON 字符串内嵌，一律用单引号；括号必须自平衡。
9. **底图库 HandChart（页面已内置，无需引入）**：凡是带坐标轴的图——函数曲线、散点、柱状、多折线——**必须**用它画底图，不要手写坐标轴与刻度：
   `var ch = HandChart.render(canvas, { type:'function', xlim:[-3,3], ylim:[-2,9], fns:[{fn:function(x){return x*x;}, label:'y = x²'}], xlabel:'x', ylabel:'y', title:'可选标题' })`
   散点 `type:'points', points:[[x,y],...]`；柱状 `type:'bars', bars:{labels:[...], values:[...], highlight:2}`；多折线 `type:'lines', series:[{name, points, color, dash}]`（自动图例）。
   `render` 返回 `ch = {ctx, sx, sy, box}`，之后用**世界坐标**叠加标注：`HandChart.marker(ch, x, y, {label:'x = 1.2'})`、`HandChart.vline(ch, x, {label})`、`HandChart.hline(ch, y, {label})`、`HandChart.segment(ch, [x1,y1], [x2,y2], {color, arrow:true})`（切线/向量）、`HandChart.note(ch, x, y, '一句标注')`。
   探针：`HandChart.attachProbe(canvas, {xlim, ylim}, function(p){ /* p={x,y} 世界坐标：重绘底图 + 叠加 + 更新读数 */ })`。每次交互都重新 `render` 再叠加（底图很便宜）。
   风格已内建（纸面/手绘墨线坐标轴/主红 HandChart.style.MAIN/辅绿 AUX/手写字）。只有非坐标轴对象（向量场、网格、几何变换、示意图）才自己画，自绘部分沿用视觉规范，字体用 `HandChart.font`。
10. **控制钩子（必做）**：实现 `window.hkControl = { set: function(obj){...}, highlight: function(sel){...}, annotate: function(sel, text){...}, reveal: function(sel){...} }`——老师讲解会用它与语音同步驱动教具（例如 set({probeX: 1.2})）。set 必须接受题目给出的全部"可控参数"键（同名变量、同名滑块），立即重绘并同步滑块位置；highlight/reveal 接收元素 id 选择器（'#eta-slider'），高亮 = 给该元素加一圈红色描边 1.5 秒；给每个可定位元素 id（命名约定见上）。

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
6. 所有代码放在一个 <script> 中，不使用任何外部图片或字体；不使用 alert/prompt；不输出解释文字；**禁用模板字符串**（反引号），动画时间单位一律是秒。
7. 相机自动取景（必做）：用 THREE.Box3().setFromObject(group) 计算包围盒，令 camDist = Math.max(2, radius * 2.4)，并把 group.position.sub(center) 居中，保证所有关键元素入画、不重叠。
8. 末尾暴露 `window.__hkScene = scene;` 供渲染检查统计场景对象数。
9. 控制钩子（必做）：实现 `window.hkControl = {{ set: function(obj){{...}}, highlight: function(sel){{...}}, reveal: function(sel){{...}} }}`——set 接受题目给出的可控参数（滑块、旋转、显示开关）并立即更新场景。

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
        if not re.search(r"pointermove|mousemove|touchmove|pointerdown|input|attachProbe", html, flags=re.I):
            return "no pointer/input interaction"
    return None


def element_inventory(html: str) -> set:
    """Real ids present in the generated widget (OpenMAIC's Element Inventory idea):
    control selectors must reference these, never invented ones."""
    return set(re.findall(r'''\bid=["']([^"']+)["']''', html))


def filter_controls(html: str, controls) -> tuple:
    """Keep only controls whose target selector exists in the widget's inventory."""
    import src.protocol.actions as actions
    ids = element_inventory(html)
    kept, dropped = [], []
    for c in controls or []:
        sel = str((c.payload or {}).get("selector") or "")
        if c.op in ("highlight", "annotate", "reveal") and sel and sel.lstrip("#.") not in ids and sel not in ids:
            dropped.append(f"{c.op}:{sel} (no such element)")
            continue
        if c.op == "set":
            unknown = [k for k in (c.payload or {}) if not re.search(r"\b" + re.escape(str(k)) + r"\b", html)]
            if unknown:
                dropped.append(f"set:{','.join(map(str, unknown))} (param not in widget)")
                continue
        kept.append(actions.WidgetControl(at_ms=c.at_ms or 0, op=c.op, payload=c.payload))
    return kept, dropped


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:html)?\s*(.*?)```", text, flags=re.S)
    html = m.group(1) if m else text
    start = html.find("<!DOCTYPE")
    return html[start:].strip() if start >= 0 else html.strip()


async def generate_widget_html(spec: WidgetSpec, llm: LLMClient, attempts: int = 2,
                               feedback: Optional[str] = None, problems: Optional[list] = None) -> Optional[str]:
    """Generate HTML for an explorable/threejs widget. Returns None on failure.
    `feedback` carries a runtime problem from a previous render check; `problems`
    (if given) collects the reason of every failed attempt for diagnostics."""
    if spec.kind == "mermaid":
        return None
    if spec.html and not feedback:
        return spec.html
    system = THREE_SYSTEM if spec.kind == "threejs" else EXPLORABLE_SYSTEM
    user = f"教具标题：{spec.title}\n教具任务：{spec.task}"
    if spec.params:
        user += ("\n可控参数（必须原样作为 hkControl.set 接受的键，并且是页面上的滑块/开关）："
                 + ", ".join(spec.params))
    if spec.controls:
        user += "\n讲解中老师会发出这些指令，教具必须能响应：" + "; ".join(
            f"{c.op} {c.payload}" for c in spec.controls)
    problem: Optional[str] = feedback
    for _ in range(attempts):
        try:
            prompt = user if not problem else f"{user}\n\n上一次生成的问题：{problem}。请修正后重新输出完整 HTML。"
            looping = bool(problem and "degenerate" in problem)
            html = _strip_fences(await llm.complete(system, prompt, temperature=0.7 if looping else 0.2, purpose="widget",
                                                    frequency_penalty=0.6 if looping else 0.0))
        except LLMError as e:
            problem = f"llm error: {e}"
            if problems is not None:
                problems.append(problem)
            continue
        problem = static_check(html, spec.kind)
        if problem is None and "hkControl" not in html:
            problem = "missing window.hkControl = { set, highlight, annotate, reveal } hook"
        if problem is None and spec.params:
            missing = [k for k in spec.params if not re.search(r"\b" + re.escape(k) + r"\b", html)]
            if missing:
                problem = "widget does not expose the required params: " + ", ".join(missing)
        if problem is None:
            return _inject_handchart(html)
        if problems is not None:
            problems.append(f"static check: {problem}")
    return None
