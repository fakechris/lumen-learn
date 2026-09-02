"""
Asynchronous widget generation (Three.js / HTML) with static checks.

Widgets are generated in a separate LLM call from the script; a failure here
degrades to `animation_failed` instead of failing the session.
"""

from __future__ import annotations

import re
from typing import Optional

from src.llm.client import LLMClient, LLMError
from src.protocol.session import WidgetSpec

THREE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
ORBIT_CDN = "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"

WIDGET_SYSTEM = f"""你是一名 Three.js 教学可视化工程师。根据"教具任务"生成一个**单文件、自包含**的 HTML 页面，
它会在无同源权限的 sandbox iframe 中运行，因此不能访问 localStorage、cookie 或父窗口。

硬性要求：
1. 只允许通过这两个 CDN 引入依赖，顺序固定：
   <script src="{THREE_CDN}"></script>
   <script src="{ORBIT_CDN}"></script>
   不得使用 ES module / import / type="module"。
2. 使用 THREE.OrbitControls(camera, renderer.domElement)，enableDamping = true，在 requestAnimationFrame 循环中调用 controls.update() 与 renderer.render()。
3. 监听 window resize 并更新相机与 renderer 尺寸；renderer.setSize(window.innerWidth, window.innerHeight)。
4. 背景浅色（#fcfcfc），包含 GridHelper 与 AxesHelper；向量用 ArrowHelper；平面用半透明 MeshBasicMaterial（transparent: true, opacity 0.25, DoubleSide）。
5. 文字标注用 Canvas 绘制的 THREE.Sprite；左上角放一个 div 提示"拖拽旋转视角"。
6. 所有代码放在一个 <script> 中，不使用任何外部图片或字体；不使用 alert/prompt；不输出解释文字。
7. 若任务涉及动画（例如向量随参数变化），用 requestAnimationFrame 平滑驱动，并保证首帧就有内容。

只输出完整 HTML，从 <!DOCTYPE html> 开始到 </html> 结束。"""


def static_check(html: str) -> Optional[str]:
    """Return a problem description, or None if the HTML looks runnable."""
    if len(html) > 60_000:
        return "html too large"
    if "<script" not in html:
        return "no <script> block"
    if 'type="module"' in html or re.search(r"^\s*import\s", html, re.M):
        return "ES modules are not allowed in the sandbox"
    if "THREE." in html and THREE_CDN not in html:
        return "missing three.js CDN include"
    if "OrbitControls" in html and ORBIT_CDN not in html:
        return "missing OrbitControls CDN include"
    if "requestAnimationFrame" not in html:
        return "no render loop"
    return None


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:html)?\s*(.*?)```", text, flags=re.S)
    html = m.group(1) if m else text
    start = html.find("<!DOCTYPE")
    return html[start:].strip() if start >= 0 else html.strip()


async def generate_widget_html(spec: WidgetSpec, llm: LLMClient, attempts: int = 2) -> Optional[str]:
    """Generate HTML for a threejs/html widget. Returns None on failure."""
    if spec.kind == "mermaid":
        return None
    if spec.html:
        return spec.html
    user = f"教具标题：{spec.title}\n教具任务：{spec.task}"
    problem: Optional[str] = None
    for _ in range(attempts):
        try:
            prompt = user if not problem else f"{user}\n\n上一次生成的问题：{problem}。请重新生成完整 HTML。"
            html = _strip_fences(await llm.complete(WIDGET_SYSTEM, prompt, temperature=0.2))
        except LLMError as e:
            problem = str(e)
            continue
        problem = static_check(html)
        if problem is None:
            return html
    return None
