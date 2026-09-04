"""
Pedagogical figures.

  kind="svg"   : the LLM draws an SVG. Deterministic structure and exact
                 (Chinese) labels, instant, no extra API. Default.
  kind="image" : MiniMax image-01 raster (hand-drawn watercolor look). Good
                 for scene metaphors, bad at exact counts/labels, ~1-2 min.

Both degrade to "no illustration" on failure; the session still compiles.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from typing import Optional

import httpx

from src.llm.client import LLMClient, LLMError
from src.protocol.session import IllustrationSpec

SVG_SYSTEM = """你是一名教科书插画师，用 SVG 画"手绘风教学示意图"。

硬性要求：
1. 只输出一个 <svg> 元素，viewBox="0 0 800 520"，不要 XML 头、不要 Markdown 代码块、不要 <script>、不要外部资源（无 href/url()）。
2. 手绘质感：线条 stroke="#4b4b4b" stroke-width="2" stroke-linecap="round"，可以用轻微不规则的路径；配色柔和：桃色 #f2d5c4、蓝灰 #c9d8e8、米色 #ece6da、强调红 #e05656、绿 #b8dcc6。背景透明，不画外框。
3. 所有标注用中文，font-family="'LXGW WenKai Lite', sans-serif"，正文 font-size 18~22，标题 24，颜色 #333；标注用短引线指向对应元素。
4. 画面必须严格按 brief 里的数量、元素和对比来画（例如 2x2 网格就是 4 个格子，8x8 就是 64 个），不要发挥。
5. 布局先分格再画：把 800×520 想成 6×6 网格（列 A~F 各 133 宽，行 1~6 各 87 高），每个主要元素占整格或相邻几格，标注放在元素所在格的空白角；两个元素不共用一格，文字距边 ≥ 20。
5. 构图：左右对比或上下递进，留白充足，元素不重叠。
6. 反抖动铁律：外层 <g transform="…"> 只负责摆放位置，任何动画/变形放在内层 <g class="node-body"> 里——否则元素会"弹回原点"。
7. 所有元素坐标必须落在 viewBox 0 0 800 520 内，文字距边 ≥ 20px；<text> 不要重叠。

只输出 SVG。"""

IMAGE_STYLE_PREFIX = ("Hand-drawn watercolor textbook illustration on a clean white background, soft muted colors, "
                      "thin pencil outlines, educational and minimal, no text. ")


def svg_check(svg: str) -> Optional[str]:
    s = svg.strip()
    if not s.startswith("<svg") or not s.endswith("</svg>"):
        return "not a single <svg> element"
    if "viewBox" not in s:
        return "missing viewBox"
    if re.search(r"<script|javascript:|on[a-z]+\s*=", s, flags=re.I):
        return "script content is not allowed"
    if re.search(r"(href|url\()\s*=?\s*['\"(]?\s*https?:", s, flags=re.I):
        return "external resources are not allowed"
    if len(s) > 60_000:
        return "svg too large"
    return None


def _strip_to_svg(text: str) -> str:
    m = re.search(r"<svg.*?</svg>", text, flags=re.S | re.I)
    return m.group(0).strip() if m else text.strip()



def svg_geometry_issues(svg: str) -> List[str]:
    """Deterministic geometry conflicts (OpenMAIC's whiteboard-conflicts idea):
    elements outside the viewBox and <text> labels sharing the same line."""
    issues: List[str] = []
    m = re.search(r"viewBox=['\"]0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)['\"]", svg)
    if not m:
        return ["missing viewBox"]
    vw, vh = float(m.group(1)), float(m.group(2))
    texts = []
    for t in re.finditer(r"<text\b([^>]*)>(.*?)</text>", svg, re.S):
        attrs, content = t.group(1), re.sub(r"<[^>]+>", "", t.group(2)).strip()
        if not content:
            continue
        x = re.search(r'\bx="([\-\d.]+)"', attrs)
        y = re.search(r'\by="([\-\d.]+)"', attrs)
        if not x or not y:
            continue
        fx, fy = float(x.group(1)), float(y.group(1))
        if fx < 8 or fy < 14 or fx > vw - 8 or fy > vh - 6:
            issues.append(f"text '{content[:12]}' at ({fx:.0f},{fy:.0f}) is outside/flush with the viewBox")
        texts.append((fx, fy, content))
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            ax, ay, ca = texts[i]
            bx, by, cb = texts[j]
            if len(ca) < 2 or len(cb) < 2:
                continue  # single characters (node digits) sit close by design
            if abs(ay - by) < 16 and abs(ax - bx) < max(120, (len(ca) + len(cb)) * 10):
                issues.append(f"text overlap: '{ca[:12]}' and '{cb[:12]}' share the same line")
                break
        else:
            continue
        break
    return issues[:4]


async def generate_svg(spec: IllustrationSpec, llm: LLMClient, attempts: int = 2) -> Optional[str]:
    user = f"标题/说明：{spec.caption}\nbrief：{spec.brief}"
    problem: Optional[str] = None
    for _ in range(attempts):
        try:
            prompt = user if not problem else f"{user}\n\n上一次的问题：{problem}。请保持全部内容，只修正位置与大小后重新输出完整 SVG。"
            svg = _strip_to_svg(await llm.complete(SVG_SYSTEM, prompt, temperature=0.3, purpose="svg"))
        except LLMError as e:
            problem = str(e)
            continue
        problem = svg_check(svg)
        if problem is None:
            geo = svg_geometry_issues(svg)
            if geo:
                problem = "；".join(geo)  # G2: deterministic conflicts feed the retry
            else:
                return svg
    return None


async def generate_image_minimax(spec: IllustrationSpec, out_path: str, api_key: Optional[str] = None,
                                 timeout_s: float = 180.0) -> Optional[str]:
    key = api_key or os.getenv("MINIMAX_API_KEY")
    if not key:
        return None
    payload = {"model": "image-01", "prompt": IMAGE_STYLE_PREFIX + spec.brief, "aspect_ratio": "4:3",
               "response_format": "base64", "prompt_optimizer": True}
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post("https://api.minimaxi.com/v1/image_generation", json=payload,
                                  headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        data = r.json()
        imgs = (data.get("data") or {}).get("image_base64") or []
        if not imgs:
            return None
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(imgs[0]))
        return out_path
    except (httpx.HTTPError, json.JSONDecodeError, ValueError, KeyError):
        return None


async def fill_illustration(spec: IllustrationSpec, llm: Optional[LLMClient], image_out_path: str,
                            image_url: str) -> IllustrationSpec:
    """Return a spec with svg or image_url populated (or unchanged if generation failed)."""
    if spec.svg or spec.image_url:
        return spec
    if spec.kind == "image":
        path = await generate_image_minimax(spec, image_out_path)
        return spec.model_copy(update={"image_url": image_url}) if path else spec
    if llm is None:
        return spec
    svg = await generate_svg(spec, llm)
    return spec.model_copy(update={"svg": svg}) if svg else spec
