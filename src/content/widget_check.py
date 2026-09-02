"""
Visual QA for generated widgets: render headlessly with Playwright (via the
globally installed node module), collect runtime errors and detect blank output.

Optional: if node or playwright are missing, `available()` is False and the
pipeline skips the check with a warning.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

TOOL = os.path.join(os.path.dirname(__file__), "..", "..", "tools", "render_widget.mjs")


@dataclass
class RenderResult:
    ok: bool
    problem: Optional[str]
    png_path: Optional[str]
    review: Optional[str] = None  # vision model's verdict text, when available


VISION_REVIEW_SYSTEM = """你是教学可视化审核员。给你一个教具的任务描述和它渲染后的截图。判断：
1. 截图是否画出了任务要求的对象（曲线/向量/网格/元素、标注、读数控件）；
2. 是否有明显问题：空白、元素重叠遮挡、标注看不清、坐标范围让关键现象看不见、与任务不符。
只输出 JSON：{"pass": true/false, "problems": ["……"], "summary": "一句话"}"""


async def vision_review(llm, task: str, png_path: str) -> tuple[bool, str]:
    """Ask the vision-tier model whether the rendered widget matches its task."""
    from src.llm.client import extract_json
    with open(png_path, "rb") as f:
        png = f.read()
    raw = await llm.complete(VISION_REVIEW_SYSTEM, f"教具任务：{task}", json_mode=True, temperature=0.1, images=[png])
    data = extract_json(raw)
    problems = "; ".join(str(p) for p in data.get("problems") or [])
    return bool(data.get("pass")), (problems or str(data.get("summary") or ""))


def _node_path() -> Optional[str]:
    if not shutil.which("node") or not shutil.which("npm"):
        return None
    try:
        root = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True, timeout=20).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    return root if root and os.path.isdir(os.path.join(root, "playwright")) else None


def available() -> bool:
    return _node_path() is not None and os.path.isfile(TOOL)


async def render_check(html: str, png_out: str, timeout_s: float = 60.0) -> RenderResult:
    node_path = _node_path()
    if not node_path:
        return RenderResult(True, None, None)
    os.makedirs(os.path.dirname(png_out), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(html)
        tmp = f.name
    try:
        proc = await asyncio.create_subprocess_exec(
            "node", TOOL, tmp, png_out, env={**os.environ, "NODE_PATH": node_path},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            return RenderResult(False, "render timed out", None)
        if proc.returncode != 0:
            return RenderResult(False, f"renderer failed: {err.decode(errors='ignore')[-300:]}", None)
        data = json.loads(out.decode().strip().splitlines()[-1])
    finally:
        os.unlink(tmp)
    if data.get("errors"):
        return RenderResult(False, "runtime error: " + "; ".join(data["errors"]), png_out)
    if data.get("blank"):
        return RenderResult(False, "rendered output is blank", png_out)
    return RenderResult(True, None, png_out)
