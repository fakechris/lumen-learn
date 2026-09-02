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
