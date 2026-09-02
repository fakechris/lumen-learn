"""
Convert narration text that may contain Markdown / LaTeX into something a TTS
engine can read aloud. Content generation is asked to write oral text already;
this is the safety net for stray formulas.
"""

from __future__ import annotations

import re

_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_LATIN = re.compile(r"[A-Za-z0-9]")

_COMMAND_WORDS = {
    r"\\cdot": " 乘 ",
    r"\\times": " 乘 ",
    r"\\in": " 属于 ",
    r"\\infty": " 无穷 ",
    r"\\neq": " 不等于 ",
    r"\\leq": " 小于等于 ",
    r"\\geq": " 大于等于 ",
    r"\\rightarrow": " 到 ",
    r"\\to": " 到 ",
    r"\\alpha": "阿尔法",
    r"\\beta": "贝塔",
    r"\\lambda": "拉姆达",
    r"\\theta": "西塔",
    r"\\pi": "派",
    r"\\sum": " 求和 ",
    r"\\int": " 积分 ",
    r"\\sqrt": " 根号 ",
    r"\\quad": " ",
    r"\\,": " ",
    r"\\;": " ",
}


def _spoken_math(tex: str) -> str:
    s = tex
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\2 分之 \1", s)
    s = re.sub(r"\\(?:text|mathrm|mathbf|vec|hat|bar|tilde|mathbb|operatorname)\{([^{}]*)\}", r"\1", s)
    for cmd, word in _COMMAND_WORDS.items():
        s = re.sub(cmd + r"(?![A-Za-z])", word, s)
    s = re.sub(r"\^\{?2\}?", "平方", s)
    s = re.sub(r"\^\{?3\}?", "立方", s)
    s = re.sub(r"\^\{([^{}]*)\}", r" 的\1次方", s)
    s = re.sub(r"\^([A-Za-z0-9])", r" 的\1次方", s)
    s = re.sub(r"_\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"_([A-Za-z0-9])", r"\1", s)
    s = s.replace("=", " 等于 ").replace("+", " 加 ")
    s = re.sub(r"(?<=[A-Za-z0-9)\]])\s*-\s*(?=[A-Za-z0-9(\[])", " 减 ", s)
    s = re.sub(r"\\[A-Za-z]+", " ", s)
    s = re.sub(r"[{}\\]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def to_spoken(text: str) -> str:
    s = text
    s = re.sub(r"\$\$(.+?)\$\$", lambda m: " " + _spoken_math(m.group(1)) + " ", s, flags=re.S)
    s = re.sub(r"\$(.+?)\$", lambda m: " " + _spoken_math(m.group(1)) + " ", s)
    s = re.sub(r"\\\[(.+?)\\\]", lambda m: " " + _spoken_math(m.group(1)) + " ", s, flags=re.S)
    s = re.sub(r"\\\((.+?)\\\)", lambda m: " " + _spoken_math(m.group(1)) + " ", s)
    s = re.sub(r"[*_`#>]+", "", s)  # markdown emphasis / headings / code
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)  # links
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def count_chars(text: str) -> tuple[int, int]:
    """(cjk_chars, latin_chars) used by the client's typewriter pacing."""
    return len(_CJK.findall(text)), len(_LATIN.findall(text))


def estimate_duration_ms(text: str, cjk_per_sec: float = 4.2, latin_per_sec: float = 11.0) -> int:
    cjk, latin = count_chars(text)
    pauses = text.count("，") * 0.2 + text.count("。") * 0.4 + text.count("？") * 0.4 + text.count("、") * 0.1
    seconds = cjk / cjk_per_sec + latin / latin_per_sec + pauses
    return int(max(0.8, seconds) * 1000)
