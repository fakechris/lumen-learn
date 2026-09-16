"""Canonical LUMEN_* environment variables (INV-581).

The HyperKnow-era HK_* names are still accepted for compatibility; LUMEN_*
takes precedence when both are set.
"""

from __future__ import annotations

import os


def output_root(default: str = "output") -> str:
    return os.getenv("LUMEN_OUTPUT_ROOT") or os.getenv("HK_OUTPUT_ROOT", default)


def courses_roots() -> list:
    """Extra course-package roots (colon or semicolon separated)."""
    raw = os.getenv("LUMEN_COURSES_ROOT") or os.getenv("HK_COURSES_ROOT", "")
    return [r for r in raw.replace(";", ":").split(":") if r]


def learner_cookie() -> str:
    """The cookie we SET (canonical); legacy clients may still send hk_learner."""
    return "lumen_learner"


def learner_cookie_legacy() -> str:
    return "hk_learner"
