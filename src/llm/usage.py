"""
Usage / cost ledger for LLM and TTS calls.

Every call is recorded with its purpose (plan, synth, widget, svg, exercise,
vision_review, caption, grade, interject, feedback, tts). Prices are USD per
1M tokens (LLM) or per 1M characters (TTS) and are ESTIMATES unless overridden
with the LLM_PRICES / TTS_PRICES env vars (JSON: {"model": [in, out]}).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# USD per 1M tokens [input, output]. DeepSeek V3-era list prices are used as the
# estimate for V4 flash; pro is assumed ~4x. Override with LLM_PRICES.
DEFAULT_LLM_PRICES: Dict[str, List[float]] = {
    "deepseek-chat": [0.27, 1.10],
    "deepseek-v4-flash": [0.27, 1.10],
    "deepseek-v4-flash-vision-exp": [0.27, 1.10],
    "deepseek-v4-pro": [1.00, 4.00],
    "gpt-4o": [2.50, 10.00],
    "claude-sonnet-5": [3.00, 15.00],
}
# USD per 1M characters
DEFAULT_TTS_PRICES: Dict[str, float] = {"say": 0.0, "silent": 0.0, "edge": 0.0, "minimax": 30.0}


def _load_prices(env: str, default: dict) -> dict:
    raw = os.getenv(env)
    if not raw:
        return dict(default)
    try:
        merged = dict(default)
        merged.update(json.loads(raw))
        return merged
    except json.JSONDecodeError:
        return dict(default)


@dataclass
class UsageRecord:
    ts: float
    kind: str            # "llm" | "tts"
    model: str
    purpose: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    chars: int = 0
    seconds: float = 0.0
    cost_usd: float = 0.0
    estimated_price: bool = True


class UsageLedger:
    def __init__(self, jsonl_path: Optional[str] = None):
        self.records: List[UsageRecord] = []
        self.jsonl_path = jsonl_path
        self._lock = threading.Lock()
        self.llm_prices = _load_prices("LLM_PRICES", DEFAULT_LLM_PRICES)
        self.tts_prices = _load_prices("TTS_PRICES", DEFAULT_TTS_PRICES)

    # ------------------------------------------------------------------ record

    def add_llm(self, model: str, purpose: str, prompt_tokens: int, completion_tokens: int,
                reasoning_tokens: int = 0, seconds: float = 0.0) -> UsageRecord:
        price = self.llm_prices.get(model)
        cost = 0.0
        if price:
            cost = (prompt_tokens * price[0] + (completion_tokens + reasoning_tokens) * price[1]) / 1_000_000
        rec = UsageRecord(time.time(), "llm", model, purpose, prompt_tokens, completion_tokens, reasoning_tokens,
                          0, seconds, cost, estimated_price=("LLM_PRICES" not in os.environ))
        self._append(rec)
        return rec

    def add_tts(self, engine: str, purpose: str, chars: int, seconds: float = 0.0) -> UsageRecord:
        cost = chars * self.tts_prices.get(engine, 0.0) / 1_000_000
        rec = UsageRecord(time.time(), "tts", engine, purpose, 0, 0, 0, chars, seconds, cost,
                          estimated_price=("TTS_PRICES" not in os.environ))
        self._append(rec)
        return rec

    def _append(self, rec: UsageRecord) -> None:
        with self._lock:
            self.records.append(rec)
        try:  # mirror into the DB, tagged with the current run
            from src.obs.db import get_db
            from src.obs.log import current_run_id
            get_db().add_usage(current_run_id.get(), rec)
        except Exception:
            pass
        with self._lock:
            if self.jsonl_path:
                try:
                    os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)
                    with open(self.jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                except OSError:
                    pass

    # ------------------------------------------------------------------ query

    def mark(self) -> int:
        return len(self.records)

    def summary(self, since: int = 0) -> dict:
        recs = self.records[since:]
        by_purpose: Dict[str, dict] = {}
        for r in recs:
            b = by_purpose.setdefault(r.purpose, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                                  "reasoning_tokens": 0, "chars": 0, "cost_usd": 0.0, "seconds": 0.0})
            b["calls"] += 1
            b["prompt_tokens"] += r.prompt_tokens
            b["completion_tokens"] += r.completion_tokens
            b["reasoning_tokens"] += r.reasoning_tokens
            b["chars"] += r.chars
            b["cost_usd"] += r.cost_usd
            b["seconds"] += r.seconds
        total = {
            "calls": len(recs),
            "prompt_tokens": sum(r.prompt_tokens for r in recs),
            "completion_tokens": sum(r.completion_tokens for r in recs),
            "reasoning_tokens": sum(r.reasoning_tokens for r in recs),
            "tts_chars": sum(r.chars for r in recs),
            "cost_usd": round(sum(r.cost_usd for r in recs), 6),
            "seconds": round(sum(r.seconds for r in recs), 1),
            "estimated_price": any(r.estimated_price for r in recs),
        }
        for b in by_purpose.values():
            b["cost_usd"] = round(b["cost_usd"], 6)
            b["seconds"] = round(b["seconds"], 1)
        return {"total": total, "by_purpose": by_purpose}

    @staticmethod
    def format_usd(v: float) -> str:
        return f"${v:.4f}" if v < 0.01 else f"${v:.3f}"


# A process-wide ledger; the pipeline and runtime share it so server jobs can
# report deltas with mark()/summary(since).
GLOBAL_LEDGER = UsageLedger(os.path.join(__import__("src.envs", fromlist=["output_root"]).output_root(), "_usage", "usage.jsonl"))
