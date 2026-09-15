"""Runtime BYOK settings (INV-570, Stage 11 头十分钟).

Stored at ``<output_root>/settings.json`` — output/ is git-ignored, so keys never
enter the repository. Values set here override the environment when building the
LLM/TTS clients; fields left empty fall back to the usual env-var resolution
(``DEEPSEEK_API_KEY`` / ``LLM_*`` / ``TTS_ENGINE``). GET responses always mask
the api_key; a PUT that echoes the mask back leaves the stored key unchanged.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

FIELDS = ("provider", "api_key", "base_url", "model", "model_pro", "tts_engine")


class Settings:
    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, str] = {}

    def load(self) -> None:
        if os.path.isfile(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    stored = json.load(f)
                self.data = {k: str(v) for k, v in stored.items() if k in FIELDS and v}
            except (OSError, ValueError):
                self.data = {}

    def update(self, changes: Dict[str, Optional[str]]) -> None:
        """None = keep, "" = clear, value = set; a masked api_key echo is ignored."""
        for k, v in changes.items():
            if k not in FIELDS or v is None:
                continue
            v = v.strip()
            if k == "api_key" and v == self.mask():
                continue
            if v:
                self.data[k] = v
            else:
                self.data.pop(k, None)
        self.save()

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def mask(self) -> str:
        key = self.data.get("api_key", "")
        return f"••••{key[-4:]}" if len(key) >= 8 else ("••••" if key else "")

    def masked(self) -> Dict[str, str]:
        out = {k: self.data.get(k, "") for k in FIELDS if k != "api_key"}
        out["api_key"] = self.mask()
        return out

    def llm_overrides(self) -> Dict[str, str]:
        """Explicit values for make_client(); empty fields fall back to env resolution."""
        return {k: self.data[k] for k in ("provider", "api_key", "base_url", "model", "model_pro") if self.data.get(k)}

    def tts_engine(self) -> str:
        return self.data.get("tts_engine", "")
