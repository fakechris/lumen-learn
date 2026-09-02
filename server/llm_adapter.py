"""
LLM Client Adapter.
Supports OpenAI, Anthropic, Gemini, DeepSeek, and local heuristic fallback.
"""

import os
import json
import re
from typing import Optional, Dict, Any


class LLMAdapter:
    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = provider or os.getenv("LLM_PROVIDER", "auto")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o")

        # Auto-detect provider if not specified
        if self.provider == "auto":
            if os.getenv("DEEPSEEK_API_KEY"):
                self.provider = "deepseek"
                self.model = self.model or "deepseek-chat"
                self.base_url = self.base_url or "https://api.deepseek.com"
            elif os.getenv("ANTHROPIC_API_KEY"):
                self.provider = "anthropic"
                self.model = self.model or "claude-3-7-sonnet-20250219"
            elif os.getenv("GEMINI_API_KEY"):
                self.provider = "gemini"
                self.model = self.model or "gemini-2.5-flash"
            else:
                self.provider = "openai"

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
        """
        Calls the configured LLM provider and returns the raw string response.
        """
        if not self.api_key:
            raise ValueError("No LLM API Key configured. Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, or DEEPSEEK_API_KEY.")

        if self.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt, temperature)
        else:
            # Default to OpenAI / OpenAI-compatible (DeepSeek, Qwen, Moonshot, etc.)
            return self._call_openai(system_prompt, user_prompt, temperature)

    def _call_openai(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI API Error ({self.model}): {str(e)}")

    def _call_anthropic(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API Error ({self.model}): {str(e)}")
