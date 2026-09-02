"""
Thin async LLM client with structured-output helpers.

Providers: any OpenAI-compatible chat API (DeepSeek, OpenAI, Qwen, Moonshot, ...)
and Anthropic. Configuration is explicit; nothing here silently falls back to
canned content — if generation fails, `LLMError` propagates.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import AsyncIterator, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    """Models by tier: `model` (default / fast), `model_pro` (planning, synthesis),
    `model_vision` (image understanding). Unset tiers fall back to `model`."""
    provider: str  # "openai" (compatible) | "anthropic"
    api_key: str
    model: str
    base_url: Optional[str] = None
    max_tokens: int = 16000
    model_pro: Optional[str] = None
    model_vision: Optional[str] = None

    def for_tier(self, tier: str) -> str:
        if tier == "pro" and self.model_pro:
            return self.model_pro
        if tier == "vision" and self.model_vision:
            return self.model_vision
        return self.model

    @classmethod
    def from_env(cls, provider: Optional[str] = None, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, model: Optional[str] = None) -> Optional["LLMConfig"]:
        """Resolve config from explicit args, then env. Returns None when no key is available."""
        provider = (provider or os.getenv("LLM_PROVIDER") or "auto").lower()
        if provider == "auto":
            if api_key:
                provider = "openai"
            elif os.getenv("DEEPSEEK_API_KEY"):
                provider = "deepseek"
            elif os.getenv("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            else:
                return None

        if provider == "deepseek":
            return cls("openai", api_key or os.environ["DEEPSEEK_API_KEY"],
                       model or os.getenv("LLM_MODEL", "deepseek-v4-flash"),
                       base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
                       model_pro=os.getenv("LLM_MODEL_PRO", "deepseek-v4-pro"),
                       model_vision=os.getenv("LLM_MODEL_VISION", "deepseek-v4-flash-vision-exp"))
        if provider == "anthropic":
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                return None
            return cls("anthropic", key, model or os.getenv("LLM_MODEL", "claude-sonnet-5"), base_url,
                       model_pro=os.getenv("LLM_MODEL_PRO"), model_vision=os.getenv("LLM_MODEL_VISION"))
        key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        if not key:
            return None
        return cls("openai", key, model or os.getenv("LLM_MODEL", "gpt-4o"),
                   base_url or os.getenv("LLM_BASE_URL"),
                   model_pro=os.getenv("LLM_MODEL_PRO"), model_vision=os.getenv("LLM_MODEL_VISION"))


_BAD_ESCAPE = re.compile(r'\\(?![\\"/bfnrtu])')
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _lenient_loads(s: str):
    """json.loads with the fixes that LaTeX-heavy model output usually needs:
    invalid backslash escapes and trailing commas."""
    first: Optional[json.JSONDecodeError] = None
    for candidate in (s, _BAD_ESCAPE.sub(r"\\\\", s), _TRAILING_COMMA.sub(r"\1", _BAD_ESCAPE.sub(r"\\\\", s))):
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError as e:
            first = first or e
    # Last resort: json-repair handles unescaped quotes inside strings (code snippets
    # with string literals), missing commas and similar LLM-typical damage.
    try:
        from json_repair import repair_json
        repaired = repair_json(s, return_objects=True)
        if isinstance(repaired, dict) and repaired:
            return repaired
    except Exception:
        pass
    raise json.JSONDecodeError(first.msg if first else "unrecoverable JSON", s, first.pos if first else 0)


def _dump_bad_output(text: str, reason: str) -> None:
    """Keep unparseable model replies for diagnosis (output/_debug/)."""
    try:
        import time
        d = os.path.join(os.getenv("HK_OUTPUT_ROOT", "output"), "_debug")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"badjson_{int(time.time() * 1000)}.txt"), "w", encoding="utf-8") as f:
            f.write(f"# {reason}\n{text}")
    except OSError:
        pass


def extract_json(text: str) -> dict:
    """Parse a JSON object out of a model reply, tolerating code fences, prose,
    bad LaTeX escapes and trailing commas."""
    cleaned = text.strip().lstrip("\ufeff")
    # Only unwrap a fence that wraps the WHOLE reply; JSON often contains inner ```code``` blocks.
    fence = re.match(r"^```(?:json)?\s*(.*)```\s*$", cleaned, flags=re.S)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return _lenient_loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1:
            raise LLMError(f"No JSON object in model reply: {text[:200]!r}")
        if end == -1 or end <= start:
            raise LLMError("Malformed JSON in model reply: no closing brace; output looks truncated")
        try:
            return _lenient_loads(cleaned[start:end + 1])
        except json.JSONDecodeError as e:
            ctx = cleaned[max(0, start + e.pos - 60): start + e.pos + 60].replace("\n", "\\n")
            reason = (f"Malformed JSON in model reply ({e.msg} at char {e.pos}: …{ctx}…); "
                      f"{'output looks truncated' if not cleaned.rstrip().endswith('}') else 'check quotes/escapes'}")
            _dump_bad_output(text, reason)
            raise LLMError(reason)


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        if config.provider == "anthropic":
            import anthropic
            self._anthropic = anthropic.AsyncAnthropic(api_key=config.api_key, base_url=config.base_url)
        else:
            from openai import AsyncOpenAI
            self._openai = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def has_vision(self) -> bool:
        return bool(self.config.model_vision)

    async def complete(self, system: str, user: str, *, json_mode: bool = False,
                       temperature: float = 0.4, tier: str = "fast",
                       images: Optional[list] = None) -> str:
        """`tier`: fast | pro | vision. `images`: list of PNG/JPEG bytes (vision tier)."""
        model = self.config.for_tier("vision" if images else tier)
        try:
            if self.config.provider == "anthropic":
                content = [{"type": "text", "text": user}]
                for img in images or []:
                    content.insert(0, {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                                    "data": base64.b64encode(img).decode()}})
                resp = await self._anthropic.messages.create(
                    model=model, max_tokens=self.config.max_tokens, system=system,
                    messages=[{"role": "user", "content": content}], temperature=temperature,
                )
                return "".join(block.text for block in resp.content if getattr(block, "text", None))
            kwargs = {}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            user_content = user
            if images:
                user_content = [{"type": "text", "text": user}] + [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(img).decode()}}
                    for img in images]
            resp = await self._openai.chat.completions.create(
                model=model, temperature=temperature, max_tokens=self.config.max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user_content}],
                **kwargs,
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            if choice.finish_reason == "length":
                raise LLMError(f"output truncated at {self.config.max_tokens} tokens; produce a shorter answer")
            return content
        except LLMError:
            raise
        except Exception as e:  # provider SDK errors
            raise LLMError(f"{self.config.provider}/{self.config.model}: {e}") from e

    async def complete_model(self, system: str, user: str, schema: Type[T], *, repairs: int = 2,
                             temperature: float = 0.4, tier: str = "fast") -> T:
        """Ask for JSON, validate against `schema`, and give the model one chance to
        repair its own output using the validation error. Raises LLMError after that."""
        try:
            raw = await self.complete(system, user, json_mode=True, temperature=temperature, tier=tier)
        except LLMError as e:  # e.g. truncated: retry once asking for brevity
            raw = await self.complete(system, user + f"\n\n注意：{e}。请精简输出。", json_mode=True,
                                      temperature=temperature, tier=tier)
        last_error: Optional[str] = None
        for attempt in range(repairs + 1):
            try:
                return schema.model_validate(extract_json(raw))
            except (ValidationError, LLMError, json.JSONDecodeError) as e:
                last_error = str(e)
                if attempt >= repairs:
                    break
                try:
                    raw = await self.complete(
                        system,
                        user + "\n\n你上一次的输出无法通过校验，错误如下，请修正后重新输出完整 JSON：\n"
                        + last_error[:2000] + "\n\n上一次输出：\n" + raw[:6000],
                        json_mode=True, temperature=0.2, tier=tier,
                    )
                except LLMError as e:
                    last_error = str(e)
                    raw = ""
        raise LLMError(f"Model output failed validation for {schema.__name__}: {last_error}")

    async def stream(self, system: str, user: str, *, temperature: float = 0.6) -> AsyncIterator[str]:
        try:
            if self.config.provider == "anthropic":
                async with self._anthropic.messages.stream(
                    model=self.config.for_tier("fast"), max_tokens=1024, system=system,
                    messages=[{"role": "user", "content": user}], temperature=temperature,
                ) as s:
                    async for delta in s.text_stream:
                        yield delta
                return
            stream = await self._openai.chat.completions.create(
                model=self.config.for_tier("fast"), temperature=temperature, max_tokens=1024, stream=True,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise LLMError(f"{self.config.provider}/{self.config.model}: {e}") from e


def make_client(**overrides) -> Optional[LLMClient]:
    cfg = LLMConfig.from_env(**overrides)
    return LLMClient(cfg) if cfg else None
