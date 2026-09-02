"""
Text-to-speech engines that return audio files with *measured* durations.

Engines:
  - MacSayEngine   : macOS `say` -> AIFF -> WAV via afconvert (offline, zero deps)
  - EdgeTtsEngine  : Microsoft Edge neural voices via edge-tts (needs network)
  - SilentEngine   : no audio, estimated duration only (client runs a virtual clock)

`choose_engine()` picks from TTS_ENGINE=say|edge|silent|auto.
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import wave
from dataclasses import dataclass
from typing import Optional, Protocol

from src.tts.spoken_text import count_chars, estimate_duration_ms, to_spoken


@dataclass
class TtsResult:
    audio_path: Optional[str]  # None for SilentEngine
    duration_ms: int
    cjk: int
    latin: int
    engine: str
    marks: Optional[list] = None  # [[char_index, start_ms], ...] when aligned


class TtsEngine(Protocol):
    name: str

    async def synthesize(self, text: str, out_stem: str, speed: float = 1.0) -> TtsResult: ...


def _env_voice(default: str) -> str:
    return os.getenv("TTS_VOICE") or default


def wav_duration_ms(path: str) -> int:
    with wave.open(path, "rb") as w:
        return int(w.getnframes() * 1000 / w.getframerate())


def _has_cjk(text: str) -> bool:
    return count_chars(text)[0] > 0


class SilentEngine:
    name = "silent"

    async def synthesize(self, text: str, out_stem: str, speed: float = 1.0) -> TtsResult:
        spoken = to_spoken(text)
        cjk, latin = count_chars(spoken)
        return TtsResult(None, int(estimate_duration_ms(spoken) / speed), cjk, latin, self.name)


class MacSayEngine:
    """macOS `say`. Duration is measured from the WAV; when ffmpeg is present the
    file is then transcoded to MP3 (~10x smaller) unless mp3=False."""
    name = "say"

    def __init__(self, zh_voice: str = "Tingting", en_voice: str = "Samantha", base_rate: int = 190,
                 mp3: Optional[bool] = None):
        self.zh_voice = _env_voice(zh_voice)
        self.en_voice = en_voice
        self.base_rate = base_rate
        self.mp3 = shutil.which("ffmpeg") is not None if mp3 is None else mp3

    @staticmethod
    def available() -> bool:
        return platform.system() == "Darwin" and shutil.which("say") is not None and shutil.which("afconvert") is not None

    async def synthesize(self, text: str, out_stem: str, speed: float = 1.0) -> TtsResult:
        spoken = to_spoken(text)
        cjk, latin = count_chars(spoken)
        voice = self.zh_voice if _has_cjk(spoken) else self.en_voice
        aiff = out_stem + ".aiff"
        wav_path = out_stem + ".wav"
        rate = str(int(self.base_rate * speed))
        await _run("say", "-v", voice, "-r", rate, "-o", aiff, spoken)
        await _run("afconvert", "-f", "WAVE", "-d", "LEI16", aiff, wav_path)
        os.remove(aiff)
        duration_ms = wav_duration_ms(wav_path)
        audio_path = wav_path
        if self.mp3:
            mp3_path = out_stem + ".mp3"
            await _run("ffmpeg", "-y", "-loglevel", "error", "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", "64k", mp3_path)
            os.remove(wav_path)
            audio_path = mp3_path
        return TtsResult(audio_path, duration_ms, cjk, latin, self.name)


class EdgeTtsEngine:
    name = "edge"
    provides_marks = True  # WordBoundary events give word-level timing

    def __init__(self, zh_voice: str = "zh-CN-XiaoxiaoNeural", en_voice: str = "en-US-AriaNeural"):
        self.zh_voice = _env_voice(zh_voice)
        self.en_voice = en_voice

    @staticmethod
    def available() -> bool:
        try:
            import edge_tts  # noqa: F401
            return True
        except ImportError:
            return False

    async def synthesize(self, text: str, out_stem: str, speed: float = 1.0) -> TtsResult:
        import edge_tts

        spoken = to_spoken(text)
        cjk, latin = count_chars(spoken)
        voice = self.zh_voice if _has_cjk(spoken) else self.en_voice
        pct = int(round((speed - 1.0) * 100))
        rate = f"{pct:+d}%"
        mp3_path = out_stem + ".mp3"
        communicate = edge_tts.Communicate(spoken, voice, rate=rate)
        end_100ns = 0
        words = []  # (text, offset_ms)
        with open(mp3_path, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    end_100ns = max(end_100ns, chunk["offset"] + chunk["duration"])
                    words.append((chunk.get("text", ""), int(chunk["offset"] / 10_000)))
        duration_ms = int(end_100ns / 10_000) + 250 if end_100ns else int(estimate_duration_ms(spoken) / speed)
        marks = _marks_from_words(text, words, duration_ms)
        return TtsResult(mp3_path, duration_ms, cjk, latin, self.name, marks=marks)


def _marks_from_words(text: str, words, duration_ms: int):
    """Map engine word boundaries back onto the original text by sequential search."""
    if not words:
        return None
    marks, pos = [[0, 0]], 0
    for w, ms in words:
        w = w.strip()
        if not w:
            continue
        i = text.find(w, pos)
        if i < 0:
            i = text.find(w[:1], pos) if w[:1] else -1
        if i >= 0:
            marks.append([i, ms])
            pos = i + len(w)
    marks.append([len(text), duration_ms])
    return marks


class MiniMaxEngine:
    """MiniMax t2a_v2 (speech-2.6-turbo by default). Needs MINIMAX_API_KEY."""
    name = "minimax"

    def __init__(self, voice: str = "male-qn-qingse", model: Optional[str] = None):
        self.voice = _env_voice(voice)
        self.model = model or os.getenv("TTS_MODEL", "speech-2.6-turbo")

    @staticmethod
    def available() -> bool:
        return bool(os.getenv("MINIMAX_API_KEY"))

    async def synthesize(self, text: str, out_stem: str, speed: float = 1.0) -> TtsResult:
        import httpx
        spoken = to_spoken(text)
        cjk, latin = count_chars(spoken)
        payload = {"model": self.model, "text": spoken, "stream": False,
                   "voice_setting": {"voice_id": self.voice, "speed": float(speed), "vol": 1.0, "pitch": 0},
                   "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": "mp3", "channel": 1}}
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post("https://api.minimaxi.com/v1/t2a_v2", json=payload,
                                  headers={"Authorization": f"Bearer {os.environ['MINIMAX_API_KEY']}",
                                           "Content-Type": "application/json"})
        data = r.json()
        audio_hex = (data.get("data") or {}).get("audio")
        if not audio_hex:
            raise RuntimeError(f"MiniMax TTS failed: {str(data.get('base_resp'))[:200]}")
        mp3_path = out_stem + ".mp3"
        with open(mp3_path, "wb") as f:
            f.write(bytes.fromhex(audio_hex))
        duration_ms = int((data.get("extra_info") or {}).get("audio_length") or estimate_duration_ms(spoken) / speed)
        return TtsResult(mp3_path, duration_ms, cjk, latin, self.name)


async def _run(*cmd: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({proc.returncode}): {err.decode(errors='ignore').strip()}")


def choose_engine(preference: Optional[str] = None) -> TtsEngine:
    pref = (preference or os.getenv("TTS_ENGINE", "auto")).lower()
    if pref == "silent":
        return SilentEngine()
    if pref == "say":
        if not MacSayEngine.available():
            raise RuntimeError("TTS_ENGINE=say requires macOS with `say` and `afconvert`")
        return MacSayEngine()
    if pref == "edge":
        if not EdgeTtsEngine.available():
            raise RuntimeError("TTS_ENGINE=edge requires `pip install edge-tts`")
        return EdgeTtsEngine()
    if pref == "minimax":
        if not MiniMaxEngine.available():
            raise RuntimeError("TTS_ENGINE=minimax requires MINIMAX_API_KEY")
        return MiniMaxEngine()
    if pref != "auto":
        raise RuntimeError(f"Unknown TTS_ENGINE {pref!r}; use say|edge|minimax|silent|auto")
    if MacSayEngine.available():
        return MacSayEngine()
    if EdgeTtsEngine.available():
        return EdgeTtsEngine()
    return SilentEngine()
