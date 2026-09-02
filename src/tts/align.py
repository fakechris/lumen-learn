"""
Sentence-level aligned synthesis for any TTS engine.

The text is split into sentences on the ORIGINAL string (so indices refer to
what the client displays), each sentence is synthesized separately and its
duration measured, the clips are concatenated, and character->time marks are
derived: exact at sentence starts, interpolated by character weight inside a
sentence (CJK 1.0, latin 0.45, punctuation pause 0.8). Engines that provide
their own word marks (edge-tts) are used as-is.

Marks format: list of [char_index, start_ms], ascending, first is [0, 0],
last is [len(text), total_ms].
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import wave
from typing import List, Optional, Tuple

from src.tts.engine import TtsEngine, TtsResult, wav_duration_ms
from src.tts.spoken_text import to_spoken

Marks = List[List[int]]

_SENT_END = re.compile(r"([。！？；!?;]+|\n+)")
_SOFT_BREAK = re.compile(r"([，,])")
MAX_SENT_CHARS = 60


def split_sentences(text: str) -> List[Tuple[int, int]]:
    """(start, end) spans covering the text, split at sentence ends; long
    sentences are further split at commas so timing stays fine-grained."""
    spans: List[Tuple[int, int]] = []
    pos = 0
    for m in _SENT_END.finditer(text):
        end = m.end()
        if end > pos:
            spans.append((pos, end))
        pos = end
    if pos < len(text):
        spans.append((pos, len(text)))
    out: List[Tuple[int, int]] = []
    for a, b in spans:
        if b - a <= MAX_SENT_CHARS:
            out.append((a, b))
            continue
        cur = a
        for m in _SOFT_BREAK.finditer(text, a, b):
            if m.end() - cur >= 20:
                out.append((cur, m.end()))
                cur = m.end()
        if cur < b:
            out.append((cur, b))
    return [(a, b) for a, b in out if text[a:b].strip()] or [(0, len(text))]


def _char_weight(ch: str) -> float:
    if "一" <= ch <= "鿿":
        return 1.0
    if ch in "，。！？；：,.!?;:、":
        return 0.8
    if ch.isspace():
        return 0.15
    return 0.45


def interpolate_marks(text: str, sentences: List[Tuple[int, int, int]]) -> Marks:
    """sentences: (start_idx, end_idx, duration_ms). Produces per-character marks."""
    marks: Marks = []
    t = 0
    for a, b, dur in sentences:
        seg = text[a:b]
        weights = [_char_weight(c) for c in seg]
        total_w = sum(weights) or 1.0
        acc = 0.0
        for i, w in enumerate(weights):
            marks.append([a + i, int(t + dur * acc / total_w)])
            acc += w
        t += dur
    marks.append([len(text), t])
    return marks


def _concat(paths: List[str], out_stem: str, want_mp3: bool = False) -> Tuple[str, int]:
    """Concatenate clips. WAV clips are joined exactly with the wave module (then optionally
    transcoded once to MP3); MP3 clips go through ffmpeg's concat *filter*, which re-decodes
    and is robust to encoder padding differences (the concat demuxer was not)."""
    if all(p.endswith(".wav") for p in paths):
        out = out_stem + ".wav"
        with wave.open(paths[0], "rb") as first:
            params = first.getparams()
        with wave.open(out, "wb") as w:
            w.setparams(params)
            for p in paths:
                with wave.open(p, "rb") as r:
                    w.writeframes(r.readframes(r.getnframes()))
        duration = wav_duration_ms(out)
        if want_mp3 and shutil.which("ffmpeg"):
            mp3 = out_stem + ".mp3"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out, "-codec:a", "libmp3lame", "-b:a", "64k", mp3],
                           check=True)
            os.remove(out)
            return mp3, duration
        return out, duration
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to concatenate mp3 clips")
    out = out_stem + ".mp3"
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for p in paths:
        cmd += ["-i", os.path.abspath(p)]
    inputs = "".join(f"[{i}:a]" for i in range(len(paths)))
    cmd += ["-filter_complex", f"{inputs}concat=n={len(paths)}:v=0:a=1[out]", "-map", "[out]",
            "-codec:a", "libmp3lame", "-b:a", "64k", out]
    subprocess.run(cmd, check=True)
    return out, 0  # duration is summed from clips by the caller


async def synthesize_aligned(engine: TtsEngine, text: str, out_stem: str, speed: float = 1.0) -> TtsResult:
    """Synthesize `text` with character-level marks. Uses the engine's own marks
    when it provides them (single call); otherwise per-sentence synthesis."""
    if getattr(engine, "provides_marks", False):
        res = await engine.synthesize(text, out_stem, speed)
        if res.marks:
            return res
    spans = split_sentences(text)
    if len(spans) == 1:
        res = await engine.synthesize(text, out_stem, speed)
        res.marks = interpolate_marks(text, [(0, len(text), res.duration_ms)])
        return res

    clips: List[str] = []
    sentences: List[Tuple[int, int, int]] = []
    cjk = latin = 0
    # macOS say: synthesize pieces as WAV so the join is exact; transcode once at the end.
    piece_kwargs = {"mp3": False} if getattr(engine, "name", "") == "say" else {}
    want_mp3 = bool(getattr(engine, "mp3", False))
    for k, (a, b) in enumerate(spans):
        piece = text[a:b]
        if not to_spoken(piece).strip():
            sentences.append((a, b, 0))
            continue
        r = await engine.synthesize(piece, f"{out_stem}_p{k}", speed, **piece_kwargs)
        cjk += r.cjk
        latin += r.latin
        if r.audio_path:
            clips.append(r.audio_path)
        sentences.append((a, b, r.duration_ms))
    total = sum(d for _, _, d in sentences)
    audio_path: Optional[str] = None
    if clips:
        audio_path, measured = _concat(clips, out_stem, want_mp3=want_mp3)
        for p in clips:
            try:
                os.remove(p)
            except OSError:
                pass
        if measured:
            # rescale sentence durations to the measured total (wav concat is exact anyway)
            if total and abs(measured - total) > 50:
                sentences = [(a, b, int(d * measured / total)) for a, b, d in sentences]
            total = measured
    marks = interpolate_marks(text, sentences)
    return TtsResult(audio_path, total, cjk, latin, engine.name, marks=marks)
