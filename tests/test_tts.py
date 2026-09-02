import os
import shutil

import pytest

from src.tts.engine import MacSayEngine, SilentEngine, choose_engine, wav_duration_ms
from src.tts.spoken_text import count_chars, estimate_duration_ms, to_spoken


def test_to_spoken_reads_formulas_aloud():
    assert to_spoken("组合 $c_1 \\vec{v}_1 + c_2 \\vec{v}_2$ 是平面") == "组合 c1 v1 加 c2 v2 是平面"
    assert to_spoken("$\\frac{a}{b}$") == "b 分之 a"
    assert to_spoken("$x^2 + y^{10}$") == "x平方 加 y 的10次方"
    assert to_spoken("**粗体** 和 `code` 与 [链接](http://x)") == "粗体 和 code 与 链接"
    assert "$" not in to_spoken("$$\\text{span}(\\vec{v}_1) = V$$")


def test_count_and_estimate():
    assert count_chars("你好 abc 12") == (2, 5)
    assert estimate_duration_ms("") == 800
    assert estimate_duration_ms("这是一句四十二个字左右的话。" * 3) > 5000


@pytest.mark.asyncio
async def test_silent_engine_estimates_duration(tmp_path):
    res = await SilentEngine().synthesize("你好，世界。", str(tmp_path / "x"))
    assert res.audio_path is None and res.duration_ms >= 800 and res.cjk == 4


@pytest.mark.asyncio
@pytest.mark.skipif(not MacSayEngine.available(), reason="macOS say not available")
async def test_mac_say_engine_produces_wav_with_measured_duration(tmp_path):
    res = await MacSayEngine(mp3=False).synthesize("两个向量张成一个平面。", str(tmp_path / "step"))
    assert res.audio_path.endswith(".wav")
    assert res.duration_ms == wav_duration_ms(res.audio_path)
    assert 1000 < res.duration_ms < 8000


@pytest.mark.asyncio
@pytest.mark.skipif(not MacSayEngine.available() or not shutil.which("ffmpeg"), reason="needs say + ffmpeg")
async def test_mac_say_engine_transcodes_to_mp3(tmp_path):
    res = await MacSayEngine(mp3=True).synthesize("你好。", str(tmp_path / "step"))
    assert res.audio_path.endswith(".mp3") and os.path.getsize(res.audio_path) > 1000
    assert not os.path.exists(str(tmp_path / "step.wav"))
    assert res.duration_ms > 200


def test_choose_engine_respects_preference():
    assert choose_engine("silent").name == "silent"
    with pytest.raises(RuntimeError):
        choose_engine("nope")
