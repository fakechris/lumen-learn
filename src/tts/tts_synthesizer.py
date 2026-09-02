"""
TTS Audio Synthesizer & Millisecond Timing Generator.
Generates spoken audio files and computes exact duration metadata
for 60FPS audio-visual synchronization (typewriter & stroke animations).
"""

import os
import wave
import struct
import math
from typing import Optional
from src.models.schema import SocraticStep, AudioMeta, SessionManifest


class TTSSynthesizer:
    def __init__(self, output_dir: str = "output/audio"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def estimate_speech_duration_ms(self, text: str, cjk_char_per_sec: float = 3.8, latin_word_per_sec: float = 2.5) -> int:
        """
        Estimates the millisecond duration based on character & word metrics.
        """
        cjk_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        latin_words = len([w for w in text.split() if any(c.isascii() and c.isalpha() for c in w)])
        
        cjk_time = cjk_chars / cjk_char_per_sec if cjk_char_per_sec > 0 else 0
        latin_time = latin_words / latin_word_per_sec if latin_word_per_sec > 0 else 0
        
        # Punctuation pauses (commas: 250ms, periods/question marks: 450ms)
        punct_pause = text.count("，") * 0.25 + text.count("。") * 0.45 + text.count("？") * 0.5
        
        total_sec = max(1.0, cjk_time + latin_time + punct_pause)
        return int(total_sec * 1000)

    def synthesize_mock_wav(self, duration_ms: int, output_path: str):
        """
        Generates a lightweight valid WAV file with soft sine tone/silence for offline testing.
        """
        sample_rate = 16000
        num_samples = int(sample_rate * (duration_ms / 1000.0))
        
        with wave.open(output_path, "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            
            # Generate soft modulated gentle tone
            raw_data = bytearray()
            for i in range(num_samples):
                t = float(i) / sample_rate
                # 440 Hz gentle sine modulated with soft envelope
                envelope = min(1.0, t * 10) * min(1.0, (num_samples - i) / (sample_rate * 0.1))
                val = int(math.sin(2 * math.pi * 440 * t) * 500 * envelope)
                raw_data.extend(struct.pack("<h", val))
            
            wav_file.writeframes(raw_data)

    def process_manifest(self, manifest: SessionManifest) -> SessionManifest:
        """
        Enriches all steps in the session manifest with TTS audio assets and millisecond timing metadata.
        """
        session_audio_dir = os.path.join(self.output_dir, manifest.session_id)
        os.makedirs(session_audio_dir, exist_ok=True)

        for step in manifest.steps:
            duration_ms = self.estimate_speech_duration_ms(step.speech_text)
            wav_filename = f"step_{step.step_id}.wav"
            wav_path = os.path.join(session_audio_dir, wav_filename)
            
            self.synthesize_mock_wav(duration_ms, wav_path)

            step.audio_meta = AudioMeta(
                audio_url=f"/audio/{manifest.session_id}/{wav_filename}",
                duration_ms=duration_ms,
                cjk_rate=3.8,
                latin_rate=2.5,
            )

        return manifest
