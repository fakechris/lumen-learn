# Lumen Learn (Socratic Whiteboard / 苏格拉底互动白板)

<p align="center">
  <b>A protocol-driven Socratic AI tutor with synchronized handwriting, voice narration, 2D/3D interactive widgets, and adaptive learning.</b><br>
  <b>基于全双工 Action 协议的苏格拉底式 AI 互动白板导师系统：音画同步、逐步板书、探针交互教具、即时打断与自适应因材施教。</b>
</p>

---

## What is this?

**Lumen Learn** compiles a textbook PDF or lecture markdown into a live, one-on-one whiteboard lesson:

- **Synchronized handwriting & audio** — notes appear line by line on a handwritten canvas, in lockstep with voice narration (character-level TTS marks drive the client's audio clock).
- **Visual signals** — formulas and key conclusions get circled, underlined and spotlighted exactly as they are spoken.
- **Interactive explorables** — sandboxed 2D canvas widgets (live probes, envelopes) and 3D scenes, with teacher-driven controls timed to speech.
- **Socratic gates** — the tutor pauses to ask prediction and concept-check questions; wrong answers climb a remediation ladder (re-ask → deeper retelling → prerequisite replay).
- **Interruptions** — speak up or type at any moment; the tutor branches into a 1–3 step detour, then returns to the exact mainline beat.
- **Adaptive teaching** — entry diagnosis, novice/standard/fast play policies, per-step deeper/compressed variants, a four-axis mastery model and a course concept graph.
- **BYOK** — bring any OpenAI-compatible key (DeepSeek / Kimi / Ollama / vLLM…) via the in-app ⚙ settings panel; keys stay in a git-ignored local file and apply without a restart. TTS engines: macOS `say` / edge-tts / MiniMax.

Everything runs on a typed wire protocol (`src/protocol/actions.py` — the single source of truth) with an explicit `action_step_complete` handshake and reveal gating.

## Repository layout

This repository holds the **runnable system only**. Course packages, lecture sources, design/decision docs, research and evaluation data live in the private companion repo **`lumen-learn-class`**:

```bash
git clone git@github.com:fakechris/lumen-learn.git
git clone git@github.com:fakechris/lumen-learn-class.git   # private: courses, docs, eval data
export HK_COURSES_ROOT="$PWD/lumen-learn-class/courses"   # serve its course catalog
```

A tiny synthetic demo course ships here (`examples/courses/course_demo`) so the UI is never empty; real content comes from the companion repo or your own generation.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# no keys needed to browse and play compiled courses
.venv/bin/uvicorn server.app:app --port 8000
# open http://localhost:8000 → ⚙ settings → paste an OpenAI-compatible key → generate

# CLI generation from your own lecture markdown / textbook PDF
.venv/bin/python -m src.content.pipeline --input my_lecture.md --mode llm

# tests
.venv/bin/python -m pytest
```

Requirements: Python ≥ 3.11, `ffmpeg` for audio transcode; Node ≥ 18 only for optional headless widget QA.

## Architecture

```
textbook.pdf / lecture.md
   ▼ [Ingest]  document parser: TOC, headings, page refs, figures
   ▼ [Plan]    curriculum planner (LLM): Unit → Session → Segments (editable plan tree)
   ▼ [Build]   session synthesizer: validators · widget/illustration/exercise generators · TTS
   ▼ [Compile] CompiledSession — ordered typed actions with step_ids, reveal gates, timings
   ▼ [Run]     FastAPI + WebSocket runtime (teaching / awaiting_answer / interjecting / paused)
   ▼ [Client]  vanilla ES modules — audio clock → handwritten page → KaTeX decorations → sandboxed iframes
```

## License

MIT. Course packages and generated content in the companion repo have their own provenance rules — see `lumen-learn-class/README.md`.
