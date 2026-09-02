# Implementation Plan: Socratic Whiteboard Rebuild

Goal: replace the demo skeleton with a working, protocol-driven Socratic whiteboard
tutor whose protocol mirrors the real Lumen Learn whiteboard (verified against the
captured bundles in `research/`, not the design doc).

## Stage 1: Protocol & compiled session model
**Goal**: Single source of truth for server→client actions and client→server messages.
`src/protocol/actions.py`, `src/protocol/session.py`. A session is an ordered list of
actions with `step_id`s; the same protocol serves offline-compiled and live sessions.
**Success Criteria**: pydantic models round-trip JSON; compiler tests pass.
**Status**: Complete

## Stage 2: Real TTS with real durations
**Goal**: Pluggable TTS (`macOS say` → WAV, edge-tts, silent fallback) returning real
`duration_ms` plus CJK/Latin character counts; LaTeX→spoken-text preprocessing.
**Success Criteria**: engine tests pass; generated WAV duration matches file header.
**Status**: Complete

## Stage 3: Content generation with validation
**Goal**: LLM structured-output planner + per-step session synthesizer, separate widget
generation with failure degradation, validators (snippet-in-board, one-correct-option,
balanced math), honest heuristic mode, stable IDs, fail-fast (no silent template fallback).
**Success Criteria**: `python -m src.content.pipeline --input examples/...` produces a
course package with DeepSeek; validators tested.
**Status**: Complete

## Stage 4: Session runtime + server
**Goal**: Per-connection asyncio state machine (teaching / awaiting_answer / interjecting /
paused) streaming actions with `action_step_complete` handshake and fail-safe timeouts;
live tutor for interjections and answer feedback; REST for course packages and async
generation jobs.
**Success Criteria**: runtime tests with fake transport; WS test via TestClient.
**Status**: Complete

## Stage 5: State-driven whiteboard client
**Goal**: ES-module client: WS client, audio-clock sync engine (typewriter, progressive
board reveal, decoration timing), column-packing layout with follow/newcol/fit-visible,
DOM-located snippet decorations, sandboxed widgets with error reporting, ask/interject UI.
**Success Criteria**: end-to-end session plays in browser with synced audio.
**Status**: Complete

## Verification log (2026-09-02)
- `pytest`: 22 passed (protocol round-trip, compiler ordering/gating, validators, TTS engines, runtime handshake/interjection/fail-safe, server REST + WebSocket).
- DeepSeek run on `examples/linear_algebra_basis.md`: 3 sessions, 4 Three.js widgets passed static checks, 4 bad snippets/trigger phrases dropped by validators.
- Headless Chromium: gated reveal in sync with clock, circles located on KaTeX formulas, sandboxed widget interactive, wrong-answer feedback narrated live, interjection streamed + spoken + resumed, pause/speed/navigation, generate modal (heuristic) end to end.
- Known gaps: no PDF parsing without `pymupdf`; no voice input for interjections; `new_column` action is compiled from `layout: "newcol"` on cards rather than emitted standalone.
