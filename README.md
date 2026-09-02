# Socratic Whiteboard

An open, protocol-driven re-creation of the Lumen Learn-style "Socratic whiteboard" tutor:
a lecture note goes in, and out comes an interactive lesson where a tutor voice narrates,
handwritten notes appear line by line in sync with the speech, formulas get circled as they are mentioned,
a sandboxed 3D manipulative shows the geometry, and the tutor stops to ask you questions.
You can interrupt at any time and ask your own.

The wire protocol mirrors the real Lumen Learn whiteboard client (verified against the
captured front-end bundles), not a guess: `speak / tts_segment / board / circle /
highlight / graph / ask / new_column / new_page / generated_animation / reward_user / done`
with a per-step `action_step_complete` handshake and reveal gates.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
export DEEPSEEK_API_KEY=...        # or OPENAI_API_KEY / ANTHROPIC_API_KEY (optional)
.venv/bin/uvicorn server.app:app --port 8000
open http://localhost:8000
```

A bundled example course (`examples/courses/`) plays without any API key. With a key the
"从讲义生成课程" button produces a full Socratic course from pasted Markdown, and the tutor
answers interruptions and grades free-text answers live.

TTS: on macOS the built-in `say` voice is used automatically (MP3 if `ffmpeg` is installed).
`pip install edge-tts` for neural voices (`TTS_ENGINE=edge`). `TTS_ENGINE=silent` runs with
a virtual clock and no audio.

Generate from the command line:

```bash
.venv/bin/python -m src.content.pipeline --input examples/linear_algebra_basis.md --mode llm
```

## Architecture

```
lecture.md ─▶ document_parser ─▶ curriculum_planner (LLM, structured) ─▶ CourseStructure
                                                 │
                       per session:  session_synthesizer (LLM) ─▶ SessionScript
                                     ├─ validators   (snippet ∈ board, math balanced, 1 correct option)
                                     ├─ widget_generator (separate LLM call, static checks, degrades on failure)
                                     └─ tts engine   (real audio + measured duration)
                                                 ▼
                                     compiler ─▶ CompiledSession (ordered actions with step_ids)
                                                 ▼
server/app.py  ──WebSocket──▶  runtime/session_runtime.py  (teaching / awaiting_answer / interjecting / paused)
                                                 ▼
client/js  (ws → audio clock → board layout → decorations → sandboxed widgets)
```

Key decisions:

- **One protocol for offline and live.** Pre-generated sessions are compiled to the same action
  stream a live agent would emit, so the client has a single consumption path.
- **The client owns the clock.** `audio.currentTime` drives the subtitle typewriter, progressive
  card reveal and annotation timing; the server only knows measured durations.
- **Content is validated, never faked.** Bad snippets are dropped with warnings, failed widgets
  degrade to `animation_failed`, and with no LLM the pipeline runs an honest read-through mode
  labelled `heuristic` instead of pretending.
- **Widgets are sandboxed.** `sandbox="allow-scripts"` only, with an injected shim that reports
  runtime errors back to the host.
- **Widgets are 2D explorables by default.** The real product's "interactive H5" is a
  function plot with dashed envelopes, a pointer probe and a live readout, not a 3D scene.
  `kind: "explorable"` generates a zero-dependency Canvas widget from a hand-written exemplar
  (`examples/authored/squeeze_explorable.html`); every generated widget is rendered headlessly
  with Playwright (`tools/render_widget.mjs`) to catch runtime errors and blank output, and is
  regenerated once with the error as feedback before being dropped. Three.js stays available
  for genuinely 3D concepts.
- **Figures are drawn, not painted.** Pedagogical diagrams need exact counts and labels, so the
  default illustration is an LLM-drawn SVG (`kind: "svg"`); MiniMax `image-01` is available for
  scene metaphors (`kind: "image"`, `MINIMAX_API_KEY`).
- **The board is one handwritten page**, not cards: telegraphic notes with indentation, a
  highlighted page title, figures with handwritten captions, and a centered subtitle. The
  session-synthesis prompt encodes this and ships a hand-authored exemplar
  (`examples/authored/`), which is what moved model output from lecture prose to the target.

## Layout

| Path | What |
| --- | --- |
| `src/protocol/` | Wire protocol (`actions.py`) and content/compiled models (`session.py`) |
| `src/content/` | Parser, planner, synthesizer, validators, widget generator, compiler, pipeline, store |
| `src/tts/` | TTS engines with measured durations; LaTeX-to-speech preprocessing |
| `src/llm/` | Async LLM client with structured output and self-repair |
| `src/runtime/` | Per-connection session state machine and live tutor |
| `server/app.py` | FastAPI: REST for packages and generation jobs, WebSocket for sessions |
| `client/` | ES-module whiteboard client (no build step) |
| `tests/` | `pytest` (`.venv/bin/python -m pytest`) |
| `tools/render_widget.mjs` | Headless widget render check (needs `node` and a global `playwright`) |
| `research/` | Captured Lumen Learn bundles and notes (git-ignored) |

Course packages live in `examples/courses/<course_id>/` (bundled) and `output/<course_id>/`
(generated): `course_structure.json`, `scripts/*.json` (editable source), `sessions/*.json`
(compiled), `audio/`.
