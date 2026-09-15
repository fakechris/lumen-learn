# Implementation Plan: Socratic Whiteboard Rebuild

Goal: replace the demo skeleton with a working, protocol-driven Socratic whiteboard
tutor with a protocol-driven interactive whiteboard architecture.

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

## Stage 6: Close the quality gap (2026-09-02, target styling and interaction parity)
**Goal**: Match the reference look and content style: handwritten single page, telegraphic
boards, narration that points at the page, figures with exact labels, colloquial prediction
questions; illustration modality (LLM-drawn SVG, optional MiniMax image); prompt with exemplar.
**Success Criteria**: DeepSeek output on the same lecture reads like the hand-authored baseline;
rendered page resembles the reference screenshots.
**Status**: Complete. Verified: boards 3-5 lines, speech points at board/figure, options colloquial,
SVG figure with labels rendered with caption, circles on formulas and text.
**Remaining**: per-student name in narration (needs live TTS or SSML placeholders); voice
input for interjections.

## Stage 7: Widgets as 2D explorables + render check (2026-09-02, from the widget screen recording)
**Goal**: Match the real interactive widget (2D plot, dashed envelopes, pointer probe, live
readout, muted beige aesthetic) and close the visual QA loop.
**Status**: Complete. `explorable` widget kind with a hand-written exemplar in the prompt;
Playwright render check (runtime errors, blank output) with one feedback regeneration;
authored 夹逼定理 session bundled as the reference.

## Stage 8: Textbook → 教案 → course, exercises, structure/polish (2026-09-02)
**Goal**: Start from a PDF textbook; plan before generating; decide media per segment; add
post-session exercises; match the remaining product screens (course structure tree with tags,
tables in notes, 课堂要点 progress list, typed transcript, end-of-session message).
**Status**: Complete. PDF parsing with figures; `ingest -> plan -> build`; staged UI with editable
plan tree (units, lectures, tags, per-segment media); exercises with semantic fill-blank grading,
choice, interactive; keypoints list; typed transcript; tables in boards.
**Verified**: CS251 lecture PDF → 3 chapters / 4 sessions / 14 segments with sensible media;
18 exercises; "变得更高" accepted for "升高" with justification.
**Remaining**: handwriting font choice (user will pick later); student name in narration;
voice interjections; multi-page (new_page) boards for very long sessions.

## Stage 9: Absorb the OSS borrow list (2026-09-03, after reviewing the course/glm-manual-lesson worktree)
**Goal**: Land the seven research items (research/COMPONENT_BORROW_CATALOG.md) on main, each verified in the browser.
**Success Criteria**: generated widgets use HandChart and expose `hkControl`; teacher controls fire at the spoken
trigger phrase; spotlight dims the page at its trigger; Feynman round runs against the live LLM; mastery bars
appear on the home after grading; concept map renders for a 3-session and a 111-session course.
**Status**: Complete. Absorbed as-is from the worktree: board contract + gadget rules + exercise audit + storage shim +
scene-object check + detour discipline (9c6c7c9), SVG geometry conflicts (7843055), spotlight (637017f).
Reworked on main: HandChart (overlays, legends, fonts in the iframe), widget controls (trigger-phrase timing, per-step
ticks, `set` op, param validation — the worktree's `set` vs `setState` mismatch meant nothing fired), Feynman
(broken home button, quality scores), mastery (worktree added score for wrong answers; now wrong earns nothing,
diminishing monotone gains, tutor-judged quality for open answers), concept map (worktree drew broken circles on a
ring with course-order edges; now an LLM-built typed graph cached per course, layered hand-drawn rendering).
Also: client js/css served no-cache (a cached module made spotlights draw as circles).

## Stage 10: 因材施教 — 教会为北极星（2026-09-03 定向，见 SYSTEM_DESIGN §10）
**Goal**: 同一份教案对不同基础的学生讲得不一样，且学不会时能反复讲到明白。
**Success Criteria**: 模拟学生评测集（20 节 × 3 种学生）上，三种学生的课后测通过率都高于"按教案直播"基线；
novice 的先修补救能回到主线断点；fast 学生一节课时长明显缩短且通过率不降。
**Order** (用户定：1 → 2 → 4 → 3):
1. 教案分 beat + 入口诊断分层 + 讲解梯子（deeper / compressed 按需生成并缓存）+ 补救回路（换法 → 回先修 → 换形式）；
2. 媒体配额与教具计划层（每节至少一个可动的量；`{objects, anchors, timeline}` 先于代码）；
4. 模拟学生评测集与北极星指标；审阅模式把人的反馈写进 DB 并触发 `--only` 重生成；
3. 声音与节奏（TTS 引擎、句级停顿、首音延迟）最后做。
**Status**: In Progress. Done (2026-09-03): beats; entry diagnosis + evidence-based level; fast skips hook/analogy,
novice gets a 先修回顾 column; 我懂了 skip with auto-promotion; remediation ladder (re-ask → deeper variant, cached →
prerequisite replay → explain); `tools/sim_student.py` harness. First run (3 sessions × 3 personas, baseline vs adaptive,
$0.05): post-test 0.89 in both modes — the 3-item post-tests are too easy to discriminate yet; the harness did surface a
content defect (sess_3 gate "为什么批处理时要把 W 写成 nin×nout" — every persona picks the "矩阵乘法" option, the key says
"每一列对应一个神经元"), now reported as 可疑提问. Then (same day): transfer post-tests generated per session + cold pre-test → learning gain; novice personas hold the
session's cognitive hurdle as a belief (sess_3 novice: pre 2/6 → post 5/6, +0.50; sess_2's hurdle does not bite, pre 6/6);
beats flow plan → synthesis with per-beat writing rules; compressed variants precomputed (`--variants-for`) and played
for fast learners (sess_3 fast: hook skipped, derive replaced by 要点, gate kept, ~25 s to the first gate);
media-quota pass in the planner; widget technical plan (grid anchors) before code; SVG grid layout rule.
Open: the LLM student still knows too much for most items (gain is only visible where a misconception is planted) —
real-learner data or misconception-targeted items are the honest ruler; 111-session variant precompute not run yet
(cost ≈ $0.5, ~40 min of `say`). Suspicious gate sess_3 step 8 rewritten (INV-258, 2026-09-15): the "why nin×nout"
stem had two defensible options (dimension legality vs column semantics) — split into a definitional column-semantics
gate (rewritten step 8) plus a new dimension-legality step 12 with its own board/narration/gate; scripts+sessions
rewritten in place, variants renumbered 9→13, suspicious-gate detection extracted to
`find_suspicious_gates()` with a fixture test reproducing the original report (see output/_evidence/inv-258/).
