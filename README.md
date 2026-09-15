# Lumen Learn (Socratic Whiteboard / 苏格拉底互动白板)

<p align="center">
  <b>A protocol-driven Socratic AI tutor with synchronized handwriting, voice narration, 2D/3D interactive widgets, and adaptive learning.</b><br>
  <b>基于全双工 Action 协议的苏格拉底式 AI 互动白板导师系统：音画同步、逐步板书、探针交互教具、即时打断与自适应因材施教。</b>
</p>

<p align="center">
  <a href="#english">English</a> •
  <a href="#中文说明">中文说明</a> •
  <a href="#milestones--roadmap-阶段里程碑">Milestones & Roadmap</a> •
  <a href="#quick-start-快速上手">Quick Start</a> •
  <a href="#architecture-系统架构">Architecture</a>
</p>

---

<a name="english"></a>
## English

### What is Lumen Learn?

**Lumen Learn** is an open-source, protocol-driven Socratic whiteboard AI tutor. Input a textbook PDF or lecture markdown, and Lumen Learn compiles it into a deeply engaging, multimodal interactive lesson:
- **Synchronized Handwriting & Audio**: Notes appear line by line on a single handwritten canvas in tight synchronization with voice narration.
- **Visual Signals**: Formulas and key concepts are circled, underlined, or spotlighted exactly as they are spoken.
- **Interactive Explorables**: Sandboxed 2D canvas manipulatives (curves, dashed envelopes, live pointer probes) and 3D scenes demonstrate dynamic principles.
- **Socratic Inquiry**: The tutor pauses to ask prediction and concept-checking questions.
- **Branch-and-Return Interruptions**: Students can speak up or type questions at any moment. The tutor temporarily branches into a detour explanation, then returns seamlessly to the lecture flow.
- **Adaptive Teaching**: Generates explanation ladders, entry diagnostic quizzes, prerequisite remediation loops, and dynamic course concept graphs.

The entire system is powered by a typed wire protocol (`speak`, `tts_segment`, `board`, `circle`, `highlight`, `graph`, `ask`, `new_column`, `new_page`, `generated_animation`, `reward_user`, `done`) with an explicit `action_step_complete` handshake and reveal gating.

---

<a name="中文说明"></a>
## 中文说明

### 项目简介

**Lumen Learn** 是一个基于第一性原理打造的、面向深度理解的苏格拉底式 AI 互动白板教学系统。系统输入教材 PDF 或讲义 Markdown，即可端到端编译为具备高度交互性的一对一私教课堂：
- **音画同频与手绘墨迹**：讲到哪里写到哪里，告别机械幻灯片，在单页手绘纸质画布上逐步呈现提纲挈领的板书。
- **动态信号系统**：公式、符号与关键结论随语音节奏精准触发红圈标注、划线、聚光灯与色块高亮。
- **探针式可交互教具（Explorables）**：内嵌免依赖的 2D Canvas 探针图（如夹逼定理、曲线包络线、实时读数探针）及 3D 几何教具，学生可亲手拨动参数观察变化。
- **苏格拉底式发问**：每节课设置认知断点与预测设问，引导学生先猜再验证，针对误概念给予阶梯提示与针对性解析。
- **随时插话与分岔回归**：支持学习者在任意节点打断提问，系统生成 1~3 步的“岔路解答”分支，讲完后精准回归主线进度。
- **因材施教与认知闭环**：结合入口诊断分层、快慢学生认知弧线裁剪、先修知识补救回路、四维掌握度模型与手绘概念图谱。

---

## Key Features / 核心特性

| Feature 模块 | Description (EN) | 核心说明 (CN) |
| :--- | :--- | :--- |
| **Audio-Clock Driven** | Client-side audio clock drives typewriter, progressive board reveal, and annotation timings. | 客户端音频时钟主导字幕打字机、渐进式板书展开与标注时序，毫秒级严丝合缝。 |
| **2D & 3D Explorables** | Zero-dependency 2D Canvas charts with `HandChart` wobbly ink axes, live probes, and teacher-driven controls. | 内置手绘风格 HandChart 探针图表与 3D WebGL，支持教师端按语音锚点驱动教具参数。 |
| **Headless Render QA** | Playwright runs headless rendering verification on generated widgets to prevent runtime errors or blank outputs. | 自动化 Playwright 无头渲染质检，在生成期捕获脚本报错或空白并自动反馈重试。 |
| **Bilingual TTS** | Measured speech durations, LaTeX-to-speech phoneme conversion, character/word-level marks (Edge-TTS / macOS / MiniMax). | 支持多引擎真实时长测量与 LaTeX 公式口语化转写，支持字符/单词级同步锚点。 |
| **Feynman Round** | "Explain to me" mode where a curious peer agent probes the user's understanding for 4 progressive rounds. | 讲给我听（费曼演练）：学生用自己的话解释，AI 好奇学伴连续 4 轮追问深度检验。 |
| **Mastery & Concept Map** | Append-only learner evidence feeding a 4-axis mastery model and dynamic typed concept graph. | 学习证据驱动的四维掌握度模型，动态生成课程概念拓扑网络与掌握度热力映射。 |
| **Printable Cheatsheet** | ≤2-page A4 cheatsheet compiled per course from the concept map, step takeaways, key boards, and misconception correctives — no LLM, pure data compilation. | 每门课可打印的考前两页速查表：概念速览、关键公式、概念关系与常见误区，纯课程包数据编译，不额外生成内容。 |
| **BYOK Settings** | Configure any OpenAI-compatible endpoint (DeepSeek / Kimi / Ollama / vLLM) and TTS engine in the UI; keys stay in a git-ignored local file and apply without a restart. | 界面内自带 Key：任意 OpenAI 兼容端点与语音引擎即配即用，密钥仅存本机且不进 git，保存后无需重启。 |

---

<a name="milestones--roadmap-阶段里程碑"></a>
## Milestones & Roadmap / 阶段里程碑

本项目严格按照分期工程推进，Stage 1 ~ 11 已全部交付（78 项自动化测试全绿）；Stage 10 北极星基线与 Stage 11 零门槛上手面均已落地：

| Milestone 里程碑 | Target & Scope 目标与交付范围 | Status 状态 | Verification 验证方式 |
| :--- | :--- | :---: | :--- |
| **Stage 1: Wire Protocol** | Single source of truth wire protocol (`actions.py`), compiled session model, JSON serialization, and step handshake. | ✅ Complete | Pydantic model roundtrips, compiler gating tests |
| **Stage 2: Real TTS Engine** | Pluggable TTS engines (`macOS say`, `edge-tts`, `minimax`, `silent`) with real measured durations and LaTeX preprocessing. | ✅ Complete | Audio duration header verification, LaTeX phoneme conversion tests |
| **Stage 3: Content Pipeline** | LLM structured-output planner, session synthesizer, and strict validators (snippet-in-board, math balance, one correct option). | ✅ Complete | E2E pipeline run on benchmark courses, failure degradation checks |
| **Stage 4: Session Runtime** | State machine (teaching, awaiting answer, interjecting, paused) streaming via WebSocket, live tutor for interjections. | ✅ Complete | WebSocket TestClient, fake transport handshake & timeout tests |
| **Stage 5: Whiteboard Client** | ES-module client, audio-clock sync engine, column-packing layout, DOM-located KaTeX annotations, sandboxed widgets. | ✅ Complete | End-to-end browser playback with synced audio, KaTeX circle placement |
| **Stage 6: Visual Styling Parity** | Handwritten single-page aesthetic (LXGW WenKai), telegraphic boards, LLM-generated SVG diagrams with geometry conflict checks. | ✅ Complete | Visual QA matching reference lessons, multi-column board packing |
| **Stage 7: 2D Explorables & QA** | Zero-dependency 2D Canvas explorables with pointer probes, Playwright headless render check with one feedback regeneration. | ✅ Complete | Playwright ink sampling check, authored squeeze theorem lesson |
| **Stage 8: Textbook Ingest & Plan Tree** | PDF textbook parsing, hierarchical course planner (`ingest → plan → build`), editable plan tree, semantic fill-blank grading. | ✅ Complete | CS251 lecture PDF parsing, 18 post-session exercises, semantic grader |
| **Stage 9: Cognitive Tools & Mastery** | `HandChart` base, teacher-driven widget hooks (`hkControl`), spotlight mask, Feynman rounds, 4-axis mastery, course concept map. | ✅ Complete | Interactive widget control at speech phrases, typed concept graph render |
| **Stage 10: Adaptive Teaching** | Cognitive-arc beats, entry diagnostic quiz, explanation ladder (deeper / compressed), remediation loops, simulated-student north-star harness; frozen evaluation contract with A/B parallel forms, leak checks and known-bad registry. | ✅ Complete | `tools/sim_student.py --eval-set` across novice/standard/fast personas; `src/content/eval_contract.py` |
| **Stage 11: First Ten Minutes** | BYOK settings panel (live reconnect, masked keys), per-course printable cheatsheets, seed catalog grown to 10 bundled courses across NN/LLM, physics, statistics, CS and math. | ✅ Complete | 78-item pytest suite, headless Chrome print check, per-course sim gating |

---

<a name="architecture-系统架构"></a>
## Architecture / 系统架构

```
textbook.pdf / lecture.md
   │
   ▼ [Ingest] document_parser: TOC / font-size headings, page refs, figure extraction
ParsedDocument
   │
   ▼ [Plan] curriculum_planner (LLM, hierarchical for long documents)
CourseStructure (Unit → Lecture → Session → Segments with media decisions)
   │  (Reviewable & editable in UI before execution)
   ▼ [Build] session_synthesizer
   ├── Validators: snippet-in-board, balanced math, unambiguous options
   ├── Widget Generator: 2D Canvas explorables / Three.js + Playwright render QA
   ├── Illustration Generator: LLM-drawn SVG / diagram geometry conflict check
   ├── Exercise Generator: fill-blank, multiple-choice, interactive + semantic grader
   └── TTS Engine: real duration synthesis + LaTeX pronunciation preprocess
   │
   ▼
CompiledSession (ordered actions, step_ids, keypoints, exercises)
   │
   ▼ [WebSocket]
server/app.py ──▶ runtime/session_runtime.py (teaching / interjecting / paused)
   │
   ▼ [Full-Duplex Stream]
client/js (audio clock → handwritten page → KaTeX decorations → sandboxed iframe)
```

---

<a name="quick-start-快速上手"></a>
## Quick Start / 快速上手

### 1. Requirements 环境要求

- Python >= 3.11
- Node.js >= 18 (用于无头交互教具渲染测试，可选)
- `ffmpeg` (用于音频转码)

### 2. Installation 安装

```bash
# Clone the repository
git clone https://github.com/fakechris/lumen-learn.git
cd lumen-learn

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Environment Configuration 环境变量

Configure LLM and TTS keys (DeepSeek / OpenAI / Anthropic / MiniMax):

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
# Optional overrides:
# export LLM_MODEL="deepseek-chat"
# export LLM_MODEL_PRO="deepseek-reasoner"
# export TTS_ENGINE="edge"   # or "say" on macOS, "minimax", "silent"
```

### 4. Run Server & Web UI 启动服务

```bash
# Start FastAPI and WebSocket server
.venv/bin/uvicorn server.app:app --port 8000 --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser:
- **Play bundled lessons**: Browse the bundled courses in `examples/courses/` directly without any API keys.
- **Generate new courses**: Paste your own lecture Markdown or upload a textbook PDF to run the automated course builder.
- **Live interruptions**: Hit the microphone or keyboard button during playback to interrupt and probe the AI tutor.

### 5. CLI Course Generation 命令行生成

```bash
# Generate a course from markdown
.venv/bin/python -m src.content.pipeline --input examples/linear_algebra_basis.md --mode llm

# Plan-only mode (generates reviewable plan tree first)
.venv/bin/python -m src.content.pipeline --input examples/linear_algebra_basis.md --plan-only
```

### 6. Run Test Suite 运行测试

```bash
.venv/bin/python -m pytest
```

---

## Directory Layout / 工程目录

```text
lumen-learn/
├── client/                 # ES-module frontend (vanilla JS, CSS, no build step needed)
│   ├── js/                 # Audio clock, board layout, decorations, widgets, Feynman
│   └── css/                # Handwritten paper styling, typography, theme
├── src/
│   ├── protocol/           # Wire action schemas (actions.py) & session models (session.py)
│   ├── content/            # Ingest, planner, synthesizer, validators, widgets, compiler
│   ├── runtime/            # Per-connection state machine, live tutor & interruption handler
│   ├── tts/                # Pluggable TTS engines & LaTeX-to-speech processor
│   ├── llm/                # Async LLM adapter with structured JSON repair
│   └── obs/                # Observability & cost accounting
├── server/
│   └── app.py              # FastAPI REST endpoints & WebSocket server
├── tools/
│   ├── render_widget.mjs   # Playwright headless widget QA runner
│   └── sim_student.py      # Multi-persona simulated-student evaluation harness
├── examples/               # Hand-authored reference lessons & bundled course packages
└── tests/                  # 64-item pytest verification suite
```

---

## License

This project is licensed under the [MIT License](LICENSE).
