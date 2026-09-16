# Lumen Learn — agent notes

Protocol source of truth: `src/protocol/actions.py`. Run tests with
`.venv/bin/python -m pytest`; the server is `.venv/bin/uvicorn server.app:app`.
Generated packages go to `output/` (git-ignored). Never add silent fallbacks to
canned content: fail loudly or label the mode (`generation_mode`).

Repository scope: this repo holds the runnable system only. Course packages,
lecture sources, design/decision docs, research and acceptance data live in the
private companion repo `fakechris/lumen-learn-class`; mount its catalogs with
`LUMEN_COURSES_ROOT` (colon-separated roots; the legacy `HK_COURSES_ROOT` is
still accepted). `examples/courses/course_demo` is a
synthetic fixture — do not replace it with real course content.
