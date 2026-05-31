# research-builder UI

Vite + React + TS frontend for the harness web UI. Served by the FastAPI
backend (`research-builder-app`), which also spawns the reproduction pipeline.

## Single-command launch (recommended)

```bash
cd custom-harness/frontend
npm install
npm run build                          # writes frontend/dist/

uv run research-builder-app                           # launcher mode — opens http://127.0.0.1:7777
```

Drop a PDF in the browser; the backend creates a workspace under
`./runs/<paper-stem>/` and spawns `research-builder --auto` against it.
Override the runs directory with `--runs-dir /path/to/runs`.

## Viewing an existing run

```bash
uv run research-builder-app /path/to/workspace        # serves the workspace read-only
```

The pipeline is NOT respawned — the UI just lets you browse a completed
or in-progress run.

## Dev (two terminals)

```bash
# 1. Backend — serves API + WS on :7777
uv run research-builder-app --no-open

# 2. Frontend — Vite dev server on :5173, proxies /api + /ws to :7777
cd custom-harness/frontend
npm install
npm run dev
# open http://localhost:5173
```

FastAPI auto-detects the built bundle at `frontend/dist/` (local dev install) or
`research_builder/web/_static/` (for installed wheels — populate by copying
`frontend/dist/*` into that dir before packaging).

## Keyboard shortcuts

- `⌘K` / `Ctrl+K` — command palette (switch tabs, retry phases, focus chat)
- `/` — focus the Paper Agent chat input
- `Esc` — close palette / modal

## Tests

```bash
cd custom-harness
uv run pytest tests/test_web.py -v
```

Covers cascade computation, command-channel writes, and path-traversal protection
on the file API.

## Stack notes

- **pdfjs-dist** directly (not the `react-pdf` wrapper) — we need page-
  navigation + text-selection events for the bidirectional PDF ↔ spec
  linking in Phase 2.
- **Tailwind CSS v4** — CSS-first config, no `tailwind.config.js`.
- **react-resizable-panels** — persists split position to localStorage
  via `autoSaveId`.
- No state library yet (just `useState` + a singleton `EventStream`).
  Reach for Zustand if/when component-prop drilling for the Paper Agent
  chat becomes painful in Phase 2.
