"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .api import build_api_router
from .events_ws import build_events_router
from .state import WebState


VITE_DEV_ORIGIN = "http://localhost:5173"


def create_app(
    runs_dir: Path,
    workspace: Path | None = None,
    dev_mode: bool = False,
    extra_allowed_dirs: list[Path] | None = None,
) -> FastAPI:
    """Build a FastAPI app.

    Two boot modes:

    - ``workspace`` supplied → app starts bound to that workspace
      (the legacy ``research-builder-app /path/to/workspace`` flow).
    - ``workspace`` is ``None`` → launcher mode. The UI shows an upload
      screen; uploading a PDF triggers ``POST /api/launch`` which
      creates ``runs_dir/<paper-stem>/`` and spawns the pipeline.

    Only one active workspace + pipeline per process. Re-uploading
    while a run is in progress is rejected by the API.
    """
    runs_dir = runs_dir.resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    state = WebState(
        runs_dir=runs_dir,
        dev_mode=dev_mode,
        extra_allowed_dirs=list(extra_allowed_dirs or []),
    )
    if workspace is not None:
        state.set_workspace(workspace.resolve())

    name = workspace.name if workspace else "launcher"
    app = FastAPI(title=f"research-builder · {name}")
    # Exposed so the CLI's Server.handle_exit can fire shutdown_event at
    # SIGINT receipt — lifespan.shutdown fires too late (after task cancel).
    app.state.web_state = state

    # Vite dev server runs on :5173 and proxies REST through itself,
    # but the WS connection (`ws://127.0.0.1:7777/ws/...`) is cross-origin
    # during dev. CORS only needs to allow the dev origin; production
    # serves the built bundle from the same FastAPI process.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[VITE_DEV_ORIGIN],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(build_api_router(state))
    app.include_router(build_events_router(state))

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "ok": True,
            "runs_dir": str(state.runs_dir),
            "workspace": str(state.workspace) if state.workspace else None,
        }

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        # 1. Flip the shutdown event FIRST. The events tail loop checks it
        #    on every iteration and via wait_for around its poll-interval
        #    sleep, so all in-flight tail loops bail within ~100ms even if
        #    they were mid-sleep when shutdown fired. Without this, the
        #    loops keep polling because ``ws.close()`` server-side doesn't
        #    interrupt an already-awaiting ``asyncio.sleep`` — uvicorn
        #    hangs on "Waiting for background tasks to complete."
        state.shutdown_event.set()

        # 2. Close every open /ws/events WebSocket so the client side sees
        #    the disconnect promptly and any send_json mid-flight fails fast.
        for ws in list(state.websockets):
            try:
                await ws.close(code=1001, reason="server shutting down")
            except Exception:
                pass
        state.websockets.clear()

        # 3. Kill the harness subprocess so we don't leave orphans.
        state.terminate()

    # ─── Static frontend ────────────────────────────────────────────────
    # Mount the built frontend at /. Two locations are probed, in order:
    #   1. ``research_builder/web/_static/`` — populated by ``npm run build``
    #      and shipped inside the wheel for production installs.
    #   2. ``<repo>/custom-harness/frontend/dist/`` — local dev convenience
    #      so contributors can ``npm run build`` without copying files.
    # If neither exists, ``/`` returns a help page directing the user at
    # ``npm run dev``; the API endpoints still work either way.
    static_dir = _find_static_dir()
    if static_dir is not None:
        # Mount last so it doesn't shadow /api or /ws routes (route lookup
        # in Starlette is order-dependent — earlier wins).
        app.mount(
            "/",
            StaticFiles(directory=str(static_dir), html=True),
            name="frontend",
        )
    else:
        @app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return _DEV_PAGE_HTML

    return app


def _find_static_dir() -> Path | None:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "_static",                                              # wheel
        here.parents[2] / "frontend" / "dist",                         # repo
    ]
    for c in candidates:
        if (c / "index.html").exists():
            return c
    return None


_DEV_PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>research-builder</title>
<style>
  body { font: 14px/1.5 -apple-system, system-ui, sans-serif;
         background:#0a0a0a; color:#fafafa; max-width:560px; margin:6rem auto; padding:0 1rem; }
  code { background:#161616; border:1px solid #2a2a2a; border-radius:4px;
         padding:.1rem .4rem; font-family:ui-monospace, monospace; }
  h1 { font-size:1.1rem; font-weight:600; letter-spacing:-.01em; }
  p { color:#a3a3a3; }
  .accent { color:#6366f1; }
</style></head>
<body>
<h1>research-builder <span class="accent">·</span> backend running</h1>
<p>The frontend bundle wasn't found at <code>web/_static/</code> or <code>frontend/dist/</code>. Two ways to view the UI:</p>
<ol>
  <li>Dev: <code>cd custom-harness/frontend && npm run dev</code>, then open <a href="http://localhost:5173" style="color:#6366f1">localhost:5173</a>.</li>
  <li>Prod: <code>cd custom-harness/frontend && npm run build</code>, then refresh this page.</li>
</ol>
<p>API endpoints are live regardless — try <code>/api/workspace</code>, <code>/healthz</code>, or <code>/ws/events</code>.</p>
</body></html>
"""
