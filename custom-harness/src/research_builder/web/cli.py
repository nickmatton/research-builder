"""``research-builder-app`` entry point.

Two boot modes:

  - ``research-builder-app``            launcher: upload a paper via the UI,
                                        harness subprocess is spawned by the
                                        backend (single-terminal flow).
  - ``research-builder-app <workspace>`` legacy: serve an existing workspace
                                         (does NOT spawn the pipeline; useful
                                         for viewing a completed run).

The launcher writes new runs under ``--runs-dir`` (default ``./runs``).
"""

from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

import click


@click.command()
@click.argument(
    "workspace",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=False,
    default=None,
)
@click.option(
    "--runs-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("./runs"),
    show_default=True,
    help="Directory under which new workspaces are created when uploading a paper.",
)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=7777, type=int, show_default=True)
@click.option(
    "--open/--no-open",
    "open_browser",
    default=True,
    help="Open the UI in the default browser on start.",
)
@click.option(
    "--log-level",
    default="info",
    type=click.Choice(["critical", "error", "warning", "info", "debug"]),
    show_default=True,
)
@click.option(
    "--dev/--no-dev",
    "dev",
    default=True,
    show_default=True,
    help=(
        "Spawn pipeline subprocesses with --dev — routes the Claude Agent "
        "SDK through the bundled `claude` CLI (Claude Code subscription) "
        "instead of requiring ANTHROPIC_API_KEY. Pass --no-dev to use a "
        "billed ANTHROPIC_API_KEY from the environment / .env instead."
    ),
)
@click.option(
    "--allow-dir",
    "allow_dirs",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    multiple=True,
    help="Extra directory the spawned pipeline's agent sandbox may read/write "
    "outside of the workspace. Repeatable. Forwarded to ``research-builder`` "
    "via --allow-dir.",
)
def main(
    workspace: Path | None,
    runs_dir: Path,
    host: str,
    port: int,
    open_browser: bool,
    log_level: str,
    dev: bool,
    allow_dirs: tuple[Path, ...],
) -> None:
    """Serve the research-builder web UI.

    WORKSPACE is optional. Without it, the UI opens a launcher screen
    where you upload a PDF; the backend creates a workspace under
    --runs-dir and spawns the pipeline.
    """
    # Populate os.environ from a local .env (ANTHROPIC_API_KEY, LAMBDA_API_KEY,
    # etc.) so spawned pipeline subprocesses — which inherit os.environ via
    # env=os.environ.copy() in web/api.py — see them without `source .env`.
    from dotenv import load_dotenv
    load_dotenv()

    # Lazy imports — keep ``research-builder-app --help`` fast and avoid
    # pulling in FastAPI/uvicorn for any other consumer of research_builder.web.
    import uvicorn

    from .app import create_app

    runs_dir = runs_dir.resolve()
    if workspace is not None:
        workspace = workspace.resolve()
    extra_allowed_dirs = [p.resolve() for p in allow_dirs]
    app = create_app(
        runs_dir=runs_dir,
        workspace=workspace,
        dev_mode=dev,
        extra_allowed_dirs=extra_allowed_dirs,
    )
    url = f"http://{host}:{port}"

    if workspace is not None:
        click.echo(f"research-builder-app · {workspace}")
    else:
        click.echo(f"research-builder-app · launcher (runs → {runs_dir})")
    if dev:
        click.echo("  • dev mode: pipelines spawned with --dev (Claude Code subscription auth)")
    click.echo(f"  → {url}")
    if open_browser and host in ("127.0.0.1", "localhost", "0.0.0.0"):
        # Browser-open is best-effort; some shells (SSH, containers)
        # will quietly no-op which is fine.
        try:
            webbrowser.open(url)
        except Exception:
            pass

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
        timeout_graceful_shutdown=2,
    )

    # Fire shutdown_event from the signal handler, not lifespan.shutdown:
    # uvicorn waits for connections to drain *before* running lifespan, so a
    # lifespan-set flag arrives after the WS task has already been cancelled.
    class _ResearchBuilderServer(uvicorn.Server):
        def handle_exit(self, sig: int, frame) -> None:  # type: ignore[override]
            try:
                app.state.web_state.shutdown_event.set()
            except Exception:
                pass
            super().handle_exit(sig, frame)

    server = _ResearchBuilderServer(config)
    try:
        server.run()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
