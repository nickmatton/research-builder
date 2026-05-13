"""CLI entry point for the research builder harness."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import shutil
import subprocess
import sys
from pathlib import Path

import click

from . import ui
from .config import Config


def _paper_stem(paper_path: Path) -> str:
    """Derive a directory-safe folder name from the paper filename."""
    stem = paper_path.stem
    sanitized = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return sanitized or "paper"


def _get_descendant_pids(root_pid: int) -> list[int]:
    """Return all descendant PIDs of *root_pid* (not including root_pid).

    Walks the process tree via ``pgrep -P`` so that children in different
    process groups (e.g. those spawned by the claude CLI's Bash tool) are
    captured — unlike a PGID-based scan which misses them.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-P", str(root_pid)],
            capture_output=True, text=True, timeout=5,
        )
        children = [int(p) for p in result.stdout.strip().split() if p.strip()]
    except Exception:
        return []
    descendants = list(children)
    for child in children:
        descendants.extend(_get_descendant_pids(child))
    return descendants


def _shutdown_pipeline(pipeline_proc: subprocess.Popen) -> None:
    """Shut down the pipeline subprocess and all of its descendants."""
    if pipeline_proc.poll() is not None:
        return

    # Snapshot the full descendant tree *before* killing the root, so we
    # can clean up children in other process groups (e.g. evaluate.py
    # spawned by the claude CLI's Bash tool).
    descendants = _get_descendant_pids(pipeline_proc.pid)

    # Terminate the pipeline process itself.
    pipeline_proc.terminate()
    try:
        pipeline_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pipeline_proc.kill()
        pipeline_proc.wait(timeout=3)

    # SIGTERM surviving descendants (they may be in different process groups).
    alive = []
    for pid in descendants:
        try:
            os.kill(pid, signal.SIGTERM)
            alive.append(pid)
        except ProcessLookupError:
            pass

    if not alive:
        return

    ui.warning(f"Terminating {len(alive)} orphaned descendant process(es)...")

    # Give them a moment to exit, then SIGKILL stragglers.
    import time
    time.sleep(2)
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    ui.success("Descendant processes terminated.")


def _copy_report(workspace, spec_manager) -> bool:
    """Find the reproduction report from the results phase and copy to report/."""
    results_phase = spec_manager.state.get_phase("results")
    if results_phase is None:
        return False

    outputs_dir = workspace.outputs_dir("results")
    if not outputs_dir.exists():
        return False

    for name in ["reproduction_report.md", "report.md"]:
        src = outputs_dir / name
        if src.exists():
            workspace.report_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, workspace.report_path)
            return True

    for md_file in sorted(outputs_dir.glob("*.md")):
        workspace.report_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md_file, workspace.report_path)
        return True

    return False


async def run_pipeline(config: Config, resume: bool = False) -> bool:
    """Execute the full paper reproduction pipeline."""
    from .commands import CommandListener, get_inbox
    from .console import InteractiveConsole
    from .interaction import (
        UserAction,
        prompt_after_phase,
        prompt_after_spec,
    )
    from .orchestrator.agent import OrchestratorAgent
    from .orchestrator.loop import ExecutionLoop
    from .orchestrator.spec_manager import SpecManager
    from .storage.spec_store import SpecStore
    from .storage.workspace import WorkspaceManager

    logger = logging.getLogger(__name__)
    log_dir = config.project_root / "logs"

    # Initialize workspace
    workspace = WorkspaceManager(config)
    workspace.initialize()
    logger.info("Workspace initialized at %s", config.project_root)

    # Initialize orchestrator
    store = SpecStore(config.spec_dir)
    orchestrator_agent = OrchestratorAgent(config)

    # Step 1: Create canonical spec from paper
    paper_path = config.paper_path
    if not paper_path.exists():
        logger.error("Paper not found: %s", paper_path)
        ui.failure(f"Paper not found at {paper_path}")
        return False

    if resume and store.state_path.exists():
        ui.step(f"Resuming existing run from {store.state_path}")
        spec_manager = SpecManager(store, store.load_state())
        ui.success(f"Resumed: {len(spec_manager.state.phases)} phases loaded")
    else:
        ui.step(f"Ingesting paper: {paper_path}")
        spec_manager = await orchestrator_agent.create_spec(paper_path, store)
        ui.success(f"Spec created: {len(spec_manager.state.phases)} phases")
    for phase in spec_manager.state.phases:
        deps = spec_manager.dep_graph.get_dependencies(phase.phase_id)
        dep_str = f" (depends on: {', '.join(deps)})" if deps else ""
        ui.info(f"  - {phase.phase_id}: {phase.title}{dep_str}")

    # Checkpoint: review spec
    if config.interactive:
        action = prompt_after_spec(workspace.spec_md_path)
        if action == UserAction.ABORT:
            ui.warning("Aborted by user.")
            return False

    # Step 2: Build phase callback for interactive mode
    async def on_phase_complete(phase_id: str, result):
        """Called after each phase. Returns UserAction."""
        if not config.interactive:
            return UserAction.CONTINUE
        return prompt_after_phase(phase_id, result)

    # Step 3: Run execution loop
    ui.header("Starting execution loop")
    console = InteractiveConsole(workspace.spec_md_path, spec_manager, config) if config.interactive else None

    # Optional cloud provisioner: only enabled when LAMBDA_API_KEY is set.
    # Without it, the loop runs phases entirely locally as before.
    import os as _os
    cloud_provisioner = None
    lambda_key = _os.environ.get("LAMBDA_API_KEY")
    if lambda_key:
        from .cloud import ApprovalRequest, BudgetLedger, CloudProvisioner
        ledger = BudgetLedger(cap_usd=config.gpu_budget_usd)

        async def _gpu_approval_callback(req: ApprovalRequest) -> float | None:
            """Bubble a budget-exceeding provision/upgrade request to the operator.

            Returns the new cap (>= current) on approval, or None to deny.
            Runs the blocking click prompt off-loop so we don't stall the
            event loop while waiting for the human.
            """
            new_total = req.current_total_usd + req.requested_cost_usd
            kind = "upgrade" if req.is_upgrade else "provision"
            ui.warning(
                f"GPU {kind} for phase '{req.phase_id}' would exceed the per-run cap.\n"
                f"  Requested: {req.instance_type} (~${req.requested_cost_usd:.2f})\n"
                f"  Reason:    {req.reason}\n"
                f"  Current:   ${req.current_total_usd:.2f} reserved\n"
                f"  After:     ${new_total:.2f}\n"
                f"  Cap:       ${req.cap_usd:.2f}"
            )
            # Suggest raising the cap to cover this request with a small cushion.
            suggested = max(req.cap_usd * 2, new_total + 5.0)

            def _ask() -> float | None:
                try:
                    if not click.confirm(
                        f"  Raise cap from ${req.cap_usd:.2f} to ${suggested:.2f}?",
                        default=False,
                    ):
                        return None
                except (EOFError, click.Abort):
                    return None
                return suggested

            return await asyncio.to_thread(_ask)

        cloud_provisioner = CloudProvisioner(
            config, lambda_key,
            ledger=ledger,
            approval_callback=_gpu_approval_callback,
        )
        ui.info(
            f"Lambda Cloud provisioner enabled (LAMBDA_API_KEY set, "
            f"GPU budget cap ${config.gpu_budget_usd:.2f})"
        )

    loop = ExecutionLoop(
        config=config,
        spec_manager=spec_manager,
        workspace=workspace,
        orchestrator_agent=orchestrator_agent,
        on_phase_complete=on_phase_complete,
        console=console,
        cloud_provisioner=cloud_provisioner,
    )

    # Wire the inbound chat command channel: listener tails commands.jsonl,
    # delivers to inbox; orchestrator-targeted messages call OrchestratorAgent.chat().
    inbox = get_inbox()
    async def _orchestrator_chat_handler(text: str) -> None:
        await orchestrator_agent.chat(text, spec_manager=spec_manager)
    inbox.register_orchestrator_handler(_orchestrator_chat_handler)

    import os as _os
    commands_path_str = _os.environ.get("RESEARCH_BUILDER_COMMAND_LOG")
    listener_task = None
    if commands_path_str:
        listener = CommandListener(Path(commands_path_str))
        listener_task = asyncio.create_task(listener.run())

    try:
        success = await loop.run()
    finally:
        if listener_task is not None:
            listener_task.cancel()
            try:
                await listener_task
            except (asyncio.CancelledError, Exception):
                pass

    # Log final status for each phase
    logger.info("=== Run Summary ===")
    for phase in spec_manager.state.phases:
        logger.info("  %s: %s", phase.phase_id, phase.status.value)

    if success:
        report_copied = _copy_report(workspace, spec_manager)

        ui.header("Run Complete")
        if loop.total_cost_usd > 0:
            ui.success(f"All phases finished. Total cost: ${loop.total_cost_usd:.2f}")
        else:
            ui.success("All phases finished.")
        if report_copied:
            ui.path_line("Report", workspace.report_path)
            report_text = workspace.report_path.read_text()
            report_lines = report_text.count("\n")
            if sys.stdout.isatty() and report_lines > 40:
                click.echo_via_pager(report_text)
            else:
                ui.divider()
                click.echo(report_text)
                ui.divider()
        else:
            ui.warning("No reproduction report found in results phase output.")
        ui.path_line("Spec", workspace.spec_md_path)
        ui.path_line("Phases", config.phases_dir)
        ui.path_line("Log", log_dir / "run.log")
    else:
        ui.failure("Run failed. Check logs for details.")
        ui.path_line("Log", log_dir / "run.log")
        ui.path_line("Revision log", workspace.revision_log_path)

    return success


@click.command()
@click.argument("paper", type=click.Path(exists=True, path_type=Path), required=False, default=None)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Base output directory (default: current directory). Runs are stored under <output>/<paper_name>/.",
)
@click.option(
    "--model", "-m",
    default=None,
    help="Claude model to use (default: Config.model, currently claude-haiku-4-5-20251001)",
)
@click.option(
    "--max-retries",
    default=3,
    type=int,
    help="Max orchestrator retries per phase",
)
@click.option(
    "--max-debug-attempts",
    default=10,
    type=int,
    help="Max debug attempts per sub-agent invocation",
)
@click.option(
    "--auto",
    is_flag=True,
    help="Run without interactive checkpoints",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose (DEBUG) logging to console.",
)
@click.option(
    "--dev",
    is_flag=True,
    help=(
        "DEV MODE for testing without burning Anthropic API tokens. "
        "Forces the Claude Agent SDK to use the user's Claude Code "
        "subscription (via the bundled `claude` CLI) by unsetting "
        "ANTHROPIC_API_KEY in this process. Also implies --verbose, "
        "and pins the model to claude-haiku-4-5 for fast iteration "
        "unless --model is passed explicitly."
    ),
)
@click.option(
    "--test",
    is_flag=True,
    help=(
        "SMOKE TEST: zero-setup smoke run. Uses the bundled 3-page "
        "test_paper.pdf, outputs to /tmp/rb-test, auto-wipes prior "
        "state, and implies --auto + --dev. A single command instead "
        "of four. Useful for iterating on harness changes."
    ),
)
@click.option(
    "--event-log",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Path to a JSONL event stream for external viewers (e.g. agent-terminal). "
        "Defaults to <output>/logs/events.jsonl. Pass --no-event-log to disable."
    ),
)
@click.option(
    "--no-event-log",
    is_flag=True,
    default=False,
    help="Disable the JSONL event stream entirely.",
)
@click.option(
    "--command-log",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Path to a JSONL command stream the pipeline tails for inbound chat "
        "messages from agent-terminal. Defaults to <output>/logs/commands.jsonl. "
        "Pass --no-command-log to disable."
    ),
)
@click.option(
    "--no-command-log",
    is_flag=True,
    default=False,
    help="Disable the inbound JSONL command stream entirely.",
)
@click.option(
    "--resume",
    "resume_flag",
    is_flag=True,
    default=False,
    help="Resume an existing partial run without prompting.",
)
@click.option(
    "--fresh",
    "fresh_flag",
    is_flag=True,
    default=False,
    help="Archive any existing partial run and start over without prompting.",
)
@click.option(
    "--wipe",
    "wipe_flag",
    is_flag=True,
    default=False,
    help="Delete any existing run entirely and start over without prompting.",
)
@click.option(
    "--project-root",
    "project_root_override",
    type=click.Path(path_type=Path),
    default=None,
    hidden=True,
    help="Use this exact directory as project_root (skip paper-stem derivation). Internal use by TUI subprocess.",
)
@click.option(
    "--gpu-budget",
    "gpu_budget_usd",
    type=float,
    default=None,
    help="Hard cap on GPU spend for this run, in USD. Defaults to $30 (or RB_GPU_BUDGET_USD env). "
    "When a sub-agent's compute request would exceed the cap mid-run, the harness asks you to "
    "approve raising it.",
)
def cli(
    paper: Path | None,
    output: Path | None,
    model: str | None,
    max_retries: int,
    max_debug_attempts: int,
    auto: bool,
    verbose: bool,
    dev: bool,
    test: bool,
    event_log: Path | None,
    no_event_log: bool,
    command_log: Path | None,
    no_command_log: bool,
    resume_flag: bool,
    fresh_flag: bool,
    wipe_flag: bool,
    project_root_override: Path | None,
    gpu_budget_usd: float | None,
) -> None:
    """Reproduce a research paper's results.

    PAPER is the path to the research paper PDF.
    If omitted, you will be prompted to provide it.

    By default, the pipeline pauses after spec creation and each phase
    for human review. Use --auto to run without prompts.
    """
    # Populate os.environ from a local .env (ANTHROPIC_API_KEY, LAMBDA_API_KEY,
    # etc.) so downstream code that reads env vars — notably the Lambda Cloud
    # GPU provisioner — sees them without requiring the user to `source .env`.
    from dotenv import load_dotenv
    load_dotenv()

    # --- TEST MODE: zero-setup smoke run ---
    if test:
        # Bundled test paper at custom-harness/paper/test_paper.pdf
        # main.py is at custom-harness/src/research_builder/main.py, so
        # parents[2] resolves to custom-harness/.
        bundled = Path(__file__).resolve().parents[2] / "paper" / "test_paper.pdf"
        if not bundled.exists():
            click.echo(
                f"\033[1;31merror:\033[0m bundled test paper not found at {bundled}. "
                "Did you delete custom-harness/paper/test_paper.pdf?",
                err=True,
            )
            sys.exit(2)
        # Apply --test overrides for any flag the user didn't explicitly set.
        if paper is None:
            paper = bundled
        if output is None:
            output = Path("/tmp/rb-test")
        auto = True
        dev = True
        wipe_flag = True   # nuke prior state, no resume prompt
        # Visible banner so it's unmistakable.
        print("\033[1;36m" + "=" * 60 + "\033[0m")
        print("\033[1;36m  SMOKE TEST MODE\033[0m")
        print(f"    • Paper:  {paper}")
        print(f"    • Output: {output}")
        print(f"    • Implies --auto, --dev, --wipe")
        print("\033[1;36m" + "=" * 60 + "\033[0m\n")

    # --- DEV MODE: route through Claude Code subscription, not direct API ---
    if dev:
        # Unset ANTHROPIC_API_KEY so the bundled `claude` CLI falls back to
        # whatever auth `claude login` configured (typically a Claude.ai Pro/Max
        # subscription). Only affects THIS process — your shell env stays intact.
        popped = os.environ.pop("ANTHROPIC_API_KEY", None)
        # --dev implies --verbose
        verbose = True
        # Pin to haiku unless user passed --model explicitly. Haiku is fast +
        # cheap, ideal for dev iteration.
        if model is None:
            model = "claude-haiku-4-5-20251001"
        # Big visible banner so it's impossible to confuse a dev run with prod.
        print("\033[1;33m" + "=" * 60 + "\033[0m")
        print("\033[1;33m  DEV MODE\033[0m")
        print(f"    • ANTHROPIC_API_KEY: {'unset (was set, popped)' if popped else 'unset (was already)'}")
        print(f"    • Backend:           Claude Code subscription (bundled `claude` CLI)")
        print(f"    • Model:             {model}")
        print(f"    • Logging:           DEBUG (verbose enabled)")
        print("\033[1;33m" + "=" * 60 + "\033[0m\n")

    # Show banner
    ui.banner()

    # Prompt for paper if not provided as argument
    if paper is None:
        import termios as _termios

        from prompt_toolkit import prompt as pt_prompt
        from prompt_toolkit.completion import PathCompleter

        completer = PathCompleter(expanduser=True)
        while True:
            raw = pt_prompt(
                "  Path to research paper PDF: ",
                completer=completer,
                complete_while_typing=True,
            ).strip()
            if not raw:
                continue
            paper = Path(raw).expanduser().resolve()
            if paper.exists():
                break
            ui.failure(f"File not found: {paper}")

        # Flush any leftover input from prompt_toolkit so downstream
        # click.prompt calls don't auto-accept on stale bytes.
        if sys.stdin.isatty():
            _termios.tcflush(sys.stdin, _termios.TCIFLUSH)

    # Build project_root: base_output / <paper_stem>
    # --project-root overrides derivation (used by TUI subprocess).
    if project_root_override is not None:
        project_root = project_root_override
    else:
        base_output = output or Path(".")
        project_root = base_output / _paper_stem(paper)

    # Detect partial run and decide resume vs fresh vs wipe BEFORE we set up
    # logging or launch the TUI — this is the only point where we can safely
    # use click prompts on the real terminal.
    from . import resume as resume_mod
    existing = resume_mod.detect(project_root.resolve())
    resume_decision = False  # default for greenfield runs
    if existing is not None:
        exclusive_flags = sum([resume_flag, fresh_flag, wipe_flag])
        if exclusive_flags > 1:
            ui.failure("Cannot pass more than one of --resume, --fresh, --wipe.")
            sys.exit(2)
        if resume_flag:
            resume_decision = True
        elif fresh_flag:
            archive = resume_mod.archive_and_clear(project_root.resolve())
            ui.info(f"Archived previous run to {archive}")
        elif wipe_flag:
            resume_mod.wipe(project_root.resolve())
            ui.info(f"Wiped previous run at {project_root.resolve()}")
        else:
            ui.warning(
                f"Found existing run in {project_root.resolve()}: "
                f"{existing.summary()}"
            )
            if auto:
                ui.failure(
                    "--auto mode cannot prompt. Pass --resume to continue, "
                    "--fresh to archive and start over, or --wipe to delete and rerun."
                )
                sys.exit(2)
            choice = click.prompt(
                "  [c]ontinue / [w]ipe and rerun / [n]ew run (archive old)",
                type=click.Choice(["c", "w", "n"], case_sensitive=False),
                default="c",
            )
            if choice == "c":
                resume_decision = True
            elif choice == "w":
                resume_mod.wipe(project_root.resolve())
                ui.info(f"Wiped previous run at {project_root.resolve()}")
            else:  # "n"
                archive = resume_mod.archive_and_clear(project_root.resolve())
                ui.info(f"Archived previous run to {archive}")

    # Set up logging — console + file.
    #
    # Console level intentionally caps at INFO even with --verbose / --dev:
    # DEBUG was flooding stdout with hundreds of "received StreamEvent" lines
    # per LLM call, making it impossible to see anything useful. The file log
    # below stays at DEBUG so the forensic trail is intact.
    console_level = logging.INFO if verbose else logging.WARNING
    log_dir = project_root.resolve() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # capture everything to the file handler

    # Console handler — INFO max (avoid the DEBUG flood)
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(console)

    # File handler (always DEBUG)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(funcName)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers in console
    for noisy in ["pdfminer", "httpx", "httpcore", "anyio"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    ui.path_line("Log file", log_file)

    # Configure JSONL event stream for external viewers (and the in-process TUI).
    # Must happen before Config / OrchestratorAgent are constructed so the
    # module-level emitter singleton picks up the env var on first read.
    import os as _os
    event_log_path = (event_log.resolve() if event_log else log_dir / "events.jsonl")
    command_log_path = (command_log.resolve() if command_log else log_dir / "commands.jsonl")

    if not no_event_log:
        _os.environ["RESEARCH_BUILDER_EVENT_LOG"] = str(event_log_path)
        ui.path_line("Event stream", event_log_path)
    else:
        _os.environ.pop("RESEARCH_BUILDER_EVENT_LOG", None)

    if not no_command_log:
        _os.environ["RESEARCH_BUILDER_COMMAND_LOG"] = str(command_log_path)
        ui.path_line("Command stream", command_log_path)
    else:
        _os.environ.pop("RESEARCH_BUILDER_COMMAND_LOG", None)

    # Default UX: launch the Textual TUI viewer in the same process and run
    # the pipeline as a background task on its event loop. The viewer reads
    # events.jsonl / commands.jsonl that the pipeline writes — same in-process
    # path the cross-process viewer uses, just no second terminal needed.
    #
    # --auto disables the TUI entirely and runs headless (the original path).
    # Resolve GPU budget: --gpu-budget > RB_GPU_BUDGET_USD env > Config default ($30).
    if gpu_budget_usd is None:
        env_budget = os.environ.get("RB_GPU_BUDGET_USD")
        gpu_budget_usd = float(env_budget) if env_budget else 30.0

    config = Config(
        paper_path=paper.resolve(),
        project_root=project_root.resolve(),
        model=model,
        max_retries=max_retries,
        max_debug_attempts=max_debug_attempts,
        # Interactive checkpoint prompts (click.prompt / InteractiveConsole)
        # cannot coexist with the Textual alt-screen. The TUI provides its
        # own interaction surface (chat pane), so we always run the pipeline
        # in non-interactive mode under the TUI.
        interactive=False,
        gpu_budget_usd=gpu_budget_usd,
    )

    # Pipeline always runs inline now (--auto or not). The inline viewer
    # (rich-based subscriber on the event emitter) renders styled activity
    # in the same terminal — no subprocess, no second window.

    def _auto_cleanup(signum, frame):
        # Kill all descendants first — catches children in different process
        # groups (e.g. evaluate.py spawned by claude CLI Bash).
        for pid in _get_descendant_pids(os.getpid()):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        # Also killpg as belt-and-suspenders for same-group children.
        try:
            os.killpg(os.getpgrp(), signal.SIGTERM)
        except ProcessLookupError:
            pass
        sys.exit(1)

    signal.signal(signal.SIGTERM, _auto_cleanup)

    try:
        from .viewer.inline import inline_viewer_for

        async def _run_with_viewer():
            with inline_viewer_for(workspace_label=project_root.name):
                return await run_pipeline(config, resume=resume_decision)

        success = asyncio.run(_run_with_viewer())
    except Exception:
        import traceback
        ui.failure("Pipeline crashed with an unhandled exception:")
        traceback.print_exc()
        logging.getLogger(__name__).exception("Fatal error in run_pipeline")
        sys.exit(1)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    cli()
