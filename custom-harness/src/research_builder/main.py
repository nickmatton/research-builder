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
    """Execute the full paper reproduction pipeline.

    The pipeline is **model-orchestrated**: this function sets up workspace +
    services + the execution-loop scaffolding, then hands off to a single
    long-running ``OrchestratorAgent.run_as_orchestrator()`` call. The model
    decides what to do next (write skeleton, author specs, run phases, ask
    user for approval) via MCP tools defined in ``orchestrator/runtime_tools.py``.
    """
    from .commands import CommandListener, get_inbox
    from .orchestrator.agent import OrchestratorAgent
    from .orchestrator.loop import ExecutionLoop
    from .orchestrator.runtime_tools import OrchestratorRuntime, deliver_user_reply
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

    paper_path = config.paper_path
    if not paper_path.exists():
        logger.error("Paper not found: %s", paper_path)
        ui.failure(f"Paper not found at {paper_path}")
        return False

    # The execution loop instance is reused by the start_phase tool — it owns
    # per-phase failure handling, retries, sub-agent dispatch, and event
    # emission. The orchestrator only schedules; the loop executes.
    spec_manager: SpecManager | None = None
    if resume and store.state_path.exists():
        pass  # spec_manager was loaded above
    # Otherwise spec_manager stays None — the orchestrator's write_skeleton
    # tool will populate it on its first turn.

    ui.header("Starting orchestrator")
    console = None

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

        from .events.emitter import get_emitter as _get_emitter
        cloud_provisioner = CloudProvisioner(
            config, lambda_key,
            ledger=ledger,
            approval_callback=_gpu_approval_callback,
            emitter=_get_emitter(),
            persistence_path=log_dir / "compute_instances.json",
        )
        ui.info(
            f"Lambda Cloud provisioner enabled (LAMBDA_API_KEY set, "
            f"GPU budget cap ${config.gpu_budget_usd:.2f})"
        )

    # ExecutionLoop is constructed here but its top-level run() is NOT called.
    # Instead, the orchestrator's start_phase tool drives _execute_phase one
    # phase at a time. Construction still wires the failure handler, cost
    # tracking, sub-agent dispatch, and event emission — all of which fire
    # from inside each start_phase invocation.
    loop = ExecutionLoop(
        config=config,
        spec_manager=spec_manager,
        workspace=workspace,
        orchestrator_agent=orchestrator_agent,
        console=console,
        cloud_provisioner=cloud_provisioner,
    )

    # Build the runtime shared by all orchestrator tools.
    runtime = OrchestratorRuntime(
        config=config,
        paper_path=paper_path,
        workspace=workspace,
        store=store,
        orchestrator_agent=orchestrator_agent,
        execution_loop=loop,
        spec_manager=spec_manager,
    )

    # Wire chat input into the runtime's approval queue. Every inbound chat
    # message is delivered to whichever request_user_approval is currently
    # awaiting (or queued for the next one). The model interprets the text —
    # there's no separate classifier; chat IS the orchestrator's input
    # channel.
    inbox = get_inbox()

    async def _orchestrator_chat_handler(text: str) -> None:
        deliver_user_reply(runtime, text)

    inbox.register_orchestrator_handler(_orchestrator_chat_handler)

    commands_path_str = os.environ.get("RESEARCH_BUILDER_COMMAND_LOG")
    listener_task = None
    if commands_path_str:
        listener = CommandListener(Path(commands_path_str))
        listener_task = asyncio.create_task(listener.run())

    # Hand off to the model. Returns when the orchestrator calls
    # pipeline_complete or pipeline_failed.
    try:
        success, final_message = await orchestrator_agent.run_as_orchestrator(runtime)
        if final_message:
            logger.info("Orchestrator finished: %s", final_message)
    finally:
        if listener_task is not None:
            listener_task.cancel()
            try:
                await listener_task
            except (asyncio.CancelledError, Exception):
                pass

    # The orchestrator may have populated spec_manager via its write_skeleton
    # tool; pull the latest into the local handle so the summary code below
    # works.
    spec_manager = runtime.spec_manager or spec_manager
    if spec_manager is None:
        ui.failure("Run ended without a spec_manager — nothing to summarize.")
        return success

    # Log final status for each phase. The [build]/[exp] tag distinguishes
    # build-the-code phases from experiment-runs-the-paper-claim phases —
    # otherwise it's not obvious from "section_4_*" vs "section_5_1_*" alone.
    logger.info("=== Run Summary ===")
    for phase in spec_manager.state.phases:
        tag = "[exp]  " if phase.kind.value == "experiment" else "[build]"
        logger.info("  %s %s: %s", tag, phase.phase_id, phase.status.value)

    # Aggregated error summary — scans run.log for every WARNING/ERROR
    # this run produced, groups them, writes a single markdown file to
    # notes/run_errors.md. Best-effort; failures here never block.
    try:
        from .storage.run_summary import write_run_summary
        summary_path = workspace.config.notes_dir / "run_errors.md"
        if write_run_summary(log_dir / "run.log", summary_path):
            ui.path_line("Run errors", summary_path)
    except Exception:
        logger.exception("Failed to write run error summary")

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
    help="Workspace directory for this run (default: ./<paper_name>/). Used as-is — no paper-name subdir is appended.",
)
@click.option(
    "--model", "-m",
    default=None,
    help="Claude model to use (default: Config.model, currently claude-opus-4-6[1m])",
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
        "ANTHROPIC_API_KEY in this process. Also implies --verbose. "
        "Uses Config.model (Opus by default); pass --model to override."
    ),
)
@click.option(
    "--test",
    is_flag=True,
    help=(
        "SMOKE TEST: zero-setup smoke run. Uses the bundled 4-page "
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
@click.option(
    "--llm-budget",
    "llm_spend_cap_usd",
    type=float,
    default=None,
    help="Hard cap on LLM spend for this run, in USD (covers sub-agent + orchestrator queries). "
    "Defaults to $20 (or RB_LLM_SPEND_CAP_USD env). When total spend crosses the cap the run "
    "aborts cleanly — remaining phases are marked failed so they can be resumed later. Set to "
    "0 to disable.",
)
@click.option(
    "--allow-dir",
    "allow_dirs",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    multiple=True,
    help="Extra directory the agent sandbox may read/write outside of the workspace cwd. "
    "Repeatable. Each path is resolved to absolute. The agent can also call "
    "mcp__access__read_outside_workspace at runtime to ask for paths not pre-allowed; "
    "in interactive mode this surfaces a yes/no prompt, in --auto it is denied.",
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
    llm_spend_cap_usd: float | None,
    allow_dirs: tuple[Path, ...],
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
        # Resolve model to Config default if not overridden, so the banner
        # below shows the real value (not None) and Config doesn't get
        # clobbered by an explicit None.
        if model is None:
            model = Config.__dataclass_fields__["model"].default
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

    # Build project_root.
    # --project-root overrides everything (used by TUI subprocess).
    # --output is used as-is when supplied (the literal workspace dir).
    # If --output is omitted, default to ./<paper_stem>/ in the cwd so a
    # bare `research-builder paper.pdf` still creates a sensibly-named dir.
    if project_root_override is not None:
        project_root = project_root_override
    elif output is not None:
        project_root = output
    else:
        project_root = Path(".") / _paper_stem(paper)

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
                    "--auto mode cannot prompt. "
                    + ("Pass --fresh to archive and start over, or --wipe to delete and rerun."
                       if existing.stale
                       else "Pass --resume to continue, --fresh to archive and start over, or --wipe to delete and rerun.")
                )
                sys.exit(2)
            # Stale workspaces aren't resumable (no parseable state.json) —
            # omit the [c]ontinue option so the operator picks wipe or archive.
            if existing.stale:
                choice = click.prompt(
                    "  [w]ipe and rerun / [n]ew run (archive old)",
                    type=click.Choice(["w", "n"], case_sensitive=False),
                    default="w",
                )
            else:
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

    # Resolve LLM budget: --llm-budget > RB_LLM_SPEND_CAP_USD env > Config default ($20).
    if llm_spend_cap_usd is None:
        env_llm_cap = os.environ.get("RB_LLM_SPEND_CAP_USD")
        llm_spend_cap_usd = float(env_llm_cap) if env_llm_cap else 20.0

    # Assistant mode: --auto opts out of interactive checkpoints. Without
    # --auto, the harness pauses for a human OK after spec creation and
    # after each phase, and offers an [a]sk option that opens a chat with
    # the orchestrator (paper + spec in context).
    config = Config(
        paper_path=paper.resolve(),
        project_root=project_root.resolve(),
        model=model,
        max_retries=max_retries,
        max_debug_attempts=max_debug_attempts,
        interactive=not auto,
        gpu_budget_usd=gpu_budget_usd,
        llm_spend_cap_usd=llm_spend_cap_usd,
        extra_allowed_dirs=[p.resolve() for p in allow_dirs],
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

        # Persistent bottom-anchored input box (Claude-Code-style ``> ``
        # prompt) is enabled in interactive mode on a real TTY. Typed
        # text is delivered to the orchestrator via the inbox — same
        # path the web chat uses.
        chat_enabled = config.interactive and sys.stdin.isatty()

        async def _run_with_viewer():
            if chat_enabled:
                from prompt_toolkit.patch_stdout import patch_stdout
                from .commands import get_inbox
                from .terminal_input import run_terminal_input

                with patch_stdout(raw=True):
                    input_task = asyncio.create_task(
                        run_terminal_input(get_inbox()),
                    )
                    try:
                        with inline_viewer_for(
                            workspace_label=project_root.name,
                            interactive=config.interactive,
                        ):
                            return await run_pipeline(config, resume=resume_decision)
                    finally:
                        input_task.cancel()
                        try:
                            await input_task
                        except (asyncio.CancelledError, Exception):
                            pass
            else:
                with inline_viewer_for(
                    workspace_label=project_root.name,
                    interactive=config.interactive,
                ):
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
