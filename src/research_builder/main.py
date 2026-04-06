"""CLI entry point for the research builder harness."""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
from pathlib import Path

import click

from .config import Config
from .interaction import (
    UserAction,
    open_in_editor,
    prompt_after_phase,
    prompt_after_spec,
    prompt_skip_which_phase,
)
from .orchestrator.agent import OrchestratorAgent
from .orchestrator.loop import ExecutionLoop
from .orchestrator.spec_manager import SpecManager
from .storage.spec_store import SpecStore
from .storage.workspace import WorkspaceManager


def _copy_report(workspace: WorkspaceManager, spec_manager: SpecManager) -> bool:
    """Find the reproduction report from the results phase and copy to report/."""
    results_phase = spec_manager.state.get_phase("results")
    if results_phase is None:
        return False

    try_num = results_phase.current_try or 1
    outputs_dir = workspace.outputs_dir("results", try_num)
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


async def run_pipeline(config: Config) -> bool:
    """Execute the full paper reproduction pipeline."""
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
        click.echo(f"Error: Paper not found at {paper_path}", err=True)
        return False

    click.echo(f"Ingesting paper: {paper_path}")
    spec_manager = await orchestrator_agent.create_spec(paper_path, store)
    click.echo(f"\nSpec created: {len(spec_manager.state.phases)} phases")
    for phase in spec_manager.state.phases:
        deps = spec_manager.dep_graph.get_dependencies(phase.phase_id)
        dep_str = f" (depends on: {', '.join(deps)})" if deps else ""
        click.echo(f"  - {phase.phase_id}: {phase.title}{dep_str}")

    # Checkpoint: review spec
    if config.interactive:
        action = prompt_after_spec(workspace.spec_md_path)
        if action == UserAction.ABORT:
            click.echo("Aborted.")
            return False

    # Step 2: Build phase callback for interactive mode
    def on_phase_complete(phase_id: str, result):
        """Called after each phase. Returns UserAction."""
        if not config.interactive:
            return UserAction.CONTINUE
        return prompt_after_phase(phase_id, result)

    # Step 3: Run execution loop
    click.echo("\nStarting execution loop...")
    loop = ExecutionLoop(
        config=config,
        spec_manager=spec_manager,
        workspace=workspace,
        orchestrator_agent=orchestrator_agent,
        on_phase_complete=on_phase_complete,
    )

    success = await loop.run()

    # Log final status for each phase
    logger.info("=== Run Summary ===")
    for phase in spec_manager.state.phases:
        logger.info("  %s: %s", phase.phase_id, phase.status.value)

    if success:
        report_copied = _copy_report(workspace, spec_manager)

        click.echo("\nRun completed successfully!")
        if report_copied:
            click.echo(f"\nReproduction report: {workspace.report_path}")
            click.echo("-" * 60)
            click.echo(workspace.report_path.read_text())
            click.echo("-" * 60)
        else:
            click.echo("\nNote: No reproduction report found in results phase output.")
        click.echo(f"\nSpec: {workspace.spec_md_path}")
        click.echo(f"Phases: {config.phases_dir}")
        click.echo(f"Log: {log_dir / 'run.log'}")
    else:
        click.echo("\nRun failed. Check logs for details.", err=True)
        click.echo(f"Log: {log_dir / 'run.log'}", err=True)
        click.echo(f"Revision log: {workspace.revision_log_path}", err=True)

    return success


@click.command()
@click.argument("paper", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: current directory)",
)
@click.option(
    "--model", "-m",
    default="claude-opus-4-6",
    help="Claude model to use",
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
    help="Enable verbose logging",
)
def cli(
    paper: Path,
    output: Path | None,
    model: str,
    max_retries: int,
    max_debug_attempts: int,
    auto: bool,
    verbose: bool,
) -> None:
    """Reproduce a research paper's results.

    PAPER is the path to the research paper PDF.

    By default, the pipeline pauses after spec creation and each phase
    for human review. Use --auto to run without prompts.
    """
    # Build config
    project_root = output or Path(".")

    # Set up logging — console + file
    console_level = logging.DEBUG if verbose else logging.INFO
    log_dir = project_root.resolve() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # capture everything

    # Console handler (respects --verbose)
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(console)

    # File handler (always DEBUG)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers in console
    for noisy in ["pdfminer", "httpx", "httpcore", "anyio"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    click.echo(f"Log file: {log_file}")
    config = Config(
        paper_path=paper.resolve(),
        project_root=project_root.resolve(),
        model=model,
        max_retries=max_retries,
        max_debug_attempts=max_debug_attempts,
        interactive=not auto,
    )

    # Run
    success = asyncio.run(run_pipeline(config))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    cli()
