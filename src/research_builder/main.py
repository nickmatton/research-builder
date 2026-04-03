"""CLI entry point for the research builder harness."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from .config import Config
from .llm.client import LLMClient
from .orchestrator.agent import OrchestratorAgent
from .orchestrator.loop import ExecutionLoop
from .storage.spec_store import SpecStore
from .storage.workspace import WorkspaceManager


async def run_pipeline(config: Config) -> bool:
    """Execute the full paper reproduction pipeline.

    1. Initialize workspace
    2. Ingest paper and create canonical spec
    3. Run execution loop (dispatch sub-agents per phase)

    Returns True if the run completed successfully.
    """
    logger = logging.getLogger(__name__)

    # Initialize workspace
    workspace = WorkspaceManager(config)
    workspace.initialize()
    logger.info("Workspace initialized at %s", config.project_root)

    # Initialize LLM client
    llm_client = LLMClient(config)

    # Initialize orchestrator
    store = SpecStore(config.spec_dir)
    orchestrator_agent = OrchestratorAgent(config, llm_client)

    # Step 1: Create canonical spec from paper
    paper_path = config.paper_path
    if not paper_path.exists():
        logger.error("Paper not found: %s", paper_path)
        click.echo(f"Error: Paper not found at {paper_path}", err=True)
        return False

    click.echo(f"Ingesting paper: {paper_path}")
    spec_manager = await orchestrator_agent.create_spec(paper_path, store)
    click.echo(f"Spec created: {len(spec_manager.state.phases)} phases")
    for phase in spec_manager.state.phases:
        deps = spec_manager.dep_graph.get_dependencies(phase.phase_id)
        dep_str = f" (depends on: {', '.join(deps)})" if deps else ""
        click.echo(f"  - {phase.phase_id}: {phase.title}{dep_str}")

    # Step 2: Run execution loop
    click.echo("\nStarting execution loop...")
    loop = ExecutionLoop(
        config=config,
        llm_client=llm_client,
        spec_manager=spec_manager,
        workspace=workspace,
        orchestrator_agent=orchestrator_agent,
    )

    success = await loop.run()

    if success:
        click.echo("\nRun completed successfully!")
        click.echo(f"Spec: {workspace.spec_md_path}")
        click.echo(f"State: {workspace.state_path}")
        click.echo(f"Revision log: {workspace.revision_log_path}")
        click.echo(f"Phases: {config.phases_dir}")
    else:
        click.echo("\nRun failed. Check the revision log for details.", err=True)
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
    verbose: bool,
) -> None:
    """Reproduce a research paper's results.

    PAPER is the path to the research paper PDF.
    """
    load_dotenv()
    # Set up logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build config
    project_root = output or Path(".")
    config = Config(
        paper_path=paper.resolve(),
        project_root=project_root.resolve(),
        model=model,
        max_retries=max_retries,
        max_debug_attempts=max_debug_attempts,
    )

    # Run
    success = asyncio.run(run_pipeline(config))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    cli()
