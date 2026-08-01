import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from .ingest import main as ingest_main
from .preprocess import main as preprocess_main
from .utils import DEFAULT_WORKERS

app = typer.Typer()

logger = logging.getLogger(__name__)


@app.callback()
def main_callback(
    ctx: typer.Context,
    env_file: Optional[Path] = typer.Option(
        None,
        "--env-file",
        help="Path to environment file (e.g. .env.development); used by ingest "
        "for database settings and FILE_SERVER_URL. "
        "Falls back to .env if present, otherwise process environment variables.",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
):
    """Environmental STAC generator CLI."""
    ctx.ensure_object(dict)
    ctx.obj["env_file"] = env_file


@app.command(help="Generate COGs and generate static JSON STAC catalog.")
def preprocess(
    forecast_frequency: str = typer.Argument(
        ..., help="The forecast frequency (e.g., 6hours, 1days, etc.)"
    ),
    input: list[str] = typer.Argument(
        ..., help="Input file, directory or wildcard pattern"
    ),
    name: str = typer.Option(
        "default", "-n", "--name", help="Collection name"
    ),
    workers: int = typer.Option(
        DEFAULT_WORKERS,
        "-w",
        "--workers",
        help=f"Max number of concurrent workers (default: CPU count, {DEFAULT_WORKERS})",
    ),
    overwrite: bool = typer.Option(
        False, "-o", "--overwrite", help="Overwrite existing COGs"
    ),
    no_compress: bool = typer.Option(
        True,
        "-c",
        "--no-compress",
        help="Disable COG compression (default is compressed)",
    ),
    stac_only: bool = typer.Option(
        False,
        "-s",
        "--stac-only",
        help="Output only the STAC files, not COGs/Thumbnails (default is not enabled)",
    ),
):
    logger.debug(f"Command line input arguments: {sys.argv}")
    preprocess_main(
        forecast_frequency=forecast_frequency,
        input=input,
        name=name,
        workers=workers,
        overwrite=overwrite,
        no_compress=no_compress,
        stac_only=stac_only,
    )


@app.command(help="Ingest generated JSON STAC catalog into pgSTAC database.")
def ingest(
    ctx: typer.Context,
    catalog: str = typer.Argument(..., help="Path to the STAC catalog JSON file."),
    overwrite: bool = typer.Option(
        False, "-o", "--overwrite", help="Overwrite any matching collections/items"
    ),
):
    logger.debug(f"Command line input arguments: {sys.argv}")
    ingest_main(
        catalog=catalog,
        overwrite=overwrite,
        env_file=ctx.obj.get("env_file"),
    )


def main():
    app()


if __name__ == "__main__":
    main()
