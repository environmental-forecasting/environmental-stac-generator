import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

from .ingest import main as ingest_main
from .preprocess import main as preprocess_main
from .utils import DEFAULT_WORKERS

app = typer.Typer()

logger = logging.getLogger(__name__)


@app.callback()
def main_callback():
    """Environmental STAC generator CLI."""


@app.command(help="Generate COGs and generate static JSON STAC catalog.")
def preprocess(
    input: Annotated[
        list[str], typer.Argument(help="Input file, directory or wildcard pattern")
    ],
    name: Annotated[
        str, typer.Option("-n", "--name", help="Collection name")
    ] = "default",
    workers: Annotated[
        int,
        typer.Option(
            "-w",
            "--workers",
            help=f"Max number of concurrent workers (default: CPU count, {DEFAULT_WORKERS})",
        ),
    ] = DEFAULT_WORKERS,
    overwrite: Annotated[
        bool, typer.Option("-o", "--overwrite", help="Overwrite existing COGs")
    ] = False,
    no_compress: Annotated[
        bool,
        typer.Option(
            "-c",
            "--no-compress",
            help="Disable COG compression (default is compressed)",
        ),
    ] = False,
    stac_only: Annotated[
        bool,
        typer.Option(
            "-s",
            "--stac-only",
            help="Output only the STAC files, not COGs/Thumbnails (default is not enabled)",
        ),
    ] = False,
):
    logger.debug(f"Command line input arguments: {sys.argv}")
    preprocess_main(
        input=input,
        name=name,
        workers=workers,
        overwrite=overwrite,
        compress=not no_compress,
        stac_only=stac_only,
    )


@app.command(help="Ingest generated JSON STAC catalog into pgSTAC database.")
def ingest(
    catalog: Annotated[str, typer.Argument(help="Path to the STAC catalog JSON file.")],
    overwrite: Annotated[
        bool,
        typer.Option(
            "-o", "--overwrite", help="Overwrite any matching collections/items"
        ),
    ] = False,
    env_file: Annotated[
        Path | None,
        typer.Option(
            "--env-file",
            help="Path to environment file (e.g. .env.dev) for database "
            "settings and FILE_SERVER_URL. Falls back to .env if present, otherwise "
            "process environment variables.",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
):
    logger.debug(f"Command line input arguments: {sys.argv}")
    ingest_main(
        catalog=catalog,
        overwrite=overwrite,
        env_file=env_file,
    )


def main():
    app()


if __name__ == "__main__":
    main()
