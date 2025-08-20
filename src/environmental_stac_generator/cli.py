import sys
import logging

import typer

from .ingest import main as ingest_main
from .preprocess import main as preprocess_main

app = typer.Typer()

logger = logging.getLogger(__name__)

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
        4, "-w", "--workers", help="Max number of concurrent workers"
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
    catalog: str = typer.Argument(..., help="Path to the STAC catalog JSON file."),
    overwrite: bool = typer.Option(
        False, "-o", "--overwrite", help="Overwrite any matching collections/items"
    ),
):
    logger.debug(f"Command line input arguments: {sys.argv}")
    ingest_main(
        catalog = catalog,
        overwrite = overwrite,
    )


def main():
    app()

if __name__ == "__main__":
    main()
