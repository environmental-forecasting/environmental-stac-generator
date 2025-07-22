import sys
import typer

from types import SimpleNamespace

from .preprocess import main as preprocess_main
# from .ingest import main as ingest_main

app = typer.Typer()


@app.command(help="Generate COGs and generate static JSON STAC catalog.")
def preprocess(
    forecast_frequency: str = typer.Argument(
        ..., help="The forecast frequency (e.g., 6hours, 1days, etc.)"
    ),
    input: list[str] = typer.Argument(
        ..., help="Input file, directory or wildcard pattern"
    ),
    name: str = typer.Option(
        "icenet", "-n", "--name", help="Collection name"
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
):
    print("Command:", " ".join(sys.argv))
    args = SimpleNamespace(
        forecast_frequency=forecast_frequency,
        input=input,
        name=name,
        overwrite=overwrite,
        no_compress=no_compress,
    )
    preprocess_main(args)


@app.command(help="Ingest generated JSON STAC catalog into pgSTAC database.")
def ingest(
    catalog: str = typer.Argument(..., help="Path to the STAC catalog JSON file."),
):
    print("Command:", " ".join(sys.argv))
    args = SimpleNamespace(
        catalog = catalog,
    )
#     ingest_main(args)


def main():
    app()

if __name__ == "__main__":
    main()
