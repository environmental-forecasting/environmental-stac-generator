import logging
from pathlib import Path

from .config import get_settings
from .stac.dataloader import PGSTACDataLoader

logger = logging.getLogger(__name__)


def main(
    catalog: str,
    overwrite: bool = False,
    env_file: str | Path | None = None,
) -> None:
    """
    Main function to ingest pre-generated STAC catalogs into pgSTAC database.

    Loads a JSON STAC catalog file using the `PGSTACDataLoader`, which communicates
    with a running instance of [stac-fastapi](https://github.com/stac-utils/stac-fastapi)
    (e.g., pgSTAC), to ingest the catalog into a PostgreSQL/PostGIS database.
    Portable asset hrefs (e.g. `data/cogs/...`) are prefixed with `FILE_SERVER_URL`
    from settings before load.

    Args:
        catalog: Path to the JSON STAC catalog file to be ingested.
        overwrite: Whether to overwrite any existing matching collections/items.
                   Defaults to False.
        env_file: Optional path to an environment file (e.g. ``.env.development``).

    Raises:
        FileNotFoundError: If no valid JSON files are found for ingestion.
        Exception: Any exception raised by the underlying `PGSTACDataLoader`.

    Examples:
        >>> envstacgen ingest --env-file .env.development data/stac/catalog.json
    """
    config = get_settings(env_file)
    pg_db_url = config.database_url

    loader = PGSTACDataLoader(
        pg_db_url,
        file_server_url=config.file_server_url,
    )

    # Actually load the STAC metadata into PgSTAC database
    loader.ingest_stac_catalog(catalog_file=catalog, overwrite=overwrite)
