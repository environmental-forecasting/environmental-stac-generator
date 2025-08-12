import logging
import os

from dotenv import load_dotenv

from .stac.dataloader import PGSTACDataLoader

logger = logging.getLogger(__name__)


def main(catalog: str, overwrite: bool = False) -> None:
    """
    Main function to ingest pre-generated STAC catalogs into pgSTAC database.

    Loads a JSON STAC catalog file using the `PGSTACDataLoader`, which communicates
    with a running instance of [stac-fastapi](https://github.com/stac-utils/stac-fastapi)
    (e.g., pgSTAC), to ingest the catalog into a PostgreSQL/PostGIS database.

    Args:
        catalog: Path to the JSON STAC catalog file to be ingested.
        overwrite: Whether to overwrite any existing matching collections/items.
                   Defaults to False.

    Raises:
        FileNotFoundError: If no valid JSON files are found for ingestion.

    Raises:
        ValueError: If the `STAC_FASTAPI_URL` environment variable is not set.
        ConnectionError: If the STAC API is unreachable or unresponsive.
        Exception: Any exception raised by the underlying `PGSTACDataLoader`.

    Examples:
        >>> dashboard ingest data/stac/catalog.json
    """

    # Configuration
    load_dotenv()
    stac_api_url = os.getenv("STAC_FASTAPI_URL", None) # E.g. "http://localhost:8081"
    if not stac_api_url:
        logger.error("No STAC API URL specified in environment variable 'STAC_FASTAPI_URL'")
        exit(1)

    loader = PGSTACDataLoader(stac_api_url)

    # Wait for "stac-fastapi" to be ready
    if not loader.wait_for_api():
        logger.error("STAC API not available, exiting")
        return

    # Actually load the STAC metadata into PgSTAC database
    loader.load_stac_catalog(catalog_file=catalog, overwrite=overwrite)
