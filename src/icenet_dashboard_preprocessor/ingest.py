import logging
import os
from types import SimpleNamespace

from dotenv import load_dotenv

from .stac.dataloader import PGSTACDataLoader

logger = logging.getLogger(__name__)


def main(args: SimpleNamespace):
    """
    Main function to ingest pre-generated STAC catalogs into PostGIS database.

    This function ingests JSON STAC catalogs into the PostGIS database.

    Args:
        args: Parsed Command line arguments. The expected keys are:
            input (List[str]): List of STAC catalog/collection/item JSON files or directories w/ them.
            --overwrite (bool): Whether to overwrite existing database entry matches.

    Raises:
        FileNotFoundError: If no valid JSON files are found for ingestion.

    Returns:
        None

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
    loader.load_stac_catalog(catalog_file=args.catalog)
