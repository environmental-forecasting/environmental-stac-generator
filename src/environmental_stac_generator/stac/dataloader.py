import logging
import time
from pathlib import Path

import requests
from pypgstac.db import PgstacDB
from pypgstac.load import Loader, Methods
from pystac import Catalog

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PGSTACDataLoader:
    """Loads STAC catalogs into a PostgreSQL database via the PgstacDB interface.

    This class provides methods to check existence of STAC collections/items,
    load STAC catalogs from files, and ingest data into the PostgreSQL database.
    It also includes functionality to wait for the STAC API to be accessible.

    Note: The legacy method of checking collection/item existence via the STAC API
    has been retained for my own reference, but is much slower than direct database
    queries. Direct database usage is recommended for performance. i.e., not
    specifying a `stac_api_url` upon instance initialisation.

    Attributes:
        db: Instance of PgstacDB for interacting with PostgreSQL.
        loader: Loader instance for ingesting STAC data.
        stac_api_url: Base URL of the STAC API (without trailing slash).
        _use_api: If set, collection/item existence checks use the legacy STAC API;
                  otherwise, direct database queries are used.
    """

    def __init__(self, pg_db_url: str, stac_api_url: str = ""):
        """Initialise PGSTACDataLoader with database and STAC API configurations.

        Args:
            pg_db_url: PostgreSQL database connection string.
            stac_api_url (optional): Base URL of the STAC API. If provided,
                collection/item existence checks will use the legacy STAC API
                for compatibility, but direct database queries are recommended
                for performance.
                Defaults to "".
        """
        self.db = PgstacDB(dsn=pg_db_url)
        self.loader = Loader(self.db)
        if stac_api_url:
            self.stac_api_url = stac_api_url.rstrip("/")
            if not self.wait_for_api():
                logger.error("STAC API not available, exiting")
                exit(1)
        self._use_api = stac_api_url if stac_api_url else None

    def collection_exists(self, collection_id: str) -> bool:
        """Check if a STAC collection exists.

        Uses the legacy STAC API for existence checks (if configured),
        or direct database queries for better performance.

        Args:
            collection_id: ID of the STAC collection to check.

        Returns:
            True if the collection exists, False otherwise.
        """
        if self._use_api:
            url = f"{self.stac_api_url}/collections/{collection_id}"
            r = requests.get(url)
            return r.status_code == 200
        else:
            query = "SELECT EXISTS (SELECT 1 FROM collections WHERE id = %s);"
            result = bool(self.db.query_one(query, [collection_id]))
            return bool(result)

    def item_exists(self, collection_id: str, item_id: str) -> bool:
        """Check if a STAC item exists.

        Uses the legacy STAC API for existence checks (if configured),
        or direct database queries for better performance.

        Args:
            collection_id: ID of the STAC collection.
            item_id: ID of the STAC item to check.

        Returns:
            True if the item exists, False otherwise.
        """
        if self._use_api:
            url = f"{self.stac_api_url}/collections/{collection_id}/items/{item_id}"
            r = requests.get(url)
            return r.status_code == 200
        else:
            query = """
                SELECT EXISTS (
                    SELECT 1 FROM items
                    WHERE id = %s AND collection = %s
                );
            """
            result = self.db.query_one(query, [item_id, collection_id])
            return bool(result)

    def ingest_stac_catalog(
        self,
        catalog_file: str | Path,
        overwrite: bool = False,
    ) -> bool:
        """Load a STAC catalog from file into the PostgreSQL database.

        Args:
            catalog_file: Path to the STAC catalog JSON file.
            overwrite: If True, existing collections/items will be overwritten.

        Returns:
            True if loading was successful, False otherwise.
        """
        catalog_file = Path(catalog_file)
        if not catalog_file.exists():
            logger.error(f"Catalog file not found: {catalog_file}")
            return False

        try:
            self.catalog = Catalog.from_file(str(catalog_file))
            self._load_collections_from_file(overwrite=overwrite)
        except Exception as e:
            logger.exception(f"Failed to read catalog: {e}")

        return False

    def _load_collections_from_file(self, overwrite: bool = False) -> None:
        """Process collections and items from the STAC catalog file.

        Skips existing collections/items if overwrite is False. Used internally
        by load_stac_catalog().

        Args:
            overwrite: If True, existing collections/items will be overwritten.
        """
        # Prepare collections to load (skip if exists and overwrite==False)
        collections_to_load = []
        for collection in self.catalog.get_all_collections():
            if not overwrite and self.collection_exists(collection.id):
                logger.info(f"Skipping existing collection {collection.id}")
                continue
            collections_to_load.append(collection.to_dict())

        # Prepare items to load (skip if exists and overwrite==False)
        items_to_load = []
        for item in self.catalog.get_all_items():
            if not overwrite and self.item_exists(item.collection_id, item.id):
                logger.info(
                    f"Skipping existing item {item.id} in collection {item.collection_id}"
                )
                continue
            items_to_load.append(item.to_dict())

        self._ingest_collection_and_items(collections_to_load, items_to_load, overwrite)

    def _ingest_collection_and_items(
        self, collections_to_load: list, items_to_load: list, overwrite: bool
    ) -> bool:
        """Ingest STAC collections and items into PostgreSQL.

        Args:
            collections_to_load: List of collection dictionaries to load.
            items_to_load: List of item dictionaries to load.
            overwrite: If True, use upsert mode; otherwise, insert only.

        Returns:
            True if ingestion was successful, False otherwise.
        """
        insert_mode = Methods.upsert if overwrite else Methods.insert
        try:
            # Load collections via your existing loader
            if collections_to_load:
                self.loader.load_collections(
                    file=iter(collections_to_load), insert_mode=insert_mode
                )
                logger.info(f"Loaded {len(collections_to_load)} collections")
            else:
                logger.info("No collections to load")

            # Load items via your existing loader
            if items_to_load:
                self.loader.load_items(
                    file=iter(items_to_load), insert_mode=insert_mode
                )
                logger.info(f"Loaded {len(items_to_load)} items")
            else:
                logger.info("No items to load")

            return True
        except Exception as e:
            logger.exception(f"Failed to load STAC catalog: {e}")
            return False

    def wait_for_api(self, max_retries: int = 30, delay: int = 10) -> bool:
        """Wait for the STAC API to become accessible.

        Periodically checks if the STAC API is reachable by making a GET request
        to its root endpoint. Retries up to max_retries times with specified delay.

        Args:
            max_retries: Maximum number of attempts.
            delay: Delay between retries in seconds.

        Returns:
            True if the API becomes accessible, False otherwise.
        """
        for i in range(max_retries):
            logging.info(f"{i}")
            try:
                response = requests.get(f"{self.stac_api_url}")
                if response.status_code == 200:
                    logger.info("STAC API is accessible")
                    return True
            except requests.exceptions.RequestException as e:
                logger.info(f"Waiting for STAC API... (attempt {i + 1}/{max_retries})")
                time.sleep(delay)

        logger.error("STAC API not accessible after maximum retries")
        return False
