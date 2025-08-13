import logging
from pathlib import Path
from typing import Union

from pypgstac.db import PgstacDB
from pypgstac.load import Loader, Methods
from pystac import Catalog

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PGSTACDataLoader:
    """
    Ingest STAC metadata into a pgSTAC database using pypgstac's bulk loader.
    """

    def __init__(self, pg_db_url: str):
        self.db = PgstacDB(dsn=pg_db_url)
        self.loader = Loader(self.db)

    def load_stac_catalog(
        self,
        catalog_file: Union[str, Path],
        overwrite: bool = False,
    ) -> bool:
        """
        Load STAC collections and items from a catalog.json into the pgSTAC database.
        """
        catalog_file = Path(catalog_file)
        if not catalog_file.exists():
            logger.error(f"Catalog file not found: {catalog_file}")
            return False

        try:
            catalog = Catalog.from_file(str(catalog_file))
        except Exception as e:
            logger.exception(f"Failed to read catalog: {e}")
            return False

        insert_mode = Methods.upsert if overwrite else Methods.insert

        try:
            # Load collections
            collections = list(catalog.get_all_collections())
            collection_dicts = [c.to_dict() for c in collections]
            self.loader.load_collections(
                file=iter(collection_dicts), insert_mode=insert_mode
            )
            logger.info(f"Loaded {len(collection_dicts)} collections")

            # Load items
            items = list(catalog.get_all_items())
            item_dicts = (i.to_dict() for i in items)
            self.loader.load_items(file=item_dicts, insert_mode=insert_mode)
            logger.info(f"Loaded {len(items)} items")

            return True

        except Exception as e:
            logger.exception(f"Failed to load STAC catalog: {e}")
            return False
