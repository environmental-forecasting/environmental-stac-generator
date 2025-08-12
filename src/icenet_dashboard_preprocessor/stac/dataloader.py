import logging
import time
from pathlib import Path

import pystac
import requests
from pystac import Catalog, Collection, Item

logger = logging.getLogger(__name__)

class PGSTACDataLoader:
    """Ingest JSON STAC metadata into PgSTAC using stac-fastapi"""

    def __init__(self, stac_api_url: str = "http://stac-fastapi:8081") -> None:
        self.stac_api_url = stac_api_url.rstrip("/")
        self.session = requests.Session()

    def wait_for_api(self, max_retries: int = 30, delay: int = 10) -> bool:
        """Wait for STAC API to be accessible"""
        for i in range(max_retries):
            logging.info(f"{i}")
            try:
                response = self.session.get(f"{self.stac_api_url}/")
                if response.status_code == 200:
                    logger.info("STAC API is accessible")
                    return True
            except requests.exceptions.RequestException as e:
                logger.info(f"Waiting for STAC API... (attempt {i + 1}/{max_retries})")
                time.sleep(delay)

        logger.error("STAC API not accessible after maximum retries")
        return False

    def create_collection(self, collection_dict: dict, overwrite: bool = False) -> bool:
        """Create a collection in PgSTAC"""
        try:
            kwargs = dict(
                url=f"{self.stac_api_url}/collections",
                json=collection_dict,
                headers={"Content-Type": pystac.MediaType.JSON},
            )
            response = self.session.post(**kwargs) # type: ignore

            # Ref: https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status
            if response.status_code in [200, 201]:
                logger.info(f"Created collection: {collection_dict['id']}")
                return True
            elif response.status_code == 409:
                message = f"Collection already exists: {collection_dict['id']}"
                if overwrite:
                    message = "Overwrite requested, " + message
                    response = self.session.put(**kwargs) # type: ignore
                logger.info(f"{message}")
                return True
            else:
                logger.error(
                    f"Failed to create collection: {response.status_code} - {response.text}"
                )
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating collection: {e}")
            return False

    def create_item(self, collection_id: str, item_dict: dict, overwrite: bool = False) -> bool:
        """Create an item in PgSTAC"""
        try:
            kwargs = dict(
                url=f"{self.stac_api_url}/collections/{collection_id}/items",
                json=item_dict,
                headers={"Content-Type": pystac.MediaType.JSON},
            )
            response = self.session.post(**kwargs) # type: ignore

            if response.status_code == 201:
                logger.info(f"Created item: {item_dict['id']}")
                return True
            elif response.status_code == 409:
                message = f"Item already exists: {item_dict['id']}"
                if overwrite:
                    message = "Overwrite requested, " + message
                    response = self.session.put(**kwargs) # type: ignore
                logger.info(f"{message}")
                return True
            else:
                logger.error(
                    f"Failed to create item {item_dict['id']}: {response.status_code} - {response.text}"
                )
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating item {item_dict['id']}: {e}")
            return False

    # def flatten_collection(self, collection: Collection):
    #     from copy import deepcopy
    #     new_collection = deepcopy(collection)
    #     for sub_collection in new_collection.get_collections():
    #         new_collection.remove_child(sub_collection.id)
    #     # print(new_collection)
    #     # exit()
    #     all_items = []
    #     all_collections = [collection]

    #     def traverse(c: pystac.Collection):
    #         for child in c.get_children():
    #             if isinstance(child, pystac.Collection):
    #                 all_collections.append(child)
    #                 traverse(child)
    #         for item in c.get_items():
    #             item.set_parent(collection)
    #             # print("Item parent:", item.get_parent())
    #             all_items.append(item)

    #     traverse(collection)
    #     return all_collections, all_items

    def load_stac_catalog(
        self,
        catalog_file: str | Path = Path("data/stac_flat/catalog.json"),
        overwrite: bool = False,
    ) -> bool:
        """Load STAC catalog into PgSTAC"""
        if isinstance(catalog_file, str):
            catalog_file = Path(catalog_file)

        if not catalog_file.exists():
            logger.error(f"Catalog file not found: {catalog_file}")
            return False

        catalog = Catalog.from_file(catalog_file)

        collection = list(catalog.get_collections())
        for collection in catalog.get_all_collections():
            # nested_collections, items = self.flatten_collection(collection)
            self._load_collection_from_file(collection, overwrite)

        return True

    def _get_item_hierarchy(self, item: Item) -> list:
        hierarchy = []
        current = item
        while current is not None:
            hierarchy.append(current)
            current = current.get_parent()
        return hierarchy

    def _load_collection_from_file(self, collection: Collection, overwrite=False) -> None:
        """Load a collection and its items from file"""

        self.create_collection(collection.to_dict(), overwrite=overwrite)

        # `get_all_items()` also walks through back-references - creates duplicates, so, need to de-duplicate.
        items = collection.get_all_items()
        unique_items = {item.id: item for item in items}

        for item in unique_items.values():
            # logging.debug("Item:", type(item))
            # catalog_hierarchy = reversed(self._get_item_hierarchy(item))
            # print(list(catalog_hierarchy))

            self.create_item(collection.id, item.to_dict(), overwrite=overwrite)
