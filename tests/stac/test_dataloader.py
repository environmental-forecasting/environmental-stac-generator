from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from environmental_stac_generator.stac.dataloader import PGSTACDataLoader
from pystac import Catalog, Collection, Item


@pytest.fixture
def mock_db():
    with patch("environmental_stac_generator.stac.dataloader.PgstacDB") as MockDB:
        mock_instance = MockDB.return_value
        yield mock_instance


@pytest.fixture
def mock_loader():
    with patch("environmental_stac_generator.stac.dataloader.Loader") as MockLoader:
        mock_instance = MockLoader.return_value
        yield mock_instance


@pytest.fixture
def loader_instance(mock_db, mock_loader):
    # Initialise without API URL for direct DB interaction
    return PGSTACDataLoader("postgresql://user:pass@localhost:5432/db")


def test_init_with_api_url_success(mock_db, mock_loader):
    with patch.object(PGSTACDataLoader, "wait_for_api", return_value=True):
        loader = PGSTACDataLoader("postgresql://user:pass@localhost:5432/db", stac_api_url="http://localhost:8000/")
        assert loader.stac_api_url == "http://localhost:8000"
        assert loader._use_api == "http://localhost:8000/"


def test_init_with_api_url_failure(mock_db, mock_loader):
    with patch.object(PGSTACDataLoader, "wait_for_api", return_value=False):
        with pytest.raises(SystemExit):
            PGSTACDataLoader("postgresql://user:pass@localhost:5432/db", stac_api_url="http://localhost:8000/")


def test_collection_exists_db(loader_instance, mock_db):
    mock_db.query_one.return_value = True
    assert loader_instance.collection_exists("test_col") is True
    mock_db.query_one.assert_called_once_with("SELECT EXISTS (SELECT 1 FROM collections WHERE id = %s);", ["test_col"])


def test_collection_exists_api(mock_db, mock_loader):
    with patch.object(PGSTACDataLoader, "wait_for_api", return_value=True):
        loader = PGSTACDataLoader("postgresql://user:pass@localhost:5432/db", stac_api_url="http://test")
        with patch("environmental_stac_generator.stac.dataloader.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            assert loader.collection_exists("test_col") is True
            mock_get.assert_called_once_with("http://test/collections/test_col")


def test_item_exists_db(loader_instance, mock_db):
    mock_db.query_one.return_value = True
    assert loader_instance.item_exists("test_col", "test_item") is True
    
    query = """
                SELECT EXISTS (
                    SELECT 1 FROM items
                    WHERE id = %s AND collection = %s
                );
            """
    mock_db.query_one.assert_called_once_with(query, ["test_item", "test_col"])


def test_ingest_stac_catalog_file_not_found(loader_instance):
    with pytest.raises(FileNotFoundError):
        loader_instance.ingest_stac_catalog("nonexistent.json")


def test_ingest_stac_catalog_success(loader_instance, tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.touch()
    
    with patch("environmental_stac_generator.stac.dataloader.Catalog.from_file") as mock_from_file, \
         patch.object(PGSTACDataLoader, "_load_collections_from_file") as mock_load:
        
        mock_from_file.return_value = MagicMock(spec=Catalog)
        
        result = loader_instance.ingest_stac_catalog(catalog_path, overwrite=True)
        
        assert result is True
        mock_from_file.assert_called_once_with(str(catalog_path))
        mock_load.assert_called_once_with(overwrite=True)


def test_ingest_collection_and_items(loader_instance, mock_loader):
    collections = [{"id": "col1"}]
    items = [{"id": "item1"}]
    
    result = loader_instance._ingest_collection_and_items(collections, items, overwrite=True)
    
    assert result is True
    assert mock_loader.load_collections.call_count == 1
    assert mock_loader.load_items.call_count == 1
