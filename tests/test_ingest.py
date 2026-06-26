import os
from unittest.mock import MagicMock, patch

import pytest
from environmental_stac_generator.ingest import main


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("HOST_IP", "127.0.0.1")
    monkeypatch.setenv("DATABASE_PORT", "5432")
    monkeypatch.setenv("DATABASE_USER", "user")
    monkeypatch.setenv("DATABASE_PASSWORD", "password")
    monkeypatch.setenv("DATABASE_DBNAME", "db")


def test_main_missing_env_vars(mock_env, monkeypatch):
    # Ensure environment variables are clear
    monkeypatch.delenv("DATABASE_USER", raising=False)
    
    with patch("environmental_stac_generator.ingest.load_dotenv"):
        with pytest.raises(ValueError, match="Missing required environment variable: DATABASE_USER"):
            main("catalog.json")


def test_main_success(mock_env):
    with patch("environmental_stac_generator.ingest.PGSTACDataLoader") as MockLoader, \
         patch("environmental_stac_generator.ingest.load_dotenv"):
        mock_instance = MockLoader.return_value
        
        main("catalog.json", overwrite=True)
        
        expected_dsn = "postgresql://user:password@127.0.0.1:5432/db"
        MockLoader.assert_called_once_with(expected_dsn)
        mock_instance.ingest_stac_catalog.assert_called_once_with(catalog_file="catalog.json", overwrite=True)
