from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from environmental_stac_generator.config import get_settings
from environmental_stac_generator.ingest import main


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("HOST_IP", "127.0.0.1")
    monkeypatch.setenv("DATABASE_PORT", "5432")
    monkeypatch.setenv("DATABASE_USER", "user")
    monkeypatch.setenv("DATABASE_PASSWORD", "password")
    monkeypatch.setenv("DATABASE_DBNAME", "db")


def test_main_missing_env_vars(mock_env, monkeypatch):
    # Drop the database user so config loading should fail inside main().
    monkeypatch.delenv("DATABASE_USER", raising=False)

    with pytest.raises(ValidationError, match="database_user"):
        main("catalog.json")


def test_main_success(mock_env):
    with patch("environmental_stac_generator.ingest.PGSTACDataLoader") as MockLoader:
        mock_instance = MockLoader.return_value

        main("catalog.json", overwrite=True)

        MockLoader.assert_called_once_with(
            "postgresql://user:password@127.0.0.1:5432/db",
            file_server_url="http://localhost:8001",
        )
        mock_instance.ingest_stac_catalog.assert_called_once_with(
            catalog_file="catalog.json", overwrite=True
        )


def test_get_settings_from_env_file(tmp_path, monkeypatch):
    # Clear process env so the file is the source of truth.
    for key in (
        "HOST_IP",
        "DATABASE_PORT",
        "DATABASE_USER",
        "DATABASE_PASSWORD",
        "DATABASE_DBNAME",
        "FILE_SERVER_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    env_path = tmp_path / ".env.development"
    env_path.write_text(
        "\n".join(
            [
                "HOST_IP=10.0.0.1",
                "DATABASE_PORT=5433",
                "DATABASE_USER=stac",
                "DATABASE_PASSWORD=secret",
                "DATABASE_DBNAME=postgis",
                "FILE_SERVER_URL=http://files.example:8001",
            ]
        )
        + "\n"
    )

    settings = get_settings(env_path)
    assert settings.host_ip == "10.0.0.1"
    assert settings.database_port == 5433
    assert settings.database_url == (
        "postgresql://stac:secret@10.0.0.1:5433/postgis"
    )
    assert settings.file_server_url == "http://files.example:8001"


def test_get_settings_missing_file():
    with pytest.raises(FileNotFoundError, match="Environment file not found"):
        get_settings(Path("/nonexistent/env.file"))
