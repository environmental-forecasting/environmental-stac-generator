import pytest


@pytest.fixture(autouse=True)
def _default_settings_env(monkeypatch):
    """Provide default DB/file-server env so Settings/get_settings work in tests."""
    monkeypatch.setenv("DATABASE_USER", "stac")
    monkeypatch.setenv("DATABASE_PASSWORD", "stac")
    monkeypatch.setenv("DATABASE_DBNAME", "postgis")
    monkeypatch.setenv("HOST_IP", "127.0.0.1")
    monkeypatch.setenv("DATABASE_PORT", "5432")
    monkeypatch.setenv("FILE_SERVER_URL", "http://localhost:8001")
