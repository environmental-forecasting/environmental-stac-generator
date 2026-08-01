from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host_ip: str = "127.0.0.1"
    database_port: int = 5432
    database_user: str
    database_password: str
    database_dbname: str = "postgis"
    file_server_url: str = "http://localhost:8001"

    # extra="ignore" so orchestrator-only keys (DOMAIN_NAME, ports, etc.) are allowed
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def database_url(self) -> str:
        """Return the PostgreSQL DSN for connecting to the database."""
        return (
            f"postgresql://{self.database_user}:{self.database_password}"
            f"@{self.host_ip}:{self.database_port}/{self.database_dbname}"
        )


def get_settings(env_file: str | Path | None = None) -> Settings:
    """
    Load settings from the process environment and an optional env file.

    If ``env_file`` is given, that file is required. Otherwise a local ``.env``
    is used when present (backwards compatible); otherwise only process env vars.
    """
    if env_file is not None:
        path = Path(env_file)
        if not path.is_file():
            raise FileNotFoundError(f"Environment file not found: {path}")
        return Settings(_env_file=path)

    if Path(".env").is_file():
        return Settings(_env_file=".env")

    return Settings(_env_file=None)
