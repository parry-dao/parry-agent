from typing import Annotated, Final, Literal

from platformdirs import user_data_dir
from pydantic import BaseModel, ConfigDict, Field
from upath import UPath
import yamling


LogFormat = Literal["chronological", "conversations"]

APP_NAME: Final = "parry-agent"
APP_AUTHOR: Final = "parry"
DATA_DIR: Final = UPath(user_data_dir(APP_NAME, APP_AUTHOR))
DEFAULT_DB_NAME: Final = "history.db"


def get_database_path() -> str:
    """Get the database file path, creating directories if needed."""
    db_path = DATA_DIR / DEFAULT_DB_NAME
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


class BaseStorageProviderConfig(BaseModel):
    type: str = Field(init=False)

    log_messages: bool = True
    """Whether to log messages"""

    log_conversations: bool = True
    """Whether to log conversations"""

    log_tool_calls: bool = True
    """Whether to log tool calls"""

    log_commands: bool = True
    """Whether to log command executions"""

    log_context: bool = True
    """Whether to log context messages."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


class SQLStorageConfig(BaseStorageProviderConfig):
    """SQL database storage configuration."""

    type: Literal["sql"] = Field("sql", init=False)
    url: str = Field(default_factory=get_database_path)
    """Database URL (e.g. sqlite:///history.db)"""
    pool_size: int = 5
    """Connection pool size"""


class TextLogConfig(BaseStorageProviderConfig):
    """Text log configuration."""

    type: Literal["text_file"] = Field("text_file", init=False)
    path: str
    """Path to log file"""
    format: LogFormat = "chronological"
    """Log format template to use"""
    template: Literal["chronological", "conversations"] | str | None = "chronological"  # noqa: PYI051
    """Template to use: either predefined name or path to custom template"""
    encoding: str = "utf-8"
    """File encoding"""


# Config:
class FileStorageConfig(BaseStorageProviderConfig):
    """File storage configuration."""

    type: Literal["file"] = Field("file", init=False)
    path: str
    """Path to storage file (extension determines format unless specified)"""
    format: yamling.FormatType = "auto"
    """Storage format (auto=detect from extension)"""
    encoding: str = "utf-8"
    """File encoding"""


class MemoryStorageConfig(BaseStorageProviderConfig):
    """In-memory storage configuration for testing."""

    type: Literal["memory"] = Field("memory", init=False)


StorageProviderConfig = Annotated[
    SQLStorageConfig | FileStorageConfig | TextLogConfig | MemoryStorageConfig,
    Field(discriminator="type"),
]


class StorageConfig(BaseModel):
    """Global storage configuration."""

    providers: list[StorageProviderConfig] | None = None
    """List of configured storage providers"""

    default_provider: str | None = None
    """Name of default provider for history queries.
    If None, uses first configured provider."""

    log_messages: bool = True
    """Whether to log messages."""

    log_conversations: bool = True
    """Whether to log conversations."""

    log_tool_calls: bool = True
    """Whether to log tool calls."""

    log_commands: bool = True
    """Whether to log command executions."""

    model_config = ConfigDict(frozen=True)

    @property
    def effective_providers(self) -> list[StorageProviderConfig]:
        """Get effective list of providers.

        Returns:
            - Default SQLite provider if providers is None
            - Empty list if providers is empty list
            - Configured providers otherwise
        """
        if self.providers is None:
            cfg = SQLStorageConfig()
            return [cfg]
        return self.providers
