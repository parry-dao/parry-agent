"""Event sources for parry agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from asyncio import Event
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field
from watchfiles import Change, awatch
from watchfiles.filters import DefaultFilter


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from watchfiles.main import FileChange


ChangeType = Literal["added", "modified", "deleted"]

CHANGE_TO_TYPE: dict[Change, ChangeType] = {
    Change.added: "added",
    Change.modified: "modified",
    Change.deleted: "deleted",
}


@dataclass(frozen=True)
class EventData:
    """Base class for event data."""

    source: str
    timestamp: datetime

    @classmethod
    def create(cls, source: str, **kwargs: Any) -> Self:
        """Create event with current timestamp."""
        return cls(source=source, timestamp=datetime.now(), **kwargs)

    @abstractmethod
    def to_prompt(self) -> str:
        """Convert event to agent prompt."""


@dataclass(frozen=True)
class FileEvent(EventData):
    """File system event."""

    path: str
    type: ChangeType

    def to_prompt(self) -> str:
        return f"File {self.type}: {self.path}"


class EventSourceConfig(BaseModel):
    """Base configuration for event sources."""

    type: str = Field(init=False)
    """Discriminator field for event source types."""

    name: str
    """Unique identifier for this event source."""

    enabled: bool = True
    """Whether this event source is active."""

    # extra_knowledge: Knowledge | None = None
    """Additional context to load when this trigger activates."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


class FileWatchConfig(EventSourceConfig):
    """File watching event source.

    Monitors file system changes and triggers agent actions when:
    - Files are created
    - Files are modified
    - Files are deleted
    """

    type: Literal["file"] = Field("file", init=False)
    """Type discriminator for file watch sources."""

    paths: list[str]
    """Paths or patterns to watch for changes."""

    extensions: list[str] | None = None
    """File extensions to monitor (e.g. ['.py', '.md'])."""

    ignore_paths: list[str] | None = None
    """Paths or patterns to ignore."""

    recursive: bool = True
    """Whether to watch subdirectories."""

    debounce: int = 1600
    """Minimum time (ms) between trigger events."""


class WebhookConfig(EventSourceConfig):
    """Webhook event source.

    Listens for HTTP requests and triggers agent actions when:
    - POST requests are received at the configured endpoint
    - Request content matches any defined filters
    """

    type: Literal["webhook"] = Field("webhook", init=False)
    """Type discriminator for webhook sources."""

    port: int
    """Port to listen on."""

    path: str
    """URL path to handle requests."""

    secret: str | None = None
    """Optional secret for request validation."""


class ManualTriggerConfig(EventSourceConfig):
    """Manual trigger configuration.

    Defines actions that can be triggered directly by users through CLI for example.

    Unlike other triggers, these don't activate automatically but provide
    a way to define reusable agent interactions.
    """

    type: Literal["manual"] = Field("manual", init=False)
    """Type discriminator for manual triggers."""

    prompt: str
    """Prompt to send to the agent when triggered."""

    description: str | None = None
    """Human-readable description of what this trigger does."""


EventConfig = Annotated[
    FileWatchConfig | WebhookConfig | ManualTriggerConfig, Field(discriminator="type")
]


class ExtensionFilter:
    """Filter for specific file extensions."""

    def __init__(self, extensions: list[str], ignore_paths: list[str] | None = None):
        """Initialize filter.

        Args:
            extensions: File extensions to watch (e.g. ['.py', '.md'])
            ignore_paths: Paths to ignore
        """
        self.extensions = tuple(
            ext if ext.startswith(".") else f".{ext}" for ext in extensions
        )
        self._default_filter = DefaultFilter(ignore_paths=ignore_paths)

    def __call__(self, change: Change, path: str) -> bool:
        """Check if file should be watched."""
        return path.endswith(self.extensions) and self._default_filter(change, path)


class EventSource(ABC):
    """Base class for event sources."""

    @abstractmethod
    async def connect(self):
        """Initialize connection to event source."""

    @abstractmethod
    async def disconnect(self):
        """Close connection to event source."""

    @abstractmethod
    def events(self) -> AsyncGenerator[EventData, None]:
        """Get event iterator.

        Returns:
            AsyncIterator yielding events from this source

        Note: This is a coroutine that returns an AsyncIterator
        """
        raise NotImplementedError


class FileSystemEventSource(EventSource):
    """Watch file system changes using watchfiles."""

    def __init__(self, config: FileWatchConfig):
        """Initialize file system watcher.

        Args:
            config: File watch configuration
        """
        self.config = config
        self._watch: AsyncIterator[set[FileChange]] | None = None
        self._stop_event: Event | None = None

    async def connect(self):
        """Set up watchfiles watcher."""
        if not self.config.paths:
            msg = "No paths specified to watch"
            raise ValueError(msg)

        self._stop_event = Event()

        # Create filter from extensions if provided
        watch_filter = None
        if self.config.extensions:
            to_ignore = self.config.ignore_paths
            watch_filter = ExtensionFilter(self.config.extensions, ignore_paths=to_ignore)

        self._watch = awatch(
            *self.config.paths,
            watch_filter=watch_filter,
            debounce=self.config.debounce,
            stop_event=self._stop_event,
            recursive=self.config.recursive,
        )

    async def disconnect(self):
        """Stop watchfiles watcher."""
        if self._stop_event:
            self._stop_event.set()
        self._watch = None
        self._stop_event = None

    async def events(self) -> AsyncGenerator[EventData, None]:
        """Get file system events."""
        watch = self._watch
        if not watch:
            msg = "Source not connected"
            raise RuntimeError(msg)

        async for changes in watch:
            for change, path in changes:
                if change not in CHANGE_TO_TYPE:
                    continue
                typ = CHANGE_TO_TYPE[change]
                yield FileEvent.create(source=self.config.name, path=str(path), type=typ)
