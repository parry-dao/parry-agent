"""Event manager for handling multiple event sources."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Self

from events.sources import (
    EventConfig,
    EventSource,
    FileSystemEventSource,
    FileWatchConfig,
    WebhookConfig,
)
from log import get_logger


if TYPE_CHECKING:
    from agent import AnyAgent
    from events.sources import EventData

logger = get_logger(__name__)


class EventManager:
    """Manages multiple event sources and their lifecycles."""

    def __init__(self, agent: AnyAgent[Any, Any], enable_events: bool = True):
        """Initialize event manager.

        Args:
            agent: Agent to manage events for
            enable_events: Whether to enable event processing
        """
        self.agent = agent
        self.enabled = enable_events
        self._sources: dict[str, EventSource] = {}
        self._tasks: set[asyncio.Task[Any]] = set()

    def create_source(self, config: EventConfig) -> EventSource:
        """Create an event source from configuration.

        Args:
            config: Event source configuration

        Returns:
            Configured event source instance

        Raises:
            ValueError: If source type is unknown or disabled
        """
        if not config.enabled:
            msg = f"Source {config.name} is disabled"
            raise ValueError(msg)

        match config:
            case FileWatchConfig():
                return FileSystemEventSource(config)
            case WebhookConfig():
                msg = "Webhook events not yet implemented"
                raise NotImplementedError(msg)
            case _:
                msg = f"Unknown event source type: {config.type}"
                raise ValueError(msg)

    async def add_source(self, config: EventConfig):
        """Add and start a new event source.

        Args:
            config: Event source configuration

        Raises:
            ValueError: If source already exists or is invalid
        """
        if not self.enabled:
            msg = "Event processing disabled, not adding source: %s"
            logger.warning(msg, config.name)
            return

        if config.name in self._sources:
            msg = f"Event source already exists: {config.name}"
            raise ValueError(msg)

        try:
            source = self.create_source(config)
            await source.connect()
            self._sources[config.name] = source

            # Start processing events
            name = f"event_processor_{config.name}"
            task = asyncio.create_task(self._process_events(source), name=name)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

            logger.debug("Added event source: %s", config.name)

        except Exception as e:
            msg = f"Failed to add event source {config.name}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

    async def remove_source(self, name: str):
        """Stop and remove an event source.

        Args:
            name: Name of source to remove
        """
        if source := self._sources.pop(name, None):
            await source.disconnect()
            logger.debug("Removed event source: %s", name)

    async def _process_events(self, source: EventSource):
        """Process events from a source.

        Args:
            source: Event source to process
        """
        try:
            # Get the async iterator from the coroutine
            async for event in source.events():
                if not self.enabled:
                    break
                await self._handle_event(event)

        except asyncio.CancelledError:
            logger.debug("Event processing cancelled")
            raise

        except Exception:
            logger.exception("Error processing events")

    async def _handle_event(self, event: EventData):
        """Handle a single event.

        This is a placeholder - in the actual implementation this would
        dispatch events to the appropriate handlers/agents.

        Args:
            event: Event to handle
        """
        prompt = event.to_prompt()
        await self.agent.run(prompt)

    async def cleanup(self):
        """Clean up all event sources and tasks."""
        self.enabled = False

        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.wait(self._tasks)

        # Disconnect all sources
        for name in list(self._sources):
            await self.remove_source(name)

    async def __aenter__(self) -> Self:
        """Allow using manager as async context manager."""
        return self

    async def __aexit__(self, *exc: object):
        """Clean up when exiting context."""
        await self.cleanup()
