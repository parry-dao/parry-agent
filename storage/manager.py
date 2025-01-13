"""Storage manager for handling multiple providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from log import get_logger
from models.storage import (
    BaseStorageProviderConfig,
    FileStorageConfig,
    MemoryStorageConfig,
    SQLStorageConfig,
    TextLogConfig,
)
from parry_agent_storage.file_provider import FileProvider
from parry_agent_storage.memory_provider import MemoryStorageProvider
from parry_agent_storage.sql_provider import SQLModelProvider
from parry_agent_storage.text_log_provider import TextLogProvider


if TYPE_CHECKING:
    from datetime import datetime

    from models.agents import ToolCallInfo
    from models.messages import ChatMessage, TokenCost
    from models.session import SessionQuery
    from models.storage import StorageConfig
    from parry_agent_storage.base import StorageProvider

logger = get_logger(__name__)


class StorageManager:
    """Manages multiple storage providers.

    Handles:
    - Provider initialization and cleanup
    - Message distribution to providers
    - History loading from capable providers
    - Global logging filters
    """

    def __init__(self, config: StorageConfig):
        """Initialize storage manager.

        Args:
            config: Storage configuration including providers and filters
        """
        self.config = config
        self.providers = [
            self._create_provider(cfg) for cfg in self.config.effective_providers
        ]

    def cleanup(self):
        """Clean up all providers."""
        for provider in self.providers:
            try:
                provider.cleanup()
            except Exception:
                logger.exception("Error cleaning up provider: %r", provider)
        self.providers.clear()

    def _create_provider(self, config: BaseStorageProviderConfig) -> StorageProvider:
        """Create provider instance from configuration."""
        # Extract common settings from BaseStorageProviderConfig
        common_settings = {
            "log_messages": config.log_messages,
            "log_conversations": config.log_conversations,
            "log_tool_calls": config.log_tool_calls,
            "log_commands": config.log_commands,
            "log_context": config.log_context,
        }

        match config:
            case SQLStorageConfig():
                from sqlmodel import create_engine

                engine = create_engine(config.url, pool_size=config.pool_size)
                return SQLModelProvider(engine, **common_settings)

            case FileStorageConfig():
                return FileProvider(
                    config.path,
                    output_format=config.format,
                    encoding=config.encoding,
                    **common_settings,
                )

            case TextLogConfig():
                return TextLogProvider(
                    config.path,
                    template=config.template,
                    encoding=config.encoding,
                    **common_settings,
                )
            case MemoryStorageConfig():
                return MemoryStorageProvider(**common_settings)
            case _:
                msg = f"Unknown provider type: {config}"
                raise ValueError(msg)

    def _get_history_provider(self, preferred: str | None = None) -> StorageProvider:
        """Get provider for loading history.

        Args:
            preferred: Optional preferred provider name

        Returns:
            First capable provider based on priority:
            1. Preferred provider if specified and capable
            2. Default provider if specified and capable
            3. First capable provider
            4. Raises error if no capable provider found
        """

        # Function to find capable provider by name
        def find_provider(name: str) -> StorageProvider | None:
            for p in self.providers:
                if (
                    not getattr(p, "write_only", False)
                    and p.can_load_history
                    and p.__class__.__name__.lower() == name.lower()
                ):
                    return p
            return None

        # Try preferred provider
        if preferred and (provider := find_provider(preferred)):
            return provider

        # Try default provider
        if self.config.default_provider:
            if provider := find_provider(self.config.default_provider):
                return provider
            logger.warning(
                "Default provider %s not found or not capable of loading history",
                self.config.default_provider,
            )

        # Find first capable provider
        for provider in self.providers:
            if not getattr(provider, "write_only", False) and provider.can_load_history:
                return provider

        msg = "No capable provider found for loading history"
        raise RuntimeError(msg)

    async def filter_messages(
        self,
        query: SessionQuery,
        preferred_provider: str | None = None,
    ) -> list[ChatMessage[str]]:
        """Get messages matching query.

        Args:
            query: Filter criteria
            preferred_provider: Optional preferred provider to use
        """
        provider = self._get_history_provider(preferred_provider)
        return await provider.filter_messages(query)

    async def log_message(
        self,
        *,
        conversation_id: str,
        content: str,
        role: str,
        name: str | None = None,
        cost_info: TokenCost | None = None,
        model: str | None = None,
        response_time: float | None = None,
        forwarded_from: list[str] | None = None,
    ):
        """Log message to all providers."""
        if not self.config.log_messages:
            return

        for provider in self.providers:
            try:
                await provider.log_message(
                    conversation_id=conversation_id,
                    content=content,
                    role=role,
                    name=name,
                    cost_info=cost_info,
                    model=model,
                    response_time=response_time,
                    forwarded_from=forwarded_from,
                )
            except Exception:
                logger.exception("Error logging message to provider: %r", provider)

    async def log_conversation(
        self,
        *,
        conversation_id: str,
        agent_name: str,
        start_time: datetime | None = None,
    ):
        """Log conversation to all providers."""
        if not self.config.log_conversations:
            return

        for provider in self.providers:
            try:
                await provider.log_conversation(
                    conversation_id=conversation_id,
                    agent_name=agent_name,
                    start_time=start_time,
                )
            except Exception:
                logger.exception("Error logging conversation to provider: %r", provider)

    async def log_tool_call(
        self,
        *,
        conversation_id: str,
        message_id: str,
        tool_call: ToolCallInfo,
    ):
        """Log tool call to all providers."""
        if not self.config.log_tool_calls:
            return

        for provider in self.providers:
            try:
                await provider.log_tool_call(
                    conversation_id=conversation_id,
                    message_id=message_id,
                    tool_call=tool_call,
                )
            except Exception:
                logger.exception("Error logging tool call to provider: %r", provider)

    async def log_command(self, *, agent_name: str, session_id: str, command: str):
        """Log command to all providers."""
        if not self.config.log_commands:
            return

        for provider in self.providers:
            try:
                await provider.log_command(
                    agent_name=agent_name,
                    session_id=session_id,
                    command=command,
                )
            except Exception:
                logger.exception("Error logging command to provider: %r", provider)

    async def get_commands(
        self,
        agent_name: str,
        session_id: str,
        *,
        limit: int | None = None,
        current_session_only: bool = False,
        preferred_provider: str | None = None,
    ) -> list[str]:
        """Get command history."""
        if not self.config.log_commands:
            return []

        provider = self._get_history_provider(preferred_provider)
        return await provider.get_commands(
            agent_name=agent_name,
            session_id=session_id,
            limit=limit,
            current_session_only=current_session_only,
        )

    async def log_context_message(
        self,
        *,
        conversation_id: str,
        content: str,
        role: str,
        name: str | None = None,
        model: str | None = None,
    ):
        """Log context message to all providers."""
        for provider in self.providers:
            try:
                await provider.log_context_message(
                    conversation_id=conversation_id,
                    content=content,
                    role=role,
                    name=name,
                    model=model,
                )
            except Exception:
                logger.exception(
                    "Error logging context message to provider: %r", provider
                )

    def log_conversation_sync(self, **kwargs):
        """Sync wrapper for log_conversation."""
        for provider in self.providers:
            provider.log_conversation_sync(**kwargs)

    def log_tool_call_sync(self, **kwargs):
        """Sync wrapper for log_tool_call."""
        for provider in self.providers:
            provider.log_tool_call_sync(**kwargs)

    def log_command_sync(self, **kwargs):
        """Sync wrapper for log_command."""
        for provider in self.providers:
            provider.log_command_sync(**kwargs)

    def get_commands_sync(self, **kwargs) -> list[str]:
        """Sync wrapper for get_commands."""
        provider = self._get_history_provider()
        return provider.get_commands_sync(**kwargs)

    def filter_messages_sync(self, **kwargs) -> list[ChatMessage[str]]:
        """Sync wrapper for filter_messages."""
        provider = self._get_history_provider()
        return provider.filter_messages_sync(**kwargs)

    def log_context_message_sync(self, **kwargs):
        """Sync wrapper for log_context_message."""
        for provider in self.providers:
            provider.log_context_message_sync(**kwargs)
