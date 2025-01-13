"""Core chat session implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import pathlib
import time
from typing import TYPE_CHECKING, Any, Literal, overload
from uuid import uuid4

from platformdirs import user_data_dir
from psygnal import Signal
from slashed import CommandError, CommandStore, DefaultOutputWriter, ExitCommandError

from agent import Agent, AnyAgent
from agent.conversation import ConversationManager
from chat_session.exceptions import ChatSessionConfigError
from chat_session.models import SessionState
from log import get_logger
from models.messages import ChatMessage, TokenCost
from tools.base import ToolInfo
from parry_agent_commands import get_commands


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from slashed import OutputWriter

    from delegation.pool import AgentPool
    from tools.manager import ToolManager


logger = get_logger(__name__)
HISTORY_DIR = pathlib.Path(user_data_dir("parry", "parry")) / "cli_history"


class AgentPoolView:
    """User's view and control point for interacting with an agent in a pool.

    This class provides a focused way to interact with one primary agent that is part
    of a larger agent pool. Through this view, users can:
    1. Interact with the primary agent directly
    2. Manage connections to other agents in the pool
    3. Control tool availability and settings
    4. Handle commands and responses

    Think of it as looking at the agent pool through the lens of one specific agent,
    while still being able to utilize the pool's collaborative capabilities.
    """

    @dataclass(frozen=True)
    class SessionReset:
        """Emitted when session is reset."""

        session_id: str
        previous_tools: dict[str, bool]
        new_tools: dict[str, bool]
        timestamp: datetime = field(default_factory=datetime.now)

    history_cleared = Signal(ConversationManager.HistoryCleared)
    session_reset = Signal(SessionReset)
    tool_added = Signal(str, ToolInfo)
    tool_removed = Signal(str)  # tool_name
    tool_changed = Signal(str, ToolInfo)  # name, new_info
    agent_connected = Signal(Agent)

    def __init__(
        self,
        agent: AnyAgent[Any, Any],
        *,
        pool: AgentPool | None = None,
        wait_chain: bool = True,
    ):
        """Initialize chat session.

        Args:
            agent: The parry agent to use
            pool: Optional agent pool for multi-agent interactions
            wait_chain: Whether to wait for chain completion
        """
        # Basic setup that doesn't need async
        self._agent = agent
        self._pool = pool
        self.wait_chain = wait_chain
        # forward ToolManager signals to ours
        self._agent.tools.events.added.connect(self.tool_added.emit)
        self._agent.tools.events.removed.connect(self.tool_removed.emit)
        self._agent.tools.events.changed.connect(self.tool_changed.emit)
        self._agent.conversation.history_cleared.connect(self.history_cleared.emit)
        self._initialized = False  # Track initialization state
        file_path = HISTORY_DIR / f"{agent.name}.history"
        self.commands = CommandStore(history_file=file_path, enable_system_commands=True)
        self.start_time = datetime.now()
        self._state = SessionState(current_model=self._agent.model_name)

    @classmethod
    async def create(
        cls,
        agent: Agent[Any],
        *,
        pool: AgentPool | None = None,
        wait_chain: bool = True,
    ) -> AgentPoolView:
        """Create and initialize a new agent pool view.

        Args:
            agent: The primary agent to interact with
            pool: Optional agent pool for multi-agent interactions
            wait_chain: Whether to wait for chain completion

        Returns:
            Initialized AgentPoolView
        """
        view = cls(agent, pool=pool, wait_chain=wait_chain)
        await view.initialize()
        return view

    @property
    def pool(self) -> AgentPool | None:
        """Get the agent pool if available."""
        return self._pool

    async def connect_to(self, target: str, wait: bool | None = None):
        """Connect to another agent.

        Args:
            target: Name of target agent
            wait: Override session's wait_chain setting

        Raises:
            ValueError: If target agent not found or pool not available
        """
        logger.debug("Connecting to %s (wait=%s)", target, wait)
        if not self._pool:
            msg = "No agent pool available"
            raise ValueError(msg)

        try:
            target_agent = self._pool.get_agent(target)
        except KeyError as e:
            msg = f"Target agent not found: {target}"
            raise ValueError(msg) from e

        self._agent.pass_results_to(target_agent)
        self.agent_connected.emit(target_agent)

        if wait is not None:
            self.wait_chain = wait

    def _ensure_initialized(self):
        """Check if session is initialized."""
        if not self._initialized:
            msg = "Session not initialized. Call initialize() first."
            raise RuntimeError(msg)

    async def initialize(self):
        """Initialize async resources and load data."""
        if self._initialized:
            return

        # Load command history
        await self.commands.initialize()
        for cmd in get_commands():
            self.commands.register_command(cmd)

        self._initialized = True
        logger.debug("Initialized chat session for agent %r", self._agent.name)

    async def cleanup(self):
        """Clean up session resources."""
        if self._pool:
            await self._agent.disconnect_all()

    def add_command(self, command: str):
        """Add command to history."""
        if not command.strip():
            return
        from storage.models import CommandHistory

        id_ = str(self._agent.conversation.id)
        CommandHistory.log(agent_name=self._agent.name, session_id=id_, command=command)

    def get_commands(
        self, limit: int | None = None, current_session_only: bool = False
    ) -> list[str]:
        """Get command history ordered by newest first."""
        from storage.models import CommandHistory

        return CommandHistory.get_commands(
            agent_name=self._agent.name,
            session_id=str(self._agent.conversation.id),
            limit=limit,
            current_session_only=current_session_only,
        )

    async def clear(self):
        """Clear chat history."""
        self._agent.conversation.clear()

    async def reset(self):
        """Reset session state."""
        old_tools = self.tools.list_tools()
        self._agent.conversation.clear()
        self.tools.reset_states()
        new_tools = self.tools.list_tools()

        event = self.SessionReset(
            session_id=str(self._agent.conversation.id),
            previous_tools=old_tools,
            new_tools=new_tools,
        )
        self.session_reset.emit(event)

    async def handle_command(
        self,
        command_str: str,
        output: OutputWriter,
        metadata: dict[str, Any] | None = None,
    ):
        """Handle a slash command.

        Args:
            command_str: Command string without leading slash
            output: Output writer implementation
            metadata: Optional interface-specific metadata
        """
        self._ensure_initialized()
        meta = metadata or {}
        ctx = self.commands.create_context(self, output_writer=output, metadata=meta)
        await self.commands.execute_command(command_str, ctx)

    async def send_slash_command(
        self,
        content: str,
        *,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage[str]:
        writer = output or DefaultOutputWriter()
        try:
            await self.handle_command(content[1:], output=writer, metadata=metadata)
            return ChatMessage(content="", role="system")
        except ExitCommandError:
            # Re-raise without wrapping in CommandError
            raise
        except CommandError as e:
            return ChatMessage(content=f"Command error: {e}", role="system")

    @overload
    async def send_message(
        self,
        content: str,
        *,
        stream: Literal[False] = False,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage[str]: ...

    @overload
    async def send_message(
        self,
        content: str,
        *,
        stream: Literal[True],
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatMessage[str]]: ...

    async def send_message(
        self,
        content: str,
        *,
        stream: bool = False,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage[str] | AsyncIterator[ChatMessage[str]]:
        """Send a message and get response(s)."""
        self._ensure_initialized()
        if not content.strip():
            msg = "Message cannot be empty"
            raise ValueError(msg)

        if content.startswith("/"):
            return await self.send_slash_command(
                content,
                output=output,
                metadata=metadata,
            )
        try:
            if stream:
                return self._stream_message(content)
            return await self._send_normal(content)

        except Exception as e:
            logger.exception("Error processing message")
            msg = f"Error processing message: {e}"
            raise ChatSessionConfigError(msg) from e

    async def _send_normal(self, content: str) -> ChatMessage[str]:
        """Send message and get single response."""
        result = await self._agent.run(content)
        text_message = result.to_text_message()

        # Update session state metrics
        self._state.message_count += 2  # User and assistant messages
        if text_message.cost_info:
            self._state.update_tokens(text_message)
            self._state.total_cost = float(text_message.cost_info.total_cost)
        if text_message.response_time:
            self._state.last_response_time = text_message.response_time

        # Add chain waiting if enabled
        if self.wait_chain and self._pool:
            await self._agent.wait_for_chain()

        return text_message

    async def _stream_message(self, content: str) -> AsyncIterator[ChatMessage[str]]:
        """Send message and stream responses."""
        async with self._agent.run_stream(content) as stream_result:
            # Stream intermediate chunks
            async for response in stream_result.stream():
                yield ChatMessage[str](content=str(response), role="assistant")

            # Final message with complete metrics after stream completes
            start_time = time.perf_counter()

            # Get usage info if available
            usage = stream_result.usage()
            cost_info = (
                await TokenCost.from_usage(
                    usage, self._agent.model_name, content, response
                )
                if usage and self._agent.model_name
                else None
            )

            # Create final status message with all metrics
            final_msg = ChatMessage[str](
                content="",  # Empty content for final status message
                role="assistant",
                name=self._agent.name,
                model=self._agent.model_name,
                message_id=str(uuid4()),
                cost_info=cost_info,
                response_time=time.perf_counter() - start_time,
            )

            # Update session state
            self._state.message_count += 2  # User and assistant messages
            self._state.update_tokens(final_msg)

            # Add chain waiting if enabled
            if self.wait_chain and self._pool:
                await self._agent.wait_for_chain()

            yield final_msg

    @property
    def tools(self) -> ToolManager:
        """Get current tool states."""
        return self._agent.tools
