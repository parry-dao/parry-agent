from __future__ import annotations

from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, overload

from psygnal import Signal
from slashed import CommandStore, DefaultOutputWriter, ExitCommandError, OutputWriter
from typing_extensions import TypeVar

from log import get_logger
from models.messages import ChatMessage


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.result import StreamedRunResult
    from slashed.events import CommandExecutedEvent
    from toprompt import AnyPromptType

    from agent import AnyAgent
    from agent.conversation import ConversationManager
    from common_types import ModelType
    from delegation.pool import AgentPool
    from models.agents import ToolCallInfo
    from models.context import AgentContext
    from tools.manager import ToolManager
    from parry_agent_providers.base import AgentProvider


logger = get_logger(__name__)

TContext = TypeVar("TContext")
TDeps = TypeVar("TDeps")
TResult = TypeVar("TResult", default=str)


OutputType = Literal[
    "message_received",
    "message_sent",
    "tool_called",
    "tool_result",
    "command_executed",
    "status",
    "error",
    "stream",
]


@dataclass
class AgentOutput:
    """Unified output event for UIs."""

    # TODO: replace with Response, also in Docs

    type: OutputType
    """Type of output event."""

    content: Any
    """Main content of the output."""

    timestamp: datetime = field(default_factory=datetime.now)
    """When this output was generated."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional context about the output."""


class SlashedAgent[TDeps, TContext]:
    """Wraps an agent with slash command support."""

    message_output = Signal(AgentOutput)
    streamed_output = Signal(AgentOutput)
    streaming_started = Signal(str)  # message_id
    streaming_stopped = Signal(str)  # message_id

    def __init__(
        self,
        agent: AnyAgent[TDeps, Any],
        *,
        command_context: TContext | None = None,
        command_history_path: str | None = None,
        output: DefaultOutputWriter | None = None,
    ):
        self.agent = agent
        assert self.agent.context, "Agent must have a context!"
        assert self.agent.context.pool, "Agent must have a pool!"

        self.commands = CommandStore(
            history_file=command_history_path,
            enable_system_commands=True,
        )
        self.commands._initialize_sync()
        self._current_stream_id: str | None = None
        self.command_context: TContext = command_context or self  # type: ignore
        self.output = output or DefaultOutputWriter()
        # Connect to agent's signals
        agent.message_received.connect(self._handle_message_received)
        agent.message_sent.connect(self._handle_message_sent)
        agent.tool_used.connect(self._handle_tool_used)
        self.commands.command_executed.connect(self._handle_command_executed)
        self.commands.output.connect(self.streamed_output)
        agent.chunk_streamed.connect(self.streamed_output)

    @overload
    async def run[TMethodResult](
        self,
        *prompt: AnyPromptType,
        result_type: type[TMethodResult],
        deps: TDeps | None = None,
        model: ModelType = None,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage[TMethodResult]: ...

    @overload
    async def run(
        self,
        *prompt: AnyPromptType,
        result_type: None = None,
        deps: TDeps | None = None,
        model: ModelType = None,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage[str]: ...

    async def run(
        self,
        *prompt: AnyPromptType,
        result_type: type[Any] | None = None,
        deps: TDeps | None = None,
        model: ModelType = None,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage[Any]:
        """Run with slash command support."""
        # First execute all commands sequentially
        remaining_prompts = []
        for p in prompt:
            if isinstance(p, str) and p.startswith("/"):
                await self.handle_command(
                    p[1:],
                    output=output or self.output,
                    metadata=metadata,
                )
            else:
                remaining_prompts.append(p)

        # Then pass remaining prompts to agent
        return await self.agent.run(
            *remaining_prompts, result_type=result_type, deps=deps, model=model
        )

    @overload
    def run_stream[TMethodResult](
        self,
        *prompt: AnyPromptType,
        result_type: type[TMethodResult],
        deps: TDeps | None = None,
        model: ModelType = None,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AbstractAsyncContextManager[
        StreamedRunResult[AgentContext[TDeps], TMethodResult]
    ]: ...

    @overload
    def run_stream(
        self,
        *prompt: AnyPromptType,
        result_type: None = None,
        deps: TDeps | None = None,
        model: ModelType = None,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AbstractAsyncContextManager[StreamedRunResult[AgentContext[TDeps], str]]: ...

    @asynccontextmanager
    async def run_stream(
        self,
        *prompt: AnyPromptType,
        result_type: type[Any] | None = None,
        deps: TDeps | None = None,
        model: ModelType = None,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamedRunResult[AgentContext[TDeps], Any]]:
        """Stream responses with slash command support."""
        # First execute all commands sequentially
        remaining_prompts: list[AnyPromptType] = []
        for p in prompt:
            if isinstance(p, str) and p.startswith("/"):
                await self.handle_command(
                    p[1:],
                    output=output or self.output,
                    metadata=metadata,
                )
            else:
                remaining_prompts.append(p)

        # Then yield from agent's stream
        async with self.agent.run_stream(
            *remaining_prompts, result_type=result_type, deps=deps, model=model
        ) as stream:
            yield stream

    async def handle_command(
        self,
        command: str,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage[str]:
        """Handle a slash command."""
        try:
            await self.commands.execute_command_with_context(
                command,
                context=self.command_context,
                output_writer=output or self.output,
                metadata=metadata,
            )
            return ChatMessage(content="", role="system")
        except ExitCommandError:
            raise
        except Exception as e:  # noqa: BLE001
            msg = f"Command error: {e}"
            return ChatMessage(content=msg, role="system")

    @property
    def tools(self) -> ToolManager:
        """Access to tool management."""
        return self.agent.tools

    @property
    def conversation(self) -> ConversationManager:
        """Access to conversation management."""
        return self.agent.conversation

    @property
    def provider(self) -> AgentProvider[TDeps]:
        """Access to the underlying provider."""
        return self.agent._provider

    @property
    def pool(self) -> AgentPool:
        """Get agent's pool from context."""
        assert self.agent.context.pool
        return self.agent.context.pool

    @property
    def model_name(self) -> str | None:
        """Get current model name."""
        return self.agent.model_name

    @property
    def context(self) -> AgentContext[TDeps]:
        """Access to agent context."""
        return self.agent.context

    def _handle_message_received(self, message: ChatMessage[str]):
        meta = {"role": message.role}
        output = AgentOutput(type="message_received", content=message, metadata=meta)
        self.message_output.emit(output)

    def _handle_message_sent(self, message: ChatMessage[Any]):
        cost = message.cost_info.total_cost if message.cost_info else None
        metadata = {"role": message.role, "model": message.model, "cost": cost}
        output = AgentOutput(type="message_sent", content=message, metadata=metadata)
        self.message_output.emit(output)

    def _handle_tool_used(self, tool_call: ToolCallInfo):
        metadata = {"tool_name": tool_call.tool_name}
        output = AgentOutput("tool_called", content=tool_call, metadata=metadata)
        self.message_output.emit(output)

    # Could also connect to Slashed's command signals
    def _handle_command_executed(self, event: CommandExecutedEvent):
        """Handle command execution events."""
        error = str(event.error) if event.error else None
        meta = {"success": event.success, "error": error, "context": event.context}
        output = AgentOutput("command_executed", content=event.command, metadata=meta)
        self.message_output.emit(output)

    def _handle_chunk_streamed(self, chunk: str, message_id: str):
        """Handle streaming chunks."""
        # If this is a new streaming session
        if message_id != self._current_stream_id:
            self._current_stream_id = message_id
            self.streaming_started.emit(message_id)

        if chunk:  # Only emit non-empty chunks
            output = AgentOutput(
                type="stream", content=chunk, metadata={"message_id": message_id}
            )
            self.streamed_output.emit(output)
        else:  # Empty chunk signals end of stream
            self.streaming_stopped.emit(message_id)
            self._current_stream_id = None
