"""Pool-level slashed interface for agent control."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Self, overload

from psygnal import Signal
from slashed import BaseCommand, CommandStore, DefaultOutputWriter, OutputWriter

from agent import Agent, AnyAgent, SlashedAgent
from agent.slashed_agent import AgentOutput
from log import get_logger
from models.messages import ChatMessage


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

    from pydantic_ai.result import StreamedRunResult
    from toprompt import AnyPromptType

    from delegation.pool import AgentPool
    from models.context import AgentContext

logger = get_logger(__name__)


@dataclass
class MultiAgentResponse[T]:
    """Collected responses from multiple agents."""

    responses: dict[str, ChatMessage[T]]
    """Individual responses from each agent."""

    start_time: datetime
    """When the multi-agent call started."""

    end_time: datetime
    """When all responses were received."""

    @property
    def duration(self) -> float:
        """Total duration of all responses."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def combined_content(self) -> str:
        """All responses formatted with agent names."""
        return "\n\n".join(
            f"{name}: {msg.content}" for name, msg in self.responses.items()
        )


class SlashedPool[TDeps]:
    """Pool-level interface that routes messages to specific or all agents.

    Provides a unified interface for both single-agent and multi-agent operations.
    Messages can be routed using:
    - @ prefix: "@agent1 hello"
    - agent parameter: run("hello", agent="agent1")
    - broadcast: run("hello") -> sends to all agents
    """

    message_output = Signal(AgentOutput)
    streamed_output = Signal(AgentOutput)
    streaming_started = Signal(str, str)  # agent_name, message_id
    streaming_stopped = Signal(str, str)  # agent_name, message_id

    agent_added = Signal(str, Agent[Any])
    agent_removed = Signal(str)
    agent_command_executed = Signal(str, str)  # agent_name, command

    def __init__(
        self,
        pool: AgentPool,
        *,
        command_history_path: str | None = None,
        output: OutputWriter | None = None,
    ):
        """Initialize pool interface.

        Args:
            pool: Agent pool to manage
            command_history_path: Optional path for command history
            output: Output writer
        """
        self.pool = pool
        self.commands = CommandStore(history_file=command_history_path)
        self._slashed_agents: dict[str, SlashedAgent[TDeps, Any]] = {}
        self._output = output

        # Create SlashedAgent wrappers for each agent
        for name, agent in pool.agents.items():
            self._add_agent(name, agent)

    def _get_output(self, output: OutputWriter | None) -> OutputWriter:
        """Get appropriate output writer."""
        return output or self._output or DefaultOutputWriter()

    def _add_agent(self, name: str, agent: AnyAgent[TDeps, Any]):
        """Add a new slashed agent with signal forwarding."""
        slashed: SlashedAgent[TDeps, Any] = SlashedAgent(agent)
        # Forward all signals with agent name context
        slashed.message_output.connect(
            lambda output: self.message_output.emit(
                dataclasses.replace(output, metadata={"agent": name, **output.metadata})
            )
        )
        slashed.streamed_output.connect(
            lambda output: self.streamed_output.emit(
                dataclasses.replace(output, metadata={"agent": name, **output.metadata})
            )
        )
        slashed.streaming_started.connect(
            lambda msg_id: self.streaming_started.emit(name, msg_id)
        )
        slashed.streaming_stopped.connect(
            lambda msg_id: self.streaming_stopped.emit(name, msg_id)
        )
        self._slashed_agents[name] = slashed

    @overload
    async def run(
        self,
        *prompt: AnyPromptType,
        agent: str,  # Specific agent
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage[str]: ...

    @overload
    async def run(
        self,
        *prompt: AnyPromptType,
        agent: None = None,  # Broadcasting
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MultiAgentResponse[str]: ...

    async def run(
        self,
        *prompt: AnyPromptType,
        agent: str | None = None,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage[str] | MultiAgentResponse[str]:
        """Run with agent routing and wait for response(s).

        Args:
            *prompt: Prompts to send
            agent: Optional specific agent to target
            output: Optional output writer
            metadata: Additional metadata for messages

        Returns:
            - Single agent: ChatMessage with response
            - Multiple agents: MultiAgentResponse with all responses

        Raises:
            ValueError: If agent not found or @ syntax is invalid
        """
        writer = self._get_output(output)
        target = agent
        remaining_prompts: list[Any] = []

        # Handle @ routing first
        for p in prompt:
            if isinstance(p, str) and p.startswith("@"):
                parts = p[1:].split(maxsplit=1)
                if len(parts) != 2:  # noqa: PLR2004
                    msg = "Usage: @agent_name message"
                    raise ValueError(msg)
                target, message = parts
                remaining_prompts.append(message)
            else:
                remaining_prompts.append(p)

        # Handle commands
        non_command_prompts = []
        for p in remaining_prompts:
            if isinstance(p, str) and p.startswith("/"):
                await self.execute_command(p[1:], output=writer, metadata=metadata)
            else:
                non_command_prompts.append(p)

        if not non_command_prompts:
            return ChatMessage(content="", role="system")

        # Single agent case
        if target:
            if target not in self._slashed_agents:
                msg = f"Agent {target} not found"
                raise ValueError(msg)
            return await self._slashed_agents[target].run(
                *non_command_prompts,
                output=writer,
                metadata=metadata,
            )

        # Multi-agent case
        start_time = datetime.now()
        tasks = [
            agent.run(
                *non_command_prompts,
                output=writer,
                metadata=metadata,
            )
            for name, agent in self._slashed_agents.items()
        ]
        results = await asyncio.gather(*tasks)
        responses = dict(zip(self._slashed_agents.keys(), results))
        return MultiAgentResponse(
            responses, start_time=start_time, end_time=datetime.now()
        )

    @asynccontextmanager
    async def run_stream(
        self,
        *prompt: AnyPromptType,
        agent: str,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamedRunResult[AgentContext[TDeps], str]]:
        """Stream responses from a specific agent.

        Args:
            *prompt: Prompts to send
            agent: Name of agent to stream from (required)
            output: Optional output writer
            metadata: Additional metadata for messages

        Yields:
            Stream of results from pydantic-ai

        Raises:
            ValueError: If agent not found or no prompts remain after command handling
        """
        writer = self._get_output(output)
        remaining_prompts = []

        # Handle @ routing first - not needed for streaming as agent is required
        if any(isinstance(p, str) and p.startswith("@") for p in prompt):
            msg = "@ routing not supported for streaming - use agent parameter"
            raise ValueError(msg)

        # Handle commands
        for p in prompt:
            if isinstance(p, str) and p.startswith("/"):
                await self.execute_command(p[1:], output=writer, metadata=metadata)
            else:
                remaining_prompts.append(p)

        if not remaining_prompts:
            msg = "No prompts remaining after command handling"
            raise ValueError(msg)

        if agent not in self._slashed_agents:
            msg = f"Agent {agent} not found"
            raise ValueError(msg)

        async with self._slashed_agents[agent].run_stream(
            *remaining_prompts,
            output=writer,
            metadata=metadata,
        ) as stream:
            yield stream

    async def run_iter(
        self,
        *prompt: AnyPromptType,
        agent: str | None = None,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatMessage[str]]:
        """Run and yield responses as they complete.

        Args:
            *prompt: Prompts to send
            agent: Optional specific agent to target
            output: Optional output writer
            metadata: Additional metadata for messages

        Yields:
            ChatMessages as they complete

        Raises:
            ValueError: If agent not found or @ syntax is invalid
        """
        writer = self._get_output(output)
        target = agent
        remaining_prompts: list[Any] = []

        # Handle @ routing first
        for p in prompt:
            if isinstance(p, str) and p.startswith("@"):
                parts = p[1:].split(maxsplit=1)
                if len(parts) != 2:  # noqa: PLR2004
                    msg = "Usage: @agent_name message"
                    raise ValueError(msg)
                target, message = parts
                remaining_prompts.append(message)
            else:
                remaining_prompts.append(p)

        # Handle commands
        non_command_prompts = []
        for p in remaining_prompts:
            if isinstance(p, str) and p.startswith("/"):
                await self.execute_command(p[1:], output=writer, metadata=metadata)
            else:
                non_command_prompts.append(p)

        if not non_command_prompts:
            yield ChatMessage(content="", role="system")
            return

        # Single agent case
        if target:
            if target not in self._slashed_agents:
                msg = f"Agent {target} not found"
                raise ValueError(msg)
            response = await self._slashed_agents[target].run(
                *non_command_prompts,
                output=writer,
                metadata=metadata,
            )
            yield response
            return

        # Multi-agent case
        for slashed_agent in self._slashed_agents.values():
            response = await slashed_agent.run(
                *non_command_prompts,
                output=writer,
                metadata=metadata,
            )
            yield response

    async def execute_command(
        self,
        command: str,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Execute command with optional agent routing.

        Args:
            command: Command to execute (can start with @agent_name)
            output: Optional output writer
            metadata: Additional metadata

        Returns:
            Command result message
        """
        output = self._get_output(output)
        ctx = self.commands.create_context(self, output_writer=output, metadata=metadata)

        # Handle agent-specific commands
        if command.startswith("@"):
            parts = command[1:].split(maxsplit=1)
            if len(parts) != 2:  # noqa: PLR2004
                return "Usage: @agent_name /command"

            agent_name, agent_command = parts
            if agent_name not in self._slashed_agents:
                return f"Agent {agent_name} not found"

            # Forward to specific agent
            slashed_agent = self._slashed_agents[agent_name]
            result = await slashed_agent.handle_command(
                agent_command,
                output=output,
                metadata=metadata,
            )
            self.agent_command_executed.emit(agent_name, agent_command)
            return str(result.content)

        # Pool-level command
        await self.commands.execute_command(command, ctx)
        return "finished"

    def register_pool_command(self, command: BaseCommand):
        """Register a pool-level command."""
        self.commands.register_command(command)

    @property
    def agents(self) -> dict[str, SlashedAgent[TDeps, Any]]:
        """Access to all slashed agents."""
        return self._slashed_agents.copy()

    async def __aenter__(self) -> Self:
        """Enter async context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        """Exit async context."""
        # Cleanup will be handled by pool
