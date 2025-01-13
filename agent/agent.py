"""The main Agent. Can do all sort of crazy things."""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from os import PathLike
import time
from typing import TYPE_CHECKING, Any, Literal, Self, cast, overload
from uuid import uuid4

from parry import (
    Config,
    DynamicPrompt,
    LLMCallableTool,
    RuntimeConfig,
    StaticPrompt,
    ToolError,
)
from .prompts.models import FilePrompt
from parry.utils.importing import import_callable
import logfire
from psygnal import Signal
from pydantic_ai import RunContext  # noqa: TC002
from tokonomics import TokenLimits, get_model_limits
from toprompt import AnyPromptType, to_prompt
from typing_extensions import TypeVar

from agent.connection import Talk, TalkManager, TeamTalk
from agent.conversation import ConversationManager
from log import get_logger
from models import AgentContext, AgentsManifest
from models.agents import ToolCallInfo
from models.mcp_server import MCPServerConfig, StdioMCPServer
from models.messages import ChatMessage, TokenCost
from responses.utils import to_type
from tools.manager import ToolManager
from utils.inspection import call_with_context
from utils.tasks import TaskManagerMixin
from parry_agent_providers import AgentProvider, HumanProvider, PydanticAIProvider


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from datetime import timedelta
    from types import TracebackType

    from parry.config.models import Resource
    from pydantic_ai.agent import EndStrategy
    from pydantic_ai.result import StreamedRunResult

    from agent import AnyAgent
    from agent.structured import StructuredAgent
    from agent.talk import Interactions
    from common_types import ModelType, SessionIdType, StrPath, ToolType
    from delegation.agentgroup import Team
    from models.context import ConfirmationCallback
    from models.forward_targets import ConnectionType
    from models.session import SessionQuery
    from models.task import AgentTask
    from responses.models import ResponseDefinition
    from tools.base import ToolInfo


logger = get_logger(__name__)

TResult = TypeVar("TResult", default=str)
TDeps = TypeVar("TDeps", default=Any)
AgentType = Literal["ai", "human", "litellm"] | AgentProvider
JINJA_PROC = "jinja_template"  # Name of builtin parry Jinja2 processor


class Agent[TDeps](TaskManagerMixin):
    """Agent for AI-powered interaction with parry resources and tools.

    Generically typed with: parryAgent[Type of Dependencies, Type of Result]

    This agent integrates parry's resource system with PydanticAI's agent capabilities.
    It provides:
    - Access to resources through RuntimeConfig
    - Tool registration for resource operations
    - System prompt customization
    - Signals
    - Message history management
    - Database logging
    """

    # this fixes weird mypy issue
    conversation: ConversationManager
    connections: TalkManager
    talk: Interactions
    description: str | None

    message_received = Signal(ChatMessage[str])  # Always string
    message_sent = Signal(ChatMessage)
    tool_used = Signal(ToolCallInfo)
    model_changed = Signal(object)  # Model | None
    chunk_streamed = Signal(str, str)  # (chunk, message_id)
    outbox = Signal(ChatMessage[Any], str)  # message, prompt

    def __init__(
        self,
        runtime: RuntimeConfig | Config | StrPath | None = None,
        *,
        context: AgentContext[TDeps] | None = None,
        agent_type: AgentType = "ai",
        session: SessionIdType | SessionQuery = None,
        model: ModelType = None,
        system_prompt: str | Sequence[str] = (),
        name: str = "parry-agent",
        description: str | None = None,
        tools: Sequence[ToolType] | None = None,
        mcp_servers: list[str | MCPServerConfig] | None = None,
        retries: int = 1,
        result_retries: int | None = None,
        tool_choice: bool | str | list[str] = True,
        end_strategy: EndStrategy = "early",
        defer_model_check: bool = False,
        enable_db_logging: bool = True,
        confirmation_callback: ConfirmationCallback | None = None,
        debug: bool = False,
        **kwargs,
    ):
        """Initialize agent with runtime configuration.

        Args:
            runtime: Runtime configuration providing access to resources/tools
            context: Agent context with capabilities and configuration
            agent_type: Agent type to use (ai: PydanticAIProvider, human: HumanProvider)
            session: Optional id or Session query to recover a conversation
            model: The default model to use (defaults to GPT-4)
            system_prompt: Static system prompts to use for this agent
            name: Name of the agent for logging
            description: Description of the Agent ("what it can do")
            tools: List of tools to register with the agent
            mcp_servers: MCP servers to connect to
            retries: Default number of retries for failed operations
            result_retries: Max retries for result validation (defaults to retries)
            tool_choice: Ability to set a fixed tool or temporarily disable tools usage.
            end_strategy: Strategy for handling tool calls that are requested alongside
                          a final result
            defer_model_check: Whether to defer model evaluation until first run
            kwargs: Additional arguments for PydanticAI agent
            enable_db_logging: Whether to enable logging for the agent
            confirmation_callback: Callback for confirmation prompts
            debug: Whether to enable debug mode
        """
        super().__init__()
        self._debug = debug
        self._result_type = None

        # save some stuff for asnyc init
        self._owns_runtime = False
        self._mcp_servers = [
            StdioMCPServer(command=s.split()[0], args=s.split()[1:])
            if isinstance(s, str)
            else s
            for s in (mcp_servers or [])
        ]

        # prepare context
        ctx = context or AgentContext[TDeps].create_default(name)
        ctx.confirmation_callback = confirmation_callback
        match runtime:
            case None:
                ctx.runtime = RuntimeConfig.from_config(Config())
            case Config():
                ctx.runtime = RuntimeConfig.from_config(runtime)
            case str() | PathLike():
                ctx.runtime = RuntimeConfig.from_config(Config.from_file(runtime))
            case RuntimeConfig():
                ctx.runtime = runtime
        # connect signals
        self.message_sent.connect(self._forward_message)

        # Initialize tool manager
        all_tools = list(tools or [])
        self._tool_manager = ToolManager(all_tools, tool_choice=tool_choice, context=ctx)

        # set up conversation manager
        config_prompts = ctx.config.system_prompts if ctx else []
        all_prompts = list(config_prompts)
        if isinstance(system_prompt, str):
            all_prompts.append(system_prompt)
        else:
            all_prompts.extend(system_prompt)
        self.conversation = ConversationManager(self, session, all_prompts)

        # Initialize provider based on type
        match agent_type:
            case "ai":
                if model and not isinstance(model, str):
                    from pydantic_ai import models

                    assert isinstance(model, models.Model)
                self._provider: AgentProvider = PydanticAIProvider(
                    model=model,  # pyright: ignore
                    system_prompt=system_prompt,
                    retries=retries,
                    end_strategy=end_strategy,
                    result_retries=result_retries,
                    defer_model_check=defer_model_check,
                    debug=debug,
                    **kwargs,
                )
            case "human":
                self._provider = HumanProvider(name=name, debug=debug)
            case "litellm":
                from parry_agent_providers.litellm_provider import LiteLLMProvider

                self._provider = LiteLLMProvider(name=name, debug=debug, retries=retries)
            case AgentProvider():
                self._provider = agent_type
            case _:
                msg = f"Invalid agent type: {type}"
                raise ValueError(msg)
        self._provider.tool_manager = self._tool_manager
        self._provider.context = ctx
        self._provider.conversation = self.conversation
        ctx.capabilities.register_capability_tools(self)

        # Forward provider signals
        self._provider.chunk_streamed.connect(self.chunk_streamed.emit)
        self._provider.model_changed.connect(self.model_changed.emit)
        self._provider.tool_used.connect(self.tool_used.emit)
        self._provider.model_changed.connect(self.model_changed.emit)

        self.name = name
        self.description = description
        msg = "Initialized %s (model=%s)"
        logger.debug(msg, self.name, model)

        from agent import AgentLogger
        from agent.talk import Interactions
        from events import EventManager

        self.connections = TalkManager(self)
        self.talk = Interactions(self)

        self._logger = AgentLogger(self, enable_db_logging=enable_db_logging)
        self._events = EventManager(self, enable_events=True)

        self._background_task: asyncio.Task[Any] | None = None

    def __repr__(self) -> str:
        desc = f", {self.description!r}" if self.description else ""
        tools = f", tools={len(self.tools)}" if self.tools else ""
        return f"Agent({self._provider!r}{desc}{tools})"

    def __prompt__(self) -> str:
        parts = [
            f"Agent: {self.name}",
            f"Type: {self._provider.__class__.__name__}",
            f"Model: {self.model_name or 'default'}",
        ]
        if self.description:
            parts.append(f"Description: {self.description}")
        parts.extend([self.tools.__prompt__(), self.conversation.__prompt__()])

        return "\n".join(parts)

    async def __aenter__(self) -> Self:
        """Enter async context and set up MCP servers."""
        try:
            # First initialize runtime
            runtime_ref = self.context.runtime
            if runtime_ref and not runtime_ref._initialized:
                self._owns_runtime = True
                await runtime_ref.__aenter__()
                runtime_tools = runtime_ref.tools.values()
                logger.debug(
                    "Registering runtime tools: %s", [t.name for t in runtime_tools]
                )
                for tool in runtime_tools:
                    self.tools.register_tool(tool, source="runtime")

            # Then setup constructor MCP servers
            if self._mcp_servers:
                await self.tools.setup_mcp_servers(self._mcp_servers)

            # Then setup config MCP servers if any
            if self.context and self.context.config and self.context.config.mcp_servers:
                await self.tools.setup_mcp_servers(self.context.config.get_mcp_servers())
        except Exception as e:
            # Clean up in reverse order
            if self._owns_runtime and runtime_ref and self.context.runtime == runtime_ref:
                await runtime_ref.__aexit__(type(e), e, e.__traceback__)
            msg = "Failed to initialize agent"
            raise RuntimeError(msg) from e
        else:
            return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        """Exit async context."""
        try:
            await self.tools.cleanup()
        finally:
            if self._owns_runtime and self.context.runtime:
                await self.context.runtime.__aexit__(exc_type, exc_val, exc_tb)

    @overload
    def __rshift__(self, other: AnyAgent[Any, Any] | str) -> Talk: ...

    @overload
    def __rshift__(self, other: Team[Any]) -> TeamTalk: ...

    def __rshift__(self, other: AnyAgent[Any, Any] | Team[Any] | str) -> Talk | TeamTalk:
        """Connect agent to another agent or group.

        Example:
            agent >> other_agent  # Connect to single agent
            agent >> (agent2 | agent3)  # Connect to group
            agent >> "other_agent"  # Connect by name (needs pool)
        """
        return self.pass_results_to(other)

    def __or__(self, other: AnyAgent[Any, Any] | Team[Any]) -> Team[TDeps]:
        """Create agent group using | operator.

        Example:
            group = analyzer | planner | executor  # Create group of 3
            group = analyzer | existing_group  # Add to existing group
        """
        from delegation.agentgroup import Team

        if isinstance(other, Team):
            return Team([self, *other.agents])
        return Team([self, other])

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._provider.name or "parry-agent"

    @name.setter
    def name(self, value: str):
        self._provider.name = value

    @property
    def context(self) -> AgentContext[TDeps]:
        """Get agent context."""
        return self._provider.context

    @context.setter
    def context(self, value: AgentContext[TDeps]):
        """Set agent context and propagate to provider."""
        self._provider.context = value
        self._tool_manager.context = value

    def set_result_type(
        self,
        result_type: type[TResult] | str | ResponseDefinition | None,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ):
        """Set or update the result type for this agent.

        Args:
            result_type: New result type, can be:
                - A Python type for validation
                - Name of a response definition
                - Response definition instance
                - None to reset to unstructured mode
            tool_name: Optional override for tool name
            tool_description: Optional override for tool description
        """
        logger.debug("Setting result type to: %s", result_type)
        self._result_type = to_type(result_type)  # to_type?

    @overload
    def to_structured(
        self,
        result_type: None,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> Self: ...

    @overload
    def to_structured[TResult](
        self,
        result_type: type[TResult] | str | ResponseDefinition,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> StructuredAgent[TDeps, TResult]: ...

    def to_structured[TResult](
        self,
        result_type: type[TResult] | str | ResponseDefinition | None,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> StructuredAgent[TDeps, TResult] | Self:
        """Convert this agent to a structured agent.

        If result_type is None, returns self unchanged (no wrapping).
        Otherwise creates a StructuredAgent wrapper.

        Args:
            result_type: Type for structured responses. Can be:
                - A Python type (Pydantic model)
                - Name of response definition from context
                - Complete response definition
                - None to skip wrapping
            tool_name: Optional override for result tool name
            tool_description: Optional override for result tool description

        Returns:
            Either StructuredAgent wrapper or self unchanged
        from agent import StructuredAgent
        """
        if result_type is None:
            return self

        from agent import StructuredAgent

        return StructuredAgent(
            self,
            result_type=result_type,
            tool_name=tool_name,
            tool_description=tool_description,
        )

    @classmethod
    @overload
    def open(
        cls,
        config_path: StrPath | Config | None = None,
        *,
        result_type: None = None,
        model: ModelType = None,
        session: SessionIdType | SessionQuery = None,
        system_prompt: str | Sequence[str] = (),
        name: str = "parry-agent",
        retries: int = 1,
        result_retries: int | None = None,
        end_strategy: EndStrategy = "early",
        defer_model_check: bool = False,
        **kwargs: Any,
    ) -> AbstractAsyncContextManager[Agent[TDeps]]: ...

    @classmethod
    @overload
    def open[TResult](
        cls,
        config_path: StrPath | Config | None = None,
        *,
        result_type: type[TResult],
        model: ModelType = None,
        session: SessionIdType | SessionQuery = None,
        system_prompt: str | Sequence[str] = (),
        name: str = "parry-agent",
        retries: int = 1,
        result_retries: int | None = None,
        end_strategy: EndStrategy = "early",
        defer_model_check: bool = False,
        **kwargs: Any,
    ) -> AbstractAsyncContextManager[StructuredAgent[TDeps, TResult]]: ...

    @classmethod
    @asynccontextmanager
    async def open[TResult](
        cls,
        config_path: StrPath | Config | None = None,
        *,
        result_type: type[TResult] | None = None,
        model: ModelType = None,
        session: SessionIdType | SessionQuery = None,
        system_prompt: str | Sequence[str] = (),
        name: str = "parry-agent",
        retries: int = 1,
        result_retries: int | None = None,
        end_strategy: EndStrategy = "early",
        defer_model_check: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[Agent[TDeps] | StructuredAgent[TDeps, TResult]]:
        """Open and configure an agent with an auto-managed runtime configuration.

        This is a convenience method that combines RuntimeConfig.open with agent creation.

        Args:
            config_path: Path to the runtime configuration file or a Config instance
                        (defaults to Config())
            result_type: Optional type for structured responses
            model: The default model to use (defaults to GPT-4)
            session: Optional id or Session query to recover a conversation
            system_prompt: Static system prompts to use for this agent
            name: Name of the agent for logging
            retries: Default number of retries for failed operations
            result_retries: Max retries for result validation (defaults to retries)
            end_strategy: Strategy for handling tool calls that are requested alongside
                        a final result
            defer_model_check: Whether to defer model evaluation until first run
            **kwargs: Additional arguments for PydanticAI agent

        Yields:
            Configured Agent instance

        Example:
            ```python
            async with Agent.open("config.yml") as agent:
                result = await agent.run("Hello!")
                print(result.data)
            ```
        """
        if config_path is None:
            config_path = Config()
        async with RuntimeConfig.open(config_path) as runtime:
            agent = cls(
                runtime=runtime,
                model=model,
                session=session,
                system_prompt=system_prompt,
                name=name,
                retries=retries,
                end_strategy=end_strategy,
                result_retries=result_retries,
                defer_model_check=defer_model_check,
                result_type=result_type,
                **kwargs,
            )
            try:
                async with agent:
                    yield (
                        agent if result_type is None else agent.to_structured(result_type)
                    )
            finally:
                # Any cleanup if needed
                pass

    @classmethod
    @overload
    def open_agent(
        cls,
        config: StrPath | AgentsManifest,
        agent_name: str,
        *,
        deps: TDeps | None = None,
        result_type: None = None,
        model: str | None = None,
        session: SessionIdType | SessionQuery = None,
        model_settings: dict[str, Any] | None = None,
        tools: list[ToolType] | None = None,
        tool_choice: bool | str | list[str] = True,
        end_strategy: EndStrategy = "early",
    ) -> AbstractAsyncContextManager[Agent[TDeps]]: ...

    @classmethod
    @overload
    def open_agent[TResult](
        cls,
        config: StrPath | AgentsManifest,
        agent_name: str,
        *,
        deps: TDeps | None = None,
        result_type: type[TResult],
        model: str | None = None,
        session: SessionIdType | SessionQuery = None,
        model_settings: dict[str, Any] | None = None,
        tools: list[ToolType] | None = None,
        tool_choice: bool | str | list[str] = True,
        end_strategy: EndStrategy = "early",
    ) -> AbstractAsyncContextManager[StructuredAgent[TDeps, TResult]]: ...

    @classmethod
    @asynccontextmanager
    async def open_agent[TResult](
        cls,
        config: StrPath | AgentsManifest,
        agent_name: str,
        *,
        deps: TDeps | None = None,  # TDeps from class
        result_type: type[TResult] | None = None,
        model: str | ModelType = None,
        session: SessionIdType | SessionQuery = None,
        model_settings: dict[str, Any] | None = None,
        tools: list[ToolType] | None = None,
        tool_choice: bool | str | list[str] = True,
        end_strategy: EndStrategy = "early",
        retries: int = 1,
        result_tool_name: str = "final_result",
        result_tool_description: str | None = None,
        result_retries: int | None = None,
        system_prompt: str | Sequence[str] | None = None,
        enable_db_logging: bool = True,
    ) -> AsyncIterator[Agent[TDeps] | StructuredAgent[TDeps, TResult]]:
        """Open and configure a specific agent from configuration."""
        """Implementation with all parameters..."""
        """Open and configure a specific agent from configuration.

        Args:
            config: Path to agent configuration file or AgentsManifest instance
            agent_name: Name of the agent to load

            # Basic Configuration
            model: Optional model override
            result_type: Optional type for structured responses
            model_settings: Additional model-specific settings
            session: Optional id or Session query to recover a conversation

            # Tool Configuration
            tools: Additional tools to register (import paths or callables)
            tool_choice: Control tool usage:
                - True: Allow all tools
                - False: No tools
                - str: Use specific tool
                - list[str]: Allow specific tools
            end_strategy: Strategy for handling tool calls that are requested alongside
                            a final result

            # Execution Settings
            retries: Default number of retries for failed operations
            result_tool_name: Name of the tool used for final result
            result_tool_description: Description of the final result tool
            result_retries: Max retries for result validation (defaults to retries)

            # Other Settings
            system_prompt: Additional system prompts
            enable_db_logging: Whether to enable logging for the agent

        Yields:
            Configured Agent instance

        Raises:
            ValueError: If agent not found or configuration invalid
            RuntimeError: If agent initialization fails

        Example:
            ```python
            async with Agent.open_agent(
                "agents.yml",
                "my_agent",
                model="gpt-4",
                tools=[my_custom_tool],
            ) as agent:
                result = await agent.run("Do something")
            ```
        """
        if isinstance(config, AgentsManifest):
            agent_def = config
        else:
            agent_def = AgentsManifest.from_file(config)

        if agent_name not in agent_def.agents:
            msg = f"Agent {agent_name!r} not found in {config}"
            raise ValueError(msg)

        agent_config = agent_def.agents[agent_name]
        resolved_type = result_type or agent_def.get_result_type(agent_name)

        # Use model from override or agent config
        actual_model = model or agent_config.model
        if not actual_model:
            msg = "Model must be specified either in config or as override"
            raise ValueError(msg)

        # Create context
        context = AgentContext[TDeps](  # Use TDeps here
            agent_name=agent_name,
            capabilities=agent_config.capabilities,
            definition=agent_def,
            config=agent_config,
            model_settings=model_settings or {},
        )

        # Set up runtime
        cfg = agent_config.get_config()
        async with RuntimeConfig.open(cfg) as runtime:
            # Create base agent with correct typing
            base_agent = cls(  # cls is Agent[TDeps]
                runtime=runtime,
                context=context,
                model=actual_model,  # type: ignore[arg-type]
                retries=retries,
                session=session,
                result_retries=result_retries,
                end_strategy=end_strategy,
                tool_choice=tool_choice,
                tools=tools,
                system_prompt=system_prompt or [],
                enable_db_logging=enable_db_logging,
            )
            try:
                async with base_agent:
                    if resolved_type is not None and resolved_type is not str:
                        # Yield structured agent with correct typing
                        from agent.structured import StructuredAgent

                        yield StructuredAgent[TDeps, TResult](  # Use TDeps and TResult
                            base_agent,
                            resolved_type,
                            tool_description=result_tool_description,
                            tool_name=result_tool_name,
                        )
                    else:
                        yield base_agent
            finally:
                # Any cleanup if needed
                pass

    def _forward_message(self, message: ChatMessage[Any]):
        """Forward sent messages."""
        logger.debug(
            "forwarding message from %s: %s (type: %s) to %d connected agents",
            self.name,
            repr(message.content),
            type(message.content),
            len(self.connections.get_targets()),
        )
        # update = {"forwarded_from": [*message.forwarded_from, self.name]}
        # forwarded_msg = message.model_copy(update=update)
        message.forwarded_from.append(self.name)
        self.outbox.emit(message, None)

    async def disconnect_all(self):
        """Disconnect from all agents."""
        for target in list(self.connections.get_targets()):
            self.stop_passing_results_to(target)

    @overload
    def pass_results_to(
        self,
        other: AnyAgent[Any, Any] | str,
        prompt: str | None = None,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
    ) -> Talk: ...

    @overload
    def pass_results_to(
        self,
        other: Team[Any],
        prompt: str | None = None,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
    ) -> TeamTalk: ...

    def pass_results_to(
        self,
        other: AnyAgent[Any, Any] | Team[Any] | str,
        prompt: str | None = None,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
    ) -> Talk | TeamTalk:
        """Forward results to another agent or all agents in a team."""
        return self.connections.connect_agent_to(
            other,
            connection_type=connection_type,
            priority=priority,
            delay=delay,
        )

    def stop_passing_results_to(self, other: AnyAgent[Any, Any]):
        """Stop forwarding results to another agent."""
        self.connections.disconnect(other)

    def is_busy(self) -> bool:
        """Check if agent is currently processing tasks."""
        return bool(self._pending_tasks or self._background_task)

    @property
    def model_name(self) -> str | None:
        """Get the model name in a consistent format."""
        return self._provider.model_name

    @logfire.instrument("Calling Agent.run: {prompt}:")
    async def run(
        self,
        *prompt: AnyPromptType,
        result_type: type[TResult] | None = None,
        deps: TDeps | None = None,
        model: ModelType = None,
        store_history: bool = True,
    ) -> ChatMessage[TResult]:
        """Run agent with prompt and get response.

        Args:
            prompt: User query or instruction
            result_type: Optional type for structured responses
            deps: Optional dependencies for the agent
            model: Optional model override
            store_history: Whether the message exchange should be added to the
                           context window

        Returns:
            Result containing response and run information

        Raises:
            UnexpectedModelBehavior: If the model fails or behaves unexpectedly
        """
        """Run agent with prompt and get response."""
        prompts = [await to_prompt(p) for p in prompt]
        final_prompt = "\n\n".join(prompts)
        if deps is not None:
            self.context.data = deps
        self.context.current_prompt = final_prompt
        self.set_result_type(result_type)
        wait_for_chain = False  # TODO

        try:
            # Create and emit user message
            user_msg = ChatMessage[str](content=final_prompt, role="user")
            self.message_received.emit(user_msg)

            # Get response through provider
            message_id = str(uuid4())
            start_time = time.perf_counter()
            result = await self._provider.generate_response(
                final_prompt,
                message_id,
                result_type=result_type,
                model=model,
                store_history=store_history,
            )

            # Get cost info for assistant response
            usage = result.usage
            cost_info = (
                await TokenCost.from_usage(
                    usage, result.model_name, final_prompt, str(result.content)
                )
                if self.model_name and usage
                else None
            )

            # Create final message with all metrics
            assistant_msg = ChatMessage[TResult](
                content=result.content,
                role="assistant",
                name=self.name,
                model=self.model_name,
                message_id=message_id,
                tool_calls=result.tool_calls,
                cost_info=cost_info,
                response_time=time.perf_counter() - start_time,
            )
            if self._debug:
                import devtools

                devtools.debug(assistant_msg)

            self.message_sent.emit(assistant_msg)

        except Exception:
            logger.exception("Agent run failed")
            raise

        else:
            if wait_for_chain:
                await self.wait_for_chain()
            return assistant_msg

    def to_agent_tool(
        self,
        *,
        name: str | None = None,
        reset_history_on_run: bool = True,
        pass_message_history: bool = False,
        share_context: bool = False,
        parent: AnyAgent[Any, Any] | None = None,
    ) -> LLMCallableTool:
        """Create a tool from this agent.

        Args:
            name: Optional tool name override
            reset_history_on_run: Clear agent's history before each run
            pass_message_history: Pass parent's message history to agent
            share_context: Whether to pass parent's context/deps
            parent: Optional parent agent for history/context sharing
        """
        tool_name = f"ask_{self.name}"

        async def wrapped_tool(ctx: RunContext[AgentContext[TDeps]], prompt: str) -> str:
            if pass_message_history and not parent:
                msg = "Parent agent required for message history sharing"
                raise ToolError(msg)

            if reset_history_on_run:
                self.conversation.clear()

            history = None
            deps = ctx.deps.data if share_context else None
            if pass_message_history and parent:
                history = parent.conversation.get_history()
                old = self.conversation.get_history()
                self.conversation.set_history(history)
            result = await self.run(prompt, deps=deps, result_type=self._result_type)
            if history:
                self.conversation.set_history(old)
            return result.data

        normalized_name = self.name.replace("_", " ").title()
        docstring = f"Get expert answer from specialized agent: {normalized_name}"
        if self.description:
            docstring = f"{docstring}\n\n{self.description}"

        wrapped_tool.__doc__ = docstring
        wrapped_tool.__name__ = tool_name

        return LLMCallableTool.from_callable(
            wrapped_tool,
            name_override=tool_name,
            description_override=docstring,
        )

    @asynccontextmanager
    async def run_stream(
        self,
        *prompt: AnyPromptType,
        result_type: type[TResult] | None = None,
        deps: TDeps | None = None,
        model: ModelType = None,
        store_history: bool = True,
    ) -> AsyncIterator[StreamedRunResult[AgentContext[TDeps], TResult]]:
        """Run agent with prompt and get a streaming response.

        Args:
            prompt: User query or instruction
            result_type: Optional type for structured responses
            deps: Optional dependencies for the agent
            model: Optional model override
            store_history: Whether the message exchange should be added to the
                           context window

        Returns:
            A streaming result to iterate over.

        Raises:
            UnexpectedModelBehavior: If the model fails or behaves unexpectedly
        """
        prompts = [await to_prompt(p) for p in prompt]
        final_prompt = "\n\n".join(prompts)
        self.set_result_type(result_type)

        if deps is not None:
            self.context.data = deps
        self.context.current_prompt = final_prompt
        try:
            # Create and emit user message
            user_msg = ChatMessage[str](content=final_prompt, role="user")
            self.message_received.emit(user_msg)
            message_id = str(uuid4())
            start_time = time.perf_counter()

            async with self._provider.stream_response(
                final_prompt,
                message_id,
                result_type=result_type,
                model=model,
                store_history=store_history,
            ) as stream:
                yield stream  # type: ignore

                # After streaming is done, create and emit final message
                usage = stream.usage()
                cost_info = (
                    await TokenCost.from_usage(
                        usage,
                        stream.model_name,  # type: ignore
                        final_prompt,
                        str(stream.formatted_content),  # type: ignore
                    )
                    if self.model_name
                    else None
                )

                assistant_msg = ChatMessage[TResult](
                    content=cast(TResult, stream.formatted_content),  # type: ignore
                    role="assistant",
                    name=self.name,
                    model=self.model_name,
                    message_id=message_id,
                    cost_info=cost_info,
                    response_time=time.perf_counter() - start_time,
                )
                self.message_sent.emit(assistant_msg)

        except Exception:
            logger.exception("Agent stream failed")
            raise

    def run_sync(
        self,
        *prompt: AnyPromptType,
        result_type: type[TResult] | None = None,
        deps: TDeps | None = None,
        model: ModelType = None,
        store_history: bool = True,
    ) -> ChatMessage[TResult]:
        """Run agent synchronously (convenience wrapper).

        Args:
            prompt: User query or instruction
            result_type: Optional type for structured responses
            deps: Optional dependencies for the agent
            model: Optional model override
            store_history: Whether the message exchange should be added to the
                           context window
        Returns:
            Result containing response and run information
        """
        try:
            return asyncio.run(
                self.run(
                    prompt,
                    deps=deps,
                    model=model,
                    store_history=store_history,
                    result_type=result_type,
                )
            )
        except KeyboardInterrupt:
            raise
        except Exception:
            logger.exception("Sync agent run failed")
            raise

    async def wait_for_chain(self, _seen: set[str] | None = None):
        """Wait for this agent and all connected agents to complete their tasks."""
        # Track seen agents to avoid cycles
        seen = _seen or {self.name}

        # Wait for our own tasks
        await self.complete_tasks()

        # Wait for connected agents
        for agent in self.connections.get_targets():
            if agent.name not in seen:
                seen.add(agent.name)
                await agent.wait_for_chain(seen)

    async def run_task[TResult](
        self,
        task: AgentTask[TDeps, TResult],
        *,
        store_history: bool = True,
        include_agent_tools: bool = True,
    ) -> ChatMessage[TResult]:
        """Execute a pre-defined task.

        Args:
            task: Task configuration to execute
            store_history: Whether the message exchange should be added to the
                           context window
            include_agent_tools: Whether to include agent tools
        Returns:
            Task execution result

        Raises:
            TaskError: If task execution fails
            ValueError: If task configuration is invalid
        """
        from tasks import TaskError

        original_result_type = self._result_type

        self.set_result_type(task.result_type)

        # Load task knowledge
        if task.knowledge:
            # Add knowledge sources to context
            resources: list[Resource | str] = list(task.knowledge.paths) + list(
                task.knowledge.resources
            )
            for source in resources:
                await self.conversation.load_context_source(source)
            for prompt in task.knowledge.prompts:
                if isinstance(prompt, StaticPrompt | DynamicPrompt | FilePrompt):
                    await self.conversation.add_context_from_prompt(prompt)
                else:
                    await self.conversation.load_context_source(prompt)

        try:
            # Register task tools temporarily
            tools = [import_callable(cfg.import_path) for cfg in task.tool_configs]
            names = [cfg.name for cfg in task.tool_configs]
            descriptions = [cfg.description for cfg in task.tool_configs]
            tools = [
                LLMCallableTool.from_callable(
                    tool, name_override=name, description_override=description
                )
                for tool, name, description in zip(tools, names, descriptions)
            ]
            with self.tools.temporary_tools(tools, exclusive=not include_agent_tools):
                # Execute task with task-specific tools
                from tasks.strategies import DirectStrategy

                strategy = DirectStrategy[TDeps, TResult]()
                return await strategy.execute(
                    task=task,
                    agent=self,
                    store_history=store_history,
                )

        except Exception as e:
            msg = f"Task execution failed: {e}"
            logger.exception(msg)
            raise TaskError(msg) from e
        finally:
            self.set_result_type(original_result_type)

    async def run_continuous(
        self,
        prompt: AnyPromptType,
        *,
        max_count: int | None = None,
        interval: float = 1.0,
        block: bool = False,
        **kwargs: Any,
    ) -> ChatMessage[TResult] | None:
        """Run agent continuously with prompt or dynamic prompt function.

        Args:
            prompt: Static prompt or function that generates prompts
            max_count: Maximum number of runs (None = infinite)
            interval: Seconds between runs
            block: Whether to block until completion
            **kwargs: Arguments passed to run()
        """

        async def _continuous():
            count = 0
            msg = "%s: Starting continuous run (max_count=%s, interval=%s)"
            logger.debug(msg, self.name, max_count, interval)
            while max_count is None or count < max_count:
                try:
                    current_prompt = (
                        call_with_context(prompt, self.context, **kwargs)
                        if callable(prompt)
                        else to_prompt(prompt)
                    )
                    msg = "%s: Generated prompt #%d: %s"
                    logger.debug(msg, self.name, count, current_prompt)

                    await self.run(current_prompt, **kwargs)
                    msg = "%s: Run continous result #%d"
                    logger.debug(msg, self.name, count)

                    count += 1
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    logger.debug("%s: Continuous run cancelled", self.name)
                    break
                except Exception:
                    logger.exception("%s: Background run failed", self.name)
                    await asyncio.sleep(interval)
            msg = "%s: Continuous run completed after %d iterations"
            logger.debug(msg, self.name, count)

        # Cancel any existing background task
        await self.stop()
        task = asyncio.create_task(_continuous(), name=f"background_{self.name}")

        if block:
            try:
                await task
                return None
            finally:
                if not task.done():
                    task.cancel()
        else:
            logger.debug("%s: Started background task %s", self.name, task.get_name())
            self._background_task = task
            return None

    async def stop(self):
        """Stop continuous execution if running."""
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            await self._background_task
            self._background_task = None

    def clear_history(self):
        """Clear both internal and pydantic-ai history."""
        self._logger.clear_state()
        self.conversation.clear()
        logger.debug("Cleared history and reset tool state")

    async def get_token_limits(self) -> TokenLimits | None:
        """Get token limits for the current model."""
        if not self.model_name:
            return None

        try:
            return await get_model_limits(self.model_name)
        except ValueError:
            logger.debug("Could not get token limits for model: %s", self.model_name)
            return None

    async def share(
        self,
        target: AnyAgent[TDeps, Any],
        *,
        tools: list[str] | None = None,
        resources: list[str] | None = None,
        history: bool | int | None = None,  # bool or number of messages
        token_limit: int | None = None,
    ) -> None:
        """Share capabilities and knowledge with another agent.

        Args:
            target: Agent to share with
            tools: List of tool names to share
            resources: List of resource names to share
            history: Share conversation history:
                    - True: Share full history
                    - int: Number of most recent messages to share
                    - None: Don't share history
            token_limit: Optional max tokens for history

        Raises:
            ValueError: If requested items don't exist
            RuntimeError: If runtime not available for resources
        """
        # Share tools if requested
        for name in tools or []:
            if tool := self.tools.get(name):
                meta = {"shared_from": self.name}
                target.tools.register_tool(tool.callable, metadata=meta)
            else:
                msg = f"Tool not found: {name}"
                raise ValueError(msg)

        # Share resources if requested
        if resources:
            if not self.runtime:
                msg = "No runtime available for sharing resources"
                raise RuntimeError(msg)
            for name in resources:
                if resource := self.runtime.get_resource(name):
                    await target.conversation.load_context_source(resource)
                else:
                    msg = f"Resource not found: {name}"
                    raise ValueError(msg)

        # Share history if requested
        if history:
            history_text = await self.conversation.format_history(
                max_tokens=token_limit,
                num_messages=history if isinstance(history, int) else None,
            )
            await target.conversation.add_context_message(
                history_text, source=self.name, metadata={"type": "shared_history"}
            )

    def register_worker(
        self,
        worker: Agent[Any],
        *,
        name: str | None = None,
        reset_history_on_run: bool = True,
        pass_message_history: bool = False,
        share_context: bool = False,
    ) -> ToolInfo:
        """Register another agent as a worker tool."""
        return self.tools.register_worker(
            worker,
            name=name,
            reset_history_on_run=reset_history_on_run,
            pass_message_history=pass_message_history,
            share_context=share_context,
            parent=self if (pass_message_history or share_context) else None,
        )

    def set_model(self, model: ModelType):
        """Set the model for this agent.

        Args:
            model: New model to use (name or instance)

        Emits:
            model_changed signal with the new model
        """
        self._provider.set_model(model)

    @property
    def runtime(self) -> RuntimeConfig:
        """Get runtime configuration from context."""
        assert self.context.runtime
        return self.context.runtime

    @runtime.setter
    def runtime(self, value: RuntimeConfig):
        """Set runtime configuration and update context."""
        self.context.runtime = value

    @property
    def tools(self) -> ToolManager:
        return self._tool_manager


if __name__ == "__main__":
    import logging

    from parry_agent import config_resources

    logging.basicConfig(level=logging.INFO)

    sys_prompt = "Open browser with google, please"
    path = config_resources.OPEN_BROWSER

    async def main():
        async with Agent[None].open(path, model="openai:gpt-4o-mini") as agent:
            result = await agent.run(sys_prompt)
            print(result.data)

    asyncio.run(main())
