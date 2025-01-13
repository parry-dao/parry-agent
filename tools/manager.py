"""Tool management for parry agents."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AsyncExitStack, contextmanager
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Self

from parry import BaseRegistry, LLMCallableTool, ToolError
from psygnal import Signal

from log import get_logger
from mcp_server.client import MCPClient
from mcp_server.tools import register_mcp_tools
from models.mcp_server import MCPServerConfig, SSEMCPServer, StdioMCPServer
from tools.base import ToolInfo


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import TracebackType

    from agent import AnyAgent
    from common_types import ToolSource, ToolType
    from models.context import AgentContext


logger = get_logger(__name__)
MAX_LEN_DESCRIPTION = 1000


class ToolManager(BaseRegistry[str, ToolInfo]):
    """Manages tool registration, enabling/disabling and access.

    Inherits from BaseRegistry providing:
    - Dict-like access: manager["tool_name"] -> ToolInfo
    - Async startup/shutdown: await manager.startup()
    - Event observation: manager.add_observer(observer)
    - Registration: manager.register("tool_name", tool)
    - Listing: manager.list_items()
    - State check: manager.is_empty, manager.has_item()
    - Async iteration: async for name, tool in manager: ...
    """

    @dataclass(frozen=True)
    class ToolStateReset:
        """Emitted when tool states are reset."""

        previous_tools: dict[str, bool]
        new_tools: dict[str, bool]
        timestamp: datetime = field(default_factory=datetime.now)

    tool_states_reset = Signal(ToolStateReset)

    def __init__(
        self,
        tools: Sequence[ToolInfo | ToolType | dict[str, Any]] | None = None,
        *,
        tool_choice: bool | str | list[str] = True,
        context: AgentContext[Any] | None = None,
    ):
        """Initialize tool manager.

        Args:
            tools: Initial tools to register
            tool_choice: Control tool usage:
                - True: Allow all tools
                - False: No tools
                - str: Use specific tool
                - list[str]: Allow specific tools
            context: Tool context
        """
        super().__init__()
        self.tool_choice = tool_choice
        self.context = context
        self._mcp_clients: dict[str, MCPClient] = {}
        self.exit_stack = AsyncExitStack()

        # Register initial tools
        for tool in tools or []:
            t = self._validate_item(tool)
            self.register(t.name, t)

    def __prompt__(self) -> str:
        enabled_tools = [t.name for t in self.values() if t.enabled]
        if not enabled_tools:
            return "No tools available"
        return f"Available tools: {', '.join(enabled_tools)}"

    async def __aenter__(self) -> Self:
        """Enter async context and set up MCP servers."""
        try:
            # Setup MCP servers if configured
            if self.context and self.context.config and self.context.config.mcp_servers:
                await self.setup_mcp_servers(self.context.config.get_mcp_servers())
        except Exception as e:
            # Clean up on error
            await self.__aexit__(type(e), e, e.__traceback__)
            msg = "Failed to initialize tool manager"
            logger.exception(msg, exc_info=e)
            raise RuntimeError(msg) from e
        else:
            return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context."""
        try:
            # Clean up MCP clients through exit stack
            await self.exit_stack.aclose()
            self._mcp_clients.clear()
        except Exception as e:
            msg = "Error during tool manager cleanup"
            logger.exception(msg, exc_info=e)
            raise RuntimeError(msg) from e

    def reset_states(self):
        """Reset all tools to their default enabled states."""
        for info in self.values():
            info.enabled = True

    @property
    def _error_class(self) -> type[ToolError]:
        """Error class for tool operations."""
        return ToolError

    def _validate_item(self, item: ToolInfo | ToolType | dict[str, Any]) -> ToolInfo:
        """Validate and convert items before registration."""
        match item:
            case ToolInfo():
                return item
            case LLMCallableTool():
                return ToolInfo(callable=item)
            case str():
                item = LLMCallableTool.from_callable(item)
                return ToolInfo(callable=item)
            case _ if callable(item):
                tool = LLMCallableTool.from_callable(item)
                return ToolInfo(callable=tool)
            case {"callable": callable_item, **config} if callable(
                callable_item
            ) or isinstance(callable_item, LLMCallableTool):
                # First convert callable to LLMCallableTool if needed
                tool = (
                    callable_item
                    if isinstance(callable_item, LLMCallableTool)
                    else LLMCallableTool.from_callable(callable_item)
                )

                # Get valid fields from ToolInfo dataclass (excluding 'callable')
                valid_keys = {f.name for f in fields(ToolInfo)} - {"callable"}
                tool_config = {k: v for k, v in config.items() if k in valid_keys}

                return ToolInfo(callable=tool, **tool_config)  # type: ignore

            case _:
                typ = type(item)
                msg = f"Item must be ToolInfo, LLMCallableTool, or callable. Got {typ}"
                raise ToolError(msg)

    def enable_tool(self, tool_name: str):
        """Enable a previously disabled tool."""
        if tool_name not in self:
            msg = f"Tool not found: {tool_name}"
            raise ToolError(msg)
        tool_info = self[tool_name]
        tool_info.enabled = True
        self.events.changed(tool_name, tool_info)
        logger.debug("Enabled tool: %s", tool_name)

    def disable_tool(self, tool_name: str):
        """Disable a tool."""
        if tool_name not in self:
            msg = f"Tool not found: {tool_name}"
            raise ToolError(msg)
        tool_info = self[tool_name]
        tool_info.enabled = False
        self.events.changed(tool_name, tool_info)
        logger.debug("Disabled tool: %s", tool_name)

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is currently enabled."""
        return self[tool_name].enabled if tool_name in self else False

    def list_tools(self) -> dict[str, bool]:
        """Get a mapping of all tools and their enabled status."""
        return {name: info.enabled for name, info in self.items()}

    def get_tools(
        self,
        state: Literal["all", "enabled", "disabled"] = "all",
        names: list[str] | None = None,
    ) -> list[LLMCallableTool]:
        """Get tool objects based on filters."""
        filtered_tools = [
            info.callable
            for info in self.values()
            if info.matches_filter(state) and (names is None or info.name in names)
        ]

        # Sort by priority (if any have non-default priority)
        if any(self[t.name].priority != 100 for t in filtered_tools):  # noqa: PLR2004
            filtered_tools.sort(key=lambda t: self[t.name].priority)

        return filtered_tools

    def get_tool_names(
        self, state: Literal["all", "enabled", "disabled"] = "all"
    ) -> set[str]:
        """Get tool names based on state."""
        return {name for name, info in self.items() if info.matches_filter(state)}

    def register_tool(
        self,
        tool: LLMCallableTool | Callable[..., Any],
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        enabled: bool = True,
        source: ToolSource = "runtime",
        priority: int = 100,
        requires_confirmation: bool = False,
        requires_capability: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ToolInfo:
        """Register a new tool with custom settings.

        Args:
            tool: Tool to register (callable, LLMCallableTool, or config dict)
            enabled: Whether tool is initially enabled
            name_override: Optional name override for the tool
            description_override: Optional description override for the tool
            source: Tool source (runtime/agent/builtin/dynamic)
            priority: Execution priority (lower = higher priority)
            requires_confirmation: Whether tool needs confirmation
            requires_capability: Optional capability needed to use tool
            metadata: Additional tool metadata

        Returns:
            Created ToolInfo instance
        """
        # First convert to basic ToolInfo
        if not isinstance(tool, LLMCallableTool):
            llm_tool = LLMCallableTool.from_callable(
                tool,
                name_override=name_override,
                description_override=description_override,
            )
        else:
            llm_tool = tool
            llm_tool.name = name_override or llm_tool.name
            llm_tool.description = description_override or llm_tool.description
        if llm_tool.description and len(llm_tool.description) > MAX_LEN_DESCRIPTION:
            msg = f"Too long description for {tool}"
            raise ToolError(msg)
        tool_info = ToolInfo(
            llm_tool,
            enabled=enabled,
            source=source,
            priority=priority,
            requires_confirmation=requires_confirmation,
            requires_capability=requires_capability,
            metadata=metadata or {},
        )
        # Register the tool
        self.register(tool_info.name, tool_info)
        return tool_info

    def register_worker(
        self,
        worker: AnyAgent[Any, Any],
        *,
        name: str | None = None,
        reset_history_on_run: bool = True,
        pass_message_history: bool = False,
        share_context: bool = False,
        parent: AnyAgent[Any, Any] | None = None,
    ) -> ToolInfo:
        """Register an agent as a worker tool.

        Args:
            worker: Agent to register as worker
            name: Optional name override for the worker tool
            reset_history_on_run: Whether to clear history before each run
            pass_message_history: Whether to pass parent's message history
            share_context: Whether to pass parent's context/deps
            parent: Optional parent agent for history/context sharing
        """
        tool = worker.to_agent_tool(
            parent=parent,
            name=name,
            reset_history_on_run=reset_history_on_run,
            pass_message_history=pass_message_history,
            share_context=share_context,
        )
        msg = "Registering worker %s as tool %s"
        logger.debug(msg, worker.name, tool.name)
        return self.register_tool(tool, source="agent", metadata={"agent": worker.name})

    def reset(self):
        """Reset tool states."""
        old_tools = self.list_tools()
        self.reset_states()
        new_tools = self.list_tools()

        event = self.ToolStateReset(old_tools, new_tools)
        self.tool_states_reset.emit(event)

    async def setup_mcp_servers(self, servers: list[MCPServerConfig]):
        """Set up multiple MCP server integrations.

        Args:
            servers: List of MCP server configurations
        """
        try:
            for server in servers:
                if not server.enabled:
                    continue

                match server:
                    case StdioMCPServer():
                        # Create client with stdio mode
                        client = MCPClient(stdio_mode=True)
                        client = await self.exit_stack.enter_async_context(client)
                        env = server.environment or {}
                        env["PYTHONIOENCODING"] = "utf-8"
                        await client.connect(server.command, args=server.args, env=env)
                        # Store client
                        client_id = f"{server.command}_{' '.join(server.args)}"
                        self._mcp_clients[client_id] = client

                        # Register tools
                        register_mcp_tools(self, client)

                    case SSEMCPServer():
                        # SSE client without stdio mode
                        client = MCPClient(stdio_mode=False)
                        client = await self.exit_stack.enter_async_context(client)

                        await client.connect(
                            command="",  # Not used for SSE
                            args=[],  # Not used for SSE
                            url=server.url,
                            env=server.environment,
                        )

                        # Store client
                        client_id = f"sse_{server.url}"
                        self._mcp_clients[client_id] = client

                        # Register tools
                        register_mcp_tools(self, client)

        except Exception as e:
            msg = "Failed to setup MCP servers"
            logger.exception(msg, exc_info=e)
            raise RuntimeError(msg) from e

    async def cleanup(self):
        """Clean up resources including all MCP clients."""
        try:
            # Store tools to remove
            to_remove = [
                name
                for name, info in self.items()
                if info.metadata.get("mcp_tool") is not None
            ]

            # Clean up exit stack (which includes MCP clients)
            await self.exit_stack.aclose()
            self._mcp_clients.clear()

            # Remove MCP tools
            for name in to_remove:
                del self._items[name]

        except Exception as e:
            msg = "Error during tool manager cleanup"
            logger.exception(msg, exc_info=e)
            raise RuntimeError(msg) from e

    @contextmanager
    def temporary_tools(
        self,
        tools: Callable[..., Any]
        | LLMCallableTool
        | Sequence[Callable[..., Any] | LLMCallableTool],
        *,
        exclusive: bool = False,
    ) -> Iterator[list[ToolInfo]]:
        """Temporarily register tools.

        Args:
            tools: Tool(s) to register
            exclusive: Whether to temporarily disable all other tools

        Yields:
            List of registered tool infos

        Example:
            ```python
            with tool_manager.temporary_tools([tool1, tool2], exclusive=True) as tools:
                # Only tool1 and tool2 are available
                await agent.run(prompt)
            # Original tool states are restored
            ```
        """
        # Normalize inputs to lists
        tools_list = [tools] if not isinstance(tools, Sequence) else list(tools)

        # Store original tool states if exclusive
        original_states: dict[str, bool] = {}
        if exclusive:
            original_states = {name: tool.enabled for name, tool in self.items()}
            # Disable all existing tools
            for t in self.values():
                t.enabled = False

        # Register all tools
        registered_tools: list[ToolInfo] = []
        try:
            for tool in tools_list:
                tool_info = self.register_tool(tool)
                registered_tools.append(tool_info)
            yield registered_tools

        finally:
            # Remove temporary tools
            for tool_info in registered_tools:
                del self[tool_info.name]

            # Restore original tool states if exclusive
            if exclusive:
                for name_, was_enabled in original_states.items():
                    if t := self.get(name_):
                        t.enabled = was_enabled
