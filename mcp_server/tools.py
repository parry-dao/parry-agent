"""MCP tool integration for parry agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from parry import LLMCallableTool

from log import get_logger


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mcp.types import Tool as MCPTool

    from mcp_server.client import MCPClient
    from tools.base import ToolInfo
    from tools.manager import ToolManager


logger = get_logger(__name__)


def create_mcp_tool_callable(
    mcp_client: MCPClient,
    tool: MCPTool,
) -> Callable[..., Awaitable[str]]:
    """Create a callable that forwards to MCP tool."""

    async def call_mcp_tool(**kwargs: Any) -> str:
        """Forward call to MCP server."""
        return await mcp_client.call_tool(tool.name, kwargs)

    # Set metadata for LLMCallableTool creation
    call_mcp_tool.__name__ = tool.name
    call_mcp_tool.__doc__ = tool.description

    return call_mcp_tool


def register_mcp_tools(
    tool_manager: ToolManager,
    mcp_client: MCPClient,
) -> list[ToolInfo]:
    """Register MCP tools with tool manager."""
    registered = []

    for mcp_tool in mcp_client._available_tools:
        # Create properly typed callable from schema
        tool_callable = mcp_client.create_tool_callable(mcp_tool)

        # The function already has proper typing, so no schema override needed
        llm_tool = LLMCallableTool.from_callable(tool_callable)

        metadata = {"mcp_tool": mcp_tool.name}
        tool_info = tool_manager.register_tool(llm_tool, source="mcp", metadata=metadata)
        registered.append(tool_info)

        logger.debug("Registered MCP tool: %s", mcp_tool.name)

    return registered
