"""MCP client integration for parry agent."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager, AsyncExitStack
import inspect
import sys
from typing import TYPE_CHECKING, Any, Self, TextIO

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import EmbeddedResource, ImageContent, Tool as MCPTool

from log import get_logger


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from types import TracebackType

    from mcp.types import Tool

logger = get_logger(__name__)


class MCPClient(AbstractAsyncContextManager["MCPClient"]):
    """MCP client for communicating with MCP servers."""

    def __init__(self, stdio_mode: bool = False):
        self.exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self._available_tools: list[Tool] = []
        self._old_stdout: TextIO | None = None
        self._stdio_mode = stdio_mode

    async def __aenter__(self) -> Self:
        """Enter context and redirect stdout if in stdio mode."""
        try:
            if self._stdio_mode:
                self._old_stdout = sys.stdout
                sys.stdout = sys.stderr
                logger.info("Redirecting stdout for stdio MCP server")
        except Exception as e:
            msg = "Failed to enter MCP client context"
            logger.exception(msg, exc_info=e)
            raise RuntimeError(msg) from e
        else:
            return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        """Restore stdout if redirected and cleanup."""
        try:
            if self._old_stdout:
                sys.stdout = self._old_stdout
            await self.cleanup()
        except Exception:
            logger.exception("Error during MCP client cleanup")
            raise

    async def connect(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        url: str | None = None,
    ):
        """Connect to an MCP server.

        Args:
            command: Command to run (for stdio servers)
            args: Command arguments (for stdio servers)
            env: Optional environment variables
            url: Server URL (for SSE servers)
        """
        if url:
            # SSE connection - just a placeholder for now
            logger.info("SSE servers not yet implemented")
            self.session = None
            return

        # Stdio connection
        params = StdioServerParameters(command=command, args=args, env=env)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
        stdio, write = stdio_transport
        session = ClientSession(stdio, write)
        self.session = await self.exit_stack.enter_async_context(session)
        assert self.session
        await self.session.initialize()

        # Get available tools
        result = await self.session.list_tools()
        self._available_tools = result.tools
        msg = "Connected to MCP server with tools: %s"
        logger.info(msg, [t.name for t in self._available_tools])

    def get_tools(self) -> list[dict]:
        """Get tools in OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "No description provided",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in self._available_tools
        ]

    def create_tool_callable(self, tool: MCPTool) -> Callable[..., Awaitable[str]]:
        """Create a properly typed callable from MCP tool schema."""
        schema = tool.inputSchema
        parameters = schema.get("properties", {})
        required = schema.get("required", [])

        # Create parameter annotations dict for the function
        annotations = {
            # Map JSON schema types to Python types
            param: str
            if details.get("type") == "string"
            else int
            if details.get("type") == "integer"
            else float
            if details.get("type") == "number"
            else bool
            if details.get("type") == "boolean"
            else Any
            for param, details in parameters.items()
        }
        annotations["return"] = str  # Return type is always str

        # Build signature parts for all parameters
        params = [
            inspect.Parameter(
                name=name,
                kind=inspect.Parameter.KEYWORD_ONLY,  # Make all params keyword-only
                annotation=typ,
                default=... if name in required else None,
            )
            for name, typ in annotations.items()
            if name != "return"
        ]

        # Create the signature
        sig = inspect.Signature(params, return_annotation=str)

        async def tool_callable(**kwargs: Any) -> str:
            """Dynamically generated MCP tool wrapper."""
            return await self.call_tool(tool.name, kwargs)

        # Set proper signature and docstring
        tool_callable.__signature__ = sig  # type: ignore
        tool_callable.__annotations__ = annotations
        tool_callable.__name__ = tool.name
        tool_callable.__doc__ = tool.description or "No description provided."

        return tool_callable

    async def call_tool(self, name: str, arguments: dict | None = None) -> str:
        """Call an MCP tool."""
        if not self.session:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)

        try:
            result = await self.session.call_tool(name, arguments or {})
            if isinstance(result.content[0], EmbeddedResource | ImageContent):
                msg = "Tool returned an embedded resource"
                raise TypeError(msg)  # noqa: TRY301
            return result.content[0].text
        except Exception as e:
            msg = f"MCP tool call failed: {e}"
            raise RuntimeError(msg) from e

    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()
        self._available_tools = []
