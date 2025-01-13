from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class MCPServerBase(BaseModel):
    """Base model for MCP server configuration."""

    type: str
    """Type discriminator for MCP server configurations."""

    enabled: bool = True
    """Whether this server is currently enabled."""

    environment: dict[str, str] | None = None
    """Environment variables to pass to the server process."""

    model_config = ConfigDict(use_attribute_docstrings=True)


class StdioMCPServer(MCPServerBase):
    """MCP server started via stdio.

    Uses subprocess communication through standard input/output streams.
    """

    type: Literal["stdio"] = Field("stdio", init=False)
    """Type discriminator for stdio servers."""

    command: str
    """Command to execute (e.g. "pipx", "python", "node")."""

    args: list[str] = Field(default_factory=list)
    """Command arguments (e.g. ["run", "some-server", "--debug"])."""


class SSEMCPServer(MCPServerBase):
    """MCP server using Server-Sent Events transport.

    Connects to a server over HTTP with SSE for real-time communication.
    """

    type: Literal["sse"] = Field("sse", init=False)
    """Type discriminator for SSE servers."""

    url: str
    """URL of the SSE server endpoint."""


MCPServerConfig = Annotated[StdioMCPServer | SSEMCPServer, Field(discriminator="type")]
