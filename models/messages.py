"""Message and token usage models."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel
import tokonomics
from typing_extensions import TypeVar
import yamling

from common_types import JsonObject, MessageRole  # noqa: TC001
from log import get_logger
from models.agents import ToolCallInfo  # noqa: TC001


TContent = TypeVar("TContent", str, BaseModel, default=str)

logger = get_logger(__name__)


class TokenUsage(TypedDict):
    """Token usage statistics from model responses."""

    total: int
    """Total tokens used"""
    prompt: int
    """Tokens used in the prompt"""
    completion: int
    """Tokens used in the completion"""


@dataclass(frozen=True)
class TokenCost:
    """Combined token and cost tracking."""

    token_usage: TokenUsage
    """Token counts for prompt and completion"""
    total_cost: float
    """Total cost in USD"""

    @classmethod
    async def from_usage(
        cls,
        usage: tokonomics.Usage | None,
        model: str,
        prompt: str,
        completion: str,
    ) -> TokenCost | None:
        """Create result from usage data.

        Args:
            usage: Token counts from model response
            model: Name of the model used
            prompt: The prompt text sent to model
            completion: The completion text received

        Returns:
            TokenCost if usage data available, None otherwise
        """
        if not (
            usage
            and usage.total_tokens is not None
            and usage.request_tokens is not None
            and usage.response_tokens is not None
        ):
            logger.debug("Missing token counts in Usage object")
            return None

        token_usage = TokenUsage(
            total=usage.total_tokens,
            prompt=usage.request_tokens,
            completion=usage.response_tokens,
        )
        logger.debug("Token usage: %s", token_usage)

        cost = await tokonomics.calculate_token_cost(
            model,
            usage.request_tokens,
            usage.response_tokens,
        )
        total_cost = cost.total_cost if cost else 0.0

        return cls(token_usage=token_usage, total_cost=total_cost)


@dataclass
class ChatMessage[TContent]:
    """Common message format for all UI types.

    Generically typed with: ChatMessage[Type of Content]
    The type can either be str or a BaseModel subclass.
    """

    content: TContent
    """Message content, typed as TContent (either str or BaseModel)."""

    role: MessageRole
    """Role of the message sender (user/assistant/system)."""

    model: str | None = None
    """Name of the model that generated this message."""

    metadata: JsonObject = field(default_factory=dict)
    """Additional metadata about the message."""

    timestamp: datetime = field(default_factory=datetime.now)
    """When this message was created."""

    cost_info: TokenCost | None = None
    """Token usage and costs for this specific message if available."""

    message_id: str = field(default_factory=lambda: str(uuid4()))
    """Unique identifier for this message."""

    response_time: float | None = None
    """Time it took the LLM to respond."""

    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    """List of tool calls made during message generation."""

    name: str | None = None
    """Display name for the message sender in UI."""

    forwarded_from: list[str] = field(default_factory=list)
    """List of agent names (the chain) that forwarded this message to the sender."""

    def to_text_message(self) -> ChatMessage[str]:
        """Convert this message to a text-only version."""
        return dataclasses.replace(self, content=str(self.content))  # type: ignore

    def _get_content_str(self) -> str:
        """Get string representation of content."""
        match self.content:
            case str():
                return self.content
            case BaseModel():
                return self.content.model_dump_json(indent=2)
            case _:
                msg = f"Unexpected content type: {type(self.content)}"
                raise ValueError(msg)

    def to_gradio_format(self) -> tuple[str | None, str | None]:
        """Convert to Gradio chatbot format."""
        content_str = self._get_content_str()
        match self.role:
            case "user":
                return (content_str, None)
            case "assistant":
                return (None, content_str)
            case "system":
                return (None, f"System: {content_str}")

    @property
    def data(self) -> TContent:
        """Get content as typed data. Provides compat to RunResult."""
        return self.content

    def format(
        self,
        style: Literal["simple", "detailed", "markdown"] = "simple",
        *,
        show_metadata: bool = False,
        show_costs: bool = False,
    ) -> str:
        """Format message with configurable style."""
        match style:
            case "simple":
                return self._format_simple()
            case "detailed":
                return self._format_detailed(show_metadata, show_costs)
            case "markdown":
                return self._format_markdown(show_metadata, show_costs)
            case _:
                msg = f"Invalid style: {style}"
                raise ValueError(msg)

    def _format_simple(self) -> str:
        """Basic format: sender and message."""
        sender = self.name or self.role.title()
        return f"{sender}: {self.content}"

    def _format_detailed(self, show_metadata: bool, show_costs: bool) -> str:
        """Detailed format with optional metadata and costs."""
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        name = self.name or self.role.title()
        parts = [f"From: {name}", f"Time: {ts}", "-" * 40, f"{self.content}", "-" * 40]

        if show_costs and self.cost_info:
            parts.extend([
                f"Tokens: {self.cost_info.token_usage['total']:,}",
                f"Cost: ${self.cost_info.total_cost:.4f}",
            ])
            if self.response_time:
                parts.append(f"Response time: {self.response_time:.2f}s")

        if show_metadata and self.metadata:
            parts.append("Metadata:")
            parts.extend(f"  {k}: {v}" for k, v in self.metadata.items())
        if self.forwarded_from:
            forwarded_from = " -> ".join(self.forwarded_from)
            parts.append(f"Forwarded via: {forwarded_from}")

        return "\n".join(parts)

    def _format_markdown(self, show_metadata: bool, show_costs: bool) -> str:
        """Markdown format for rich display."""
        name = self.name or self.role.title()
        timestamp = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        parts = [f"## {name}", f"*{timestamp}*", "", str(self.content), ""]

        if show_costs and self.cost_info:
            parts.extend([
                "---",
                "**Stats:**",
                f"- Tokens: {self.cost_info.token_usage['total']:,}",
                f"- Cost: ${self.cost_info.total_cost:.4f}",
            ])
            if self.response_time:
                parts.append(f"- Response time: {self.response_time:.2f}s")

        if show_metadata and self.metadata:
            meta = yamling.dump_yaml(self.metadata)
            parts.extend(["", "**Metadata:**", "```", meta, "```"])

        if self.forwarded_from:
            parts.append(f"\n*Forwarded via: {' → '.join(self.forwarded_from)}*")

        return "\n".join(parts)


@dataclass
class Response[TContent]:
    """Response from any source in the agent system."""

    content: ChatMessage[TContent] | str
    """The actual response content (either a ChatMessage or raw text)."""

    source: str
    """Identifies where this response came from (agent/command/tool/stream)."""

    agent_name: str
    """Name of the agent that generated or handled this response."""

    timing: float | None = None
    """Time taken to generate this response in seconds."""

    error: str | None = None
    """Error message if the response generation failed."""

    @property
    def success(self) -> bool:
        """Whether the response was generated successfully."""
        return self.error is None

    @property
    def data(self) -> TContent | str:
        """Direct access to the response content data."""
        return (
            self.content.content
            if isinstance(self.content, ChatMessage)
            else self.content
        )

    def format(
        self,
        style: Literal["simple", "detailed", "markdown"] = "simple",
        *,
        include_context: bool = False,
        **kwargs: Any,
    ) -> str:
        """Format response as string with optional context info."""
        # Get base message formatting
        msg = (
            self.content.format(style, **kwargs)
            if isinstance(self.content, ChatMessage)
            else str(self.content)
        )

        if not include_context:
            return msg

        context_parts = []
        if self.error:
            context_parts.append(f"Error: {self.error}")
        else:
            context_parts.append(f"Source: {self.source}")
            if self.timing:
                context_parts.append(f"Duration: {self.timing:.2f}s")

        return f"{' | '.join(context_parts)}\n{msg}"
