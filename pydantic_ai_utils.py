"""Utilities for working with pydantic-ai types and objects."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai import messages as _messages, models
from pydantic_ai.messages import (
    ArgsDict,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from upath import UPath

from log import get_logger
from models.agents import ToolCallInfo
from models.messages import ChatMessage


if TYPE_CHECKING:
    from collections.abc import Sequence

    from common_types import MessageRole
    from storage.models import Message
    from tools.base import ToolInfo


logger = get_logger(__name__)

# Type definitions
type ContentType = Literal["text", "image", "audio", "video"]
type ContentSource = str | bytes | Path | Any


def to_base64(data: bytes) -> str:
    """Convert bytes to base64 string."""
    return base64.b64encode(data).decode()


def format_part(  # noqa: PLR0911
    response: str | _messages.ModelRequestPart | _messages.ModelResponsePart,
) -> str:
    """Format any kind of response in a readable way.

    Args:
        response: Response part to format

    Returns:
        A human-readable string representation
    """
    match response:
        case str():
            return response
        case _messages.TextPart():
            return response.content
        case _messages.ToolCallPart():
            if isinstance(response.args, _messages.ArgsJson):
                args = response.args.args_json
            else:
                args = str(response.args.args_dict)
            return f"Tool call: {response.tool_name}\nArgs: {args}"
        case _messages.ToolReturnPart():
            return f"Tool {response.tool_name} returned: {response.content}"
        case _messages.RetryPromptPart():
            if isinstance(response.content, str):
                return f"Retry needed: {response.content}"
            return f"Validation errors:\n{response.content}"
        case _:
            return str(response)


def find_last_assistant_message(messages: Sequence[ModelMessage]) -> str | None:
    """Find the last assistant message in history."""
    for msg in reversed(messages):
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                match part:
                    case TextPart():
                        return part.content
                    case ToolCallPart() as tool_call:
                        # Format tool calls nicely
                        args = (
                            tool_call.args.args_dict
                            if isinstance(tool_call.args, ArgsDict)
                            else json.loads(tool_call.args.args_json)
                        )
                        return f"Tool: {tool_call.tool_name}\nArgs: {args}"
    return None


def get_tool_calls(
    messages: list[ModelMessage],
    tools: dict[str, ToolInfo] | None = None,
    context_data: Any | None = None,
) -> list[ToolCallInfo]:
    """Extract tool call information from messages.

    Args:
        messages: Messages from captured run
        tools: Original ToolInfo set to enrich ToolCallInfos with additional info
        context_data: Optional context data to attach to tool calls
    """
    tools = tools or {}
    parts = [part for message in messages for part in message.parts]
    call_parts = {
        part.tool_call_id: part
        for part in parts
        if isinstance(part, ToolCallPart) and part.tool_call_id
    }
    return [
        parts_to_tool_call_info(
            call_parts[part.tool_call_id], part, tools.get(part.tool_name), context_data
        )
        for part in parts
        if isinstance(part, ToolReturnPart) and part.tool_call_id in call_parts
    ]


def parts_to_tool_call_info(
    call_part: ToolCallPart,
    return_part: ToolReturnPart,
    tool_info: ToolInfo | None,
    context_data: Any | None = None,
) -> ToolCallInfo:
    """Convert matching tool call and return parts into a ToolCallInfo."""
    args = (
        call_part.args.args_dict
        if isinstance(call_part.args, ArgsDict)
        else json.loads(call_part.args.args_json)
    )

    return ToolCallInfo(
        tool_name=call_part.tool_name,
        args=args,
        result=return_part.content,
        tool_call_id=call_part.tool_call_id,
        timestamp=return_part.timestamp,
        context_data=context_data,
        agent_tool_name=tool_info.agent_name if tool_info else None,
    )


def convert_model_message(
    message: ModelMessage | ModelRequestPart | ModelResponsePart,
    tools: dict[str, ToolInfo] | None = None,
) -> ChatMessage:
    """Convert a pydantic-ai message to our ChatMessage format.

    Also supports converting parts of a message (with limitations then of course)

    Args:
        message: Message to convert (ModelMessage or its parts)
        tools: Original ToolInfo set to enrich ToolCallInfos with additional info

    Returns:
        Converted ChatMessage

    Raises:
        ValueError: If message type is not supported
    """
    match message:
        case ModelRequest():
            # Collect content from all parts
            content_parts = []
            role: MessageRole = "system"
            for part in message.parts:
                match part:
                    case UserPromptPart():
                        content_parts.append(str(part.content))
                        role = "user"
                    case SystemPromptPart():
                        content_parts.append(str(part.content))
            return ChatMessage(content="\n".join(content_parts), role=role)

        case ModelResponse():
            # Collect content and tool calls from all parts
            tool_calls = get_tool_calls([message], tools, None)
            parts = [format_part(p) for p in message.parts if isinstance(p, TextPart)]
            content = "\n".join(parts)
            return ChatMessage(content=content, role="assistant", tool_calls=tool_calls)

        case TextPart() | UserPromptPart() | SystemPromptPart() as part:
            role = "assistant" if isinstance(part, TextPart) else "user"
            return ChatMessage(content=format_part(part), role=role)

        case ToolCallPart():
            args = (
                message.args.args_dict
                if isinstance(message.args, ArgsDict)
                else json.loads(message.args.args_json)
            )
            info = ToolCallInfo(
                tool_name=message.tool_name,
                args=args,
                result=None,  # Not available yet
                tool_call_id=message.tool_call_id,
            )
            content = f"Tool call: {message.tool_name}\nArgs: {args}"
            return ChatMessage(content=content, role="assistant", tool_calls=[info])

        case ToolReturnPart():
            info = ToolCallInfo(
                tool_name=message.tool_name,
                args={},  # No args in return part
                result=message.content,
                tool_call_id=message.tool_call_id,
                timestamp=message.timestamp,
            )
            content = f"Tool {message.tool_name} returned: {message.content}"
            return ChatMessage(content=content, role="assistant", tool_calls=[info])

        case RetryPromptPart():
            error_content = (
                message.content
                if isinstance(message.content, str)
                else "\n".join(
                    f"- {error['loc']}: {error['msg']}" for error in message.content
                )
            )
            return ChatMessage(content=f"Retry needed: {error_content}", role="assistant")

        case _:
            msg = f"Unsupported message type: {type(message)}"
            raise ValueError(msg)


def models_equal(
    self, a: str | models.Model | None, b: str | models.Model | None
) -> bool:
    """Compare models by their string representation."""
    match (a, b):
        case (None, None):
            return True
        case (None, _) | (_, None):
            return False
        case _:
            # Compare string representations (using model.name() for Model objects)
            name_a = a if isinstance(a, str | None) else a.name()
            name_b = b if isinstance(b, str | None) else b.name()
            return name_a == name_b


def get_message_role(msg: ModelMessage) -> str:
    """Get appropriate name for message source."""
    match msg:
        case ModelRequest():
            return "User"
        case ModelResponse():
            return "Assistant"
        case SystemPromptPart():
            return "System"
        case _:
            return "Unknown"


def create_message(
    contents: list[tuple[ContentType, ContentSource]] | str,
    role: MessageRole = "user",
) -> ModelMessage:
    """Create a message from content pairs.

    For multi-modal content, creates a JSON string that models like GPT-4V
    can interpret. For simple text, creates a plain text message.
    """
    # Handle simple text case
    if isinstance(contents, str):
        part = (
            UserPromptPart(content=contents)
            if role == "user"
            else SystemPromptPart(content=contents)
        )
        return ModelRequest(parts=[part])

    # For multi-modal, convert to a JSON string
    content_list = []
    for type_, content in contents:
        match type_:
            case "text":
                content_list.append({"type": "text", "text": str(content)})
            case "image":
                url = prepare_image_url(content)
                content_list.append({"type": "image", "url": url})
            case "audio":
                url = prepare_audio_url(content)
                content_list.append({"type": "audio", "url": url})
            case _:
                msg = f"Unsupported content type: {type_}"
                raise ValueError(msg)

    # Convert to JSON string and create appropriate message part
    content_str = json.dumps({"content": content_list})
    return ModelRequest(
        parts=[
            UserPromptPart(content=content_str)
            if role == "user"
            else SystemPromptPart(content=content_str)
        ]
    )


def prepare_image_url(content: ContentSource) -> str:
    """Convert image content to URL or data URL."""
    match content:
        case str() if content.startswith(("http://", "https://")):
            return content
        case str() | os.PathLike():
            # Read file and convert to data URL
            path = UPath(content)
            content_b64 = to_base64(path.read_bytes())
            return f"data:image/png;base64,{content_b64}"
        case bytes():
            content_b64 = to_base64(content)
            return f"data:image/png;base64,{content_b64}"
        case _:
            msg = f"Unsupported image content type: {type(content)}"
            raise ValueError(msg)


def prepare_audio_url(content: ContentSource) -> str:
    """Convert audio content to URL or data URL.

    Supports common audio formats (mp3, wav, ogg, m4a).
    Uses content-type detection when possible.
    """
    import mimetypes

    def get_audio_mime(path: str | Path) -> str:
        """Get MIME type for audio file."""
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type or not mime_type.startswith("audio/"):
            # Default to mp3 if we can't detect or it's not audio
            return "audio/mpeg"
        return mime_type

    match content:
        case str() if content.startswith(("http://", "https://")):
            return content
        case str() | os.PathLike():
            path = UPath(content)
            content_b64 = to_base64(path.read_bytes())
            mime_type = get_audio_mime(path)
            return f"data:{mime_type};base64,{content_b64}"
        case bytes():
            # For raw bytes, default to mp3 as it's most common
            content_b64 = to_base64(content)
            return f"data:audio/mpeg;base64,{content_b64}"
        case _:
            msg = f"Unsupported audio content type: {type(content)}"
            raise ValueError(msg)


def db_message_to_pydantic_ai_message(msg: Message) -> ModelMessage:
    """Convert a database message to a pydantic-ai message."""
    match msg.role:
        case "user":
            return ModelRequest(parts=[UserPromptPart(content=msg.content)])
        case "assistant":
            return ModelResponse(parts=[TextPart(content=msg.content)])
        case "system":
            return ModelRequest(parts=[SystemPromptPart(content=msg.content)])
    error_msg = f"Unknown message role: {msg.role}"
    raise ValueError(error_msg)
