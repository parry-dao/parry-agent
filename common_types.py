from __future__ import annotations

from collections.abc import Awaitable, Callable
import os
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import UUID

from parry import LLMCallableTool
from typing_extensions import TypeVar


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for model objects."""

    def name(self) -> str: ...


# Define what we consider JSON-serializable
type JsonPrimitive = None | bool | int | float | str
type JsonValue = JsonPrimitive | JsonArray | JsonObject
type JsonObject = dict[str, JsonValue]
type JsonArray = list[JsonValue]

type ToolType = str | Callable[..., Any] | LLMCallableTool

type StrPath = str | os.PathLike[str]
type SessionIdType = str | UUID | None
MessageRole = Literal["user", "assistant", "system"]
PartType = Literal["text", "image", "audio", "video"]
ModelType = ModelProtocol | str | None
EnvironmentType = Literal["file", "inline"]
ToolSource = Literal["runtime", "agent", "builtin", "dynamic", "task", "mcp"]

T = TypeVar("T")
type OptionalAwaitable[T] = T | Awaitable[T]

# P = ParamSpec("P")
# SyncAsync = Callable[P, OptionalAwaitable[T]]
