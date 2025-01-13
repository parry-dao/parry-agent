"""Base types for delegation functionality."""

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from typing_extensions import TypeVar

from delegation.router import AgentRouter, Decision


if TYPE_CHECKING:
    from delegation.pool import AgentPool


TMessage = TypeVar("TMessage", default=str)
DecisionCallback = Callable[
    [TMessage, "AgentPool", AgentRouter], Decision | Awaitable[Decision]
]
