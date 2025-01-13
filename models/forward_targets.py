"""Forward target models."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from upath import UPath


ConnectionType = Literal["run", "context", "forward"]


class ForwardTarget(BaseModel):
    """Base model for message forwarding targets."""

    type: str = Field(init=False)
    """Discriminator field for forward target types."""

    wait_for_completion: bool = Field(True)
    """Whether to wait for the result before continuing.

    If True, message processing will wait for the target to complete.
    If False, message will be forwarded asynchronously.
    """
    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


class AgentTarget(ForwardTarget):
    """Forward messages to another agent."""

    type: Literal["agent"] = Field("agent", init=False)
    """Type discriminator for agent targets."""

    name: str
    """Name of target agent."""

    connection_type: ConnectionType = "run"
    """How messages should be handled by the target agent."""

    priority: int = 0
    """Priority of the task. Lower = higher priority."""

    delay: timedelta | None = None
    """Delay before running the task."""


class FileTarget(ForwardTarget):
    """Save messages to a file."""

    type: Literal["file"] = Field("file", init=False)

    path: str
    """Path to output file. Supports variables: {date}, {time}, {agent}"""

    def resolve_path(self, context: dict[str, str]) -> UPath:
        """Resolve path template with context variables."""
        now = datetime.now()
        variables = {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H-%M-%S"),
            **context,
        }
        return UPath(self.path.format(**variables))


ForwardingTarget = Annotated[AgentTarget | FileTarget, Field(discriminator="type")]
