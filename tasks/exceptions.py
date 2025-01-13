from __future__ import annotations

from parry import parryError


class TaskError(parryError):
    """General task-related exception."""


class ToolSkippedError(TaskError):
    """Tool execution was skipped by user."""


class RunAbortedError(TaskError):
    """Run was aborted by user."""


class ChainAbortedError(TaskError):
    """Agent chain was aborted by user."""


class TaskRegistrationError(TaskError):
    """Task could not get registered."""
