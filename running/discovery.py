"""Function discovery and metadata handling."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import TYPE_CHECKING, Any

from log import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = get_logger(__name__)


@dataclass
class AgentFunction:
    """Metadata for a function that uses agents."""

    func: Callable[..., Any]
    """The actual function to execute."""

    depends_on: list[str] = field(default_factory=list)
    """Names of functions this one depends on."""

    default_inputs: dict[str, Any] = field(default_factory=dict)
    """Default parameter values."""

    name: str = field(init=False)
    """Function name (from function.__name__)."""

    def __post_init__(self):
        """Set name and validate dependencies."""
        self.name = self.func.__name__
        # Extract default inputs from function signature

        sig = inspect.signature(self.func)
        self.default_inputs = {
            name: param.default
            for name, param in sig.parameters.items()
            if param.default is not param.empty
        }
        msg = "Registered agent function %s (deps=%s)"
        logger.debug(msg, self.name, self.depends_on)


def agent_function(
    func: Callable | None = None,
    *,
    depends_on: str | Sequence[str] | None = None,
) -> Callable:
    """Mark a function for automatic agent execution.

    Can be used as simple decorator or with arguments:

    @agent_function
    async def func(): ...

    @agent_function(order=1, depends_on="other_func")
    async def func(): ...

    Args:
        func: Function to mark
        depends_on: Names of functions this one depends on

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        deps = [depends_on] if isinstance(depends_on, str) else list(depends_on or [])
        metadata = AgentFunction(func=func, depends_on=deps)
        func._agent_function = metadata  # type: ignore
        return func

    return decorator(func) if func is not None else decorator
