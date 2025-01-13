"""Agent injection utilities."""

from __future__ import annotations

import inspect
import typing
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from agent import Agent
from log import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable

    from delegation.pool import AgentPool

P = ParamSpec("P")
T = TypeVar("T")

logger = get_logger(__name__)


class AgentInjectionError(Exception):
    """Raised when agent injection fails."""


def inject_agents(
    func: Callable[P, T],
    pool: AgentPool,
    provided_kwargs: dict[str, Any],
) -> dict[str, Agent[Any]]:
    """Get agents to inject based on function signature."""
    hints = typing.get_type_hints(func)
    params = inspect.signature(func).parameters
    msg = "Injecting agents for %s.%s"
    logger.debug(msg, func.__module__, func.__qualname__)
    logger.debug("Type hints: %s", hints)
    logger.debug("Available agents in pool: %s", sorted(pool.agents))

    agents: dict[str, Agent[Any]] = {}
    for name, param in params.items():
        # Only look at normal keyword/positional params
        if param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            logger.debug("Skipping %s: wrong parameter kind %s", name, param.kind)
            continue

        # Check if parameter should be an agent
        hint = hints.get(name)
        if hint is None:
            logger.debug("Skipping %s: no type hint", name)
            continue

        # Handle both Agent and Agent[Any] | None
        origin = getattr(hint, "__origin__", None)
        args = getattr(hint, "__args__", ())

        # Check various Agent type patterns
        is_agent = (
            hint is Agent  # Direct Agent
            or origin is Agent  # Generic Agent[T]
            or (  # Optional[Agent[T]] or Union containing Agent
                origin is not None
                and any(
                    arg is Agent or getattr(arg, "__origin__", None) is Agent
                    for arg in args
                )
            )
        )

        if not is_agent:
            msg = "Skipping %s: not an agent type (hint=%s, origin=%s, args=%s)"
            logger.debug(msg, name, hint, origin, args)
            continue

        logger.debug("Found agent parameter: %s", name)

        # Check for duplicate parameters
        if name in provided_kwargs and provided_kwargs[name] is not None:
            msg = (
                f"Cannot inject agent '{name}': Parameter already provided.\n"
                f"Remove the explicit argument or rename the parameter."
            )
            logger.error(msg)
            raise AgentInjectionError(msg)

        # Get agent from pool
        if name not in pool.agents:
            available = ", ".join(sorted(pool.agents))
            msg = (
                f"No agent named '{name}' found in pool.\n"
                f"Available agents: {available}\n"
                f"Check your YAML configuration or agent name."
            )
            logger.error(msg)
            raise AgentInjectionError(msg)

        agents[name] = pool.get_agent(name)
        logger.debug("Injecting agent %s for parameter %s", agents[name], name)

    logger.debug("Injection complete. Injected agents: %s", sorted(agents))
    return agents
