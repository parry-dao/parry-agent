"""Automatic agent function execution."""

from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING

from delegation import AgentPool
from running.discovery import agent_function
from running.executor import discover_functions, execute_functions

if TYPE_CHECKING:
    from models.agents import AgentsManifest
    from common_types import StrPath


async def run_agents_async(
    config: StrPath | AgentsManifest,
    *,
    module: str | None = None,
    functions: list[str] | None = None,
    inputs: dict[str, Any] | None = None,
    parallel: bool = False,
) -> dict[str, Any]:
    """Run agent functions asynchronously."""
    # Find functions to run
    if module:
        discovered = discover_functions(module)
    else:
        # Use calling module
        import inspect

        frame = inspect.currentframe()
        while frame:
            if frame.f_globals.get("__name__") != __name__:
                break
            frame = frame.f_back
        if not frame:
            msg = "Could not determine calling module"
            raise RuntimeError(msg)
        discovered = discover_functions(frame.f_globals["__file__"])

    if functions:
        discovered = [f for f in discovered if f.name in functions]

    # Run with pool
    async with AgentPool.open(config) as pool:
        return await execute_functions(
            discovered,
            pool,
            inputs=inputs,
            parallel=parallel,
        )


def run_agents(
    config: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run agent functions (sync version)."""
    return asyncio.run(run_agents_async(config, **kwargs))


__all__ = ["agent_function", "run_agents", "run_agents_async"]
