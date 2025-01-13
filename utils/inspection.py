from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable

    from models import AgentContext


T = TypeVar("T")


def has_argument_type(func: Callable[..., Any], arg_type: str | type) -> bool:
    """Checks whether any argument of func is of type arg_type."""
    sig = inspect.signature(func)
    arg_str = arg_type if isinstance(arg_type, str) else arg_type.__name__
    return any(arg_str in str(param.annotation) for param in sig.parameters.values())


def call_with_context(
    func: Callable[..., T],
    context: AgentContext[Any],
    **kwargs: Any,
) -> T:
    """Call function with appropriate context injection.

    Handles:
    - Simple functions
    - Bound methods
    - Functions expecting AgentContext
    - Functions expecting context data
    """
    from models import AgentContext

    if inspect.ismethod(func):
        if has_argument_type(func, AgentContext):
            return func(context)
        return func()
    if has_argument_type(func, AgentContext):
        return func(context, **kwargs)
    return func(context.data)
