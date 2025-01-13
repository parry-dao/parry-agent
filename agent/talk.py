"""Agent interaction patterns."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast, overload

from pydantic import BaseModel
from toprompt import to_prompt
from typing_extensions import TypeVar

from delegation.agentgroup import Team
from delegation.controllers import interactive_controller
from delegation.pool import AgentPool
from delegation.router import CallbackRouter
from log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence

    from toprompt import AnyPromptType

    from agent import Agent, AnyAgent, StructuredAgent
    from delegation.callbacks import DecisionCallback
    from delegation.router import (
        AgentRouter,
        ChatMessage,
        Decision,
    )


logger = get_logger(__name__)
TResult = TypeVar("TResult")
TDeps = TypeVar("TDeps")


class LLMPick(BaseModel):
    """Decision format for LLM response."""

    selection: str  # The label/name of the selected option
    reason: str


class Pick[T](BaseModel):
    """Type-safe decision with original object."""

    selection: T
    reason: str


class LLMMultiPick(BaseModel):
    """Multiple selection format for LLM response."""

    selections: list[str]  # Labels of selected options
    reason: str


class MultiPick[T](BaseModel):
    """Type-safe multiple selection with original objects."""

    selections: list[T]
    reason: str


def get_label(item: Any) -> str:
    """Get label for an item to use in selection.

    Args:
        item: Item to get label for

    Returns:
        Label to use for selection

    Strategy:
        - strings stay as-is
        - types use __name__
        - others use __repr__ for unique identifiable string
    """
    from agent import Agent, StructuredAgent

    match item:
        case str():
            return item
        case type():
            return item.__name__
        case Agent() | StructuredAgent():
            return item.name or "unnamed_agent"
        case _:
            return repr(item)  # More precise than str() for identification


class Interactions[TDeps, TResult]:
    """Manages agent communication patterns."""

    def __init__(self, agent: AnyAgent[TDeps, TResult]):
        self.agent = agent

    def _resolve_agent(self, target: str | AnyAgent[TDeps, Any]) -> AnyAgent[TDeps, Any]:
        """Resolve string agent name to instance."""
        if isinstance(target, str):
            if not self.agent.context.pool:
                msg = "Pool required for resolving agent names"
                raise ValueError(msg)
            return self.agent.context.pool.get_agent(target)
        return target

    @overload
    async def ask(
        self,
        target: str | Agent[TDeps],
        message: str,
        *,
        include_history: bool = False,
        max_tokens: int | None = None,
    ) -> ChatMessage[str]: ...

    @overload
    async def ask[TOtherResult](
        self,
        target: str | StructuredAgent[TDeps, TOtherResult],
        message: TOtherResult,
        *,
        include_history: bool = False,
        max_tokens: int | None = None,
    ) -> ChatMessage[TOtherResult]: ...

    async def ask[TOtherResult](
        self,
        target: str | AnyAgent[TDeps, TOtherResult],
        message: str | TOtherResult,
        *,
        include_history: bool = False,
        max_tokens: int | None = None,
    ) -> ChatMessage[TOtherResult]:
        """Send message to another agent and wait for response."""
        target_agent = self._resolve_agent(target)

        if include_history:
            history = await self.agent.conversation.format_history(max_tokens=max_tokens)
            await target_agent.conversation.add_context_message(
                history, source=self.agent.name, metadata={"type": "conversation_history"}
            )

        return await target_agent.run(message)

    @overload
    async def controlled(
        self,
        message: str,
        decision_callback: DecisionCallback[str] = interactive_controller,
    ) -> tuple[ChatMessage[str], Decision]: ...

    @overload
    async def controlled(
        self,
        message: TResult,
        decision_callback: DecisionCallback[TResult],
    ) -> tuple[ChatMessage[TResult], Decision]: ...

    async def controlled(
        self,
        message: str | TResult,
        decision_callback: DecisionCallback[Any] = interactive_controller,
        router: AgentRouter | None = None,
    ) -> tuple[ChatMessage[Any], Decision]:
        """Get response with routing decision."""
        assert self.agent.context.pool
        router = router or CallbackRouter(self.agent.context.pool, decision_callback)

        response = await self.agent.run(message)
        decision = await router.decide(response.content)

        return response, decision

    @overload
    async def pick[T: AnyPromptType](
        self,
        selections: Sequence[T],
        task: str,
        prompt: AnyPromptType | None = None,
    ) -> Pick[T]: ...

    @overload
    async def pick[T: AnyPromptType](
        self,
        selections: Sequence[T],
        task: str,
        prompt: AnyPromptType | None = None,
    ) -> Pick[T]: ...

    @overload
    async def pick[T: AnyPromptType](
        self,
        selections: Mapping[str, T],  # Changed from dict to Mapping
        task: str,
        prompt: AnyPromptType | None = None,
    ) -> Pick[T]: ...

    @overload
    async def pick(
        self,
        selections: AgentPool,
        task: str,
        prompt: AnyPromptType | None = None,
    ) -> Pick[AnyAgent[Any, Any]]: ...

    @overload
    async def pick(
        self,
        selections: Team[TDeps],
        task: str,
        prompt: AnyPromptType | None = None,
    ) -> Pick[AnyAgent[TDeps, Any]]: ...

    async def pick[T](
        self,
        selections: Sequence[T] | Mapping[str, T] | AgentPool | Team[TDeps],
        task: str,
        prompt: AnyPromptType | None = None,
    ) -> Pick[T]:
        """Pick from available options with reasoning.

        Args:
            selections: What to pick from:
                - Sequence of items (auto-labeled)
                - Dict mapping labels to items
                - AgentPool
                - Team
            task: Task/decision description
            prompt: Optional custom selection prompt

        Returns:
            Decision with selected item and reasoning

        Raises:
            ValueError: If no choices available or invalid selection
        """
        # Get items and create label mapping
        match selections:
            case dict():
                label_map = selections
                items = list(selections.values())
            case Team():
                items = list(selections.agents)
                label_map = {get_label(item): item for item in items}
            case AgentPool():
                items = list(selections.agents.values())
                label_map = {get_label(item): item for item in items}
            case _:
                items = list(selections)
                label_map = {get_label(item): item for item in items}

        if not items:
            msg = "No choices available"
            raise ValueError(msg)

        # Get descriptions for all items
        descriptions = []
        for label, item in label_map.items():
            item_desc = await to_prompt(item)
            descriptions.append(f"{label}:\n{item_desc}")

        default_prompt = f"""Task/Decision: {task}

Available options:
{"-" * 40}
{"\n\n".join(descriptions)}
{"-" * 40}

Select ONE option by its exact label."""

        # Get LLM's string-based decision
        result = await self.agent.to_structured(LLMPick).run(prompt or default_prompt)

        # Convert to type-safe decision
        if result.content.selection not in label_map:
            msg = f"Invalid selection: {result.content.selection}"
            raise ValueError(msg)

        selected = cast(T, label_map[result.content.selection])
        return Pick(selection=selected, reason=result.content.reason)

    @overload
    async def pick_multiple[T: AnyPromptType](
        self,
        selections: Sequence[T],
        task: str,
        *,
        min_picks: int = 1,
        max_picks: int | None = None,
        prompt: AnyPromptType | None = None,
    ) -> MultiPick[T]: ...

    @overload
    async def pick_multiple[T: AnyPromptType](
        self,
        selections: Mapping[str, T],
        task: str,
        *,
        min_picks: int = 1,
        max_picks: int | None = None,
        prompt: AnyPromptType | None = None,
    ) -> MultiPick[T]: ...

    async def pick_multiple[T](
        self,
        selections: Sequence[T] | Mapping[str, T] | AgentPool | Team[TDeps],
        task: str,
        *,
        min_picks: int = 1,
        max_picks: int | None = None,
        prompt: AnyPromptType | None = None,
    ) -> MultiPick[T]:
        """Pick multiple options from available choices.

        Args:
            selections: What to pick from
            task: Task/decision description
            min_picks: Minimum number of selections required
            max_picks: Maximum number of selections (None for unlimited)
            prompt: Optional custom selection prompt
        """
        match selections:
            case Mapping():
                label_map = selections
                items = list(selections.values())
            case Team():
                items = list(selections.agents)
                label_map = {get_label(item): item for item in items}
            case AgentPool():
                items = list(selections.agents.values())
                label_map = {get_label(item): item for item in items}
            case _:
                items = list(selections)
                label_map = {get_label(item): item for item in items}

        if not items:
            msg = "No choices available"
            raise ValueError(msg)

        if max_picks is not None and max_picks < min_picks:
            msg = f"max_picks ({max_picks}) cannot be less than min_picks ({min_picks})"
            raise ValueError(msg)

        descriptions = []
        for label, item in label_map.items():
            item_desc = await to_prompt(item)
            descriptions.append(f"{label}:\n{item_desc}")

        picks_info = (
            f"Select between {min_picks} and {max_picks}"
            if max_picks is not None
            else f"Select at least {min_picks}"
        )

        default_prompt = f"""Task/Decision: {task}

Available options:
{"-" * 40}
{"\n\n".join(descriptions)}
{"-" * 40}

{picks_info} options by their exact labels.
List your selections, one per line, followed by your reasoning."""

        result = await self.agent.to_structured(LLMMultiPick).run(
            prompt or default_prompt
        )

        # Validate selections
        invalid = [s for s in result.content.selections if s not in label_map]
        if invalid:
            msg = f"Invalid selections: {', '.join(invalid)}"
            raise ValueError(msg)
        num_picks = len(result.content.selections)
        if num_picks < min_picks:
            msg = f"Too few selections: got {num_picks}, need {min_picks}"
            raise ValueError(msg)

        if max_picks and num_picks > max_picks:
            msg = f"Too many selections: got {num_picks}, max {max_picks}"
            raise ValueError(msg)

        selected = [cast(T, label_map[label]) for label in result.content.selections]
        return MultiPick(selections=selected, reason=result.content.reason)

    async def extract[T](
        self,
        text: str,
        as_type: type[T],
        prompt: AnyPromptType | None = None,
        include_tools: bool = False,
    ) -> T:
        """Extract single instance of type from text."""
        from py2openai import create_constructor_schema

        schema = create_constructor_schema(as_type).model_dump_openai()["function"]

        async def construct(**kwargs: Any) -> T:
            """Construct instance from extracted data."""
            return as_type(**kwargs)

        # Use structured agent for extraction
        structured = self.agent.to_structured(as_type)
        structured.tools.register_tool(
            construct,
            name_override=schema["name"],
            description_override=schema["description"],
        )
        final_prompt = prompt or f"Extract {as_type.__name__} from: {text}"
        with structured.tools.temporary_tools(
            construct,
            exclusive=not include_tools,
        ):
            result = await structured.run(final_prompt)
        return result.content

    async def extract_multiple[T](
        self,
        text: str,
        as_type: type[T],
        *,
        min_items: int = 1,
        max_items: int | None = None,
        prompt: AnyPromptType | None = None,
        include_tools: bool = False,
    ) -> list[T]:
        """Extract multiple instances of type from text."""
        from py2openai import create_constructor_schema

        instances: list[T] = []

        async def add_instance(**kwargs: Any) -> str:
            """Add an extracted instance."""
            instance = as_type(**kwargs)
            if max_items and len(instances) >= max_items:
                msg = f"Maximum number of items ({max_items}) reached"
                raise ValueError(msg)
            instances.append(instance)
            return f"Added {instance}"

        # Get class and init documentation for better prompting
        schema = create_constructor_schema(as_type).model_dump_openai()["function"]

        instructions = "\n".join([
            f"You are an expert at extracting {as_type.__name__} instances from text.",
            "You must:",
            f"1. Extract at least {min_items} instances",
            f"2. Extract at most {max_items} instances" if max_items else "",
            "3. Use add_instance for EACH instance found",
            "",
            f"Type information:\n{schema['description']}",
            "\nText to process:",
            text,
        ])

        structured = self.agent.to_structured(as_type)
        with structured.tools.temporary_tools(
            add_instance,
            exclusive=not include_tools,
        ):
            await structured.run(prompt or instructions)

        if len(instances) < min_items:
            msg = f"Found only {len(instances)} instances, need at least {min_items}"
            raise ValueError(msg)

        return instances
