"""Task execution strategies."""

from __future__ import annotations

import abc
from collections.abc import Awaitable, Callable
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypeVar

from agent.agent import Agent
from models.messages import ChatMessage
from models.task import AgentTask


TResult = TypeVar("TResult")
TDeps = TypeVar("TDeps")

ExecuteFunction = Callable[
    [AgentTask[TDeps, TResult], Agent[TDeps]],
    ChatMessage[TResult] | Awaitable[ChatMessage[TResult]],
]


class TaskStrategy[TDeps, TResult](BaseModel, abc.ABC):
    """Base class for task execution strategies."""

    type: str = Field(init=False)
    """Discriminator field for strategy types."""

    model_config = ConfigDict(frozen=True)

    @abc.abstractmethod
    async def execute(
        self,
        task: AgentTask[TDeps, TResult],
        agent: Agent[TDeps],
    ) -> ChatMessage[TResult]:
        """Execute task according to strategy."""


class DirectStrategy[TDeps, TResult](TaskStrategy[TDeps, TResult]):
    """Execute task prompt directly."""

    type: Literal["direct"] = Field("direct", init=False)

    async def execute(
        self,
        task: AgentTask[TDeps, TResult],
        agent: Agent[TDeps],
        store_history: bool = True,
    ) -> ChatMessage[TResult]:
        """Direct execution of task prompt."""
        return await agent.run(task.prompt, store_history=store_history)


class StepByStepStrategy[TDeps, TResult](TaskStrategy[TDeps, TResult]):
    """Break task into sequential steps."""

    type: Literal["step_by_step"] = Field("step_by_step", init=False)
    max_steps: int = Field(default=5, gt=0)
    combine_results: bool = True

    planning_prompt_template: str = Field(
        "Break this task into sequential steps:\n{prompt}\n\nMaximum steps: {max_steps}"
    )

    combination_prompt_template: str = Field(
        "Combine these intermediate results into a final output:\n"
        "Step Results:\n{results}\n\n"
        "Original Task:\n{prompt}"
    )

    async def execute(
        self,
        task: AgentTask[TDeps, TResult],
        agent: Agent[TDeps],
        store_history: bool = True,
    ) -> ChatMessage[TResult]:
        # Get step breakdown using configurable prompt
        planning_prompt = self.planning_prompt_template.format(
            prompt=task.prompt, max_steps=self.max_steps
        )
        msg = await agent.run(
            planning_prompt,
            result_type=list[str],
            store_history=store_history,
        )

        # Execute each step
        results = []
        for step in msg.content:
            step_result = await agent.run(step)
            results.append(step_result)

        if not self.combine_results:
            return cast(ChatMessage[TResult], results[-1])

        # Combine results using configurable prompt
        combine_prompt = self.combination_prompt_template.format(
            results="\n".join(str(r) for r in results), prompt=task.prompt
        )
        return await agent.run(combine_prompt, store_history=store_history)


class CustomStrategy[TDeps, TResult](TaskStrategy[TDeps, TResult]):
    """Execute task using imported function."""

    type: Literal["custom"] = Field("custom", init=False)
    execute_path: str = Field(description="Import path to execution function")

    async def execute(
        self,
        task: AgentTask[TDeps, TResult],
        agent: Agent[TDeps],
        store_history: bool = True,
    ) -> ChatMessage[TResult]:
        """Execute using imported function."""
        from parry.utils.importing import import_callable

        func = import_callable(self.execute_path)
        result = func(task, agent)

        if isinstance(result, Awaitable):
            return await result
        return result


class ResearchStrategy[TDeps, TResult](TaskStrategy[TDeps, TResult]):
    """Research first, then execute."""

    type: Literal["research"] = Field("research", init=False)
    research_prompt_template: str = Field(
        "Research task requirements using provided sources:\n{prompt}"
    )
    execution_prompt_template: str = Field(
        "Execute task with research context:\n{research}\n\nTask:\n{prompt}"
    )

    async def execute(
        self,
        task: AgentTask[TDeps, TResult],
        agent: Agent[TDeps],
        store_history: bool = True,
    ) -> ChatMessage[TResult]:
        # Research phase
        research_prompt = self.research_prompt_template.format(prompt=task.prompt)
        research = await agent.run(research_prompt)
        # Execution phase
        execution_prompt = self.execution_prompt_template.format(
            research=research.content,
            prompt=task.prompt,
        )
        return await agent.run(execution_prompt, store_history=store_history)


# Register all strategies in type union
StrategyUnion = Annotated[
    DirectStrategy[TDeps, TResult]
    | StepByStepStrategy[TDeps, TResult]
    | ResearchStrategy[TDeps, TResult]
    | CustomStrategy[TDeps, TResult],
    Field(discriminator="type"),
]
