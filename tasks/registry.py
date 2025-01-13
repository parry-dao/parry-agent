"""Task definition and registry for agents."""

from __future__ import annotations

from typing import Any

from parry.core.baseregistry import BaseRegistry

from models.task import AgentTask
from tasks.exceptions import TaskRegistrationError


class TaskRegistry(BaseRegistry[str, AgentTask[Any, Any]]):
    """Registry for managing tasks."""

    @property
    def _error_class(self) -> type[TaskRegistrationError]:
        return TaskRegistrationError

    def _validate_item(self, item: Any) -> AgentTask[Any, Any]:
        if not isinstance(item, AgentTask):
            msg = f"Expected AgentTask, got {type(item)}"
            raise self._error_class(msg)
        return item

    def register(self, name: str, task: AgentTask[Any, Any], replace: bool = False):
        """Register a task with name.

        Creates a copy of the task with the name set.
        """
        task_copy = task.model_copy(update={"name": name})
        super().register(name, task_copy, replace)
