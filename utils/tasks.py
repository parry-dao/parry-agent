from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import heapq
from typing import TYPE_CHECKING, Any, TypeVar

from log import get_logger


if TYPE_CHECKING:
    from collections.abc import Coroutine


T = TypeVar("T")
logger = get_logger(__name__)


@dataclass(order=True)
class PrioritizedTask:
    """Task with priority and optional delay."""

    priority: int
    execute_at: datetime
    coroutine: Coroutine[Any, Any, Any] = field(compare=False)
    name: str | None = field(default=None, compare=False)


class TaskManagerMixin:
    """Mixin for managing async tasks.

    Provides utilities for:
    - Creating and tracking tasks
    - Fire-and-forget task execution
    - Running coroutines in sync context
    - Cleanup of pending tasks
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._pending_tasks: set[asyncio.Task[Any]] = set()
        self._task_queue: list[PrioritizedTask] = []  # heap queue
        self._scheduler_task: asyncio.Task[Any] | None = None

    def create_task(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        name: str | None = None,
        priority: int = 0,
        delay: timedelta | None = None,
    ) -> asyncio.Task[T]:
        """Create and track a new task with optional priority and delay.

        Args:
            coro: Coroutine to run
            name: Optional name for the task
            priority: Priority (lower = higher priority, default 0)
            delay: Optional delay before execution
        """
        task = asyncio.create_task(coro, name=name)
        logger.debug(
            "Created task: %s (priority=%d, delay=%s)", task.get_name(), priority, delay
        )

        def _done_callback(t: asyncio.Task[Any]) -> None:
            logger.debug("Task completed: %s", t.get_name())
            self._pending_tasks.discard(t)
            if t.exception():
                logger.error("Task failed with error: %s", t.exception())

        task.add_done_callback(_done_callback)
        self._pending_tasks.add(task)

        if delay is not None:
            execute_at = datetime.now() + delay
            # Store the coroutine instead of the task
            heapq.heappush(
                self._task_queue, PrioritizedTask(priority, execute_at, coro, name)
            )
            # Start scheduler if not running
            if not self._scheduler_task:
                self._scheduler_task = asyncio.create_task(self._run_scheduler())
            # Cancel the original task since we'll run it later
            task.cancel()
            return task

        return task

    async def _run_scheduler(self):
        """Run scheduled tasks when their time comes."""
        try:
            while self._task_queue:
                # Get next task without removing
                next_task = self._task_queue[0]
                now = datetime.now()

                if now >= next_task.execute_at:
                    # Remove and execute
                    heapq.heappop(self._task_queue)
                    # Create new task from stored coroutine
                    new_task = asyncio.create_task(
                        next_task.coroutine,
                        name=next_task.name,
                    )
                    self._pending_tasks.add(new_task)
                    new_task.add_done_callback(self._pending_tasks.discard)
                else:
                    # Wait until next task is due
                    await asyncio.sleep((next_task.execute_at - now).total_seconds())

        except Exception:
            logger.exception("Task scheduler error")
        finally:
            self._scheduler_task = None

    def fire_and_forget(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Run coroutine without waiting for result."""
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(coro)
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)
        except RuntimeError:
            # No running loop - use new loop
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(coro)
            finally:
                loop.close()

    def run_task_sync(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(coro)
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)
            return loop.run_until_complete(task)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

    def run_background(
        self,
        coro: Coroutine[Any, Any, Any],
        name: str | None = None,
        priority: int = 0,
        delay: timedelta | None = None,
    ) -> None:
        """Run a coroutine in the background and track it."""
        try:
            self.create_task(coro, name=name, priority=priority, delay=delay)

        except RuntimeError:
            # No running loop - use fire_and_forget
            self.fire_and_forget(coro)

    async def cleanup_tasks(self) -> None:
        """Wait for all pending tasks to complete."""
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
        self._pending_tasks.clear()

    async def complete_tasks(self, cancel: bool = False):
        """Wait for all pending tasks to complete."""
        if cancel:
            for task in self._pending_tasks:
                task.cancel()
        if self._pending_tasks:
            await asyncio.wait(self._pending_tasks)
