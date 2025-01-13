from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import App
from textual.widgets import Input, RichLog

from delegation.commands import ListAgentsCommand
from delegation.supervisor import (
    InputHandler,
    OutputHandler,
    PoolSupervisor,
)
from log import get_logger


if TYPE_CHECKING:
    from delegation.pool import AgentPool


logger = get_logger(__name__)


class TextualInputHandler(InputHandler):
    """Input handler using Textual Input widget."""

    def __init__(self, app: SupervisorApp):
        self.app = app

    async def get_input(self, prompt: str) -> str:
        """Get input from Textual widget."""
        return self.app.input.value


class TextualOutputHandler(OutputHandler):
    """Output handler using Textual RichLog."""

    def __init__(self, app: SupervisorApp):
        self.app = app

    async def display(self, message: str):
        """Display message in RichLog."""
        self.app.call_later(self.app.output.write, message)


class SupervisorApp(App):
    """Basic supervision interface."""

    CSS = """
    RichLog { height: 1fr; }
    Input { dock: bottom; }
    """

    def __init__(self, pool: AgentPool):
        super().__init__()
        self.pool = pool
        self.supervisor: PoolSupervisor | None = None

    def compose(self):
        """Create and yield widgets."""
        self.output = RichLog()
        self.input = Input(placeholder="@agent message or /command")
        yield self.output
        yield self.input

    async def on_mount(self):
        """Start supervision when app mounts."""
        # Create supervisor with proper handlers
        self.supervisor = PoolSupervisor(
            pool=self.pool,
            input_handler=TextualInputHandler(self),
            output_handler=TextualOutputHandler(self),
        )
        # Register commands
        self.supervisor.commands.register_command(ListAgentsCommand())
        await self.supervisor.start()

    async def on_input_submitted(self, event: Input.Submitted):
        """Process submitted input."""
        if not self.supervisor:
            return

        text = event.value.strip()
        if not text:
            return

        if text.lower() == "exit":
            self.exit()
        else:
            await self.supervisor._handle_command(text)
        self.input.value = ""

    async def on_unmount(self):
        """Clean up when app exits."""
        if self.supervisor:
            await self.supervisor.stop()


if __name__ == "__main__":
    import asyncio
    import sys

    from delegation.pool import AgentPool
    from models.agents import AgentConfig

    async def main():
        # Create a pool with some example agents
        async with AgentPool.open() as pool:
            # Create a couple of agents
            analyzer = await pool.create_agent(
                "analyzer",
                AgentConfig(
                    name="analyzer",
                    model="openai:gpt-4o-mini",
                    system_prompts=["You are a pirate."],
                ),
            )

            _planner = await pool.create_agent(
                "planner",
                AgentConfig(
                    name="planner",
                    model="openai:gpt-4o-mini",
                    system_prompts=["You are a hippie."],
                ),
            )
            await analyzer.run_continuous("Tell a joke", max_count=5, interval=10)
            pool.start_supervision()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSupervision ended.")
    except Exception:
        logger.exception("Supervision failed")
        sys.exit(1)
