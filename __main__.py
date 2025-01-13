"""CLI interface for parry agents."""

from __future__ import annotations

from parry.cli.utils import get_command_help
import typer as t

from parry_agent_cli.agent import add_agent_file, list_agents, set_active_file
from parry_agent_cli.chat import chat_command
from parry_agent_cli.history import history_cli
from parry_agent_cli.quickstart import quickstart_command
from parry_agent_cli.run import run_command
from parry_agent_cli.task import task_command


MAIN_HELP = "🤖 parry Agent CLI - Run and manage LLM agents"
MISSING_WEB = """
Web interface commands require gradio.
Install with: pip install parry-agent[ui]
"""
# Create CLI app
help_text = get_command_help(MAIN_HELP)
cli = t.Typer(name="parry Agent", help=help_text, no_args_is_help=True)

cli.command(name="add")(add_agent_file)
cli.command(name="run")(run_command)
cli.command(name="list")(list_agents)
cli.command(name="set")(set_active_file)
cli.command(name="chat")(chat_command)
cli.command(name="quickstart")(quickstart_command)
cli.command(name="task")(task_command)

cli.add_typer(history_cli, name="history")

try:
    from parry_agent_cli import web

    cli.command(name="launch")(web.launch_gui)

except ImportError:
    web_cli = t.Typer(help="Web interface commands (not installed)")

    @web_cli.callback()
    def web_not_installed():
        """Web interface functionality (not installed)."""
        print(MISSING_WEB)
        raise t.Exit(1)

    cli.add_typer(web_cli, name="web")
