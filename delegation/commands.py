from __future__ import annotations

from typing import TYPE_CHECKING

from slashed import CommandContext, SlashedCommand

from log import get_logger


if TYPE_CHECKING:
    from delegation.supervisor import PoolSupervisor


logger = get_logger(__name__)


class ListAgentsCommand(SlashedCommand):
    """List all agents in the pool with their status."""

    name = "list-agents"
    category = "pool"

    async def execute_command(
        self,
        ctx: CommandContext[PoolSupervisor],
        show_connections: bool = False,
    ):
        """List all agents and their current status.

        Args:
            ctx: Command context with supervisor
            show_connections: Whether to show agent connections
        """
        supervisor = ctx.get_data()
        header = "\nAvailable Agents:"
        await ctx.output.print(header)
        await ctx.output.print("=" * len(header))

        for name in supervisor.pool.list_agents():
            agent = supervisor.pool.get_agent(name)

            # Build status info
            status = "🔄 busy" if agent.is_busy() else "⏳ idle"

            # Add connections if requested
            connections = []
            if show_connections and agent.connections.get_targets():
                connections = [a.name for a in agent.connections.get_targets()]
                conn_str = f" → {', '.join(connections)}"
            else:
                conn_str = ""

            # Add description if available
            desc = f" - {agent.description}" if agent.description else ""

            await ctx.output.print(f"{name} ({status}){conn_str}{desc}")
