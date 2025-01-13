"""Agent capabilities definition."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timedelta
import io
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from parry import LLMCallableTool, ToolError
from psygnal import EventedModel
from pydantic import ConfigDict
from pydantic_ai import RunContext  # noqa: TC002

from models.context import AgentContext  # noqa: TC001


if TYPE_CHECKING:
    from agent.agent import Agent


AccessLevel = Literal["none", "own", "all"]


class Capabilities(EventedModel):
    """Defines what operations an agent is allowed to perform."""

    # Agent discovery and delegation
    can_list_agents: bool = False
    """Whether the agent can discover other available agents."""

    can_delegate_tasks: bool = False
    """Whether the agent can delegate tasks to other agents."""

    can_observe_agents: bool = False
    """Whether the agent can monitor other agents' activities."""

    # History and statistics access
    history_access: AccessLevel = "none"
    """Level of access to conversation history."""

    stats_access: AccessLevel = "none"
    """Level of access to usage statistics."""

    # Resource capabilities
    can_load_resources: bool = False
    """Whether the agent can load and access resource content."""

    can_list_resources: bool = False
    """Whether the agent can discover available resources."""

    # Tool management
    can_register_tools: bool = False
    """Whether the agent can register importable functions as tools."""

    can_register_code: bool = False
    """Whether the agent can create new tools from provided code."""

    can_install_packages: bool = False
    """Whether the agent can install Python packages for tools."""

    can_chain_tools: bool = False
    """Whether the agent can chain multiple tool calls into one."""

    # Execution

    can_execute_code: bool = False
    """Whether the agent can execute Python code (WARNING: No sandbox)."""

    can_execute_commands: bool = False
    """Whether the agent can execute CLI commands (use at your own risk)."""

    # Agent creation
    can_create_workers: bool = False
    """Whether the agent can create worker agents (as tools)."""

    can_create_delegates: bool = False
    """Whether the agent can spawn temporary delegate agents."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    def __contains__(self, required: Capabilities) -> bool:
        """Check if these capabilities contain all required capabilities.

        Example:
            required in agent.capabilities  # Can agent fulfill requirements?
        """
        # Check all boolean capabilities
        for field in self.__fields__:
            if isinstance(getattr(required, field), bool):  # noqa: SIM102
                if getattr(required, field) and not getattr(self, field):
                    return False

        # Check access levels (none < own < all)
        access_order = {"none": 0, "own": 1, "all": 2}
        for field in ("history_access", "stats_access"):
            required_level = access_order[getattr(required, field)]
            self_level = access_order[getattr(self, field)]
            if required_level > self_level:
                return False

        return True

    def has_capability(self, capability: str) -> bool:
        """Check if a specific capability is enabled.

        Args:
            capability: Name of capability to check.
                      Can be a boolean capability (e.g., "can_delegate_tasks")
                      or an access level (e.g., "history_access")
        """
        match capability:
            case str() if hasattr(self, capability):
                value = getattr(self, capability)
                return bool(value) if isinstance(value, bool) else value != "none"
            case _:
                msg = f"Unknown capability: {capability}"
                raise ValueError(msg)

    def register_capability_tools(self, agent: Agent[Any]):
        """Register all capability-based tools."""
        # Resource tools - always register, enable based on capabilities
        agent.tools.register_tool(
            LLMCallableTool.from_callable(agent.runtime.load_resource),
            enabled=self.can_load_resources,
            source="builtin",
            requires_capability="can_load_resources",
        )
        agent.tools.register_tool(
            LLMCallableTool.from_callable(agent.runtime.get_resources),
            enabled=self.can_list_resources,
            source="builtin",
            requires_capability="can_list_resources",
        )

        # Tool management
        agent.tools.register_tool(
            LLMCallableTool.from_callable(agent.runtime.register_tool),
            enabled=self.can_register_tools,
            source="builtin",
            requires_capability="can_register_tools",
        )
        agent.tools.register_tool(
            LLMCallableTool.from_callable(agent.runtime.register_code_tool),
            enabled=self.can_register_code,
            source="builtin",
            requires_capability="can_register_code",
        )
        agent.tools.register_tool(
            LLMCallableTool.from_callable(agent.runtime.install_package),
            enabled=self.can_install_packages,
            source="builtin",
            requires_capability="can_install_packages",
        )

        # Agent discovery and delegation
        agent.tools.register_tool(
            LLMCallableTool.from_callable(self.create_worker_agent),
            enabled=self.can_create_workers,
            source="builtin",
            requires_capability="can_create_workers",
        )
        agent.tools.register_tool(
            LLMCallableTool.from_callable(self.spawn_delegate),
            enabled=self.can_create_delegates,
            source="builtin",
            requires_capability="can_create_delegates",
        )
        agent.tools.register_tool(
            LLMCallableTool.from_callable(self.list_available_agents),
            enabled=self.can_list_agents,
            source="builtin",
            requires_capability="can_list_agents",
        )
        agent.tools.register_tool(
            LLMCallableTool.from_callable(self.delegate_to),
            enabled=self.can_delegate_tasks,
            source="builtin",
            requires_capability="can_delegate_tasks",
        )
        # History tools
        agent.tools.register_tool(
            LLMCallableTool.from_callable(self.search_history),
            enabled=self.history_access != "none",
            source="builtin",
            requires_capability="history_access",
        )
        agent.tools.register_tool(
            LLMCallableTool.from_callable(self.show_statistics),
            enabled=self.stats_access != "none",
            source="builtin",
            requires_capability="stats_access",
        )

    # IMPLEMENTATIONS
    @staticmethod
    async def delegate_to(  # noqa: D417
        ctx: RunContext[AgentContext],
        agent_name: str,
        prompt: str,
    ) -> str:
        """Delegate a task. Allows to assign a task to task to a specific agent.

        If an action requires you to delegate a task, this tool can be used to assign and
        execute a task. Instructions can be passed via the prompt parameter.

        Args:
            agent_name: The agent to delegate the task to
            prompt: Instructions for the agent to delegate.

        Returns:
            The result of the task you delegated.
        """
        if not ctx.deps.pool:
            msg = "Agent needs to be in a pool to delegate tasks"
            raise ToolError(msg)
        specialist = ctx.deps.pool.get_agent(agent_name)
        result = await specialist.run(prompt)
        return str(result.data)

    @staticmethod
    async def list_available_agents(  # noqa: D417
        ctx: RunContext[AgentContext],
        only_idle: bool = False,
    ) -> list[str]:
        """List all agents available in the current pool.

        Args:
            only_idle: If True, only returns agents that aren't currently busy.
                      Use this to find agents ready for immediate tasks.

        Returns:
            List of agent names that you can use with delegate_to
        """
        if not ctx.deps.pool:
            msg = "Agent needs to be in a pool to list agents"
            raise ToolError(msg)

        agents = ctx.deps.pool.list_agents()
        if only_idle:
            return [n for n in agents if not ctx.deps.pool.get_agent(n).is_busy()]
        return agents

    @staticmethod
    async def create_worker_agent(
        ctx: RunContext[AgentContext],
        name: str,
        system_prompt: str,
        model: str | None = None,
    ) -> str:
        """Create a new agent and register it as a tool.

        The new agent will be available as a tool for delegating specific tasks.
        It inherits the current model unless overridden.
        """
        from models.agents import AgentConfig

        if not ctx.deps.pool:
            msg = "No agent pool available"
            raise ToolError(msg)
        model = model or ctx.model.name()
        config = AgentConfig(name=name, system_prompts=[system_prompt], model=model)
        worker = await ctx.deps.pool.create_agent(name, config)
        assert ctx.deps.agent
        tool_info = ctx.deps.agent.register_worker(worker)
        return f"Created worker agent and registered as tool: {tool_info.name}"

    @staticmethod
    async def spawn_delegate(
        ctx: RunContext[AgentContext],
        task: str,
        system_prompt: str,
        model: str | None = None,
        capabilities: dict[str, bool] | None = None,
        connect_back: bool = False,
    ) -> str:
        """Spawn a temporary agent for a specific task.

        Creates an ephemeral agent that will execute the task and clean up automatically
        Optionally connects back to receive results.
        """
        from models.agents import AgentConfig

        if not ctx.deps.pool:
            msg = "No agent pool available"
            raise ToolError(msg)

        name = f"delegate_{uuid4().hex[:8]}"
        model = model or ctx.model.name()
        config = AgentConfig(name=name, system_prompts=[system_prompt], model=model)

        agent = await ctx.deps.pool.create_agent(name, config, temporary=True)
        if connect_back:
            assert ctx.deps.agent
            ctx.deps.agent.pass_results_to(agent)
        await agent.run(task)
        return f"Spawned delegate {name} for task"

    @staticmethod
    async def search_history(
        ctx: RunContext[AgentContext],
        query: str | None = None,
        hours: int = 24,
        limit: int = 5,
    ) -> str:
        """Search conversation history."""
        from history.formatters import format_output
        from history.queries import get_filtered_conversations

        if ctx.deps.capabilities.history_access == "none":
            msg = "No permission to access history"
            raise ToolError(msg)

        results = get_filtered_conversations(
            query=query,
            period=f"{hours}h",
            limit=limit,
            include_tokens=True,
        )
        return format_output(results, output_format="text")

    @staticmethod
    async def show_statistics(
        ctx: RunContext[AgentContext],
        group_by: Literal["agent", "model", "hour", "day"] = "model",
        hours: int = 24,
    ) -> str:
        """Show usage statistics for conversations."""
        from history.formatters import format_output, format_stats
        from history.models import StatsFilters
        from history.stats import get_conversation_stats

        if ctx.deps.capabilities.stats_access == "none":
            msg = "No permission to view statistics"
            raise ToolError(msg)

        cutoff = datetime.now() - timedelta(hours=hours)
        filters = StatsFilters(cutoff=cutoff, group_by=group_by)
        stats = get_conversation_stats(filters)
        formatted = format_stats(stats, f"{hours}h", group_by)
        return format_output(formatted, output_format="text")

    @staticmethod
    async def execute_python(ctx: RunContext[AgentContext], code: str) -> str:
        """Execute Python code directly."""
        if not ctx.deps.capabilities.can_execute_code:
            msg = "No permission to execute code"
            raise ToolError(msg)

        try:
            # Capture output
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                exec(code, {"__builtins__": __builtins__})
                return buf.getvalue() or "Code executed successfully"
        except Exception as e:  # noqa: BLE001
            return f"Error executing code: {e}"

    @staticmethod
    async def execute_command(ctx: RunContext[AgentContext], command: str) -> str:
        """Execute a shell command."""
        if not ctx.deps.capabilities.can_execute_commands:
            msg = "No permission to execute commands"
            raise ToolError(msg)

        try:
            pipe = asyncio.subprocess.PIPE
            proc = await asyncio.create_subprocess_shell(
                command, stdout=pipe, stderr=pipe
            )
            stdout, stderr = await proc.communicate()
            return stdout.decode() or stderr.decode() or "Command completed"
        except Exception as e:  # noqa: BLE001
            return f"Error executing command: {e}"
