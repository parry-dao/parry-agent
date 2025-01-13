"""Runtime context models for Agents."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import cached_property
import json
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai import RunContext
from typing_extensions import TypeVar

from tools.base import ToolInfo


if TYPE_CHECKING:
    from parry import RuntimeConfig

    from agent.agent import Agent
    from config.capabilities import Capabilities
    from delegation.pool import AgentPool
    from models.agents import AgentConfig, AgentsManifest
    from storage.manager import StorageManager


TDeps = TypeVar("TDeps", default=Any)


@dataclass
class AgentContext[TDeps]:
    """Runtime context for agent execution.

    Generically typed with AgentContext[Type of Dependencies]
    """

    agent_name: str
    """Name of the current agent."""

    capabilities: Capabilities
    """Current agent's capabilities."""

    definition: AgentsManifest
    """Complete agent definition with all configurations."""

    config: AgentConfig
    """Current agent's specific configuration."""

    current_prompt: str | None = None
    """Current prompt text for the agent."""

    model_settings: dict[str, Any] = field(default_factory=dict)
    """Model-specific settings."""

    data: TDeps | None = None
    """Custom context data."""

    runtime: RuntimeConfig | None = None
    """Reference to the runtime configuration."""

    pool: AgentPool | None = None
    """Pool the agent is part of."""

    confirmation_callback: ConfirmationCallback | None = None
    """Optional confirmation handler for tool execution."""

    in_async_context: bool = False
    """Whether we're running in an async context."""

    @classmethod
    def create_default(
        cls,
        name: str,
        capabilities: Capabilities | None = None,
    ) -> AgentContext:
        """Create a default agent context with minimal privileges.

        Args:
            name: Name of the agent
            capabilities: Optional custom capabilities (defaults to minimal access)
        """
        from config.capabilities import Capabilities
        from models import AgentConfig, AgentsManifest

        caps = capabilities or Capabilities()
        defn = AgentsManifest[Any, Any](agents={})
        cfg = AgentConfig(name=name)
        return cls(agent_name=name, capabilities=caps, definition=defn, config=cfg)

    # TODO: perhaps add agent directly to context?
    @property
    def agent(self) -> Agent[TDeps] | None:
        """Get the agent instance from the pool."""
        if not self.pool or not self.agent_name:
            return None
        return self.pool.get_agent(self.agent_name)

    @cached_property
    def storage(self) -> StorageManager:
        """Storage manager from pool or config."""
        from storage.manager import StorageManager

        if self.pool:
            return self.pool.storage
        return StorageManager(self.definition.storage)

    async def handle_confirmation(
        self,
        ctx: RunContext[AgentContext] | AgentContext,
        tool: ToolInfo,
        args: dict[str, Any],
    ) -> ConfirmationResult:
        """Handle tool execution confirmation.

        Returns True if:
        - No confirmation handler is set
        - Handler confirms the execution
        """
        if not self.confirmation_callback:
            return "allow"
        result = self.confirmation_callback(ctx, tool, args)
        if isinstance(result, str):
            return result
        return await result


ConfirmationResult = Literal["allow", "skip", "abort_run", "abort_chain"]

ConfirmationCallback = Callable[
    [RunContext[AgentContext] | AgentContext, ToolInfo, dict[str, Any]],
    ConfirmationResult | Awaitable[ConfirmationResult],
]


async def simple_confirmation(
    ctx: RunContext[AgentContext] | AgentContext,
    tool: ToolInfo,
    args: dict[str, Any],
) -> ConfirmationResult:
    """Simple confirmation handler using input() in executor."""
    # Get agent name regardless of context type
    agent_name = ctx.deps.agent_name if isinstance(ctx, RunContext) else ctx.agent_name

    prompt = dedent(f"""
        Tool Execution Confirmation
        -------------------------
        Tool: {tool.name}
        Description: {tool.description or "No description"}
        Agent: {agent_name}

        Arguments:
        {json.dumps(args, indent=2)}

        Additional Context:
        """).strip()

    # Add run-specific info if available
    if isinstance(ctx, RunContext):
        prompt += dedent(f"""
            - Model: {ctx.model.name()}
            - Prompt: "{ctx.prompt[:100]}..."
        """)

    prompt += "\nAllow this tool execution? [y/N]: "

    response = input(prompt + "\n")
    # # Run input() in executor to avoid blocking
    # loop = asyncio.get_running_loop()
    # response = await loop.run_in_executor(None, input, prompt + "\n")
    return "allow" if response.lower().startswith("y") else "skip"
