from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from agent import AnyAgent


@dataclass
class WelcomeInfo:
    """Container for welcome message information."""

    header: list[str]
    agent_info: list[str]
    config_info: list[str]
    tools_info: list[str]
    prompts_info: list[str]
    usage_info: list[str]
    footer: list[str]

    def all_sections(self) -> list[tuple[str, list[str]]]:
        """Get all sections with their titles."""
        return [
            ("", self.header),
            ("", self.agent_info),
            ("", self.config_info),
            ("", self.tools_info),
            ("", self.prompts_info),
            ("", self.usage_info),
            ("", self.footer),
        ]


def create_welcome_messages(
    agent: AnyAgent[Any, Any],
    *,
    streaming: bool = False,
    rich_format: bool = True,
) -> WelcomeInfo:
    """Create welcome messages for any interface.

    Args:
        agent: The agent to create welcome messages for
        streaming: Whether streaming mode is enabled
        rich_format: Whether to include rich text formatting

    Returns:
        Structured welcome information
    """

    # Helper for conditional rich formatting
    def fmt(text: str, style: str) -> str:
        """Apply rich formatting if enabled."""
        return f"[{style}]{text}[/]" if rich_format else text

    # Build all sections
    header = [fmt("parry Agent Chat", "bold blue"), "=" * 50]

    agent_info = [f"{fmt('Agent:', 'bold')} {agent.name}"]
    if agent.context.config.description:
        agent_info.append(fmt(agent.context.config.description, "dim"))

    model = agent.model_name or "default"
    mode = "streaming" if streaming else "non-streaming"
    config_info = [f"{fmt('Model:', 'bold')} {model}", f"{fmt('Mode:', 'bold')} {mode}"]

    enabled_tools = [t.name for t in agent.tools.values() if t.enabled]
    disabled_tools = [t.name for t in agent.tools.values() if not t.enabled]

    tools_info = [fmt("Tools:", "bold")]
    if enabled_tools:
        tools_info.append(fmt("Enabled:", "green"))
        tools_info.extend(f"  • {tool}" for tool in enabled_tools)
    if disabled_tools:
        tools_info.append(fmt("Disabled:", "yellow"))
        tools_info.extend(f"  • {tool}" for tool in disabled_tools)

    prompts_info = []
    if agent.context.config.system_prompts:
        prompts_info.append(fmt("System Prompts:", "bold"))
        prompts_info.extend(
            fmt(f"  {prompt.split()[0]}...", "dim")
            for prompt in agent.context.config.system_prompts
        )

    disable_str = fmt("/disable-tool", "green")
    enable_str = fmt("/enable-tool", "green")
    usage_info = [
        fmt("Usage:", "bold"),
        f"• Type {fmt('/help', 'green')} for available commands",
        f"• Type {fmt('/exit', 'green')} to quit",
        f"• Use {enable_str} or {disable_str} to manage tools",
    ]

    footer = ["=" * 50, ""]

    return WelcomeInfo(
        header=header,
        agent_info=agent_info,
        config_info=config_info,
        tools_info=tools_info,
        prompts_info=prompts_info,
        usage_info=usage_info,
        footer=footer,
    )
