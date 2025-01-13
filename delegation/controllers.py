"""Controller implementations for agent conversations."""

from typing import TYPE_CHECKING

from delegation.router import (
    AgentRouter,
    AwaitResponseDecision,
    Decision,
    EndDecision,
    RouteDecision,
)
from log import get_logger


if TYPE_CHECKING:
    from delegation.pool import AgentPool


logger = get_logger(__name__)


async def interactive_controller(
    message: str, pool: "AgentPool", agent_router: AgentRouter
) -> Decision:
    """Interactive conversation control through console input."""
    print(f"\nMessage: {message}")
    print("\nWhat would you like to do?")
    print("1. Forward message (no wait)")
    print("2. Route and wait for response")
    print("3. End conversation")

    try:
        match choice := int(input("> ")):
            case 1 | 2:  # Route or TalkBack
                print("\nAvailable agents:")
                # Use pool's list_agents instead of passed list
                agents = pool.list_agents()
                for i, name in enumerate(agents, 1):
                    agent = pool.get_agent(name)
                    print(f"{i}. {name} ({agent.description or 'No description'})")

                agent_idx = int(input("Select agent: ")) - 1
                target = agents[agent_idx]
                reason = input("Reason: ")

                if choice == 1:
                    return RouteDecision(target_agent=target, reason=reason)
                return AwaitResponseDecision(target_agent=target, reason=reason)

            case 3:  # End
                reason = input("Reason for ending: ")
                return EndDecision(reason=reason)

            case _:
                return EndDecision(reason="Invalid choice")

    except (ValueError, IndexError):
        return EndDecision(reason="Invalid input")
