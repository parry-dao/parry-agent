"""Routing configuration and decision models."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import logging
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from agent import AnyAgent
    from delegation.callbacks import DecisionCallback
    from delegation.pool import AgentPool
    from models.messages import ChatMessage


logger = logging.getLogger(__name__)


class Decision(BaseModel):
    """Base class for all routing decisions."""

    type: str = Field(init=False)
    """Discriminator field for decision types."""

    reason: str
    """Reason for this routing decision."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    async def execute(
        self,
        message: ChatMessage[Any],
        source_agent: AnyAgent[Any, Any],
        pool: AgentPool,
    ):
        """Execute this routing decision."""
        raise NotImplementedError


class RouteDecision(Decision):
    """Forward message without waiting for response."""

    type: Literal["route"] = Field("route", init=False)
    """Type discriminator for routing decisions."""

    target_agent: str
    """Name of the agent to forward the message to."""

    async def execute(
        self,
        message: ChatMessage[Any],
        source_agent: AnyAgent[Any, Any],
        pool: AgentPool,
    ):
        """Forward message and continue."""
        target = pool.get_agent(self.target_agent)
        target.outbox.emit(message, None)


class AwaitResponseDecision(Decision):
    """Forward message and wait for response."""

    type: Literal["await_response"] = Field("await_response", init=False)
    """Type discriminator for await decisions."""

    target_agent: str
    """Name of the agent to forward the message to."""

    talk_back: bool = False
    """Whether to send response back to original agent."""

    async def execute(
        self,
        message: ChatMessage[Any],
        source_agent: AnyAgent[Any, Any],
        pool: AgentPool,
    ):
        """Forward message and wait for response."""
        target = pool.get_agent(self.target_agent)
        response = await target.run(str(message))
        if self.talk_back:
            source_agent.outbox.emit(response, None)


class EndDecision(Decision):
    """End the conversation."""

    type: Literal["end"] = Field("end", init=False)
    """Type discriminator for end decisions."""

    async def execute(
        self,
        message: ChatMessage[Any],
        source_agent: AnyAgent[Any, Any],
        pool: AgentPool,
    ):
        """End the conversation."""


class AgentRouter:
    """Base class for routing messages between agents."""

    async def decide(self, message: Any) -> Decision:
        """Make routing decision for message."""
        raise NotImplementedError

    def get_wait_decision(
        self, target: str, reason: str, talk_back: bool = False
    ) -> Decision:
        """Create decision to route and wait for response."""
        return AwaitResponseDecision(
            target_agent=target, reason=reason, talk_back=talk_back
        )

    def get_route_decision(self, target: str, reason: str) -> Decision:
        """Create decision to route without waiting."""
        return RouteDecision(target_agent=target, reason=reason)

    def get_end_decision(self, reason: str) -> Decision:
        """Create decision to end routing."""
        return EndDecision(reason=reason)


class CallbackRouter[TMessage](AgentRouter):
    """Router using callback function for decisions."""

    def __init__(
        self,
        pool: AgentPool,
        decision_callback: DecisionCallback[TMessage],
    ):
        self.pool = pool
        self.decision_callback = decision_callback

    async def decide(self, message: TMessage) -> Decision:
        """Execute callback and handle sync/async appropriately."""
        result = self.decision_callback(message, self.pool, self)
        if inspect.isawaitable(result):
            return await result
        return result


class RuleRouter(AgentRouter):
    """Router using predefined rules."""

    def __init__(self, pool: AgentPool, config: RoutingConfig):
        self.pool = pool
        self.config = config

    async def decide(self, message: str) -> Decision:
        """Make decision based on configured rules."""
        msg = message if self.config.case_sensitive else message.lower()

        # Check each rule in priority order
        for rule in sorted(self.config.rules, key=lambda r: r.priority):
            keyword = rule.keyword if self.config.case_sensitive else rule.keyword.lower()

            if keyword not in msg:
                continue

            # Skip if target doesn't exist
            if rule.target not in self.pool.list_agents():
                msg = "Target agent %s not available for rule: %s"
                logger.debug(msg, rule.target, rule.keyword)
                continue

            # Skip if capability required but not available
            if rule.requires_capability:
                agent = self.pool.get_agent(rule.target)
                if not agent.context.capabilities.has_capability(
                    rule.requires_capability
                ):
                    msg = "Agent %s missing required capability: %s"
                    logger.debug(msg, rule.target, rule.requires_capability)
                    continue

            # Create appropriate decision using base class methods
            if rule.wait_for_response:
                return self.get_wait_decision(target=rule.target, reason=rule.reason)
            return self.get_route_decision(target=rule.target, reason=rule.reason)

        # Use default route if configured
        if self.config.default_target:
            return self.get_wait_decision(
                target=self.config.default_target,
                reason=self.config.default_reason,
            )

        # End if no route found
        return self.get_end_decision(reason="No matching rule or default route")


@dataclass
class RoutingRule:
    """Rule for routing messages based on content matching.

    Defines when and how messages should be routed to specific agents.
    Rules are evaluated in priority order, with the first matching rule
    being applied.
    """

    keyword: str
    """Keyword to match in the message content."""

    target: str
    """Name of the agent to route to when rule matches."""

    reason: str
    """Reason for this routing decision."""

    wait_for_response: bool = True
    """Whether to wait for the target agent's response."""

    priority: int = 100
    """Rule priority (lower numbers = higher priority)."""

    requires_capability: str | None = None
    """Optional capability the target agent must have."""


class RoutingConfig(BaseModel):
    """Complete routing configuration for an agent.

    Defines a set of rules for message routing and default behavior
    when no rules match. Rules are evaluated in priority order.
    """

    rules: list[RoutingRule]
    """Ordered list of routing rules to evaluate."""

    default_target: str | None = None
    """Agent to route to when no rules match."""

    default_reason: str = "No specific rule matched"
    """Reason to use for default routing."""

    case_sensitive: bool = False
    """Whether keyword matching should be case-sensitive."""

    model_config = ConfigDict(use_attribute_docstrings=True)
