"""Agent delegation and collaboration functionality."""

from delegation.pool import AgentPool
from delegation.router import AgentRouter, CallbackRouter, RuleRouter
from delegation.controllers import interactive_controller
from delegation.router import (
    Decision,
    EndDecision,
    RouteDecision,
    AwaitResponseDecision,
    RoutingRule,
    RoutingConfig,
)
from delegation.callbacks import DecisionCallback
from delegation.injection import AgentInjectionError, inject_agents
from delegation.decorators import with_agents
from delegation.agentgroup import Team

__all__ = [
    "AgentInjectionError",
    "AgentPool",
    "AgentRouter",
    "AwaitResponseDecision",
    "CallbackRouter",
    "Decision",
    "DecisionCallback",
    "EndDecision",
    "RouteDecision",
    "RoutingConfig",
    "RoutingRule",
    "RuleRouter",
    "Team",
    "inject_agents",
    "interactive_controller",
    "with_agents",
]
