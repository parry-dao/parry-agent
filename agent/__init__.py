"""CLI commands for parry-agent."""

from __future__ import annotations
from typing_extensions import TypeVar

from agent.agent import Agent
from agent.structured import StructuredAgent

type AnyAgent[TDeps, TResult] = Agent[TDeps] | StructuredAgent[TDeps, TResult]

from agent.agent_logger import AgentLogger
from agent.conversation import ConversationManager
from agent.container import AgentContainer
from agent.slashed_agent import SlashedAgent
from agent.talk import Interactions


TDeps = TypeVar("TDeps")
TResult = TypeVar("TResult", default=str)


__all__ = [
    "Agent",
    "AgentContainer",
    "AgentLogger",
    "AnyAgent",
    "ConversationManager",
    "Interactions",
    "SlashedAgent",
    "StructuredAgent",
]
