"""Manages message flow between agents/groups."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import cached_property
from typing import TYPE_CHECKING, Any, Self, overload

from psygnal import Signal
from typing_extensions import TypeVar

from log import get_logger
from models.messages import ChatMessage


if TYPE_CHECKING:
    from agent import AnyAgent
    from delegation.agentgroup import Team
    from models.forward_targets import ConnectionType

TContent = TypeVar("TContent")
FilterFn = Callable[[ChatMessage[Any]], bool]
TransformFn = Callable[[ChatMessage[TContent]], ChatMessage[TContent]]

logger = get_logger(__name__)


@dataclass(frozen=True)
class TalkStats:
    """Statistics for a single connection."""

    message_count: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_message_time: datetime | None = None
    token_count: int = 0
    byte_count: int = 0
    source_name: str | None = None
    target_names: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class TeamTalkStats:
    """Statistics aggregated from multiple connections."""

    stats: list[TalkStats | TeamTalkStats] = field(default_factory=list)

    @cached_property
    def message_count(self) -> int:
        """Total messages across all connections."""
        return sum(stat.message_count for stat in self.stats)

    @cached_property
    def token_count(self) -> int:
        """Total tokens across all connections."""
        return sum(stat.token_count for stat in self.stats)

    @cached_property
    def byte_count(self) -> int:
        """Total bytes forwarded across all connections."""
        return sum(stat.byte_count for stat in self.stats)

    @property
    def num_connections(self) -> int:
        """Number of active connections."""
        return len(self.stats)

    @cached_property
    def start_time(self) -> datetime:
        """Earliest connection start time."""
        if not self.stats:
            return datetime.now()
        return min(stat.start_time for stat in self.stats)

    @cached_property
    def last_message_time(self) -> datetime | None:
        """Most recent message across all connections."""
        times = [s.last_message_time for s in self.stats if s.last_message_time]
        return max(times) if times else None

    @cached_property
    def source_names(self) -> set[str]:
        """Set of unique source names recursively."""

        def _collect_source_names(stat: TalkStats | TeamTalkStats) -> set[str]:
            """Recursively collect source names."""
            if isinstance(stat, TalkStats):
                return {stat.source_name} if stat.source_name else set()
            # It's a TeamTalkStats, recurse
            names = set()
            for s in stat.stats:
                names.update(_collect_source_names(s))
            return names

        names = set()
        for stat in self.stats:
            names.update(_collect_source_names(stat))
        return names

    @cached_property
    def target_names(self) -> set[str]:
        """Set of all target names across connections."""
        return {name for s in self.stats for name in s.target_names}


class Talk:
    """Manages message flow between agents/groups."""

    message_received = Signal(ChatMessage[Any])  # Original message
    message_forwarded = Signal(ChatMessage[Any])  # After any transformation

    def __init__(
        self,
        source: AnyAgent[Any, Any],
        targets: list[AnyAgent[Any, Any]],
        group: TeamTalk | None = None,
        *,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
    ):
        self.source = source
        self.targets = targets
        self.group = group
        self.priority = priority
        self.delay = delay
        self.active = True
        self.connection_type = connection_type
        names = {t.name for t in targets}
        self._stats = TalkStats(source_name=source.name, target_names=names)
        self._filter: FilterFn | None = None
        self._transformer: TransformFn | None = None
        source.outbox.connect(self._handle_message)

    def _handle_message(self, message: ChatMessage[Any], prompt: str | None = None):
        # logger.debug(
        #     "Message from %s to %s: %r (type: %s) (prompt: %s)",
        #     self.source.name,
        #     [t.name for t in self.targets],
        #     message.content,
        #     self.connection_type,
        #     prompt,
        # )
        if not self.active or (self.group and not self.group.active):
            return
        if self._filter and not self._filter(message):
            return

        # Update stats
        totals = message.cost_info.token_usage["total"] if message.cost_info else 0
        self._stats = TalkStats(
            message_count=self._stats.message_count + 1,
            start_time=self._stats.start_time,
            last_message_time=datetime.now(),
            token_count=self._stats.token_count + totals,
            byte_count=self._stats.byte_count + len(str(message.content).encode()),
            source_name=self._stats.source_name,
            target_names=self._stats.target_names,
        )

        # Handle message based on connection type
        match self.connection_type:
            case "run":
                for target in self.targets:
                    prompts = [str(message.content)]
                    if prompt:
                        prompts.append(prompt)
                    target.run_background(
                        target.run(*prompts), priority=self.priority, delay=self.delay
                    )

            case "context":
                for target in self.targets:
                    coro = target.conversation.add_context_message(
                        str(message.content),
                        source=self.source.name,
                        metadata={
                            "type": "forwarded_message",
                            "role": message.role,
                            "model": message.model,
                            "cost_info": message.cost_info,
                            "timestamp": message.timestamp.isoformat(),
                            "prompt": prompt,  # Include original prompt in metadata
                        },
                    )
                    target.run_background(coro, priority=self.priority, delay=self.delay)

            case "forward":
                for target in self.targets:
                    if self.delay is not None or self.priority != 0:

                        async def delayed_emit(target=target):
                            target.outbox.emit(message, prompt)

                        target.run_background(
                            delayed_emit(),
                            priority=self.priority,
                            delay=self.delay,
                        )
                    else:
                        target.outbox.emit(message, prompt)

        self.message_forwarded.emit(message)

    def when(self, condition: FilterFn) -> Self:
        """Add condition for message forwarding."""
        self._filter = condition
        return self

    @asynccontextmanager
    async def paused(self):
        """Temporarily set inactive."""
        previous = self.active
        self.active = False
        try:
            yield self
        finally:
            self.active = previous

    def disconnect(self):
        """Permanently disconnect the connection."""
        self.source.outbox.disconnect(self._handle_message)
        self.active = False

    @property
    def stats(self) -> TalkStats:
        """Get current connection statistics."""
        return self._stats


class TeamTalk(list["Talk | TeamTalk"]):
    """Group of connections with aggregate operations."""

    def __init__(self, talks: Sequence[Talk | TeamTalk]):
        super().__init__(talks)
        self._filter: FilterFn | None = None
        self.active = True

    @property
    def targets(self) -> list[AnyAgent[Any, Any]]:
        """Get all targets from all connections."""
        return [t for talk in self for t in talk.targets]

    def _handle_message(self, message: ChatMessage[Any], prompt: str | None = None):
        for talk in self:
            talk._handle_message(message, prompt)

    @classmethod
    def from_agents(
        cls,
        agents: Sequence[AnyAgent[Any, Any]],
        targets: list[AnyAgent[Any, Any]] | None = None,
    ) -> TeamTalk:
        """Create TeamTalk from a collection of agents."""
        return cls([Talk(agent, targets or []) for agent in agents])

    @asynccontextmanager
    async def paused(self):
        """Temporarily set inactive."""
        previous = self.active
        self.active = False
        try:
            yield self
        finally:
            self.active = previous

    def has_active_talks(self) -> bool:
        """Check if any contained talks are active."""
        return any(talk.active for talk in self)

    def get_active_talks(self) -> list[Talk | TeamTalk]:
        """Get list of currently active talks."""
        return [talk for talk in self if talk.active]

    @property
    def stats(self) -> TeamTalkStats:
        """Get aggregated statistics for all connections."""
        return TeamTalkStats(stats=[talk.stats for talk in self])

    def when(self, condition: FilterFn) -> Self:
        """Add condition to all connections in group."""
        for talk in self:
            talk.when(condition)
        return self

    def disconnect(self):
        """Disconnect all connections in group."""
        for talk in self:
            talk.disconnect()


class TalkManager:
    """Manages connections for both Agents and Teams."""

    def __init__(self, owner: AnyAgent[Any, Any] | Team[Any]):
        self.owner = owner
        self._connections: list[Talk | TeamTalk] = []

    def _route_message(self, message: ChatMessage[Any], prompt: str | None):
        # Each Talk already knows its targets
        for talk in self._connections:
            talk._handle_message(message, prompt)

    def get_targets(self) -> set[AnyAgent[Any, Any]]:
        """Get all currently connected target agents."""
        return {t for conn in self._connections for t in conn.targets if conn.active}

    def has_connection_to(self, target: AnyAgent[Any, Any]) -> bool:
        """Check if target is connected."""
        return any(target in conn.targets for conn in self._connections if conn.active)

    @overload
    def connect_agent_to(
        self, other: AnyAgent[Any, Any] | str, **kwargs: Any
    ) -> Talk: ...

    @overload
    def connect_agent_to(self, other: Team[Any], **kwargs: Any) -> TeamTalk: ...

    def connect_agent_to(
        self,
        other: AnyAgent[Any, Any] | Team[Any] | str,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
        **kwargs: Any,
    ) -> Talk | TeamTalk:
        """Handle single agent connections."""
        from agent import Agent, StructuredAgent
        from delegation.agentgroup import Team

        if not isinstance(self.owner, Agent | StructuredAgent):
            msg = "connect_agent_to can only be used with single agents"
            raise TypeError(msg)

        targets = self._resolve_targets(other)
        if isinstance(other, Team):
            conns = [
                Talk(
                    self.owner,
                    [target],
                    connection_type=connection_type,
                    priority=priority,
                    delay=delay,
                )
                for target in targets
            ]
            connections = TeamTalk(conns)
            self._connections.extend(connections)
            return connections

        connection = Talk(
            self.owner,
            targets,
            connection_type=connection_type,
            priority=priority,
            delay=delay,
        )
        self._connections.append(connection)
        return connection

    def connect_group_to(
        self,
        other: AnyAgent[Any, Any] | Team[Any] | str,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
        **kwargs: Any,
    ) -> TeamTalk:
        """Handle group connections."""
        from delegation.agentgroup import Team

        if not isinstance(self.owner, Team):
            msg = "connect_group_to can only be used with agent groups"
            raise TypeError(msg)

        targets = self._resolve_targets(other)
        conns = [
            Talk(
                src,
                [t],
                connection_type=connection_type,
                priority=priority,
                delay=delay,
            )
            for src in self.owner.agents
            for t in targets
        ]
        connections = TeamTalk(conns)
        ## using extend() here flattens the list
        self._connections.extend(connections)
        return connections

    def _resolve_targets(
        self, other: AnyAgent[Any, Any] | Team[Any] | str
    ) -> list[AnyAgent[Any, Any]]:
        """Resolve target(s) to connect to."""
        from agent import Agent, StructuredAgent
        from delegation.agentgroup import Team

        match other:
            case str():
                if (
                    not isinstance(self.owner, Agent | StructuredAgent)
                    or not self.owner.context.pool
                ):
                    msg = "Pool required for forwarding to agent by name"
                    raise ValueError(msg)
                return [self.owner.context.pool.get_agent(other)]
            case Team():
                return list(other.agents)
        return [other]

    def disconnect_all(self):
        """Disconnect all managed connections."""
        for conn in self._connections:
            conn.disconnect()
        self._connections.clear()

    def disconnect(self, agent: AnyAgent[Any, Any]):
        """Disconnect a specific agent."""
        to_disconnect: list[Talk | TeamTalk] = []
        for talk in self._connections:
            match talk:
                case Talk():
                    if agent in talk.targets or agent == talk.source:
                        to_disconnect.append(talk)
                case TeamTalk():
                    if agent in talk.targets:
                        to_disconnect.append(talk)

        for talk in to_disconnect:
            talk.active = False
            self._connections.remove(talk)

    @property
    def stats(self) -> TeamTalkStats:
        """Get aggregated statistics for all connections."""
        return TeamTalkStats(stats=[conn.stats for conn in self._connections])
