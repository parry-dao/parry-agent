from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import desc
from sqlmodel import Session, select

from history.filters import parse_time_period
from history.formatters import format_conversation
from history.models import ConversationData, QueryFilters, StatsFilters
from storage import Conversation, Message, engine


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlmodel.sql.expression import SelectOfScalar

    from models.messages import TokenUsage


def build_conversation_query(filters: QueryFilters) -> SelectOfScalar[Conversation]:
    """Build base conversation query with filters."""
    stmt = select(Conversation).order_by(desc(Conversation.start_time))  # type: ignore[arg-type]
    if filters.agent_name:
        stmt = stmt.where(Conversation.agent_name == filters.agent_name)
    if filters.since:
        stmt = stmt.where(Conversation.start_time >= filters.since)
    if filters.limit:
        stmt = stmt.limit(filters.limit)

    return stmt  # type: ignore[return-value]


def build_message_query(
    conversation_id: str,
    filters: QueryFilters,
) -> SelectOfScalar[Message]:
    """Build message query with filters."""
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.timestamp)  # type: ignore[arg-type]
    )

    if filters.query:
        stmt = stmt.where(Message.content.contains(filters.query))  # type: ignore[attr-defined]
    if filters.model:
        stmt = stmt.where(Message.model == filters.model)

    return stmt  # type: ignore[return-value]


def get_conversations(
    filters: QueryFilters,
) -> list[tuple[Conversation, Sequence[Message]]]:
    """Get filtered conversations with their messages."""
    with Session(engine) as session:
        conv_stmt = build_conversation_query(filters)
        conversations = session.exec(conv_stmt).all()
        results = []

        for conv in conversations:
            msg_stmt = build_message_query(conv.id, filters)
            msgs = session.exec(msg_stmt).all()

            # Skip conversations with no matching messages if content filtered
            if filters.query and not msgs:
                continue

            results.append((conv, msgs))

        return results


def get_filtered_conversations(
    agent_name: str | None = None,
    period: str | None = None,
    since: datetime | None = None,
    query: str | None = None,
    model: str | None = None,
    limit: int | None = None,
    *,
    compact: bool = False,
    include_tokens: bool = False,
) -> list[ConversationData]:
    """Get filtered conversations with formatted output.

    Args:
        agent_name: Filter by agent name
        period: Time period to include (e.g. "1h", "2d")
        since: Only show conversations after this time
        query: Search in message content
        model: Filter by model used
        limit: Maximum number of conversations
        compact: Only show first/last message of each conversation
        include_tokens: Include token usage statistics
    """
    if period:
        since = datetime.now() - parse_time_period(period)

    filters = QueryFilters(
        agent_name=agent_name,
        since=since,
        query=query,
        model=model,
        limit=limit,
    )
    conversations = get_conversations(filters)
    return [
        format_conversation(conv, msgs, compact=compact, include_tokens=include_tokens)
        for conv, msgs in conversations
    ]


def get_stats_data(
    filters: StatsFilters,
) -> list[tuple[str | None, str | None, datetime, TokenUsage | None]]:
    """Get raw statistics data."""
    with Session(engine) as session:
        query = (
            select(
                Message.model,
                Conversation.agent_name,
                Message.timestamp,
                Message.total_tokens.label("total"),  # type: ignore
                Message.prompt_tokens.label("prompt"),  # type: ignore
                Message.completion_tokens.label("completion"),  # type: ignore
            )
            .join(Conversation, Message.conversation_id == Conversation.id)  # type: ignore[arg-type]
            .where(Message.timestamp > filters.cutoff)
        )

        if filters.agent_name:
            query = query.where(Conversation.agent_name == filters.agent_name)

        results = session.exec(query).all()
        return [
            (
                str(model) if model else None,
                str(agent) if agent else None,
                timestamp,
                {
                    "total": total or 0,
                    "prompt": prompt or 0,
                    "completion": completion or 0,
                }
                if (total or prompt or completion)
                else None,
            )
            for model, agent, timestamp, total, prompt, completion in results
        ]
