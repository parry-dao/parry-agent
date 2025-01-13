from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from history.queries import get_stats_data


if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from history.models import GroupBy, StatsFilters
    from models.messages import TokenUsage


def aggregate_stats(
    rows: Sequence[tuple[str | None, str | None, datetime, TokenUsage | None]],
    group_by: GroupBy,
) -> dict[str, dict[str, Any]]:
    """Aggregate statistics data by specified grouping.

    Args:
        rows: Raw stats data (model, agent, timestamp, token_usage)
        group_by: How to group the statistics
    """
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total_tokens": 0, "messages": 0, "models": set()}
    )

    for model, agent, timestamp, token_usage in rows:
        match group_by:
            case "agent":
                key = agent or "unknown"
            case "model":
                key = model or "unknown"
            case "hour":
                key = timestamp.strftime("%Y-%m-%d %H:00")
            case "day":
                key = timestamp.strftime("%Y-%m-%d")

        entry = stats[key]
        entry["messages"] += 1
        if token_usage:
            entry["total_tokens"] += token_usage.get("total", 0)
        if model:
            entry["models"].add(model)

    return stats


def get_conversation_stats(filters: StatsFilters) -> dict[str, dict[str, Any]]:
    """Get conversation statistics grouped by specified criterion.

    Args:
        filters: Filters for statistics query
    """
    rows = get_stats_data(filters)
    return aggregate_stats(rows, filters.group_by)
