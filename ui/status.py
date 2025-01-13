"""Status bar for interactive session."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal


if TYPE_CHECKING:
    from chat_session.models import SessionState


Alignment = Literal["left", "center", "right"]


@dataclass
class BaseField:
    """Field in a status bar."""

    label: ClassVar[str]
    style: str = "dim"
    align: Alignment = "right"
    icon: str | None = None
    tooltip: str | None = None
    value: str = ""
    visible: bool = True


class ModelField(BaseField):
    label = "Model"
    align = "left"
    icon = "🤖"
    tooltip = "Current language model"


class TokensField(BaseField):
    label = "Tokens"
    icon = "🎯"
    tooltip = "Token usage"


class CostField(BaseField):
    label = "Cost"
    icon = "💰"
    tooltip = "Total cost in USD"


class MessagesField(BaseField):
    label = "Messages"
    icon = "💬"
    tooltip = "Message count"


class TimeField(BaseField):
    label = "Time"
    icon = "⏱️"
    tooltip = "Session duration"


class StatusBar:
    """UI-agnostic status bar."""

    def __init__(self):
        self.model = ModelField()
        self.tokens = TokensField()
        self.cost = CostField()
        self.messages = MessagesField()
        self.time = TimeField()

    def update(self, state: SessionState):
        """Update field values from state."""
        self.model.value = state.current_model or "default"
        self.tokens.value = f"{state.total_tokens:,}"
        self.cost.value = f"${state.total_cost:.3f}"
        self.messages.value = str(state.message_count)
        self.time.value = state.duration

    @property
    def fields(self) -> list[BaseField]:
        """Get all fields."""
        return [self.model, self.tokens, self.cost, self.messages, self.time]
