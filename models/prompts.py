"""Models for system and user prompts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class SystemPrompt(BaseModel):
    """System prompt configuration for agent behavior control.

    Defines prompts that set up the agent's behavior and context.
    Supports multiple types:
    - Static text prompts
    - Dynamic function-based prompts
    - Template prompts with variable substitution
    """

    type: Literal["text", "function", "template"]
    """Type of system prompt: static text, function call, or template"""

    value: str
    """The prompt text, function path, or template string"""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)
