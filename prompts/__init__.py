"""Prompt template management."""

from __future__ import annotations

import importlib.resources
from typing import Final

from prompts.models import PromptLibrary, PromptTemplate

# Default prompt paths
DEFAULT_PROMPTS: Final[str] = str(
    importlib.resources.files("prompts") / "default.yml"
)

__all__ = ["DEFAULT_PROMPTS", "PromptLibrary", "PromptTemplate"]
