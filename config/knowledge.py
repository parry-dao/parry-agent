from __future__ import annotations

from parry.config.models import Resource  # noqa: TC002
from parry.prompts import PromptType  # noqa: TC002
from pydantic import BaseModel, ConfigDict, Field


class Knowledge(BaseModel):
    """Collection of context sources for an agent.

    Supports both simple paths and rich resource types for content loading,
    plus parry's prompt system for dynamic content generation.
    """

    paths: list[str] = Field(default_factory=list)
    """Quick access to files and URLs."""

    resources: list[Resource] = Field(default_factory=list)
    """Rich resource definitions supporting:
    - PathResource: Complex file patterns, watching
    - TextResource: Raw content
    - CLIResource: Command output
    - RepositoryResource: Git repos
    - SourceResource: Python source
    - CallableResource: Function results
    """

    prompts: list[PromptType] = Field(default_factory=list)
    """Prompts for dynamic content generation:
    - StaticPrompt: Fixed message templates
    - DynamicPrompt: Python function-based
    - FilePrompt: File-based with template support
    """

    convert_to_markdown: bool = False
    """Whether to convert content to markdown when possible."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)
