"""Models for prompt templates and libraries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field
import yamling


if TYPE_CHECKING:
    from common_types import ModelType, StrPath


class PromptTemplate(BaseModel):
    """Template for generating prompts."""

    description: str
    system: str
    template: str | None = None
    variables: list[str] = Field(default_factory=list)
    defaults: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    async def apply(
        self,
        goal: str,
        model: ModelType = None,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Apply this template to generate a new prompt."""
        from pydantic_ai import Agent as PydanticAiAgent

        template_vars = {"goal": goal, **self.defaults, **kwargs}
        if max_length:
            template_vars["max_length"] = max_length

        # Format template, create temporary pydantic agent to generate prompt.
        content = self.template.format(**template_vars) if self.template else goal
        agent = PydanticAiAgent(model=model, system_prompt=self.system)
        result = await agent.run(content)
        return str(result.data)


class PromptLibrary(BaseModel):
    """Collection of organized prompt templates."""

    meta_prompts: dict[str, PromptTemplate]
    user_prompts: dict[str, PromptTemplate] | None = None
    system_prompts: dict[str, PromptTemplate]

    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_file(cls, path: StrPath) -> PromptLibrary:
        """Load prompts from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Loaded prompt library

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file content is invalid
        """
        try:
            data = yamling.load_yaml_file(path)
            return cls.model_validate(data)
        except Exception as e:
            msg = f"Failed to load prompts from {path}"
            raise ValueError(msg) from e

    def get_meta_prompt(self, name: str) -> PromptTemplate:
        """Get a meta prompt by name."""
        if name not in self.meta_prompts:
            msg = f"Meta prompt not found: {name}"
            raise KeyError(msg)
        return self.meta_prompts[name]

    def get_user_prompt(self, name: str) -> PromptTemplate:
        """Get a user prompt by name."""
        if not self.user_prompts or name not in self.user_prompts:
            msg = f"User prompt not found: {name}"
            raise KeyError(msg)
        return self.user_prompts[name]

    def get_system_prompt(self, name: str) -> PromptTemplate:
        """Get a system prompt by name."""
        if name not in self.system_prompts:
            msg = f"System prompt not found: {name}"
            raise KeyError(msg)
        return self.system_prompts[name]
