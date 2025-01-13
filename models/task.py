from __future__ import annotations

from parry.config.models import ToolConfig
from pydantic import BaseModel, ConfigDict, Field, ImportString
from typing_extensions import TypeVar

from config.knowledge import Knowledge  # noqa: TC001


TResult = TypeVar("TResult", default=str)


class AgentTask[TDeps, TResult](BaseModel):
    """Definition of a task that can be executed by an agent.

    Can be used both programmatically and defined in YAML:

    tasks:
      analyze_code:
        prompt: "Analyze the code in src directory"
        result_type: "myapp.types.AnalysisResult"
        knowledge:
          paths: ["src/**/*.py"]
          resources:
            - type: cli
              command: "mypy src/"
        tools: ["analyze_code", "check_types"]
    """

    name: str | None = Field(None, exclude=True)
    """Technical identifier (automatically set from config key during registration)"""

    description: str | None = None
    """Human-readable description of what this task does"""

    prompt: str | ImportString[str]
    """The task instruction/prompt."""

    result_type: ImportString[type[TResult]] = "str"  # type: ignore
    """Expected type of the task result."""

    deps: TDeps | None = None
    """Dependencies or context data needed for task execution"""

    knowledge: Knowledge | None = None
    """Optional knowledge sources for this task:
    - Simple file/URL paths
    - Rich resource definitions
    - Prompt templates
    """

    tools: list[ImportString | ToolConfig] = Field(default_factory=list)
    """Tools needed for this task."""

    min_context_tokens: int | None = None
    """Minimum amount of required context size."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    @property
    def tool_configs(self) -> list[ToolConfig]:
        """Get all tools as ToolConfig instances."""
        return [
            tool if isinstance(tool, ToolConfig) else ToolConfig(import_path=str(tool))
            for tool in self.tools
        ]
