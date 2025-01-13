"""Models for agent configuration."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any, Literal, Self

from parry import (
    Config,
    ConfigModel,
    ConfigStore,
    GlobalSettings,
    LLMCapabilitiesConfig,
)
from parry_models.model_types import AnyModel  # noqa: TC002
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai.agent import EndStrategy  # noqa: TC002
from pydantic_ai.models.test import TestModel
from toprompt import render_prompt
from typing_extensions import TypeVar
from upath.core import UPath
import yamling

from common_types import SessionIdType  # noqa: TC001
from config import Capabilities, Knowledge
from environment import AgentEnvironment, FileEnvironment, InlineEnvironment
from events.sources import EventConfig  # noqa: TC001
from models.forward_targets import ForwardingTarget  # noqa: TC001
from models.mcp_server import MCPServerBase, MCPServerConfig, StdioMCPServer
from models.providers import ProviderConfig  # noqa: TC001
from models.session import SessionQuery
from models.storage import StorageConfig
from models.task import AgentTask  # noqa: TC001
from responses import InlineResponseDefinition, ResponseDefinition


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from parry_agent import AgentPool
    from agent import AnyAgent
    from common_types import StrPath
    from parry_agent_providers.base import AgentProvider


TDeps = TypeVar("TDeps", default=Any)
TResult = TypeVar("TResult", default=Any)
TResultOverride = TypeVar("TResultOverride")

logger = logging.getLogger(__name__)


class WorkerConfig(BaseModel):
    """Configuration for a worker agent.

    Worker agents are agents that are registered as tools with a parent agent.
    This allows building hierarchies and specializations of agents.
    """

    name: str
    """Name of the agent to use as a worker"""

    reset_history_on_run: bool = True
    """Whether to clear worker's conversation history before each run.
    True (default): Fresh conversation each time
    False: Maintain conversation context between runs"""

    pass_message_history: bool = False
    """Whether to pass parent agent's message history to worker.
    True: Worker sees parent's conversation context
    False (default): Worker only sees current request"""

    share_context: bool = False
    """Whether to share parent agent's context/dependencies with worker.
    True: Worker has access to parent's context data
    False (default): Worker uses own isolated context"""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    @classmethod
    def from_str(cls, name: str) -> WorkerConfig:
        """Create config from simple string form."""
        return cls(name=name)


class AgentConfig(BaseModel):
    """Configuration for a single agent in the system.

    Defines an agent's complete configuration including its model, environment,
    capabilities, and behavior settings. Each agent can have its own:
    - Language model configuration
    - Environment setup (tools and resources)
    - Response type definitions
    - System prompts and default user prompts
    - Role-based capabilities

    The configuration can be loaded from YAML or created programmatically.
    """

    type: ProviderConfig | Literal["ai", "human", "litellm"] = "ai"
    """Provider configuration or shorthand type"""

    name: str | None = None
    """Name of the agent"""

    inherits: str | None = None
    """Name of agent config to inherit from"""

    description: str | None = None
    """Optional description of the agent's purpose"""

    model: str | AnyModel | None = None  # pyright: ignore[reportInvalidTypeForm]
    """The model to use for this agent. Can be either a simple model name
    string (e.g. 'openai:gpt-4') or a structured model definition."""

    environment: str | AgentEnvironment | None = None
    """Environment configuration (path or object)"""

    capabilities: Capabilities = Field(default_factory=Capabilities)
    """Current agent's capabilities."""

    mcp_servers: list[str | MCPServerConfig] = Field(default_factory=list)
    """List of MCP server configurations:
    - str entries are converted to StdioMCPServer
    - MCPServerConfig for full server configuration
    """

    session: str | SessionQuery | None = None
    """Session configuration for conversation recovery."""

    enable_db_logging: bool = True
    """Enable session database logging."""

    result_type: str | ResponseDefinition | None = None
    """Name of the response definition to use"""

    retries: int = 1
    """Number of retries for failed operations (maps to pydantic-ai's retries)"""

    result_tool_name: str = "final_result"
    """Name of the tool used for structured responses"""

    result_tool_description: str | None = None
    """Custom description for the result tool"""

    result_retries: int | None = None
    """Max retries for result validation"""

    end_strategy: EndStrategy = "early"
    """The strategy for handling multiple tool calls when a final result is found"""

    # defer_model_check: bool = False
    # """Whether to defer model evaluation until first run"""

    avatar: str | None = None
    """URL or path to agent's avatar image"""

    system_prompts: list[str] = Field(default_factory=list)
    """System prompts for the agent"""

    user_prompts: list[str] = Field(default_factory=list)
    """Default user prompts for the agent"""

    # context_sources: list[ContextSource] = Field(default_factory=list)
    # """Initial context sources to load"""

    include_role_prompts: bool = True
    """Whether to include default prompts based on the agent's role."""

    model_settings: dict[str, Any] = Field(default_factory=dict)
    """Additional settings to pass to the model"""

    config_file_path: str | None = None
    """Config file path for resolving environment."""

    triggers: list[EventConfig] = Field(default_factory=list)
    """Event sources that activate this agent"""

    knowledge: Knowledge | None = None
    """Knowledge sources for this agent."""

    forward_to: list[ForwardingTarget] = Field(default_factory=list)
    """Targets to forward results to."""

    workers: list[WorkerConfig] = Field(default_factory=list)
    """Worker agents which will be available as tools."""

    debug: bool = False
    """Enable debug output for this agent."""

    def is_structured(self) -> bool:
        """Check if this config defines a structured agent."""
        return self.result_type is not None

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        use_attribute_docstrings=True,
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_workers(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert string workers to WorkerConfig."""
        if workers := data.get("workers"):
            data["workers"] = [
                WorkerConfig.from_str(w)
                if isinstance(w, str)
                else w
                if isinstance(w, WorkerConfig)  # Keep existing WorkerConfig
                else WorkerConfig(**w)  # Convert dict to WorkerConfig
                for w in workers
            ]
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_result_type(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert result type and apply its settings."""
        result_type = data.get("result_type")
        if isinstance(result_type, dict):
            # Extract response-specific settings
            tool_name = result_type.pop("result_tool_name", None)
            tool_description = result_type.pop("result_tool_description", None)
            retries = result_type.pop("result_retries", None)

            # Convert remaining dict to ResponseDefinition
            if "type" not in result_type:
                result_type["type"] = "inline"
            data["result_type"] = InlineResponseDefinition(**result_type)

            # Apply extracted settings to agent config
            if tool_name:
                data["result_tool_name"] = tool_name
            if tool_description:
                data["result_tool_description"] = tool_description
            if retries is not None:
                data["result_retries"] = retries

        return data

    @model_validator(mode="before")
    @classmethod
    def handle_model_types(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert model inputs to appropriate format."""
        model = data.get("model")
        match model:
            case str():
                data["model"] = {"type": "string", "identifier": model}
            case TestModel():
                # Wrap TestModel in our custom wrapper
                data["model"] = {"type": "test", "model": model}
        return data

    def get_session_query(self) -> SessionQuery | None:
        """Get session query from config."""
        if self.session is None:
            return None
        if isinstance(self.session, str):
            return SessionQuery(name=self.session)
        return self.session

    def get_provider(self) -> AgentProvider:
        """Get resolved provider instance.

        Creates provider instance based on configuration:
        - Full provider config: Use as-is
        - Shorthand type: Create default provider config
        """
        # If string shorthand is used, convert to default provider config
        from models.providers import (
            AIProviderConfig,
            HumanProviderConfig,
            LiteLLMProviderConfig,
        )

        provider_config = self.type
        if isinstance(provider_config, str):
            match provider_config:
                case "ai":
                    provider_config = AIProviderConfig()
                case "human":
                    provider_config = HumanProviderConfig()
                case "litellm":
                    provider_config = LiteLLMProviderConfig()
                case _:
                    msg = f"Invalid provider type: {provider_config}"
                    raise ValueError(msg)

        # Create provider instance from config
        return provider_config.get_provider()

    def get_mcp_servers(self) -> list[MCPServerConfig]:
        """Get processed MCP server configurations.

        Converts string entries to StdioMCPServer configs by splitting
        into command and arguments.

        Returns:
            List of MCPServerConfig instances

        Raises:
            ValueError: If string entry is empty
        """
        configs: list[MCPServerConfig] = []

        for server in self.mcp_servers:
            match server:
                case str():
                    parts = server.split()
                    if not parts:
                        msg = "Empty MCP server command"
                        raise ValueError(msg)

                    configs.append(StdioMCPServer(command=parts[0], args=parts[1:]))
                case MCPServerBase():
                    configs.append(server)

        return configs

    def render_system_prompts(self, context: dict[str, Any] | None = None) -> list[str]:
        """Render system prompts with context."""
        if not context:
            # Default context
            context = {"name": self.name, "id": 1, "model": self.model}
        return [render_prompt(p, {"agent": context}) for p in self.system_prompts]

    def get_config(self) -> Config:
        """Get configuration for this agent."""
        match self.environment:
            case None:
                # Create minimal config
                caps = LLMCapabilitiesConfig()
                global_settings = GlobalSettings(llm_capabilities=caps)
                return Config(global_settings=global_settings)
            case str() as path:
                # Backward compatibility: treat as file path
                resolved = self._resolve_environment_path(path, self.config_file_path)
                return Config.from_file(resolved)
            case FileEnvironment(uri=uri) as env:
                # Handle FileEnvironment instance
                resolved = env.get_file_path()
                return Config.from_file(resolved)
            case {"type": "file", "uri": uri}:
                # Handle raw dict matching file environment structure
                return Config.from_file(uri)
            case {"type": "inline", "config": config}:
                return config
            case InlineEnvironment() as config:
                return config
            case _:
                msg = f"Invalid environment configuration: {self.environment}"
                raise ValueError(msg)

    def get_environment_path(self) -> str | None:
        """Get environment file path if available."""
        match self.environment:
            case str() as path:
                return self._resolve_environment_path(path, self.config_file_path)
            case {"type": "file", "uri": uri} | FileEnvironment(uri=uri):
                return uri
            case _:
                return None

    def get_environment_display(self) -> str:
        """Get human-readable environment description."""
        match self.environment:
            case str() as path:
                return f"File: {path}"
            case {"type": "file", "uri": uri} | FileEnvironment(uri=uri):
                return f"File: {uri}"
            case {"type": "inline", "uri": uri} | InlineEnvironment(uri=uri) if uri:
                return f"Inline: {uri}"
            case {"type": "inline"} | InlineEnvironment():
                return "Inline configuration"
            case None:
                return "No environment configured"
            case _:
                return "Invalid environment configuration"

    @staticmethod
    def _resolve_environment_path(env: str, config_file_path: str | None = None) -> str:
        """Resolve environment path from config store or relative path."""
        try:
            config_store = ConfigStore()
            return config_store.get_config(env)
        except KeyError:
            if config_file_path:
                base_dir = UPath(config_file_path).parent
                return str(base_dir / env)
            return env

    @model_validator(mode="before")
    @classmethod
    def resolve_paths(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Store config file path for later use."""
        if "environment" in data:
            # Just store the config path for later use
            data["config_file_path"] = data.get("config_file_path")
        return data

    def get_agent_kwargs(self, **overrides) -> dict[str, Any]:
        """Get kwargs for Agent constructor.

        Returns:
            dict[str, Any]: Kwargs to pass to Agent
        """
        # Include only the fields that Agent expects
        dct = {
            "name": self.name,
            "description": self.description,
            "agent_type": self.type,
            "model": self.model,
            "system_prompt": self.system_prompts,
            "retries": self.retries,
            "enable_db_logging": self.enable_db_logging,
            # "result_tool_name": self.result_tool_name,
            "session": self.session,
            # "result_tool_description": self.result_tool_description,
            "result_retries": self.result_retries,
            "end_strategy": self.end_strategy,
            "debug": self.debug,
            # "defer_model_check": self.defer_model_check,
            **self.model_settings,
        }
        # Note: result_type is handled separately as it needs to be resolved
        # from string to actual type in Agent initialization

        dct.update(overrides)
        return dct


# TODO: python 3.13: set defaults here
class AgentsManifest[TDeps, TResult](ConfigModel):
    """Complete agent configuration manifest defining all available agents.

    This is the root configuration that:
    - Defines available response types (both inline and imported)
    - Configures all agent instances and their settings
    - Sets up custom role definitions and capabilities
    - Manages environment configurations

    A single manifest can define multiple agents that can work independently
    or collaborate through the orchestrator.
    """

    agents: dict[str, AgentConfig] = Field(default_factory=dict)
    """Mapping of agent IDs to their configurations"""

    storage: StorageConfig = Field(default_factory=StorageConfig)
    """Storage provider configuration."""

    responses: dict[str, ResponseDefinition] = Field(default_factory=dict)
    """Mapping of response names to their definitions"""

    tasks: dict[str, AgentTask] = Field(default_factory=dict)
    """Pre-defined tasks, ready to be used by agents."""

    mcp_servers: list[str | MCPServerConfig] = Field(default_factory=list)
    """List of MCP server configurations:
    - str entries are converted to StdioMCPServer
    - MCPServerConfig for full server configuration
    """
    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    def clone_agent_config(
        self,
        name: str,
        new_name: str | None = None,
        *,
        template_context: dict[str, Any] | None = None,
        **overrides: Any,
    ) -> str:
        """Create a copy of an agent configuration.

        Args:
            name: Name of agent to clone
            new_name: Optional new name (auto-generated if None)
            template_context: Variables for template rendering
            **overrides: Configuration overrides for the clone

        Returns:
            Name of the new agent

        Raises:
            KeyError: If original agent not found
            ValueError: If new name already exists or if overrides invalid
        """
        if name not in self.agents:
            msg = f"Agent {name} not found"
            raise KeyError(msg)

        actual_name = new_name or f"{name}_copy_{len(self.agents)}"
        if actual_name in self.agents:
            msg = f"Agent {actual_name} already exists"
            raise ValueError(msg)

        # Deep copy the configuration
        config = self.agents[name].model_copy(deep=True)

        # Apply overrides
        for key, value in overrides.items():
            if not hasattr(config, key):
                msg = f"Invalid override: {key}"
                raise ValueError(msg)
            setattr(config, key, value)

        # Handle template rendering if context provided
        if template_context:
            # Apply name from context if not explicitly overridden
            if "name" in template_context and "name" not in overrides:
                config.name = template_context["name"]

            # Render system prompts
            config.system_prompts = config.render_system_prompts(template_context)

        self.agents[actual_name] = config
        return actual_name

    @model_validator(mode="before")
    @classmethod
    def resolve_inheritance(cls, data: dict) -> dict:
        """Resolve agent inheritance chains."""
        agents = data.get("agents", {})
        resolved: dict[str, dict] = {}
        seen: set[str] = set()

        def resolve_agent(name: str) -> dict:
            if name in resolved:
                return resolved[name]

            if name in seen:
                msg = f"Circular inheritance detected: {name}"
                raise ValueError(msg)

            seen.add(name)
            config = (
                agents[name].model_copy()
                if hasattr(agents[name], "model_copy")
                else agents[name].copy()
            )
            inherit = (
                config.get("inherits") if isinstance(config, dict) else config.inherits
            )
            if inherit:
                if inherit not in agents:
                    msg = f"Parent agent {inherit} not found"
                    raise ValueError(msg)

                # Get resolved parent config
                parent = resolve_agent(inherit)
                # Merge parent with child (child overrides parent)
                merged = parent.copy()
                merged.update(config)
                config = merged

            seen.remove(name)
            resolved[name] = config
            return config

        # Resolve all agents
        for name in agents:
            resolved[name] = resolve_agent(name)

        # Update agents with resolved configs
        data["agents"] = resolved
        return data

    # @model_validator(mode="after")
    # def validate_response_types(self) -> AgentsManifest:
    #     """Ensure all agent result_types exist in responses or are inline."""
    #     for agent_id, agent in self.agents.items():
    #         if (
    #             isinstance(agent.result_type, str)
    #             and agent.result_type not in self.responses
    #         ):
    #             msg = f"'{agent.result_type=}' for '{agent_id=}' not found in responses"
    #             raise ValueError(msg)
    #     return self

    @classmethod
    def from_file(cls, path: StrPath) -> Self:
        """Load agent configuration from YAML file.

        Args:
            path: Path to the configuration file

        Returns:
            Loaded agent definition

        Raises:
            ValueError: If loading fails
        """
        try:
            data = yamling.load_yaml_file(path)
            # Set identifier as name if not set
            for identifier, config in data["agents"].items():
                if not config.get("name"):
                    config["name"] = identifier
            agent_def = cls.model_validate(data)
            # Update all agents with the config file path and ensure names
            agents = {
                name: config.model_copy(update={"config_file_path": str(path)})
                for name, config in agent_def.agents.items()
            }
            return agent_def.model_copy(update={"agents": agents})
        except Exception as exc:
            msg = f"Failed to load agent config from {path}"
            raise ValueError(msg) from exc

    async def create_pool(
        self,
        *,
        agents_to_load: list[str] | None = None,
        connect_agents: bool = True,
        session_id: SessionIdType = None,
    ) -> AgentPool:
        """Create an agent pool from this manifest.

        Args:
            agents_to_load: Optional list of agents to initialize
            connect_agents: Whether to set up forwarding connections
            session_id: Optional session ID for conversation recovery

        Returns:
            Configured agent pool
        """
        from delegation import AgentPool

        pool = AgentPool(
            manifest=self,
            agents_to_load=agents_to_load,
            connect_agents=connect_agents,
        )

        # Initialize agents with knowledge
        for name, agent in pool.agents.items():
            if (cfg := self.agents.get(name)) and cfg.knowledge:
                for source in (
                    cfg.knowledge.paths + cfg.knowledge.resources + cfg.knowledge.prompts
                ):
                    await agent.conversation.load_context_source(source)  # type: ignore

        return pool

    @asynccontextmanager
    async def open_agent(
        self,
        agent_name: str,
        *,
        model: str | None = None,
        session: SessionIdType | SessionQuery = None,
    ) -> AsyncIterator[AnyAgent[TDeps, Any]]:
        """Open and configure a specific agent from configuration.

        Creates the agent in the context of a single-agent pool.

        Args:
            agent_name: Name of the agent to load
            model: Optional model override
            session: Optional ID or SessionQuery to recover a previous state

        Example:
            manifest = AgentsManifest[Any, str].from_file("agents.yml")
            async with manifest.open_agent("my-agent") as agent:
                result = await agent.run("Hello!")
        """
        from parry_agent import Agent
        from delegation import AgentPool

        # Create empty pool just for context
        pool = AgentPool(manifest=self, agents_to_load=[], connect_agents=False)
        try:
            async with Agent[TDeps].open_agent(  # type: ignore
                self,
                agent_name,
                model=model,
                session=session,
            ) as agent:
                if agent.context:
                    agent.context.pool = pool
                pool.agents[agent_name] = agent
                yield agent
        finally:
            await pool.cleanup()

    def get_result_type(self, agent_name: str) -> type[Any] | None:
        """Get the resolved result type for an agent.

        Returns None if no result type is configured.
        """
        agent_config = self.agents[agent_name]
        if not agent_config.result_type:
            return None
        logger.debug("Building response model for %r", agent_config.result_type)
        if isinstance(agent_config.result_type, str):
            response_def = self.responses[agent_config.result_type]
            return response_def.create_model()  # type: ignore
        return agent_config.result_type.create_model()  # type: ignore


class ToolCallInfo(BaseModel):
    """Information about an executed tool call."""

    tool_name: str
    """Name of the tool that was called."""

    args: dict[str, Any]
    """Arguments passed to the tool."""

    result: Any
    """Result returned by the tool."""

    tool_call_id: str | None
    """ID provided by the model (e.g. OpenAI function call ID)."""

    timestamp: datetime = Field(default_factory=datetime.now)
    """When the tool was called."""

    message_id: str | None = None
    """ID of the message that triggered this tool call."""

    context_data: Any | None = None
    """Optional context data that was passed to the agent's run() method."""

    error: str | None = None
    """Error message if the tool call failed."""

    timing: float | None = None
    """Time taken for this specific tool call in seconds."""

    agent_tool_name: str | None = None
    """If this tool is agent-based, the name of that agent."""

    model_config = ConfigDict(use_attribute_docstrings=True)
