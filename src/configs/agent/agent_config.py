from dataclasses import dataclass
from src.configs import EnvConfig, TrainableConfig

@dataclass
class AgentConfig(TrainableConfig):
    """Abstract base class for actor-critic agent configurations."""

    actor: str
    critic: str
    environment: EnvConfig = lambda : EnvConfig()

    _registry = {}

    def __init_subclass__(cls):
        AgentConfig._registry[cls.__name__.removesuffix('Config')] = cls

    def __post_init__(self) -> None:
        """Automatically converts an environment dict to EnvConfig."""
        if isinstance(self.environment, dict):
            self.environment = EnvConfig(**self.environment)
