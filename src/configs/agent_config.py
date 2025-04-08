from abc import ABC
from dataclasses import dataclass
from src.configs import EnvConfig, TrainableConfig

@dataclass
class AgentConfig(TrainableConfig):
    """Abstract base class for actor-critic agent configurations."""

    actor: str
    critic: str
    environment: EnvConfig = lambda : EnvConfig()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        AgentConfig._registry[cls.__name__.removesuffix('Agent')] = cls

    def __post_init__(self) -> None:
        """Automatically converts an environment dict to EnvConfig."""
        if isinstance(self.environment, dict):
            self.environment = EnvConfig(**self.environment)
