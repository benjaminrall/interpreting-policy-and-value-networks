from abc import ABC, abstractmethod
from src.configs import EnvConfig
from gymnasium.vector import VectorEnv

class EnvBuilder(ABC):
    """Base class for building different types of environments."""

    _registry: dict[str, 'EnvBuilder'] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        """Automatically registers environment builder subclasses."""
        super().__init_subclass__(**kwargs)
        EnvBuilder._registry[cls.__name__] = cls

    @staticmethod
    @abstractmethod
    def _construct_env(config: EnvConfig) -> VectorEnv:
        """Abstract static method that subclasses must implement to construct the environment."""
        pass

    @classmethod
    def build(cls, config: EnvConfig) -> VectorEnv:
        """
        Builds an environment based on the config's type.
        Looks up the correct EnvBuilder subclass and calls its `_construct_env` method.
        """
        # Gets the name of the builder class
        builder_type = config.type + 'EnvBuilder'

        # Checks that the builder type exists
        if builder_type not in cls._registry:
            raise ValueError(f"Unknown environment builder type: {config.type}")
        
        # Returns the constructed environment
        return cls._registry[builder_type]._construct_env(config)