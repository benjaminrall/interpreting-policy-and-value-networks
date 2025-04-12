from abc import ABC, abstractmethod
from dataclasses import dataclass
from src import Trainable

@dataclass
class TrainableConfig(ABC):
    """Abstract base class for config classes for all trainable models."""

    type: str

    _registry = {}

    def __init_subclass__(cls):
        TrainableConfig._registry[cls.__name__.removesuffix('Config').lower()] = cls

    @classmethod
    def subclasses(cls) -> dict[str, 'TrainableConfig']:
        return cls._registry

    @classmethod
    def subclass(cls, name) -> 'TrainableConfig':
        return cls._registry[name]

    @classmethod
    def from_dict(cls, trainable_dict: dict):
        type = trainable_dict.get('type')

        if type is None:
            raise ValueError(f'Trainable dict must contain a `type` field.')

        trainable_config = cls.subclass(type)(**trainable_dict)
        return trainable_config
    
    @abstractmethod
    def get_model_type(self) -> 'Trainable':
        """Returns the Trainable model type associated with the config."""
        pass