from abc import ABC
from dataclasses import dataclass

@dataclass
class TrainableConfig(ABC):
    """Abstract base class for all """

    type: str

    _registry = {}

    @classmethod
    def from_dict(cls, trainable_dict: dict):
        type = trainable_dict.get('type')

        if type is None:
            raise ValueError(f'Trainable dict must contain a `type` field.')
        # DO THINGY HERE, BECUASE ITS A CLASS METHOD THE REGISTRY WILL BE DEFINED IN EACH SUBCLASS

        trainable_config = cls._registry[type](**trainable_dict)

        return trainable_config
