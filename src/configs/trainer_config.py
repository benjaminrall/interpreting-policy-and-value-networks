from abc import ABC
from dataclasses import dataclass
from src.configs import TrainableConfig

class TrainerConfig:
    """Config class for experiments run by the Trainer."""
    
    OBJECTIVE_REGISTRY = {
        'agent',
        'sverl'
    }

    def __init__(self, run_name: str, seed: int, checkpoint_interval: int, **kwargs) -> None:
        self.run_name = run_name
        self.seed = seed
        self.checkpoint_interval = checkpoint_interval
        self.objective = self._get_objective_config(**kwargs)

    @classmethod
    def _get_objective_config(cls, **kwargs):
        print(kwargs)
        for key in cls.OBJECTIVE_REGISTRY:
            print(key)
            if key in kwargs:
                print("Creating TrainableConfig")
                return TrainableConfig.from_dict(kwargs[key])