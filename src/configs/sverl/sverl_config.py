from dataclasses import dataclass
from src.configs import TrainableConfig

@dataclass(kw_only=True)
class SVERLConfig(TrainableConfig):
    """Base class for SVERL function configurations."""

    model: str
    target: str
    state_sampler: str
    batch_size: int = 64
    validation_sampler: str = 'ProceduralSampler'
    validation_samples: int = 5000
    validation_batch_size: int = 64
    agent_checkpoint: str
    epochs: int = 400
    samples_per_epoch: int = 10000
    learning_rate: float = 0.001
    anneal_lr: bool = True
    anneal_factor: float = 0.8
    anneal_patience: int = 5

    _registry = {}

    def __post_init__(self):
        if self.target not in ['actor', 'critic']:
            raise ValueError(f'SVERL target must be \'actor\' or \'critic\', got: {self.target}')

    def __init_subclass__(cls):
        SVERLConfig._registry[cls.__name__.removesuffix('Config')] = cls