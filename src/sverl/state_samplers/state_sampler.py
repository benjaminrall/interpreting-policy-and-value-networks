from abc import ABC, abstractmethod
from torch.types import Tensor

class StateSampler(ABC):
    """Base class for SVERL state samplers."""

    _registry = {}

    def __init_subclass__(cls, **kwargs) -> None:
        """Automatically registers sampler subclasses."""
        super().__init_subclass__(**kwargs)
        StateSampler._registry[cls.__name__] = cls
        
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.setup(**kwargs)

    @classmethod
    def from_name(cls, name: str, **kwargs) -> 'StateSampler':
        # Checks that the specified type name exists in the state sampler registry
        if name not in cls._registry:
            raise ValueError(f'Unknown state sampler type: {name}')
        return cls._registry[name](**kwargs)
    
    def setup(self, **_) -> None:
        """Sets up the state sampler for sampling."""
        pass

    @abstractmethod
    def sample(self, n: int, batch_size: int = 64, shuffle: bool = True) -> Tensor:
        """Samples `n` states from the sampler, in optionally shuffled mini batches."""
        pass