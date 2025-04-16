from abc import ABC, abstractmethod
from torch.types import Tensor

class Masker(ABC):
    """Base class for SVERL masking methods."""

    _registry = {}

    def __init_subclass__(cls, **kwargs) -> None:
        """Automatically registers sampler subclasses."""
        super().__init_subclass__(**kwargs)
        Masker._registry[cls.__name__] = cls
        
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.setup(**kwargs)

    @classmethod
    def from_name(cls, name: str, **kwargs) -> 'Masker':
        # Checks that the specified type name exists in the masker registry
        if name not in cls._registry:
            raise ValueError(f'Unknown masker type: {name}')
        return cls._registry[name](**kwargs)
    
    def setup(self, **kwargs) -> None:
        """Sets up the state sampler for sampling."""
        pass

    @abstractmethod
    def mask(self, x: Tensor, mask: Tensor) -> None:
        """Applies the given mask to tensor `x`."""
        pass

    @abstractmethod
    def masked_like(self, x: Tensor) -> Tensor:
        """Returns a completely masked tensor with the shape of `x`."""
        pass