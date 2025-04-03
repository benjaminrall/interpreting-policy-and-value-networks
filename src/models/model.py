from abc import ABC, abstractmethod
import numpy as np
from torch import nn
from torch.types import Tensor

class Model(ABC, nn.Module):
    """Base class for all torch models to be used in agents or explanation techniques."""

    _registry = {}

    def __init_subclass__(cls, **kwargs) -> None:
        """Automatically registers model subclasses."""
        super().__init_subclass__(**kwargs)
        Model._registry[cls.__name__] = cls

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.model = self._construct_model(**kwargs)

    @staticmethod
    def _init_layer(layer: nn.Module, weight_std=np.sqrt(2), bias=0) -> nn.Module:
        """
        Initialises a module layer.
        Uses orthogonal initialisation for weights, and constant initialisation for biases.
        Works for any layer type that has 'weight' and 'bias' attributes, e.g. nn.Linear, nn.Conv2d.
        """
        # Initialises the weights
        if hasattr(layer, 'weight'):
            nn.init.orthogonal_(layer.weight, weight_std)

        # Initialises the bias
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.constant_(layer.bias, bias)

        return layer
     
    @abstractmethod
    def _construct_model(self, **kwargs) -> nn.Module:
        """Constructs and returns the model."""
        pass

    def forward(self, x: Tensor) -> Tensor:
        """Passes an input forward through the model."""
        return self.model(x)

    def to_dict(self) -> dict:
        """Saves a model to a model dict."""
        # Gets the model's name and state dict to be stored
        name = type(self).__name__
        state_dict = self.state_dict()

        # Constructs and returns the model dict
        model_dict = {
            'name': name,
            'state_dict': state_dict,
            'kwargs': self.kwargs
        }
        return model_dict

    @classmethod
    def from_dict(cls, model_dict: dict):
        """Instantiates a registered model from a model dict."""
        # Gets the name and state dict from the model dict
        name = model_dict.get('name')
        state_dict = model_dict.get('state_dict')
        kwargs = model_dict.get('kwargs', {})

        # Ensures that a type name was specified in the model dict
        if name is None:
            raise ValueError(f'Model dict must contain a `name` field.')

        # Checks that the specified type name exists in the model registry
        if name not in cls._registry:
            raise ValueError(f'Unknown model type: {name}')
        
        # Loads the model and (optionally) its state dictionary
        model: Model = cls._registry[name](**kwargs)
        if state_dict:
            model.load_state_dict(state_dict)
        return model
