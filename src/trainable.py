from abc import ABC, abstractmethod
import torch
from torch import nn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src import Trainer
    from src.configs import TrainableConfig

class Trainable(ABC, nn.Module):
    """Abstract base class for all trainable models."""

    def __init__(self, cfg: 'TrainableConfig', state: dict = None):
        super().__init__()
        self.cfg = cfg
        self.state = state

    def save(self, path: str) -> None:
        """Saves the trainable model's dictionary to the specified path."""
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str) -> 'Trainable':
        """Loads a trainable model from the specified path."""
        return cls.from_dict(torch.load(path, weights_only=False))

    def to_dict(self) -> dict:
        """Serialises a trainable model to a dictionary."""
        return {
            'type': type(self),
            'params': {
                'cfg': self.cfg,
                'state': self.get_state_dict(),
            }
        }

    @staticmethod
    def from_dict(d: dict) -> 'Trainable':
        """Deserialises a trainable model from a dictionary."""
        return d['type'](**d['params'])

    @abstractmethod
    def get_state_dict(self) -> dict:
        """Returns the state dict required to restore a trainable model."""
        return {}

    @abstractmethod
    def train(self, trainer: 'Trainer'):
        """Trains the trainable model, using the given Trainer instance."""
        pass