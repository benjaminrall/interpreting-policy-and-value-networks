from abc import ABC, abstractmethod
import torch
from torch import nn
from torch.nn import Sequential
from torch.utils.tensorboard import SummaryWriter
from torch.types import Tensor
from gymnasium.vector import VectorEnv

class Agent(ABC, nn.Module):
    """Abstract base class for all actor-critic agents."""

    def __init__(self, cfg) -> None:
        super().__init__()
        
        