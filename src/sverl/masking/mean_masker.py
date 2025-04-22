from src.sverl.masking import Masker
from torch.types import Tensor
from src.agents import Agent
from src.utils import get_device
from src.env_builders import EnvBuilder
import numpy as np
import torch

class MeanMasker(Masker):
    """Class to perform state masking with the mean value of the initial state."""

    def setup(self, agent: Agent, **_) -> None:
        env_cfg = agent.cfg.environment
        env_cfg.num_envs = 1
        envs = EnvBuilder.build(env_cfg)
        self.mean = Tensor(envs.reset(seed=agent.cfg.environment.seed)[0]).mean().item()
        self.device = get_device()

    def mask(self, x: Tensor, mask: Tensor) -> None:
        x[mask] = self.mean

    def masked_like(self, x):
        return torch.full_like(x, self.mean)