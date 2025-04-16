from src.sverl.masking import Masker
from torch.types import Tensor
from src.agents import Agent
from src.utils import get_device

class NoiseMasker(Masker):
    """Class to perform state masking with uniformly random noise from the state space."""

    def setup(self, agent: Agent, **_) -> None:
        self.state_space = agent.envs.single_observation_space
        self.device = get_device()

    def mask(self, x: Tensor, mask: Tensor) -> None:
        sample = Tensor(self.state_space.sample()).to(self.device)
        x[mask] = sample.unsqueeze(0).expand_as(x)[mask]

    def masked_like(self, x):
        sample = Tensor(self.state_space.sample()).to(self.device)
        return sample.unsqueeze(0).expand_as(x)