from src import Trainable
from src.agents import Agent
from src.models.model import Model
from src.sverl.state_samplers import StateSampler
from torch.types import Tensor
import torch
from torch.utils.data import DataLoader

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.configs.sverl import SVERLConfig

class SVERLFunction(Trainable):
    """Abstract base class for all SVERL functions."""

    def __init__(self, cfg: 'SVERLConfig', state: dict = None):
        super().__init__(cfg, state)
        self.cfg = cfg

        # Loads the agent to be evaluated from the given checkpoint path
        self.agent: Agent = Trainable.load_checkpoint(cfg.agent_checkpoint)

        # Gets the target network for explaining with SVERL
        self.target = self.agent.actor if cfg.target == 'actor' else self.agent.critic

        # Builds the function's model
        self.model = Model.from_name(cfg.type + cfg.model, output_size=self.target.output_size)

        # Gets the state sampler for model training and validation
        self.state_sampler = StateSampler.from_name(cfg.state_sampler, agent=self.agent, target=self.target)
        self.validation_sampler = StateSampler.from_name(cfg.validation_sampler, agent=self.agent, target=self.target)
        
        # Loads state if it was provided
        if state is not None:
            self.model.load_state_dict(state['model'])

    def generate_validation_data(self) -> None:
        xs = self.validation_sampler.sample(self.cfg.validation_samples, self.cfg.validation_minibatch_size, shuffle=False)
        masks = [torch.rand(x.shape) < 0.5 for x in xs]
        return xs, masks

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict['model'] = self.model.state_dict()
        return state_dict