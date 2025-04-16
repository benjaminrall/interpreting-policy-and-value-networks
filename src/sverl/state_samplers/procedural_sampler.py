from src.sverl.state_samplers import StateSampler
from src.agents import Agent
from src.models import Model
from src.utils import get_device
from src.env_builders import EnvBuilder
import torch
import numpy as np
from torch.types import Tensor
from torch.utils.data import DataLoader, TensorDataset

class ProceduralSampler(StateSampler):
    """Samples states by generating them procedurally from agent interactions."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, agent: Agent, target: Model, **_):
        # Stores agent, target, and current device
        self.agent = agent
        self.target = target
        self.output_size = target.model[-1].out_features
        self.device = get_device()

        # Gets the env config and overrides the record setting to rebuild the environments
        self.env_config = agent.cfg.environment
        self.env_config.record = False
        self.envs = EnvBuilder.build(self.env_config)

        # Gets the initial state observation from the environments
        self.state = Tensor(self.envs.reset(seed=self.env_config.seed)[0]).to(self.device)

    def sample(self, n: int, batch_size: int = 64, shuffle: bool = True) -> list[Tensor]:
        """Samples states by stepping through the environment following the optimal policy."""
        # Initialises arrays to collect sample x and y values
        n = (n // self.envs.num_envs) * self.envs.num_envs
        sampled_xs = np.zeros((n,) + self.state.shape[1:])
        sampled_ys = np.zeros((n, self.output_size))

        for i in range(n // self.envs.num_envs):
            # Computes the action following the agent's policy, and the result of the target network
            with torch.no_grad():
                action = self.agent.get_action(self.state)
                y = self.target.forward(self.state)

            # Saves the sampled results
            sampled_xs[i * self.envs.num_envs:(i + 1) * self.envs.num_envs] = self.state.cpu().numpy()
            sampled_ys[i * self.envs.num_envs:(i + 1) * self.envs.num_envs] = y.cpu().numpy()

            # Progresses to the next state following the optimal policy
            new_state, _, _, _, _ = self.envs.step(action.cpu().numpy())
            self.state = Tensor(new_state).to(self.device)

        # Converts sampled data to tensors
        sampled_xs = Tensor(sampled_xs).to(self.device)
        sampled_ys = Tensor(sampled_ys).to(self.device)

        # Creates and returns a data loader for the samples
        dataset = TensorDataset(sampled_xs, sampled_ys)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)