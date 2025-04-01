from .ppo_agent import PPOAgent
from gymnasium.vector import VectorEnv
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from torch.types import Tensor
from torch.distributions.categorical import Categorical

class AtariPPOAgent(PPOAgent):
    """PPO Agent that supports Atari environments."""
    
    def _construct_critic(self) -> None:
        """Constructs and stores the critic network for the agent."""
        self.critic = nn.Sequential(
            self._init_layer(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            self._init_layer(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self._init_layer(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self._init_layer(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            self._init_layer(nn.Linear(512, 1), weight_std=1)
        )

    def _construct_actor(self) -> None:
        """Constructs and stores the actor network for the agent."""
        self.output_size = self.envs.single_action_space.n
        self.actor = nn.Sequential(
            self._init_layer(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            self._init_layer(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self._init_layer(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self._init_layer(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            self._init_layer(nn.Linear(512, self.output_size), weight_std=0.01)
        )

    def get_value(self, observation):
        return super().get_value(observation / 255.0)

    def get_action_and_value(self, observation, action = None):
        return super().get_action_and_value(observation / 255.0, action)