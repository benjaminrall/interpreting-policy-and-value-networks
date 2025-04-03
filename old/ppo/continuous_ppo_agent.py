from .ppo_agent import PPOAgent
from gymnasium.vector import VectorEnv
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from torch.types import Tensor

class ContinuousPPOAgent(PPOAgent):
    """PPO Agent that supports continuous action spaces."""
    
    def _construct_critic(self) -> None:
        """Constructs and stores the critic network for the agent."""
        input_size = np.array(self.envs.single_observation_space.shape).prod()
        self.critic = nn.Sequential(
            self._init_layer(nn.Linear(input_size, 128)),
            nn.Tanh(),
            self._init_layer(nn.Linear(128, 128)),
            nn.Tanh(),
            self._init_layer(nn.Linear(128, 128)),
            nn.Tanh(),
            self._init_layer(nn.Linear(128, 128)),
            nn.Tanh(),
            self._init_layer(nn.Linear(128, 1), weight_std=1)
        )

    def _construct_actor(self) -> None:
        """Constructs and stores the actor network for the agent."""
        input_size = np.array(self.envs.single_observation_space.shape).prod()
        output_size = np.array(self.envs.single_action_space.shape).prod()
        self.actor_mean = nn.Sequential(
            self._init_layer(nn.Linear(input_size, 256)),
            nn.Tanh(),
            self._init_layer(nn.Linear(256, 256)),
            nn.Tanh(),
            self._init_layer(nn.Linear(256, 256)),
            nn.Tanh(),
            self._init_layer(nn.Linear(256, output_size), weight_std=0.01)
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1, output_size))


    def get_action_and_value(
        self, observation: Tensor, action: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Bundles actor and critic evaluation to return a sampled action and value.

        Parameters
        ----------
        observation : Tensor
            Current observation from the environment.
        action : Tensor | None, optional
            Specified action to be taken in the current state, by default None,
            in which case the action is sampled from the actor network for rollout

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            - Sampled action to be taken given the observation (or the specified value)
            - Log probability of the returned action
            - Entropies of the action probability distribution
            - Critic value of the observation
        """

        # Gets means and standard deviation for the action distribution from the actor networks
        action_mean = self.actor_mean(observation)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        # Uses a normal distribution as probabilities for the actions
        probs = Normal(action_mean, action_std)

        # If no action was specified, sample it from the probabilities
        action = action if action is not None else probs.sample()

        # Returns the tuple of action details and value
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(observation)
