import torch
from torch import nn
from torch.optim import Adam
from torch.types import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from src.agents import Agent
from src.utils import get_device
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from src.env_builders import EnvBuilder
from src.models.model import Model
from torch.distributions import Categorical

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.configs.agent import QLearningConfig
    from src import Trainer

class QLearning(Agent):
    """Test agent."""
    
    def __init__(self, cfg: 'QLearningConfig', state: dict = None):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.device = get_device()
        self.to(self.device)
        self.state = state
        
        # Builds the environments from the env config
        self.envs = EnvBuilder.build(cfg.environment)
        self.output_size = self.envs.single_action_space.n

        self.Q_table = defaultdict(lambda: np.zeros(self.output_size))
        self.updates_completed = 0

        # Builds the actor and critic models
        self.actor = Model.from_name(cfg.type + cfg.actor, output_size=self.output_size, agent=self)
        self.critic = Model.from_name(cfg.type + cfg.critic, agent=self)

        # Loads state if it was provided
        if state is not None:
            self.Q_table = defaultdict(lambda: np.zeros(self.output_size), state['Q_table'])
            self.updates_completed = state['updates_completed']

    def get_key(self, obs):
        return [int(v) for v in obs]

    def set_Q(self, obs, values: np.ndarray):
        self.Q_table[*self.get_key(obs)] = values

    def get_Q(self, obs) -> np.ndarray:
        return self.Q_table[*self.get_key(obs)]

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict['Q_table'] = dict(self.Q_table)
        state_dict['updates_completed'] = self.updates_completed
        return state_dict

    def get_action(self, observation):
        return self.actor.get_action(observation).to(self.device)
    
    def get_value(self, observation):
        return super().get_value(observation).to(self.device)

    def update(self, obs, action, reward, n_obs, terminated):
        if terminated:
            q_max = 0
        else:
            q_max = self.get_Q(n_obs).max()

        td_error = reward + self.cfg.gamma * q_max - self.get_Q(obs)[action]
        self.Q_table[*self.get_key(obs)][action] += self.cfg.alpha * td_error

    def train(self, trainer: 'Trainer') -> None:
        obs = self.envs.reset(seed=trainer.cfg.seed)[0]
        avg_r, r, count = 0, 0, 0

        pbar = tqdm(range(1 + self.updates_completed, self.cfg.total_timesteps + 1), 'Q-Learning', initial=self.updates_completed, total=self.cfg.total_timesteps)
        for i in pbar:
            action = self.actor.get_action(obs, exp=True)
            n_obs, reward, terminated, truncated, _ = self.envs.step(action.cpu().numpy())
            r += reward[0]

            self.update(obs[0], action[0], reward[0], n_obs[0], terminated[0])

            if terminated[0] or truncated[0]:
                trainer.log('charts/episodic_return', r, i)

                avg_r = avg_r * count + r
                count += 1
                avg_r /= count
                r = 0

                if count % 1000 == 0:
                    pbar.set_description(f'Q-Learning; Avg Return: {avg_r:0.5f}')

            obs = n_obs

            trainer.update(i)

        trainer.save_checkpoint(i)
