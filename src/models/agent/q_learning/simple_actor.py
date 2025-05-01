from src.models import Model
import numpy as np
import torch
from torch.distributions import Categorical

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.agents import QLearning

S = 8
K = 1

class QLearningSimpleActor(Model):
    """Test model for use on testing SVERL environments"""

    def _construct_model(self, output_size: int, agent: 'QLearning'):
        self.output_size = output_size
        self.actions = np.arange(output_size, dtype=int)
        self.get_Q = agent.get_Q
        self.epsilon = agent.cfg.epsilon
        self.device = agent.device
    
    def get_action(self, x, exp=False):
        out = torch.zeros(x.shape[0], dtype=int)
        for i, obs in enumerate(x):
            if np.random.rand() < self.epsilon and exp:
                action = np.random.choice(self.actions)
            else:
                q_values = self.get_Q(obs)
                action = np.random.choice(self.actions[q_values == q_values.max()])
            out[i] = action
        return out
    
    def forward(self, x):
        out = torch.zeros((x.shape[0], self.output_size)).to(self.device)
        for i, obs in enumerate(x):
            q_values = self.get_Q(obs).round(2)
            pi = q_values == q_values.max()
            dist = Categorical(probs=torch.tensor(pi).to(self.device))
            logits = S + K * (dist.logits - torch.max(dist.logits))
            out[i, :] = logits.unsqueeze(0)
        return out
    
    def state_dict(self):
        return {}
