from src.models import Model
import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.agents import QLearning

class QLearningSimpleCritic(Model):
    """Test model for use on testing SVERL environments"""

    def _construct_model(self, agent: 'QLearning'):
        self.get_Q = agent.get_Q
    
    def forward(self, x):
        out = torch.zeros((x.shape[0], 1)).to(self.device)
        for i, obs in enumerate(x):
            q_values = self.get_Q(obs)
            out[i, 0] = q_values.max()
        return out

    def state_dict(self):
        return {}