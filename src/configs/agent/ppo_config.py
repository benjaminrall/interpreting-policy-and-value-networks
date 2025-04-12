from dataclasses import dataclass
from src.configs.agent import AgentConfig
from src.agents import PPO

@dataclass
class PPOConfig(AgentConfig):
    """Configuration settings for training a PPO agent."""
    
    learning_rate: float = 2.5e-4
    adam_eps: float = 1e-5
    total_timesteps: int = 25000
    n_steps: int = 128
    n_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    clip_coefficient: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    anneal_lr: bool = True

    @property
    def batch_size(self) -> int:
        return self.environment.num_envs * self.n_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.n_minibatches

    def get_model_type(self) -> PPO:
        return PPO