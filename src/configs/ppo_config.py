from dataclasses import dataclass
from src.configs import AgentConfig

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
    value_coefficient: float = 0.4
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
