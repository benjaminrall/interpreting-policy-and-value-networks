from dataclasses import dataclass
from src.configs.agent import PPOConfig
from src.agents import PPG

@dataclass
class PPGConfig(PPOConfig):
    """Configuration settings for training a PPO agent."""
    
    adv_norm_fullbatch: bool = True
    n_policy_iterations: int = 32
    auxiliary_update_epochs: int = 6
    beta_clone: float = 1.0
    n_auxiliary_rollouts: int = 4
    n_aux_grad_accum: int = 1

    @property
    def n_iterations(self) -> int:
        return self.total_timesteps // self.batch_size

    @property
    def n_phases(self) -> int:
        return self.n_iterations // self.n_policy_iterations
    
    @property
    def aux_batch_rollouts(self) -> int:
        return self.environment.num_envs * self.n_policy_iterations

    def get_model_type(self) -> PPG:
        return PPG