from dataclasses import dataclass
from src.configs.agent import AgentConfig
from src.agents import QLearning

@dataclass
class QLearningConfig(AgentConfig):
    """Configuration settings for training a Q-Learning agent."""
    
    total_timesteps: int = 10000
    epsilon: float = 0.05
    gamma: float = 1
    alpha: float = 0.2

    def get_model_type(self) -> QLearning:
        return QLearning