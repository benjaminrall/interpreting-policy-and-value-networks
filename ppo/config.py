import yaml
from dataclasses import dataclass
from tabulate import tabulate
from io import TextIOWrapper

@dataclass
class PPOConfig:
    """Class to represent the configuration of a PPO run."""

    run_name: str = 'cartpole'
    environment: str = 'CartPole-v1'
    learning_rate: float = 2.5e-4
    adam_eps: float = 1e-5
    seed: int = 42
    total_timesteps: int = 25000
    n_environments: int = 4
    n_steps: int = 128
    n_recorded_steps: int = 128
    n_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    clip_coefficient: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    atari: bool = False
    track_wandb: bool = True
    wandb_project_name: str = "Year 3 Project"
    wandb_entity: str = None
    save_checkpoints: bool = True
    checkpoint_updates: int = 1

    @property
    def batch_size(self) -> int:
        return self.n_environments * self.n_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.n_minibatches

    def to_table(self) -> str:
        """Returns the configuration as a string formatted table."""
        data = self.__dict__.items()
        headers = ['name', 'value']
        return tabulate(data, headers, 'github')

    @staticmethod
    def from_yaml(file: TextIOWrapper | None) -> 'PPOConfig':
        """Constructs a Config object from a given YAML file."""
        # Returns the default configuration if no file is given
        if file is None:
            return PPOConfig()
        
        # Attempts to load the data from the YAML file and unpack into a Config instance
        data = yaml.safe_load(file)
        return PPOConfig(**data)
