from dataclasses import dataclass

@dataclass
class EnvConfig:
    """Data class to represent the configuration of an environment."""

    type: str = 'Atari'                     # Type of environment builder to use
    name: str = 'BreakoutNoFrameskip-v4'    # Name of the environment
    num_envs: int = 8                       # Number of environments to use
    record: bool = False                    # Whether to record videos of the environment
    record_interval: int = 10               # How many episodes to leave between each video
    seed: int = 42                          # Seed for the environments
