import gymnasium as gym
from gymnasium.vector import VectorEnv
from src.configs import EnvConfig
from src.env_builders import EnvBuilder
import sverl

class SimpleEnvBuilder(EnvBuilder):
    """Class to build Atari environments."""
    
    @staticmethod
    def _construct_env(cfg: EnvConfig) -> VectorEnv:
        """Constructs a vectorised Atari environment based on the config."""
        # Constructs the vectorised environment from a gymnasium environment name
        envs = gym.make_vec(
            cfg.name,
            num_envs=1,
            vectorization_mode='sync',
        )

        # Seeds the environments' actions and observations
        envs.action_space.seed(cfg.seed)
        envs.observation_space.seed(cfg.seed)
        return envs