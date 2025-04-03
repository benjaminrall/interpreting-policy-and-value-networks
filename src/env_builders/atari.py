import gymnasium as gym
from gymnasium.vector import VectorEnv
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    ClipRewardEnv,
)
from src.configs import EnvConfig
from src.env_builders import EnvBuilder
from src.utils import RecordAtariVideo
import ale_py

class AtariEnvBuilder(EnvBuilder):
    """Class to build Atari environments."""
    
    @staticmethod
    def _construct_env(cfg: EnvConfig) -> VectorEnv:
        """Constructs a vectorised Atari environment based on the config."""
        # Registers Atari environments from the ALE
        gym.register_envs(ale_py)

        # Function that determines when to record videos of the environment
        video_trigger = lambda episode : episode % cfg.record_interval

        # Constructs the vectorised environment from a gymnasium environment name
        envs = gym.make_vec(
            cfg.name,
            num_envs=cfg.num_envs,
            vectorization_mode='sync',
            render_mode='rgb_array',
            wrappers=[
                gym.wrappers.RecordEpisodeStatistics,
                lambda envs: RecordAtariVideo(envs, episode_trigger=video_trigger) if cfg.record else envs,
                lambda envs: NoopResetEnv(envs, noop_max=30),
                lambda envs: MaxAndSkipEnv(envs, skip=4),
                EpisodicLifeEnv,
                ClipRewardEnv,
                gym.wrappers.GrayscaleObservation,
                lambda envs: gym.wrappers.ResizeObservation(envs, (84, 84)),
                lambda envs: gym.wrappers.FrameStackObservation(envs, 4),
            ]
        )

        # Seeds the environments' actions and observations
        envs.action_space.seed(cfg.seed)
        envs.observation_space.seed(cfg.seed)
        return envs