import gymnasium as gym
import numpy as np
from gymnasium.vector import VectorEnv
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    ClipRewardEnv,
)
from ppo.config import PPOConfig
import ale_py
from gymnasium import error, logger, Env
import os
import wandb

def init_envs(cfg: PPOConfig) -> VectorEnv:
    """Creates a vectorised gymnasium environment based on the config."""
    # Function that determines when to record videos of the environment
    video_trigger = lambda t : t % 100 == 0

    # Constructs the vectorised environment from a gymnasium environment name
    envs = gym.make_vec(
        cfg.environment,
        num_envs=cfg.n_environments,
        vectorization_mode='sync',
        render_mode="rgb_array",
        wrappers=[
            gym.wrappers.RecordEpisodeStatistics,
            lambda envs : gym.wrappers.RecordVideo(envs, "videos/", episode_trigger=video_trigger),
        ],
    )

    # Adds extra wrappers for normalisation and clipping when using continuous actions
    if isinstance(envs.single_action_space, gym.spaces.Box):
        envs = gym.wrappers.vector.ClipAction(envs)
        envs = gym.wrappers.vector.NormalizeObservation(envs)
        envs = gym.wrappers.vector.TransformObservation(envs, lambda obs: np.clip(obs, -10, 10))
        envs = gym.wrappers.vector.NormalizeReward(envs)
        envs = gym.wrappers.vector.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

    # Seeds the environments' actions and observations
    envs.action_space.seed(cfg.seed)
    envs.observation_space.seed(cfg.seed)
    return envs

class RecordAtariVideo(gym.wrappers.RecordVideo):
    counter: int = 0

    def __init__(self, env, video_folder, episode_trigger = None, step_trigger = None, video_length = 0, name_prefix = "rl-video", fps = None, disable_logger = True):
        self.counter_value = RecordAtariVideo.counter
        RecordAtariVideo.counter += 1
        super().__init__(env, video_folder, episode_trigger, step_trigger, video_length, name_prefix, fps, disable_logger)

    def stop_recording(self):
        """Stop current recording and saves the video."""
        assert self.recording, "stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    'MoviePy is not installed, run `pip install "gymnasium[other]"`'
                ) from e

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            moviepy_logger = None if self.disable_logger else "bar"
            path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
            # clip.write_videofile(path, logger=moviepy_logger)
            # print(np.array(self.recorded_frames).shape)
            wandb.log({'videos': wandb.Video(np.transpose(self.recorded_frames, (0, 3, 1, 2)), fps=self.frames_per_sec)})

        # Bonus deletion logic to ensure memory remains cleared
        while self.recorded_frames:
            frame = self.recorded_frames.pop()
            del frame
        
        self.recorded_frames = []
        self.recording = False
        self._video_name = None

    def start_recording(self, video_name):
        if self.counter_value == 0:
            super().start_recording(video_name)

def init_atari_envs(cfg: PPOConfig, record=True) -> VectorEnv:
    """Creates a vectorised gymnasium environment based on the config."""
    gym.register_envs(ale_py)

    # Function that determines when to record videos of the environment
    video_trigger = lambda t : t % 10 == 0

    # Constructs the vectorised environment from a gymnasium environment name
    envs = gym.make_vec(
        cfg.environment,
        num_envs=cfg.n_environments,
        vectorization_mode='sync',
        render_mode='rgb_array',
        wrappers=[
            gym.wrappers.RecordEpisodeStatistics,
            lambda envs: RecordAtariVideo(envs, "videos/", episode_trigger=video_trigger) if record else envs,
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

def init_atari_playback_env(cfg: PPOConfig) -> Env:
    """Creates a vectorised gymnasium environment based on the config."""
    gym.register_envs(ale_py)

    env = gym.make(cfg.environment, render_mode='rgb_array')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env)
    env = EpisodicLifeEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.FrameStackObservation(env, 4)

    # Seeds the environments' actions and observations
    env.action_space.seed(cfg.seed)
    env.observation_space.seed(cfg.seed)
    return env
