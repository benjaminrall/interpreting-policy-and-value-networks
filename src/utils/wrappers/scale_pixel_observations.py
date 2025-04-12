import gymnasium as gym
import numpy as np

class ScalePixelObservations(gym.ObservationWrapper):
    """Wrapper to normalise pixel observations to be between 0 and 1, instead of 0 and 255."""

    def __init__(self, env):
        super().__init__(env)

        # Updates the observation space to account for the normalised observations
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Scales incoming observations by 1 / 255 and converts to float32 space."""
        return obs.astype(np.float32) / 255.0
