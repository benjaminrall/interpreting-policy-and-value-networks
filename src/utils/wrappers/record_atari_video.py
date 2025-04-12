import gymnasium as gym
from gymnasium.vector import VectorEnv
import numpy as np
from gymnasium import logger
import wandb
from gymnasium.core import ActType, ObsType, RenderFrame
from typing import Any, SupportsFloat

class RecordAtariVideo(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """
    Modified video recorder wrapper specifically for recording videos of Atari 
    vector environments and logging them to W&B instead of saving them locally.
    """

    counter: int = 0

    def __init__(self, env: VectorEnv, episode_trigger = None, fps = None):
        self.counter_value = RecordAtariVideo.counter
        RecordAtariVideo.counter += 1

        gym.Wrapper.__init__(self, env)

        if env.render_mode in {None, "human", "ansi"}:
            raise ValueError(
                f"Render mode is {env.render_mode}, which is incompatible with RecordVideo.",
                "Initialize your environment with a render_mode that returns an image, such as rgb_array.",
            )

        if episode_trigger is None:
            from gymnasium.utils.save_video import capped_cubic_video_schedule
            episode_trigger = capped_cubic_video_schedule

        self.episode_trigger = episode_trigger

        if fps is None:
            fps = self.metadata.get("render_fps", 30)
        self.frames_per_sec: int = fps
        self.recording: bool = False
        self.recorded_frames: list[RenderFrame] = []

        self.episode_id = -1

    def _capture_frame(self):
        assert self.recording, "Cannot capture a frame, recording wasn't started."

        frame = self.env.render()
        if isinstance(frame, list):
            if len(frame) == 0:  # render was called
                return
            frame = frame[-1]

        if isinstance(frame, np.ndarray):
            self.recorded_frames.append(frame)
        else:
            self.stop_recording()
            logger.warn(
                f"Recording stopped: expected type of frame returned by render to be a numpy array, got instead {type(frame)}."
            )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and eventually starts a new recording."""
        obs, info = super().reset(seed=seed, options=options)
        self.episode_id += 1

        if self.recording:
            self.stop_recording()

        if self.episode_trigger and self.episode_trigger(self.episode_id):
            self.start_recording()
        if self.recording:
            self._capture_frame()

        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        obs, rew, terminated, truncated, info = self.env.step(action)

        if self.recording:
            self._capture_frame()

        return obs, rew, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame]:
        """Compute the render frames as specified by render_mode attribute during initialization of the environment."""
        render_out = super().render()
        if self.recording and isinstance(render_out, list):
            self.recorded_frames += render_out
        return render_out

    def close(self):
        """Closes the wrapper then the video recorder."""
        super().close()
        if self.recording:
            self.stop_recording()

    def start_recording(self):
        """Starts a new recording. If already recording, stops the current recording before starting the new one."""
        # Modified recording functionality only records 1 environment in a VectorEnv.
        if self.counter_value > 0:
            return
        
        # Stops the current recording if one exists
        if self.recording:
            self.stop_recording()

        # Flags recording as true 
        self.recording = True

    def stop_recording(self):
        """Stops current recording and saves the video to W&B."""
        assert self.recording, 'stop_recording was called, but no recording was started'

        # Saves the recording to the current W&B instance instead of a local video file
        if len(self.recorded_frames) == 0:
            logger.warn('Ignored saving a video as there were zero frames to save.')
        else:
            wandb.log({'videos': wandb.Video(np.transpose(self.recorded_frames, (0, 3, 1, 2)), fps=self.frames_per_sec)})

        # Bonus deletion logic to ensure memory remains cleared
        while self.recorded_frames:
            frame = self.recorded_frames.pop()
            del frame
        
        # Resets recording variables
        self.recorded_frames = []
        self.recording = False

    def __del__(self):
        """Warn the user in case last video wasn't saved."""
        if len(self.recorded_frames) > 0:
            logger.warn("Unable to save last video! Did you call close()?")

