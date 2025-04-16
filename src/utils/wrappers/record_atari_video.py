import gymnasium as gym
import numpy as np
from gymnasium import error, logger
import wandb

class RecordAtariVideo(gym.wrappers.RecordVideo):
    counter: int = 0

    def __init__(self, env, episode_trigger = None, step_trigger = None, video_length = 0, name_prefix = "rl-video", fps = None, disable_logger = True):
        self.counter_value = RecordAtariVideo.counter
        self.enabled = False
        RecordAtariVideo.counter += 1
        super().__init__(env, "logs", episode_trigger, step_trigger, video_length, name_prefix, fps, disable_logger)

    def stop_recording(self):
        """Stop current recording and saves the video."""
        if self.counter_value > 0:
            return
        assert self.recording, "stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            try:
                if wandb.run:
                    wandb.log({'videos': wandb.Video(np.transpose(self.recorded_frames, (0, 3, 1, 2)), fps=self.frames_per_sec)})
            except:
                logger.warn("Failed to save video to W&B.")

        # Bonus deletion logic to ensure memory remains cleared
        while self.recorded_frames:
            frame = self.recorded_frames.pop()
            del frame
        
        self.recorded_frames = []
        self.recording = False
        self.enabled = False
        self._video_name = None

    def start_recording(self, video_name):
        if self.counter_value == 0:
            super().start_recording(video_name)
            self.enabled = self.recording
