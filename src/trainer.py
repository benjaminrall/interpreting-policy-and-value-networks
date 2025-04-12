from src import Trainable
from dataclasses import asdict
from src.configs import TrainerConfig
import random
import torch
import numpy as np
import time
import wandb
from torch.utils.tensorboard.writer import SummaryWriter
from src.utils import to_nested_dict, get_random_state, restore_random_state
import os
from io import TextIOWrapper

class Trainer:
    """Object that wraps trainable objects to provide checkpoints and tracking."""

    def __init__(self, cfg: TrainerConfig, saved_objective: Trainable = None):
        self.cfg = cfg
        self.trained = False
        self.objective = self.get_objective() if saved_objective is None else saved_objective
        self.cfg.wandb_id = self.cfg.wandb_id if self.cfg.wandb_id is not None else str(int(time.time()))
        self.run_name = f"{self.cfg.run_name}-{self.cfg.wandb_id}"

    @classmethod
    def from_yaml(cls, file: TextIOWrapper) -> 'Trainer':
        """Constructs a Trainer object from a given YAML file."""
        cfg = TrainerConfig.from_yaml(file)
        return cls(cfg)
    
    def save_checkpoint(self, progress: int):
        """Saves a checkpoint of the current training progress."""
        if not self.trained:
            raise RuntimeError("Checkpoint cannot be saved for a Trainer that hasn't been used.")
        
        # Check if the checkpoint folder exists, if not create it
        checkpoint_folder = os.path.join(self.cfg.checkpoint_folder, self.run_name)
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
            
        # Creates the checkpoint dictionary
        checkpoint = {
            'cfg': self.cfg.to_dict(),
            'objective': self.objective.to_dict(),
            'random_state': get_random_state()
        }

        # Saves the checkpoint using torch
        torch.save(checkpoint, os.path.join(checkpoint_folder, f"{progress}.pt"))

    @classmethod
    def load_checkpoint(cls, path: str) -> 'Trainer':
        """Loads a trainer checkpoint from the specified path."""
        # Loads the checkpoint dictionary from the path
        checkpoint = torch.load(path, weights_only=False)

        # Gets the trainable objective from a dictionary
        objective = Trainable.from_dict(checkpoint['objective'])

        # Creates the trainer instance from the saved config and objective
        trainer_cfg = TrainerConfig(**checkpoint['cfg'])
        trainer = cls(trainer_cfg, saved_objective=objective)

        # Restores random state to continue training
        restore_random_state(checkpoint['random_state'])
        return trainer

    def _reset_seed(self) -> None:
        """Resets all relevant random states from the trainer's seed."""
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)
        torch.backends.cudnn.deterministic = True

    def get_objective(self) -> Trainable:
        """Constructs the trainer's objective."""
        self._reset_seed()
        objective = self.cfg.objective_type(self.cfg.objective_config)
        return objective

    def log(self, tag: str, value, global_step: int) -> None:
        """Logs the given value to Tensorboard as a scalar."""
        self.writer.add_scalar(tag, value, global_step)

    def update(self, progress: int) -> None:
        """Updates the trainer with the current progress, used for checkpointing."""
        if not self.trained:
            raise RuntimeError("Updates cannot be given to a Trainer that hasn't been used.")
        if self.cfg.save_checkpoints and progress % self.cfg.checkpoint_interval == 0:
            self.save_checkpoint(progress)

    def train(self):
        """Performs training."""
        # Prevents a trainer instance from being used more than once
        if self.trained:
            raise RuntimeError("Trainer instance has already been used - reuse is not allowed.")
        self.trained = True

        # Initialises W&B run and Tensorboard writer for the training run
        if self.cfg.track_wandb:
            self.run = wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                id=self.cfg.wandb_id,
                sync_tensorboard=True,
                name=self.run_name,
                config=to_nested_dict(self.cfg),
                monitor_gym=True,
                dir='./logs'
            )
        self.writer = SummaryWriter(f'logs/tensorboard/{self.run_name}')

        # Resets the seed and trains the objective
        self._reset_seed()
        self.objective.train(self)

    def __del__(self) -> None:
        """Ensures the W&B run is closed once the trainer is finished with."""
        if self.trained and self.cfg.track_wandb:
            self.run.finish()
