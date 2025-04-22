import yaml
from io import TextIOWrapper
from src.configs import TrainableConfig
from dataclasses import asdict

class TrainerConfig:
    """Config class for experiments run by the Trainer."""

    def __init__(
            self, 
            run_name: str, 
            run_id: str = "",
            seed: int = 42,
            save_checkpoints: bool = True,
            checkpoint_interval: int = 1,
            checkpoint_folder: str =  './checkpoints',
            track_wandb: bool = True,
            wandb_project: str = 'Year 3 Project',
            wandb_entity: str = None,
            **kwargs
            ) -> None:
        # Stores given arguments
        self.run_name = run_name
        self.run_id = run_id
        self.seed = seed
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_folder = checkpoint_folder
        self.track_wandb = track_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

        # Gets the object config and type from remaining kwargs
        self.kwargs_type = None
        self.objective_config = self._get_objective_config(**kwargs)
        if self.objective_config is not None:
            self.objective_type = self.objective_config.get_model_type()
    
    def to_dict(self, include_objective: bool = False) -> dict:
        """Converts the config to a dict that can be used to reconstruct itself."""
        trainer_dict = {
            'run_name': self.run_name,
            'seed': self.seed,
            'save_checkpoints': self.save_checkpoints,
            'checkpoint_interval': self.checkpoint_interval,
            'checkpoint_folder': self.checkpoint_folder,
            'track_wandb': self.track_wandb,
            'wandb_project': self.wandb_project,
            'wandb_entity': self.wandb_entity,
        }
        if include_objective:
            objective_dict = { 
                self.kwargs_type: asdict(self.objective_config) 
            }
            trainer_dict.update(objective_dict)
        return trainer_dict

    def _get_objective_config(self, **kwargs) -> TrainableConfig:
        """Gets the config for the trainer's objective from given kwargs."""
        for key in TrainableConfig.subclasses():
            if key in kwargs:
                self.kwargs_type = key
                return TrainableConfig.subclass(key).from_dict(kwargs[key])

    @classmethod
    def from_yaml(cls, file: TextIOWrapper) -> 'TrainerConfig':
        """Constructs a Config object from a given YAML file."""
        # Returns the default configuration if no file is given
        if file is None:
            raise ValueError('No file specified')
        
        # Attempts to load the data from the YAML file and unpack into a Config instance
        data = yaml.safe_load(file)
        return cls(**data)
    
