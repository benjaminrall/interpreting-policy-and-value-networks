from src.models.sverl.characteristic import SVERLStandardAtariPC

print(SVERLStandardAtariPC(output_size=4))

from dataclasses import dataclass
import yaml
from src.configs import TrainerConfig

def from_yaml(file) -> TrainerConfig:
    """Constructs a Config object from a given YAML file."""
    # Returns the default configuration if no file is given
    if file is None:
        raise ValueError('No file specified')
    # Attempts to load the data from the YAML file and unpack into a Config instance
    data = yaml.safe_load(file)
    print(data)
    return TrainerConfig(**data)

with open('configs/breakout/breakout_new.yaml', 'r') as f:
    cfg = from_yaml(f)

print(cfg.objective)