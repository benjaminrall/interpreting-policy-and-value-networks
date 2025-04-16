from dataclasses import dataclass
from src.configs.sverl import SVERLConfig
from src.sverl import Shapley

@dataclass
class ShapleyConfig(SVERLConfig):
    """Configuration settings for approximating a SVERL shapley contribution function."""

    characteristic_checkpoint: str

    def get_model_type(self) -> Shapley:
        return Shapley