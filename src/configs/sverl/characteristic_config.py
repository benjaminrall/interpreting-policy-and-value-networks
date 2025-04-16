from dataclasses import dataclass
from src.configs.sverl import SVERLConfig
from src.sverl import Characteristic

@dataclass
class CharacteristicConfig(SVERLConfig):
    """Configuration settings for approximating a SVERL characteristic function."""

    masker: str
    mask_kwargs: dict = None
    
    def __post_init__(self):
        if self.mask_kwargs is None:
            self.mask_kwargs = {}
    
    def get_model_type(self) -> Characteristic:
        return Characteristic