from src.sverl.masking import Masker
from torch.types import Tensor
import torch

class ConstantMasker(Masker):
    """Class to perform state masking with a constant value, to be set to a value not in the state space."""

    def setup(self, value: float = 0, **_) -> None:
        self.value = value

    def mask(self, x: Tensor, mask: Tensor) -> None:
        x[mask] = self.value

    def masked_like(self, x):
        return torch.full_like(x, self.value)