from src.sverl.masking import Masker
from torch.types import Tensor
import torch

class ZeroMasker(Masker):
    """Class to perform state masking with zeroes."""

    def mask(self, x: Tensor, mask: Tensor) -> None:
        x[mask] = 0

    def masked_like(self, x: Tensor):
        return torch.zeros_like(x)