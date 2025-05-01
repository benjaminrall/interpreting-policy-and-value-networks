from src.models import Model
from torch import nn

class CharacteristicStandardSimple(Model):
    """Standard characteristic function model for SVERL in simple environments."""

    def _construct_model(self, output_size) -> nn.Module:
        input_size = 12
        model = nn.Sequential(
            self._init_layer(nn.Linear(input_size, 128), weight_std=0.01),
            nn.ReLU(),
            self._init_layer(nn.Linear(128, 128), weight_std=0.01),
            nn.ReLU(),
            self._init_layer(nn.Linear(128, output_size), weight_std=0.01)
        )
        return model