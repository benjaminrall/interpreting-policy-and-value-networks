from src.models import Model
from torch import nn

class CharacteristicLargerAtari(Model):
    """Larger characteristic function model for SVERL in Atari environments."""

    def _construct_model(self, output_size) -> nn.Module:
        model = nn.Sequential(
            self._init_layer(nn.Conv2d(4, 128, 8, stride=4)),
            nn.ReLU(),
            self._init_layer(nn.Conv2d(128, 256, 4, stride=2)),
            nn.ReLU(),\
            self._init_layer(nn.Conv2d(256, 256, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self._init_layer(nn.Linear(256 * 7 * 7, 1024)),
            nn.ReLU(),
            self._init_layer(nn.Linear(1024, 512)),
            nn.ReLU(),
            self._init_layer(nn.Linear(512, output_size), weight_std=0.01),
        )
        return model