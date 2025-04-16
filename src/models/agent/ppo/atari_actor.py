from src.models import Model
from torch import nn

class PPOAtariActor(Model):
    """PPO actor model for use in Atari environments."""

    def _construct_model(self, output_size) -> nn.Module:
        model = nn.Sequential(
            self._init_layer(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            self._init_layer(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self._init_layer(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self._init_layer(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            self._init_layer(nn.Linear(512, output_size), weight_std=0.01)
        )
        return model