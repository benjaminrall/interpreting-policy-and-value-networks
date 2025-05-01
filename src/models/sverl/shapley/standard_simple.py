from src.models import Model
from torch import nn

class ShapleyStandardSimple(Model):
    """Standard shapley function model for SVERL in simple environments."""

    def _construct_model(self, output_size) -> nn.Module:
        self.input_size = 12
        self.output_size = output_size
        model = nn.Sequential(
            self._init_layer(nn.Linear(self.input_size, 48), weight_std=0.01),
            nn.ReLU(),
            self._init_layer(nn.Linear(48, 48), weight_std=0.01),
            nn.ReLU(),
            self._init_layer(nn.Linear(48, self.input_size * self.output_size), weight_std=0.01),
        )
        return model
    
    def forward(self, x):
        out = self.model(x)
        out = out.view(-1, self.input_size, self.output_size)
        return out
