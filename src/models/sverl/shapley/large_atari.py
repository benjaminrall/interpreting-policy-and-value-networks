from src.models import Model
from torch import nn
import torch

def conv_block(in_channels, out_channels):
    """Returns a convolutional block for the UNet architecture."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(),
    )

class ShapleyLargeAtari(Model):
    """Standard shapley function model for SVERL in Atari environments."""

    def _construct_model(self, output_size) -> nn.Module:
        self.output_size = output_size
        self.down_1 = conv_block(4, 64)
        self.downsample_1 = nn.MaxPool2d(2)
        self.down_2 = conv_block(64, 128)
        self.downsample_2 = nn.MaxPool2d(2)
        self.down_3 = conv_block(128, 256)
        self.downsample_3 = nn.MaxPool2d(3)
        self.bottleneck = conv_block(256, 512)
        self.upsample_1 = nn.ConvTranspose2d(512, 256, 3, stride=3)
        self.up_1 = conv_block(512, 256)
        self.upsample_2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_2 = conv_block(256, 128)
        self.upsample_3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_3 = conv_block(128, 64)
        self.final = nn.Conv2d(64, 4 * output_size, 1)
        return None
    
    def forward(self, x):
        batch_size = x.shape[0]
        step_1 = self.down_1(x)
        step_2 = self.downsample_1(step_1)
        step_3 = self.down_2(step_2)
        step_4 = self.downsample_2(step_3)
        step_5 = self.down_3(step_4)
        step_6 = self.downsample_3(step_5)
        step_7 = self.bottleneck(step_6)
        step_8 = self.upsample_1(step_7)
        step_9 = self.up_1(torch.cat((step_5, step_8), dim=1))
        step_10 = self.upsample_2(step_9)
        step_11 = self.up_2(torch.cat((step_3, step_10), dim=1))
        step_12 = self.upsample_3(step_11)
        step_13 = self.up_3(torch.cat((step_1, step_12), dim=1))
        step_14 = self.final(step_13)
        res = step_14.view(batch_size, 4, self.output_size, 84, 84).permute(0, 1, 3, 4, 2)
        return res