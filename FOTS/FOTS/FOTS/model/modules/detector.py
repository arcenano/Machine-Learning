import math  # Pi
import torch.nn as nn  # Layers
import torch  # Activation functions
from ...base.base_model import BaseModel  # Logging

class Detector(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.scoreMap = nn.Conv2d(32, 1, kernel_size=1)  # 160 x 160 x 1 output
        self.geoMap = nn.Conv2d(32, 4, kernel_size=1)  # 160 x 160 x 4 output
        self.angleMap = nn.Conv2d(32, 1, kernel_size=1)  # 160 x 160 x 1 output
        self.size = config.data_loader.size

    def forward(self, *input):
        (final,) = input

        score = self.scoreMap(final)
        score = torch.sigmoid(score)

        geoMap = self.geoMap(final)
        geoMap = torch.sigmoid(geoMap) * self.size  # TODO: 640 is the image size

        angleMap = self.angleMap(final)
        angleMap = (torch.sigmoid(angleMap) - 0.5) * math.pi

        geometry = torch.cat([geoMap, angleMap], dim=1)  # 320

        return score, geometry
