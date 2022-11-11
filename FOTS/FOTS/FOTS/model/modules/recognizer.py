import torch.nn as nn  # Layers
import torch  # Activation functions
from ...base.base_model import BaseModel  # Logging
from .crnn import CRNN

class Recognizer(BaseModel):

    def __init__(self, nclass, config):
        super().__init__(config)
        self.crnn = CRNN(8, 32, nclass, 256)

    def forward(self, rois, lengths):
        return self.crnn(rois, lengths)


