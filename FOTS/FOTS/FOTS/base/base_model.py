from loguru import logger  # Simplified Logging
import torch.nn as nn # Basemodel Module inheritance
import numpy as np # Summing model parameters

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.logger = logger

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)
