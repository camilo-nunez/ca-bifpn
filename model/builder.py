from .backbones.builder import Backbone, AVAILABLE_BACKBONES
from .necks.builder import Neck, AVAILABLE_NECKS

import torch.nn as nn

from omegaconf.dictconfig import DictConfig

class BackboneNeck(nn.Module):
    
    def __init__(self,
                 base_config: DictConfig
                ):
        super(BackboneNeck, self).__init__()
        
        self.backbone = Backbone(model_name = base_config.BACKBONE.MODEL_NAME,
                                 out_indices = base_config.BACKBONE.OUT_INDICES)

        self.neck = Neck(model_name = base_config.NECK.MODEL_NAME,
                         in_channels = base_config.NECK.IN_CHANNELS,
                         num_channels = base_config.NECK.NUM_CHANNELS,
                         num_layers = base_config.NECK.NUM_LAYERS )

    def forward(self, x):
        features = self.backbone(x)
        fusioned_features = self.neck(features)
        
        return fusioned_features