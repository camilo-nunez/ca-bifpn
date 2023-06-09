from .builder_internimage import builder_im
from .builder_fpn import builder_fpn

import torch.nn as nn

from collections import OrderedDict

class Backbone(nn.Module):
    
    def __init__(self, base_config):
        super().__init__()
        
        self.im_backbone = builder_im(base_config.MODEL.BACKBONE)
        self.fpn_backbone = nn.Sequential(*builder_fpn(base_config))

    def forward(self, x):
        features = self.im_backbone(x)
        intern_features = self.fpn_backbone(features)
        
        return OrderedDict(zip([f'P{i}' for i in range(len(intern_features))], intern_features))