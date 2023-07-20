from .bifpn.model import BiFPN
from .cabifpn.model import CABiFPN

from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork

import torch.nn as nn

from typing import List

AVAILABLE_NECKS = ['fpn', 'bifpn', 'cabifpn']

class Neck(nn.Module):

    def __init__(self, 
                 model_name: str,
                 in_channels: List[int],
                 num_channels: int,
                 num_layers: int,
                ):
        super(Neck, self).__init__()

        mid_channels = [num_channels]*len(in_channels)
        
        self.neck = None
        
        if model_name == 'bifpn': 
            self.neck = nn.Sequential(*[BiFPN(num_channels, in_channels, first_time=True if _ == 0 else False) 
                                        for _ in range(num_layers)])
        elif model_name == 'cabifpn':
            self.neck = nn.Sequential(*[CABiFPN(num_channels, in_channels, first_time=True if _ == 0 else False) 
                                        for _ in range(num_layers)])
        elif model_name == 'fpn':
            self.neck = nn.Sequential(*[FeaturePyramidNetwork(in_channels if _ == 0 else mid_channels, num_channels)
                                        for _ in range(num_layers)])
        else:
            raise Exception("[+] Neck does not exist !. The neck name should be \'bifpn\' (refer to baseline) or \'cabifpn\' (refer to context agregation) or \'fpn\'.")

    def forward(self, x):
        features = self.neck(x)
        
        if isinstance(features, OrderedDict): return features
        else: return OrderedDict(zip([f'P{i}' for i in range(len(features))], features))