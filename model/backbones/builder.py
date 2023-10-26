from .internimage.intern_image import InternImage

from torch.hub import load_state_dict_from_url
from omegaconf import OmegaConf
from timm import create_model
from collections import OrderedDict
from typing import List

import os
import torch.nn as nn

DIRNAME = os.path.dirname(__file__)

TIMM_AVAILABLE_MODELS = ['convnext_tiny',
                         'convnext_small',
                         'convnext_base',
                         'efficientnetv2_s',
                         'efficientnetv2_m',
                         'efficientnetv2_l',
                        ]

INTERNIMAGE_AVAILABLE_MODELS = {'internimage_t':'./internimage/configs/00_internimage_t_1k_224.yaml',
                                'internimage_s':'./internimage/configs/01_internimage_s_1k_224.yaml',
                                'internimage_b':'./internimage/configs/02_internimage_b_1k_224.yaml'}

AVAILABLE_BACKBONES = list(INTERNIMAGE_AVAILABLE_MODELS.keys()) + TIMM_AVAILABLE_MODELS

# Create InternImage
def builder_im(model_name: str, out_indices=(0, 1, 2, 3)):
    
    config = OmegaConf.load(os.path.join(DIRNAME, INTERNIMAGE_AVAILABLE_MODELS[model_name]))

    model = InternImage(norm_layer='LN',
                        drop_path_rate=config.DROP_PATH_RATE,
                        layer_scale=config.LAYER_SCALE,
                        post_norm=config.POST_NORM,
                        with_cp=config.WITH_CP,
                        core_op=config.CORE_OP,
                        channels=config.CHANNELS,
                        depths=config.DEPTHS,
                        groups=config.GROUPS,
                        mlp_ratio=config.MLP_RATIO,
                        offset_scale=config.OFFSET_SCALE,
                        out_indices=out_indices,
                       )
    checkpoint = load_state_dict_from_url(url=config.URL, map_location="cpu")
    out_n = model.load_state_dict(checkpoint['model'], strict=False)

    return model

# General builder
class Backbone(nn.Module):
    
    def __init__(self, 
                 model_name: str,
                 out_indices: List[int] = [0, 1, 2, 3],
                ):
        super(Backbone, self).__init__()
        
        if model_name in INTERNIMAGE_AVAILABLE_MODELS: # Create InternImage models
            self.backbone = builder_im(model_name, out_indices)
        elif model_name in TIMM_AVAILABLE_MODELS: # Create timm models
            self.backbone = create_model(model_name, pretrained=True, features_only=True, out_indices=out_indices)
        else:
            raise Exception(f'The model name does not exist. The available models are: {AVAIBLE_MODELS}')
        
    def forward(self, x):
        features = self.backbone(x)
        
        return OrderedDict(zip([f'P{i}' for i in range(len(features))], features))