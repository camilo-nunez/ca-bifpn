import argparse
import os
from datetime import datetime
import numpy as np
from omegaconf import OmegaConf

import torch
import torchvision
from torchinfo import summary

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model.builder import BackboneNeck, AVAILABLE_NECKS, AVAILABLE_BACKBONES
from config.init import create_val_config
from utils.datasets import CocoDetectionV2
from utils.coco_eval import evaluateCOCO

## Customs configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AVAILABLE_DATASETS = ['coco2017']

def parse_option():
    parser = argparse.ArgumentParser(
        'Thesis cnunezf validation Mask-RCNN script - A', add_help=True)
    
    parser.add_argument('--path_checkpoint',
                        type=str,
                        metavar="FILE",
                        required=True,
                        help="Checkpoint filename.")

    parser.add_argument('--summary',
                        action='store_true',
                        help="Display the summary of the model.")
    
    parser.add_argument('--dataset_path',
                        type=str,
                        default='/thesis/classical', 
                        help='Path to complete DATASET.')
    
    parser.add_argument('--batch_size',
                        type=int,
                        default=2)

    args, unparsed = parser.parse_known_args()
    config, checkpoint = create_val_config(args)
    
    return args, config, checkpoint
    
if __name__ == '__main__':

    # Load configs for the model and the dataset
    args, base_config, checkpoint = parse_option()
    
    # Check the principal exceptions
    if not torch.cuda.is_available(): raise Exception('This script is only available to run in GPU.')
    if base_config.DATASET.NAME not in AVAILABLE_DATASETS: raise Exception(f'This script only work with the datasets {AVAILABLE_DATASETS}.')
        
    # Create the backbone and neck model
    print(f'[+] Configuring backbone and neck models with variables: {base_config.MODEL}')
    backbone_neck = BackboneNeck(base_config.MODEL)
    ## freeze the backbone
    for param in backbone_neck.backbone.parameters():
        param.requires_grad = False
    backbone_neck.out_channels = base_config.MODEL.NECK.NUM_CHANNELS
    print('[+] Ready !')

    # MaskRCNN's head config
    print('[+] Building the base model with MaskRCNN head ...')
    anchor_sizes = ((32),(64),(128),(256)) 
    aspect_ratios = ((0.5,1.0,1.5,2.0,)) * len(anchor_sizes)
    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(anchor_sizes, aspect_ratios)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['P0','P1','P2','P3'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['P0','P1','P2','P3'],
                                                         output_size=14,
                                                         sampling_ratio=2)

    # Create de base model with the FasterRCNN's head
    _num_classes = len(base_config.DATASET.OBJ_LIST)
    print(f'[++] Numbers of classes: {_num_classes}')
    base_model = torchvision.models.detection.MaskRCNN(backbone_neck,
                                                       num_classes=_num_classes,
                                                       rpn_anchor_generator=anchor_generator,
                                                       box_roi_pool=roi_pooler,
                                                       mask_roi_pool=mask_roi_pooler).to(device)
    print('[+] Ready !')

    # Display the summary of the net
    if args.summary: summary(base_model)
        
    print('[+] Loading checkpoint...')
    out_n = base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if len(out_n.unexpected_keys)!=0: 
        print(f'[++] The unexpected keys was: {out_n.unexpected_keys}')
    else:
        print('[++] All keys matched successfully')
    print(f"[+] Ready. last_epoch: {checkpoint['epoch']} - last_loss: {checkpoint['best_loss']}")
    
    # Load the dataset
    print(f'[+] Loading {base_config.DATASET.NAME} dataset...')
    print(f'[++] Using batch_size: {base_config.TRAIN.ENV.BATCH_SIZE}')
    
    ## Albumentations to use
    train_transform = A.Compose([A.Resize(base_config.DATASET.IMAGE_SIZE, base_config.DATASET.IMAGE_SIZE),
                                 A.Normalize(mean=base_config.DATASET.MEAN,
                                             std=base_config.DATASET.STD,
                                             max_pixel_value=255.0),
                                 ToTensorV2()
                                ],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
                               )
    ## Training dataset
    print('[++] Loading validation dataset...')
    val_params = {'batch_size': base_config.TRAIN.ENV.BATCH_SIZE,
                   'shuffle': False,
                   'drop_last': True,
                   'collate_fn': lambda batch: tuple(zip(*batch)),
                   'num_workers': 4,
                   'pin_memory':True,
                  }

    val_dataset = CocoDetectionV2(root=os.path.join(base_config.DATASET.PATH,'coco2017/train2017'),
                                  annFile=os.path.join(base_config.DATASET.PATH,'coco2017/annotations/instances_train2017.json'),
                                  transform = train_transform)

    val_loader = torch.utils.data.DataLoader(val_dataset, **val_params)
    print('[++] Ready !')
    print('[+] Ready !')
    
    # Validation model phase
    base_model.eval()
    
    print('[+] Starting validation ...')
    start_t = datetime.now()
    evaluateCOCO(base_model, val_loader, device=device)
    end_t = datetime.now()
    print('[+] Ready, the validation phase took:', (end_t - start_t))