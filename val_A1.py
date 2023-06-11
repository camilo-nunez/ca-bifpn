#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
from omegaconf import OmegaConf

import torch
import torchvision
from torchinfo import summary

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model.builder_backbone import Backbone
from config.basic import default_config
from utils.datasets import VOCDetectionV2
from utils.coco_eval import evaluateCOCO

## Customs configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECK_PATH = os.path.join('/thesis/checkpoint')

def parse_option():
    parser = argparse.ArgumentParser(
        'Thesis cnunezf validation BiFPN + Fast-RCNN script - A1', add_help=True)
    
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

    print('[+] Loading checkpoint...')
    checkpoint = torch.load(os.path.join(CHECK_PATH, args.path_checkpoint))
    print('[+] Ready !')
    
    print('[+] Preparing base configs...')
    model_conf = OmegaConf.load(checkpoint['fn_cfg_model'])
    dataset_conf = OmegaConf.load(checkpoint['fn_cfg_dataset'])
    def_config = default_config()
    
    base_config = OmegaConf.merge(def_config, model_conf, dataset_conf)
    
    base_config.MODEL.BIFPN.TYPE = checkpoint['fpn_type']
    
    if hasattr(args, 'batch_size') and args.batch_size:
        base_config.TRAIN.BATCH_SIZE = args.batch_size
    if hasattr(args, 'dataset_path') and args.dataset_path:
        base_config.DATASET.PATH = args.dataset_path
    
    print('[+] Ready !')

    return args, base_config, checkpoint


if __name__ == '__main__':

    # Load configs for the model and the dataset
    args, base_config, checkpoint = parse_option()
    
    # Check the principal exceptions
    if not torch.cuda.is_available(): raise Exception('This script is only available to run in GPU.')
    if base_config.DATASET.NAME!='voc2012': raise Exception('This script only work with the dataset VOC2012.')
        
    print(f'[+] backbone used: {base_config.MODEL.BACKBONE.NAME} - bifpn used: {base_config.MODEL.BIFPN.NAME} ')
    
    faster_rcnn_backbone = Backbone(base_config).to(device)
    faster_rcnn_backbone.out_channels = base_config.MODEL.BIFPN.NUM_CHANNELS 

    # FasterRCNN's head config
    print('[+] Building the base model with FasterRCNN head ...')
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['P0','P1','P2','P3'],
                                                    output_size=7, ## 3 o 7 o 14
                                                    sampling_ratio=2)
    anchor_sizes = ((32), (64), (128),(256) ) 
    aspect_ratios = ((0.5,1.0, 1.5,2.0,)) * len(anchor_sizes)
    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(anchor_sizes, aspect_ratios)
    
    # Create de base model with the FasterRCNN's head
    _num_classes = len(base_config.DATASET.OBJ_LIST)
    print(f'[++] Numbers of classes: {_num_classes}')
    base_model = torchvision.models.detection.FasterRCNN(faster_rcnn_backbone, 
                                                         num_classes=_num_classes,
                                                         rpn_anchor_generator=anchor_generator,
                                                         box_roi_pool=roi_pooler).to(device)
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
    
    # Load VOC 2012 dataset
    ## Albumentations to use
    print('[+] Loading VOC 2012 dataset...')
    val_transform = A.Compose([
                            A.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225],
                                        max_pixel_value=255.0,
                                       ),
                            ToTensorV2()
                          ],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    ## Training dataset
    print('[++] Loading validation dataset...')
    val_params = {'batch_size': base_config.TRAIN.BATCH_SIZE,
                   'shuffle': False,
                   'drop_last': True,
                   'collate_fn': lambda batch: tuple(zip(*batch)),
                   'num_workers': 8,
                   'pin_memory':True,
                  }
    dataset_val = VOCDetectionV2(root=os.path.join(base_config.DATASET.PATH), image_set="val", transform=val_transform)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, **val_params)
    print('[++] Ready !')
    
    print('[+] Ready !')

    # Validation model phase
    base_model.eval()
    
    print('[+] Starting validation ...')
    start_t = datetime.now()
    evaluateCOCO(base_model, data_loader_val, device=device)
    end_t = datetime.now()
    print('[+] Ready, the validation phase took:', (end_t - start_t))
    