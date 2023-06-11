#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from tqdm import tqdm
from datetime import datetime
import numpy as np

import torch
import torchvision
from torchinfo import summary

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model.builder_backbone import Backbone
from config.basic import create_config
from utils.datasets import VOCDetectionV2

## Customs configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECK_PATH = os.path.join('/thesis/checkpoint')

def parse_option():
    parser = argparse.ArgumentParser(
        'Thesis cnunezf training BiFPN + Fast-RCNN script - A1', add_help=True)
    
    parser.add_argument('--cfg_model',
                        type=str,
                        required=True,
                        metavar="FILE",
                        help='Path to MODEL config file. Must be a YAML file.'
                       )
    
    parser.add_argument('--cfg_dataset',
                        type=str,
                        required=True,
                        metavar="FILE",
                        help='Path to DATASET config file. Must be a YAML file.'
                       )
    
    parser.add_argument('--fpn_type',
                        type=str,
                        required=True,
                        help='Select the FPN backbone, this should be \'bs\' (refer to baseline) or \'ca\' (refer to context agregation).')
                        

    parser.add_argument('--lr', 
                        type=float, 
                        default=1e-4,
                        help='Learning rate used by the optimizer. Default is 1e-4.'
                       )
    
    parser.add_argument('--optim', 
                        type=str,
                        default='adamw', 
                        help='Select optimizer for training, suggest using \'admaw\' until the very final stage then switch to \'sgd\'.')
    
    parser.add_argument('--use_scheduler',
                        action='store_true',
                        help="Use the scheduler \'CyclicLR\', with \'triangular\' mode.")

    parser.add_argument('--num_epochs',
                        type=int,
                        default=40)
    
    parser.add_argument('--use_checkpoint',
                        action='store_true',
                        help="Load a checkpoint.")
    parser.add_argument('--path_checkpoint',
                        type=str,
                        metavar="FILE",
                        help="Path to the checkpoint file.")
    
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
    
    parser.add_argument('--dont_do_it',
                        action='store_true',
                        help='Dont use this arg !')
        
    args, unparsed = parser.parse_known_args()
    config = create_config(args)

    return args, config

if __name__ == '__main__':
                        
    # Load configs for the model and the dataset
    args, base_config = parse_option()
    
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
    
    # Load VOC 2012 dataset
    
    ## Albumentations to use
    print('[+] Loading VOC 2012 dataset...')
    print(f'[++] Using batch_size: {base_config.TRAIN.BATCH_SIZE}')
    train_transform = A.Compose([A.RandomBrightnessContrast(p=0.4),
                                 A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.6),
                                 A.InvertImg(p=0.5),
                                 A.Blur(p=0.5),
                                 A.GaussNoise(p=0.6),
                                 A.Flip(p=0.4),
                                 A.RandomRotate90(p=0.4),
                                 A.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225],
                                             max_pixel_value=255.0),
                                 ToTensorV2()
                                ],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
                               )
    
    ## Training dataset
    print('[++] Loading training dataset...')
    training_params = {'batch_size': base_config.TRAIN.BATCH_SIZE,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': lambda batch: tuple(zip(*batch)),
                       'num_workers': 8,
                       'pin_memory':True,
                      }
    train_dataset = VOCDetectionV2(root=os.path.join(base_config.DATASET.PATH), transform=train_transform)
    training_generator = torch.utils.data.DataLoader(train_dataset, **training_params)
    print('[++] Ready !')
    
    print('[+] Ready !')
    
    # General train variables
    ## Cofig the optimizer
    params = [p for p in base_model.parameters() if p.requires_grad]

    if args.optim == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.05)
        print('[+] Using AdamW optimizer')
    else:
        optimizer = torch.optim.SGD(params, base_config.TRAIN.BASE_LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
        print('[+] Using SGD optimizer')
        
    ## Learning rate scheduler
    if args.use_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2, step_size_up=2500, mode="triangular", cycle_momentum=False,)
        print('[+] Using CyclicLR scheduler')

    start_epoch = 1
    end_epoch = base_config.TRAIN.NUM_EPOCHS
    best_loss = 1e5
    loss_mean = 0
    
    ## Load the checkpoint if is need it
    if base_config.TRAIN.USE_CHECKPOINT:
        print('[+] Loading checkpoint...')
        checkpoint = torch.load(os.path.join(CHECK_PATH, base_config.TRAIN.CHECKPOINT_PATH))
        
        base_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if args.use_scheduler and checkpoint['lr_scheduler_state_dict']!=None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch'] + 1
        print(f'[+] Ready. start_epoch: {start_epoch} - best_loss: {best_loss}')

    # Train the model
    base_model.train()
    
    print('[+] Starting training ...')
    start_t = datetime.now()
    
    for epoch in range(start_epoch, end_epoch + 1):
        loss_l = []
        with tqdm(training_generator, unit=" batch") as tepoch:
            for images, targets in tepoch:

                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items() if (k=='boxes' or k=='labels')} for t in targets]

                if not all(('boxes' in d.keys() and 'labels' in d.keys()) for d in targets): continue


                loss_dict = base_model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                current_lr = optimizer.param_groups[0]['lr']

                loss_l.append(losses.item())
                loss_median = np.median(np.array(loss_l))


                tepoch.set_description('Epoch: {}/{}. lr: {:1.8f} loss_classifier: {:1.8f} - loss_box_reg: {:1.8f}'\
                                       ' - loss_objectness: {:1.8f} - loss_rpn_box_reg: {:1.8f}'\
                                       ' - total loss: {:1.8f} - median loss: {:1.8f}'\
                                       .format(epoch,end_epoch,current_lr,*loss_dict.values(),losses.item(), loss_median))
                if args.use_scheduler:
                    lr_scheduler.step()
                
                if args.dont_do_it: print('chao !'); exit();

        if loss_median < best_loss:
            best_loss = loss_median

            torch.save({'model_state_dict': base_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict() if args.use_scheduler else None,
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'fn_cfg_dataset': str(args.cfg_dataset), 
                        'fn_cfg_model': str(args.cfg_model),
                        'fpn_type': base_config.MODEL.BIFPN.TYPE,
                       },
                       os.path.join(CHECK_PATH, f'{datetime.utcnow().strftime("%Y%m%d_%H%M")}_{base_config.MODEL.BIFPN.TYPE}_{base_config.MODEL.BACKBONE.NAME}_{base_config.MODEL.BIFPN.NAME}_{epoch}.pth'))
    
    end_t = datetime.now()
    print('[+] Ready, the train phase took:', (end_t - start_t))