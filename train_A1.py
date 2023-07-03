import argparse
import os
from tqdm import tqdm
from datetime import datetime
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchinfo import summary

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model.builder_backbone import Backbone
from config.basic import create_train_config
from utils.datasets import VOCDetectionV2, CocoDetectionV2

AVAILABLE_DATASETS = ['coco2017', 'voc2012']

## Customs configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                        default=1e-3,
                        help='Learning rate used by the \'admaw\' optimizer. Default is 1e-3.'
                       )
    parser.add_argument('--wd', 
                        type=float, 
                        default=1e-5,
                        help='Weight decay used by the \'admaw\' optimizer. Default is 1e-5.'
                       )
    
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100)
    
    parser.add_argument('--use_checkpoint',
                        action='store_true',
                        help="Load a checkpoint.")
    parser.add_argument('--checkpoint_fn',
                        type=str,
                        metavar="FILE",
                        help="Checkpoint filename.")
    parser.add_argument('--checkpoint_path',
                        type=str,
                        default='/thesis/checkpoint', 
                        help='Path to complete DATASET.')
    
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
    config = create_train_config(args)

    return args, config

if __name__ == '__main__':
    
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
                        
    # Load configs for the model and the dataset
    args, base_config = parse_option()

    # Check the principal exceptions
    if not torch.cuda.is_available(): raise Exception('This script is only available to run in GPU.')
    if base_config.DATASET.NAME not in AVAILABLE_DATASETS: raise Exception('This script only work with the dataset VOC2012.')
        
    print(f'[+] backbone used: {base_config.MODEL.BACKBONE.NAME} - bifpn used: {base_config.MODEL.BIFPN.NAME} ')
    
    faster_rcnn_backbone = Backbone(base_config).to(device)
    faster_rcnn_backbone.out_channels = base_config.MODEL.BIFPN.NUM_CHANNELS 

    # FasterRCNN's head config
    print('[+] Building the base model with FasterRCNN head ...')
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['P0','P1','P2','P3'],
                                                    output_size=7,
                                                    sampling_ratio=2)
    anchor_sizes = ((32),(64),(128),(256)) 
    aspect_ratios = ((0.5,1.0,1.5,2.0,)) * len(anchor_sizes)
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
    
    # Load the dataset
    
    ## Albumentations to use
    print(f'[+] Loading {base_config.DATASET.NAME} dataset...')
    print(f'[++] Using batch_size: {base_config.TRAIN.BATCH_SIZE}')

    if base_config.DATASET.NAME == 'voc2012':
        bbox_dataset_params = A.BboxParams(format='pascal_voc', label_fields=['labels'])
    elif base_config.DATASET.NAME == 'coco2017':
        bbox_dataset_params = A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
    
    
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
                                ],bbox_params=bbox_dataset_params
                               )

    ## Training dataset
    print('[++] Loading training dataset...')
    training_params = {'batch_size': base_config.TRAIN.BATCH_SIZE,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': lambda batch: tuple(zip(*batch)),
                       'num_workers': 4,
                       'pin_memory':True,
                      }

    if base_config.DATASET.NAME == 'voc2012':
        train_dataset = VOCDetectionV2(root=os.path.join(base_config.DATASET.PATH), transform=train_transform)
    elif base_config.DATASET.NAME == 'coco2017':
        train_dataset = CocoDetectionV2(root=os.path.join(base_config.DATASET.PATH,'coco2017/train2017'),
                                        annFile=os.path.join(base_config.DATASET.PATH,'coco2017/annotations/instances_train2017.json'),
                                        transform = train_transform)

    training_loader = torch.utils.data.DataLoader(train_dataset, **training_params)
    print('[++] Ready !')
    
    print('[+] Ready !')
    
    # General train variables
    ## Cofig the optimizer
    params = [p for p in base_model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=base_config.TRAIN.OPTIM.BASE_LR,
                                  weight_decay=base_config.TRAIN.OPTIM.WEIGHT_DECAY)
    print(f'[+] Using AdamW optimizer. Configs:{base_config.TRAIN.OPTIM}')

    start_epoch = 1
    end_epoch = base_config.TRAIN.NUM_EPOCHS
    best_loss = 1e5
    global_steps = 0

    ## Load the checkpoint if is need it
    if base_config.TRAIN.USE_CHECKPOINT:
        print('[+] Loading checkpoint...')
        checkpoint = torch.load(os.path.join(args.checkpoint_path, base_config.TRAIN.CHECKPOINT_PATH))
        
        base_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch'] + 1
        print(f'[+] Ready. start_epoch: {start_epoch} - best_loss: {best_loss}')
        


    # Train the model
    base_model.train()
    
    print('[+] Starting training ...')
    start_t = datetime.now()
    
    from torchvision.transforms.functional import pil_to_tensor
    
    for epoch in range(start_epoch, end_epoch + 1):
        loss_l = []
        with tqdm(training_loader, unit=" batch") as tepoch:
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
                
                description_s = 'Epoch: {}/{}. lr: {:1.6f} loss_classifier: {:1.8f} - loss_box_reg: {:1.8f}'\
                                       ' - loss_objectness: {:1.8f} - loss_rpn_box_reg: {:1.8f}'\
                                       ' - total loss: {:1.8f} - median loss: {:1.8f}'\
                                       .format(epoch,end_epoch,current_lr,*loss_dict.values(),losses.item(), loss_median)

                tepoch.set_description(description_s)
                
                ## to board
                writer.add_text(f"A1_{base_config.MODEL.BIFPN.TYPE}_{base_config.MODEL.BACKBONE.NAME}_{base_config.MODEL.BIFPN.NAME}_{epoch}", description_s, global_steps)
                writer.add_scalar('Loss/loss_classifier', loss_dict['loss_classifier'], global_steps)
                writer.add_scalar('Loss/loss_box_reg', loss_dict['loss_box_reg'], global_steps)
                writer.add_scalar('Loss/loss_objectness', loss_dict['loss_objectness'], global_steps)
                writer.add_scalar('Loss/loss_rpn_box_reg', loss_dict['loss_rpn_box_reg'], global_steps)
                writer.add_scalar('Loss/total_loss', losses.item(), global_steps)
                writer.add_scalar('Loss/median_loss', loss_median, global_steps)
                
                global_steps+=1

        if loss_median < best_loss:
            best_loss = loss_median

            torch.save({'model_state_dict': base_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'fn_cfg_dataset': str(args.cfg_dataset), 
                        'fn_cfg_model': str(args.cfg_model),
                        'fpn_type': base_config.MODEL.BIFPN.TYPE,
                       },
                       os.path.join(args.checkpoint_path, f'{datetime.utcnow().strftime("%Y%m%d_%H%M")}_A1_{base_config.MODEL.BIFPN.TYPE}_{base_config.MODEL.BACKBONE.NAME}_{base_config.MODEL.BIFPN.NAME}_{epoch}.pth'))
    
    end_t = datetime.now()
    print('[+] Ready, the train phase took:', (end_t - start_t))
    
    writer.close()