import argparse
import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
from omegaconf import OmegaConf

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchinfo import summary

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model.builder_backbone import Backbone
from config.basic import default_config
from utils.datasets import VOCDetectionV2, CocoDetectionV2

## Customs configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_option():
    parser = argparse.ArgumentParser(
        'Thesis cnunezf traning BiFPN + Fast-RCNN script - A2', add_help=True)
    
    parser.add_argument('--checkpoint_fn',
                        type=str,
                        metavar="FILE",
                        required=True,
                        help="Checkpoint filename.")
    parser.add_argument('--checkpoint_path',
                        type=str,
                        default='/thesis/checkpoint', 
                        help='Path to complete DATASET.')

    parser.add_argument('--num_epochs',
                        type=int,
                        required=True)

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
    
    parser.add_argument('--lr', 
                        type=float, 
                        default=5e-3,
                        help='Learning rate used by the \'sgd\' optimizer. Default is 5e-3.'
                       )
    parser.add_argument('--wd', 
                        type=float, 
                        default=5e-4,
                        help='Weight decay used by the \'sgd\' optimizer. Default is 5e-4.'
                       )
    parser.add_argument('--m', 
                        type=float, 
                        default=0.3,
                        help='Momentum used by the \'sgd\' optimizer. Default is 0.3.'
                       )
    
    parser.add_argument('--t0', 
                        type=int, 
                        default=5,
                        help='Number of iterations for the first restart, used by the \'CosineAnnealingWarmRestarts\' scheduler. Default is 5.'
                       )
    parser.add_argument('--t_mult', 
                        type=int, 
                        default=1,
                        help='A factor increases after a restart, used by the \'CosineAnnealingWarmRestarts\' scheduler. Default is 1.'
                       )

    parser.add_argument('--freeze_neck',
                        action='store_true',
                        help="Freeze the FPN neck.")

    args, unparsed = parser.parse_known_args()

    print('[+] Loading checkpoint...')
    checkpoint = torch.load(os.path.join(args.checkpoint_path, args.checkpoint_fn))
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
    if hasattr(args, 'num_epochs') and args.num_epochs:
        base_config.TRAIN.NUM_EPOCHS = args.num_epochs
        
    if hasattr(args, 'lr') and args.lr:
        base_config.TRAIN.OPTIM.BASE_LR = args.lr
    if hasattr(args, 'wd') and args.wd:
        base_config.TRAIN.OPTIM.WEIGHT_DECAY = args.wd
    if hasattr(args, 'm') and args.m:
        base_config.TRAIN.OPTIM.MOMENTUM = args.m
        
    if hasattr(args, 't0') and args.t0:
        base_config.TRAIN.SCHEDULER.T_0 = args.t0
    if hasattr(args, 't_mult') and args.t_mult:
        base_config.TRAIN.SCHEDULER.T_MULT = args.t_mult
    
    print('[+] Ready !')

    return args, base_config, checkpoint

if __name__ == '__main__':
    
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    # Load configs for the model and the dataset
    args, base_config, checkpoint = parse_option()
    
    # Check the principal exceptions
    if not torch.cuda.is_available(): raise Exception('This script is only available to run in GPU.')
        
    print(f'[+] backbone used: {base_config.MODEL.BACKBONE.NAME} - bifpn used: {base_config.MODEL.BIFPN.NAME} ')
    
    faster_rcnn_backbone = Backbone(base_config).to(device)
    faster_rcnn_backbone.out_channels = base_config.MODEL.BIFPN.NUM_CHANNELS 
    
    if hasattr(args, 'freeze_neck') and args.freeze_neck:
        for param in faster_rcnn_backbone.fpn_backbone.parameters():
            param.requires_grad = False
    
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
    print(f'[++] Number of classes: {_num_classes}')
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
    
    
    # Load dataset
    
    ## Albumentations to use
    print(f'[+] Loading {base_config.DATASET.NAME} dataset...')
    print(f'[++] Using batch_size: {base_config.TRAIN.BATCH_SIZE}')

    if base_config.DATASET.NAME == 'voc2012':
        bbox_dataset_params = A.BboxParams(format='pascal_voc', label_fields=['labels'])
    elif base_config.DATASET.NAME == 'coco2017':
        bbox_dataset_params = A.BboxParams(format='pascal_voc', label_fields=['category_ids'])

    train_transform = A.Compose([A.RandomBrightnessContrast(p=0.4),
                                 A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.4),
                                 A.InvertImg(p=0.4),
                                 A.Blur(p=0.4),
                                 A.GaussNoise(p=0.3),
                                 A.Flip(p=0.3),
                                 A.RandomRotate90(p=0.3),
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

    optimizer = torch.optim.SGD(params,
                                lr=base_config.TRAIN.OPTIM.BASE_LR,
                                momentum=base_config.TRAIN.OPTIM.MOMENTUM,
                                weight_decay=base_config.TRAIN.OPTIM.WEIGHT_DECAY,
                                nesterov=True)
    print(f'[+] Using SGD optimizer. Configs:{base_config.TRAIN.OPTIM}')

    start_epoch = checkpoint['epoch'] + 1
    end_epoch = base_config.TRAIN.NUM_EPOCHS
    if end_epoch < start_epoch: raise Exception(f'The number of epochs must be greater than {start_epoch} of the orignal train A1 epochs')
    best_loss = checkpoint['best_loss']
    global_steps = 0
    
    config_scheduler = {'T_0' : base_config.TRAIN.SCHEDULER.T_0,
                        'T_mult': base_config.TRAIN.SCHEDULER.T_MULT,
                       }
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **config_scheduler)
    print(f'[+] Using CosineAnnealingWarmRestarts scheduler - Configs: {config_scheduler} - epochs:{(end_epoch-start_epoch+1)}')

    # Train the model
    base_model.train()
    
    print('[+] Starting training ...')
    start_t = datetime.now()
    
    for epoch in range(start_epoch, end_epoch + 1):
        loss_l = []
        with tqdm(training_loader, unit=" batch") as tepoch:
            for i, data in enumerate(tepoch):
                images, targets = data

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
                                       ' - median loss: {:1.8f}'\
                                       .format(epoch,end_epoch,current_lr,*loss_dict.values(), loss_median))
                scheduler.step(epoch + i/len(tepoch))
                
                ## to board
                writer.add_scalar('Loss/loss_classifier', loss_dict['loss_classifier'], global_steps)
                writer.add_scalar('Loss/loss_box_reg', loss_dict['loss_box_reg'], global_steps)
                writer.add_scalar('Loss/loss_objectness', loss_dict['loss_objectness'], global_steps)
                writer.add_scalar('Loss/loss_rpn_box_reg', loss_dict['loss_rpn_box_reg'], global_steps)
                writer.add_scalar('Loss/median_loss', loss_median, global_steps)
                writer.add_scalar('current_lr', current_lr, global_steps)
                
                global_steps+=1

        if loss_median < best_loss:
            best_loss = loss_median

            torch.save({'model_state_dict': base_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'fn_cfg_dataset': checkpoint['fn_cfg_dataset'], 
                        'fn_cfg_model': checkpoint['fn_cfg_model'],
                        'fpn_type': base_config.MODEL.BIFPN.TYPE,
                       },
                       os.path.join(args.checkpoint_path, f'{datetime.utcnow().strftime("%Y%m%d_%H%M")}_A2_{base_config.MODEL.BIFPN.TYPE}_{base_config.MODEL.BACKBONE.NAME}_{base_config.MODEL.BIFPN.NAME}_{epoch}.pth'))
    
    end_t = datetime.now()
    print('[+] Ready, the train phase took:', (end_t - start_t))
    
    writer.close()