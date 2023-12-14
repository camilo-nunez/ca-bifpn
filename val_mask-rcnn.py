import argparse
import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from pprint import pprint
import uuid

import torch
import torchvision
from torchinfo import summary
from torchmetrics.detection import MeanAveragePrecision

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model.builder import BackboneNeck
from config.init import create_val_config
from utils.datasets import CocoDetectionV2, LVISDetection

AVAILABLE_DATASETS = ['coco2017', 'lvisv1']

## Customs configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parse_option():
    parser = argparse.ArgumentParser(
        'Thesis cnunezf validation Mask-RCNN script - B', add_help=True)
    
    parser.add_argument('--path_checkpoint',
                        action="extend",
                        nargs="+",
                        type=str,
                        required=True,
                        help="Checkpoint filename.")

    parser.add_argument('--summary',
                        action='store_true',
                        help="Display the summary of the model.")
    
    parser.add_argument('--save',
                        action='store_true',
                        help="Save the results in a .csv file.")
    
    parser.add_argument('--dataset_path',
                        type=str,
                        default='/thesis/classical', 
                        help='Path to complete DATASET.')
    
    parser.add_argument('--batch_size',
                        type=int,
                        default=1)

    args, unparsed = parser.parse_known_args()
    
    config, checkpoint = create_val_config(args)
    
    return args, config, checkpoint

def _create_model(base_config):
        # Create the backbone and neck model
        print(f'[i+] Configuring backbone and neck models with variables: {base_config.MODEL}')
        backbone_neck = BackboneNeck(base_config.MODEL)
        ## freeze the backbone
        for param in backbone_neck.backbone.parameters():
            param.requires_grad = False
        backbone_neck.out_channels = base_config.MODEL.NECK.NUM_CHANNELS
        print('[i+] Ready !')

        # MaskRCNN's head config
        print('[i+] Building the base model with MaskRCNN head ...')
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
                                                           num_classes=_num_classes + 1, # +1 = background
                                                           rpn_anchor_generator=anchor_generator,
                                                           box_roi_pool=roi_pooler,
                                                           mask_roi_pool=mask_roi_pooler).to(device)
        print('[i+] Ready !')
        
        return base_model
    
def _check_model_compatibility(l_bc):
    
    ## check if all baseconfig was train with the same dataset
    assert all([bc_i.DATASET.NAME for bc_i in l_bc])
    
if __name__ == '__main__':

    # Load configs for the model and the dataset
    args, l_bc, l_ch = parse_option()

    _check_model_compatibility(l_bc)
    
    # Check the principal exceptions outer
    if not torch.cuda.is_available(): raise Exception('This script is only available to run the inference mode in GPU.')
    
    # Load the dataset
    base_config = l_bc[0]
    print(f'[+] Loading {base_config.DATASET.NAME} dataset...')
    print(f'[++] Using batch_size: {base_config.TRAIN.ENV.BATCH_SIZE}')
    
    ## Albumentations to use
    val_transform = A.Compose([A.Resize(base_config.DATASET.IMAGE_SIZE, base_config.DATASET.IMAGE_SIZE),
                               A.Normalize(mean=base_config.DATASET.MEAN,
                                             std=base_config.DATASET.STD,
                                             max_pixel_value=255.0),
                               ToTensorV2()
                              ],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
                             )

    ## Validation dataset
    print('[++] Loading validation dataset...')
    val_params = {'batch_size': base_config.TRAIN.ENV.BATCH_SIZE,
                   'shuffle': False,
                   'drop_last': True,
                   'collate_fn': lambda batch: tuple(zip(*batch)),
                   'num_workers': 4,
                   'pin_memory':True,
                  }

    if base_config.DATASET.NAME == 'coco2017':
        val_dataset = CocoDetectionV2(root=os.path.join(base_config.DATASET.PATH,'coco2017/val2017'),
                                  annFile=os.path.join(base_config.DATASET.PATH,'coco2017/annotations/instances_val2017.json'),
                                  transform = val_transform
                                 )
    elif base_config.DATASET.NAME == 'lvisv1':
        val_dataset = LVISDetection(root=os.path.join(base_config.DATASET.PATH,'lvisdataset/val2017'),
                                        annFile=os.path.join(base_config.DATASET.PATH,'lvisdataset/lvis_v1_val.json'),
                                        transform = val_transform)


    val_loader = torch.utils.data.DataLoader(val_dataset, **val_params)
    print('[++] Ready !')
    print('[+] Ready !')
    
    df_l_metric_segm = []
    df_l_metric_bbox = []
    
    for i,data in enumerate(zip(l_bc, l_ch)):
        base_config, checkpoint = data
        
        # Check the principal exceptions inner
        if base_config.DATASET.NAME not in AVAILABLE_DATASETS: raise Exception(f'This script only work with the datasets {AVAILABLE_DATASETS}.')
        
        str_arc_name = f"{base_config.MODEL.BACKBONE.MODEL_NAME}_{base_config.MODEL.NECK.MODEL_NAME}"
        
        print(f'[+] Preparing model {i}_{str_arc_name}...')        
        base_model = _create_model(base_config)
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
    
        # Validation model phase
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")

        base_model.eval()

        print('[+] Starting validation ...')
        start_t = datetime.now()

        metric_segm = MeanAveragePrecision(box_format="xyxy", iou_type = "segm")
        metric_bbox = MeanAveragePrecision(box_format="xyxy",  iou_type = "bbox")

        for batch_idx, sample in enumerate(tqdm(val_loader)):
            images, targets = sample

            if None in images and None in targets: continue
            if not all(('boxes' in d.keys() and 'labels' in d.keys() and 'masks' in d.keys()) for d in targets): continue

            images = [image.to(device) for image in images]
            preds = base_model(images)
            preds = [{k: v.detach().to(cpu_device) for k, v in t.items()} for t in preds]

            torch.cuda.synchronize()

            for p in preds:
                p['labels'] = p['labels'].type(torch.IntTensor)
                p['masks'] = p['masks'].type(torch.BoolTensor).squeeze(1)

            for t in targets:
                t['labels'] = t['labels'].type(torch.IntTensor)
                t['masks'] = t['masks'].type(torch.BoolTensor)

            metric_segm.update(preds, targets)
            metric_bbox.update(preds, targets)

        end_t = datetime.now()
        
        df_metric_segm = metric_segm.compute()
        df_metric_bbox = metric_bbox.compute()
        
        df_metric_segm['name'] = str_arc_name + '_segm'
        df_metric_bbox['name'] = str_arc_name + '_bbox'
        
        df_l_metric_segm.append(pd.DataFrame.from_dict(df_metric_segm, orient='index'))
        df_l_metric_bbox.append(pd.DataFrame.from_dict(df_metric_bbox, orient='index'))

        print('[+] Ready, the validation phase took:', (end_t - start_t))
    
    pprint(pd.concat(df_l_metric_segm, axis=1).T)
    pprint(pd.concat(df_l_metric_bbox, axis=1).T)
    
    if args.save:
        rand_id = str(uuid.uuid1())[:8]
        pd.concat(df_l_metric_segm, axis=1).T.to_csv(f'{rand_id}_metric_segm.csv')
        pd.concat(df_l_metric_bbox, axis=1).T.to_csv(f'{rand_id}_metric_bbox.csv')