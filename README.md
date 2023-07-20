# Enhanced Feature Fusion Neck: A Comprehensive Implementation with Context Aggregation from Scratch
# Development and Deployment of a Context-Aware Feature Fusion Neck for Object Detection and Mask Segmentation
# Design, Implementation, and Application of a Contextually-Enhanced Feature Fusion Approach for Object Detection and Mask Segmentation

# Introduction

# Usage

## Install

## Data Preparation

## Training from Scratch

### Training Faster-RCNN A.K.A. `train_A`

```
python train_A.py --cfg_model FILE --cfg_dataset FILE [--dataset_path DATASET_PATH] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE] [--checkpoint_path CHECKPOINT_PATH] [--use_checkpoint] [--checkpoint_fn FILE] [--lr LR] [--wd WD] [--summary]
```

Variables description:
* `--cfg_model`:
* `--cfg_dataset`:
* `--dataset_path`:
* `--num_epochs`:
* `--batch_size`:
* `--checkpoint_path`:
* `--use_checkpoint`:
* `--checkpoint_fn`:
* `--lr`:
* `--wd`:
* `--summary`: Display the summary of the `base model` (backbone + neck + head) using [pytorch-summary](https://github.com/sksq96/pytorch-summary).

Available `cfg_model` configs files:

|              cfg_model                |
|:-------------------------------------:|
| `00_internimage_t_fpn.yaml          ` |
| `01_internimage_t_bifpn.yaml        ` |
| `02_internimage_t_cabifpn.yaml      ` |
| `10_internimage_s_fpn.yaml          ` |
| `11_internimage_s_bifpn.yaml        ` |
| `12_internimage_s_cabifpn.yaml      ` |
| `20_internimage_b_fpn.yaml          ` |
| `21_internimage_b_bifpn.yaml        ` |
| `22_internimage_b_cabifpn.yaml      ` |
| `30_convnext_tiny_fpn.yaml          ` |
| `31_convnext_tiny_bifpn.yaml        ` |
| `32_convnext_tiny_cabifpn.yaml      ` |
| `40_convnext_small_fpn.yaml         ` |
| `41_convnext_small_bifpn.yaml       ` |
| `42_convnext_small_cabifpn.yaml     ` |
| `50_convnext_base_fpn.yaml          ` |
| `51_convnext_base_bifpn.yaml        ` |
| `52_convnext_base_cabifpn.yaml      ` |
| `60_tf_efficientnetv2_s_fpn.yaml    ` |
| `61_tf_efficientnetv2_s_bifpn.yaml  ` |
| `62_tf_efficientnetv2_s_cabifpn.yaml` |
| `70_tf_efficientnetv2_m_fpn.yaml    ` |
| `71_tf_efficientnetv2_m_bifpn.yaml  ` |
| `72_tf_efficientnetv2_m_cabifpn.yaml` |
| `80_tf_efficientnetv2_l_fpn.yaml    ` |
| `81_tf_efficientnetv2_l_bifpn.yaml  ` |
| `82_tf_efficientnetv2_l_cabifpn.yaml` |


### Training MASK-RCNN A.K.A. `train_B`



```
python train_B1.py --cfg_dataset config/files/dataset/coco.yaml --cfg_model config/files/model/00_fpn_5l_internimage_t_1k_224.yaml --fpn_type bs --num_epochs 100 --batch_size 4 --with_neck_checkpoint
python train_B1.py --cfg_dataset config/files/dataset/coco.yaml --cfg_model config/files/model/01_fpn_5l_internimage_s_1k_224.yaml --fpn_type bs --num_epochs 100 --batch_size 4 --with_neck_checkpoint
python train_B1.py --cfg_dataset config/files/dataset/coco.yaml --cfg_model config/files/model/02_fpn_5l_internimage_b_1k_224.yaml --fpn_type bs --num_epochs 100 --batch_size 4 --with_neck_checkpoint
```