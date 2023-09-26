# Design, Implementation, and Application of a Context Aggregation-Enhanced Feature Fusion Approach for Object Detection and Instance Segmentation

# Abstract
Feature fusion in Convolutional Neural Networks (CNNs) plays a critical role in improving the performance of specialized tasks, such as Object Detection and Instance Segmentation. This process integrates information from multiple hierarchical levels within the network to achieve a more comprehensive understanding of complex data. The incorporation of context features enriches the network's perspective, enabling more nuanced decision-making and thereby enhancing its ability to detect objects and segment instances in intricate scenes.

Despite the potential advantages of feature fusion, existing network architectures often exhibit limitations. While they are proficient in leveraging hierarchical data, these architectures frequently overlook the rich contextual information dispersed across different levels. Additionally, many feature fusion modules rely on elementary operations such as addition or concatenation, or employ rigid linear aggregation schemes for feature maps. Such practices can inadvertently impede the effective interplay between global and local context features, thereby constraining the network's capabilities.

To address these shortcomings, this research aims to design, implement, and evaluate a novel feature fusion neck architecture. Named CABiFPN, this architecture is engineered to seamlessly integrate contextual information from various receptive fields, with a particular focus on optimizing object recognition across different scales of feature maps. Our goal is to significantly boost performance in tasks like Object Detection and Instance Segmentation.

To rigorously assess its effectiveness, CABiFPN will be evaluated and validated using the COCO dataset. It will be compared against a range of existing feature extraction and feature fusion methodologies that represent the current state of the art.

This comparative analysis is designed to deepen our understanding of how context feature fusion operates within Convolutional Neural Networks, and to determine whether CABiFPN can offer a significant performance boost in specialized tasks like Object Detection and Instance Segmentation.

*Keywords*: Object Detection, Instance Segmentation, Feature Fusion, Context Feature.

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
Local
```
python train_B.py --cfg_model config/files/model/00_internimage_t_fpn.yaml --cfg_dataset config/files/dataset/coco.yaml --summary
```

server universidad
```
python train_B.py --cfg_model config/files/model/10_internimage_s_fpn.yaml --cfg_dataset config/files/dataset/coco.yaml --summary --dataset_path /chi2ad/thesis/datasets/ --checkpoint_path /chi2ad/thesis/checkpoint/ --batch_size 8
```
