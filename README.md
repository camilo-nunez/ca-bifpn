# ca-bifpn

# Training

Iniciar entrenamiento con 10 epochs para `01_fpn_5l_internimage_t_1k_224.yaml`.

## Traning Baseline BS
```
root@35f9a50f23da:/workspace/ca-bifpn# python3.8 train_A1.py --cfg_dataset config/files/dataset/voc.yaml --cfg_model config/files/model/00_fpn_5l_internimage_t_1k_224.yaml --fpn_type bs --num_epochs 5 --summary
[+] backbone used: internimage_t_1k_224 - bifpn used: fpn_5l 
using core type: DCNv3
using activation layer: GELU
using main norm layer: LN
using dpr: linear, 0.2
level2_post_norm: False
level2_post_norm_block_ids: None
res_post_norm: False
[+] The unexpected keys was: ['conv_head.0.weight', 'conv_head.1.0.weight', 'conv_head.1.0.bias', 'conv_head.1.0.running_mean', 'conv_head.1.0.running_var', 'conv_head.1.0.num_batches_tracked', 'head.weight', 'head.bias']
[+] BiFPN loaded
[+] Building the base model with FasterRCNN head ...
[++] Numbers of classes: 21
[+] Ready !
====================================================================================================
Layer (type:depth-idx)                                                      Param #
====================================================================================================
FasterRCNN                                                                  --
├─GeneralizedRCNNTransform: 1-1                                             --
├─Backbone: 1-2                                                             --
│    └─InternImage: 2-1                                                     --
│    │    └─StemLayer: 3-1                                                  (19,584)
│    │    └─Dropout: 3-2                                                    --
│    │    └─ModuleList: 3-3                                                 (28,749,936)
│    └─Sequential: 2-2                                                      --
│    │    └─BiFPN: 3-4                                                      192,207
│    │    └─BiFPN: 3-5                                                      83,343
│    │    └─BiFPN: 3-6                                                      83,343
│    │    └─BiFPN: 3-7                                                      83,343
│    │    └─BiFPN: 3-8                                                      83,343
├─RegionProposalNetwork: 1-3                                                --
│    └─AnchorGenerator: 2-3                                                 --
│    └─RPNHead: 2-4                                                         --
│    │    └─Sequential: 3-9                                                 113,008
│    │    └─Conv2d: 3-10                                                    1,808
│    │    └─Conv2d: 3-11                                                    7,232
├─RoIHeads: 1-4                                                             --
│    └─MultiScaleRoIAlign: 2-5                                              --
│    └─TwoMLPHead: 2-6                                                      --
│    │    └─Linear: 3-12                                                    5,620,736
│    │    └─Linear: 3-13                                                    1,049,600
│    └─FastRCNNPredictor: 2-7                                               --
│    │    └─Linear: 3-14                                                    21,525
│    │    └─Linear: 3-15                                                    86,100
====================================================================================================
Total params: 36,195,108
Trainable params: 7,425,588
Non-trainable params: 28,769,520
====================================================================================================
[+] Loading VOC 2012 dataset...
[++] Using batch_size: 2
[++] Loading training dataset...
[++] Ready !
[++] Ready !
[+] Using AdamW optimizer
[+] Starting training ...
Epoch: 1/5. lr: 0.00010000 loss_classifier: 0.08039355 - loss_box_reg: 0.04585745 - loss_objectness: 0.02746056 - loss_rpn_box_reg: 0.29348168 - total loss: 0.44719326 - median loss: 0.72017482: 100%|█████████████████████| 2858/2858 [07:57<00:00,  5.99 batch/s]
Epoch: 2/5. lr: 0.00010000 loss_classifier: 0.04102251 - loss_box_reg: 0.02024660 - loss_objectness: 0.05082167 - loss_rpn_box_reg: 0.12687318 - total loss: 0.23896396 - median loss: 0.62789923: 100%|█████████████████████| 2858/2858 [07:53<00:00,  6.03 batch/s]
Epoch: 3/5. lr: 0.00010000 loss_classifier: 0.20362535 - loss_box_reg: 0.12495139 - loss_objectness: 0.28768075 - loss_rpn_box_reg: 0.35361683 - total loss: 0.96987432 - median loss: 0.61523396: 100%|█████████████████████| 2858/2858 [07:53<00:00,  6.04 batch/s]
Epoch: 4/5. lr: 0.00010000 loss_classifier: 0.06914519 - loss_box_reg: 0.04815124 - loss_objectness: 0.03362034 - loss_rpn_box_reg: 0.37971252 - total loss: 0.53062928 - median loss: 0.60944918: 100%|█████████████████████| 2858/2858 [07:50<00:00,  6.07 batch/s]
Epoch: 5/5. lr: 0.00010000 loss_classifier: 0.08535144 - loss_box_reg: 0.04934930 - loss_objectness: 0.02790023 - loss_rpn_box_reg: 0.54119432 - total loss: 0.70379531 - median loss: 0.59856185: 100%|█████████████████████| 2858/2858 [07:50<00:00,  6.08 batch/s]
[+] Ready, the train phase took: 0:39:26.644003
```

## Traning Context Agregation
```
root@35f9a50f23da:/workspace/ca-bifpn# python3.8 train_A1.py --cfg_dataset config/files/dataset/voc.yaml --cfg_model config/files/model/00_fpn_5l_internimage_t_1k_224.yaml --fpn_type ca --num_epochs 5 --summary
[+] backbone used: internimage_t_1k_224 - bifpn used: fpn_5l 
using core type: DCNv3
using activation layer: GELU
using main norm layer: LN
using dpr: linear, 0.2
level2_post_norm: False
level2_post_norm_block_ids: None
res_post_norm: False
[+] The unexpected keys was: ['conv_head.0.weight', 'conv_head.1.0.weight', 'conv_head.1.0.bias', 'conv_head.1.0.running_mean', 'conv_head.1.0.running_var', 'conv_head.1.0.num_batches_tracked', 'head.weight', 'head.bias']
[+] CABiFPN loaded
[+] Building the base model with FasterRCNN head ...
[++] Numbers of classes: 21
[+] Ready !
====================================================================================================
Layer (type:depth-idx)                                                      Param #
====================================================================================================
FasterRCNN                                                                  --
├─GeneralizedRCNNTransform: 1-1                                             --
├─Backbone: 1-2                                                             --
│    └─InternImage: 2-1                                                     --
│    │    └─StemLayer: 3-1                                                  (19,584)
│    │    └─Dropout: 3-2                                                    --
│    │    └─ModuleList: 3-3                                                 (28,749,936)
│    └─Sequential: 2-2                                                      --
│    │    └─CABiFPN: 3-4                                                    484,736
│    │    └─CABiFPN: 3-5                                                    376,320
│    │    └─CABiFPN: 3-6                                                    376,320
│    │    └─CABiFPN: 3-7                                                    376,320
│    │    └─CABiFPN: 3-8                                                    376,320
├─RegionProposalNetwork: 1-3                                                --
│    └─AnchorGenerator: 2-3                                                 --
│    └─RPNHead: 2-4                                                         --
│    │    └─Sequential: 3-9                                                 113,008
│    │    └─Conv2d: 3-10                                                    1,808
│    │    └─Conv2d: 3-11                                                    7,232
├─RoIHeads: 1-4                                                             --
│    └─MultiScaleRoIAlign: 2-5                                              --
│    └─TwoMLPHead: 2-6                                                      --
│    │    └─Linear: 3-12                                                    5,620,736
│    │    └─Linear: 3-13                                                    1,049,600
│    └─FastRCNNPredictor: 2-7                                               --
│    │    └─Linear: 3-14                                                    21,525
│    │    └─Linear: 3-15                                                    86,100
====================================================================================================
Total params: 37,659,545
Trainable params: 8,890,025
Non-trainable params: 28,769,520
====================================================================================================
[+] Loading VOC 2012 dataset...
[++] Using batch_size: 2
[++] Loading training dataset...
[++] Ready !
[++] Ready !
[+] Using AdamW optimizer
[+] Starting training ...
Epoch: 1/5. lr: 0.00010000 loss_classifier: 0.13541234 - loss_box_reg: 0.05948814 - loss_objectness: 0.03292695 - loss_rpn_box_reg: 0.45184922 - total loss: 0.67967665 - median loss: 0.79978964: 100%|█████████████████████| 2858/2858 [14:00<00:00,  3.40 batch/s]
Epoch: 2/5. lr: 0.00010000 loss_classifier: 0.05348901 - loss_box_reg: 0.02548135 - loss_objectness: 0.05439838 - loss_rpn_box_reg: 0.47745311 - total loss: 0.61082184 - median loss: 0.64161423: 100%|█████████████████████| 2858/2858 [14:09<00:00,  3.36 batch/s]
Epoch: 3/5. lr: 0.00010000 loss_classifier: 0.05787423 - loss_box_reg: 0.05553241 - loss_objectness: 0.02102301 - loss_rpn_box_reg: 0.39709628 - total loss: 0.53152597 - median loss: 0.61585134: 100%|█████████████████████| 2858/2858 [14:12<00:00,  3.35 batch/s]
Epoch: 4/5. lr: 0.00010000 loss_classifier: 0.03772048 - loss_box_reg: 0.01710714 - loss_objectness: 0.03345412 - loss_rpn_box_reg: 0.11658329 - total loss: 0.20486504 - median loss: 0.60549048: 100%|█████████████████████| 2858/2858 [14:08<00:00,  3.37 batch/s]
Epoch: 5/5. lr: 0.00010000 loss_classifier: 0.10939035 - loss_box_reg: 0.05217648 - loss_objectness: 0.02602961 - loss_rpn_box_reg: 0.39112452 - total loss: 0.57872093 - median loss: 0.59114042: 100%|█████████████████████| 2858/2858 [14:04<00:00,  3.39 batch/s]
[+] Ready, the train phase took: 1:10:36.094595
```

TODO: probar dos opciones: 1) continuar sin lr_scheduler 2)continar con lr_scheduler CyclicLR triangular o triangular2


# Validation

```
root@35f9a50f23da:/workspace/ca-bifpn# python3.8 val_A1.py --path_checkpoint 20230610_1731_bs_internimage_t_1k_224_fpn_5l_5.pth      [+] Loading checkpoint...
[+] Ready !
[+] Preparing base configs...
[+] Ready !
[+] backbone used: internimage_t_1k_224 - bifpn used: fpn_5l 
using core type: DCNv3
using activation layer: GELU
using main norm layer: LN
using dpr: linear, 0.2
level2_post_norm: False
level2_post_norm_block_ids: None
res_post_norm: False
[+] The unexpected keys was: ['conv_head.0.weight', 'conv_head.1.0.weight', 'conv_head.1.0.bias', 'conv_head.1.0.running_mean', 'conv_head.1.0.running_var', 'conv_head.1.0.num_batches_tracked', 'head.weight', 'head.bias']
[+] BiFPN loaded
[+] Building the base model with FasterRCNN head ...
[++] Numbers of classes: 21
[+] Ready !
[+] Loading checkpoint...
[++] All keys matched successfully
[+] Ready. last_epoch: 5 - last_loss: 0.5985618531703949
[+] Loading VOC 2012 dataset...
[++] Loading validation dataset...
[++] Ready !
[+] Ready !
[+] Starting validation ...
creating index...
index created!
Accumulating evaluation results...
DONE (t=1.65s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.031
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.076
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.020
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.002
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.008
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.039
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.091
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.100
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.100
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.003
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.038
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.123
[+] Ready, the validation phase took: 0:04:43.652532
```


```
root@35f9a50f23da:/workspace/ca-bifpn# python3.8 val_A1.py --path_checkpoint 20230610_1920_ca_internimage_t_1k_224_fpn_5l_5.pth
[+] Loading checkpoint...
[+] Ready !
[+] Preparing base configs...
[+] Ready !
[+] backbone used: internimage_t_1k_224 - bifpn used: fpn_5l 
using core type: DCNv3
using activation layer: GELU
using main norm layer: LN
using dpr: linear, 0.2
level2_post_norm: False
level2_post_norm_block_ids: None
res_post_norm: False
[+] The unexpected keys was: ['conv_head.0.weight', 'conv_head.1.0.weight', 'conv_head.1.0.bias', 'conv_head.1.0.running_mean', 'conv_head.1.0.running_var', 'conv_head.1.0.num_batches_tracked', 'head.weight', 'head.bias']
[+] CABiFPN loaded
[+] Building the base model with FasterRCNN head ...
[++] Numbers of classes: 21
[+] Ready !
[+] Loading checkpoint...
[++] All keys matched successfully
[+] Ready. best_epoch: 6 - best_loss: 0.5911404192447662
[+] Loading VOC 2012 dataset...
[++] Loading validation dataset...
[++] Ready !
[+] Ready !
[+] Starting validation ...
creating index...
index created!
Accumulating evaluation results...
DONE (t=1.49s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.047
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.118
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.026
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.005
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.008
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.063
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.107
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.115
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.115
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.013
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.033
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.146
[+] Ready, the validation phase took: 0:05:44.538580
```