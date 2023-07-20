# Results using `val_A1.py`

## Config: internimage_t_1k_224

### Neck: BS - OPT: AdamW - Last Epoch: 100
```
root@9ba0fdc560d5:/thesis/ca-bifpn# python val_A1.py --path_checkpoint /thesis/checkpoint/base/20230621_1631_A1_bs_internimage_t_1k_224_fpn_5l_100.pth --summary --batch_size 4
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
[+] Loading checkpoint...
[++] All keys matched successfully
[+] Ready. last_epoch: 100 - last_loss: 0.43225789070129395
[+] Loading VOC 2012 dataset...
[++] Loading validation dataset...
[++] Ready !
[+] Ready !
[+] Starting validation ...
Creating index...
index created!
Accumulating evaluation results...
DONE (t=1.91s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.15011
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.29567
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.13889
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.00210
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.09097
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.18517
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.23293
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.28825
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.28852
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.00162
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.17781
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.34990
[+] Ready, the validation phase took: 0:05:01.237517

```


### Neck: BS - OPT: SGD - Last Epoch: 184
```
root@9ba0fdc560d5:/thesis/ca-bifpn# python val_A1.py --path_checkpoint /thesis/checkpoint/A2/20230705_1313_A2_bs_internimage_t_1k_224_fpn_5l_184.pth --summary --batch_size 4
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
[+] Loading checkpoint...
[++] All keys matched successfully
[+] Ready. last_epoch: 184 - last_loss: 0.40924888849258423
[+] Loading VOC 2012 dataset...
[++] Loading validation dataset...
[++] Ready !
[+] Ready !
[+] Starting validation ...
Creating index...
index created!
Accumulating evaluation results...
DONE (t=1.86s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.16356
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.30718
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.15972
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.00174
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.09405
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.20310
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.24740
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.30184
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.30210
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.00127
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.18226
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.36850
[+] Ready, the validation phase took: 0:05:01.413078
```

### Neck: CA - OPT: AdamW - Last Epoch: 99
```
root@9ba0fdc560d5:/thesis/ca-bifpn# python val_A1.py --path_checkpoint /thesis/checkpoint/base/20230622_2148_A1_ca_internimage_t_1k_224_fpn_5l_99.pth --summary --batch_size 4
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
[+] Loading checkpoint...
[++] All keys matched successfully
[+] Ready. last_epoch: 99 - last_loss: 0.405975341796875
[+] Loading VOC 2012 dataset...
[++] Loading validation dataset...
[++] Ready !
[+] Ready !
[+] Starting validation ...
Creating index...
index created!
Accumulating evaluation results...
DONE (t=1.86s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.20919
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.39096
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.20795
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.05524
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.11845
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.25611
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.28667
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.37395
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.37606
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.13702
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.27285
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.42340
[+] Ready, the validation phase took: 0:06:04.776708

```

### Neck: CA - OPT: SGD - Last Epoch: 185
```
root@9ba0fdc560d5:/thesis/ca-bifpn# python val_A1.py --path_checkpoint /thesis/checkpoint/A2/20230706_0632_A2_ca_internimage_t_1k_224_fpn_5l_185.pth --summary --batch_size 4
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
[+] Loading checkpoint...
[++] All keys matched successfully
[+] Ready. last_epoch: 185 - last_loss: 0.3688197731971741
[+] Loading VOC 2012 dataset...
[++] Loading validation dataset...
[++] Ready !
[+] Ready !
[+] Starting validation ...
Creating index...
index created!
Accumulating evaluation results...
DONE (t=1.97s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.21254
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.38885
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.21180
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.05912
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.12113
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.25790
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.28953
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.37590
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.37736
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.13722
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.27156
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.42578
[+] Ready, the validation phase took: 0:06:08.637081
```

## Config: internimage_s_1k_224

### Neck: BS - OPT: AdamW - Last Epoch: 99
```
root@9ba0fdc560d5:/thesis/ca-bifpn# python val_A1.py --summary --batch_size 4 --path_checkpoint /thesis/checkpoint/base/20230623_1802_A1_bs_internimage_s_1k_224_fpn_5l_99.pth
[+] Loading checkpoint...
[+] Ready !
[+] Preparing base configs...
[+] Ready !
[+] backbone used: internimage_s_1k_224 - bifpn used: fpn_5l 
using core type: DCNv3
using activation layer: GELU
using main norm layer: LN
using dpr: linear, 0.3
level2_post_norm: False
level2_post_norm_block_ids: None
res_post_norm: False
Downloading: "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_s_1k_224.pth" to /root/.cache/torch/hub/checkpoints/internimage_s_1k_224.pth
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 191M/191M [00:08<00:00, 23.5MB/s]
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
│    │    └─StemLayer: 3-1                                                  (30,240)
│    │    └─Dropout: 3-2                                                    --
│    │    └─ModuleList: 3-3                                                 (48,472,320)
│    └─Sequential: 2-2                                                      --
│    │    └─BiFPN: 3-4                                                      219,087
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
Total params: 55,955,028
Trainable params: 7,452,468
Non-trainable params: 48,502,560
====================================================================================================
[+] Loading checkpoint...
[++] All keys matched successfully
[+] Ready. last_epoch: 99 - last_loss: 0.4144124984741211
[+] Loading VOC 2012 dataset...
[++] Loading validation dataset...
[++] Ready !
[+] Ready !
[+] Starting validation ...
Creating index...
index created!
Accumulating evaluation results...
DONE (t=1.33s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.21943
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.39686
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.22161
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.00588
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.12094
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.28719
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.28294
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.34795
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.34873
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.00770
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.17925
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.44514
[+] Ready, the validation phase took: 0:06:14.277201
```

### Neck: BS - OPT: SGD - Last Epoch: 197
```
root@9ba0fdc560d5:/thesis/ca-bifpn# python val_A1.py --summary --batch_size 4 --path_checkpoint /thesis/checkpoint/A2/20230707_0638_A2_bs_internimage_s_1k_224_fpn_5l_197.pth
[+] Loading checkpoint...
[+] Ready !
[+] Preparing base configs...
[+] Ready !
[+] backbone used: internimage_s_1k_224 - bifpn used: fpn_5l 
using core type: DCNv3
using activation layer: GELU
using main norm layer: LN
using dpr: linear, 0.3
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
│    │    └─StemLayer: 3-1                                                  (30,240)
│    │    └─Dropout: 3-2                                                    --
│    │    └─ModuleList: 3-3                                                 (48,472,320)
│    └─Sequential: 2-2                                                      --
│    │    └─BiFPN: 3-4                                                      219,087
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
Total params: 55,955,028
Trainable params: 7,452,468
Non-trainable params: 48,502,560
====================================================================================================
[+] Loading checkpoint...
[++] All keys matched successfully
[+] Ready. last_epoch: 197 - last_loss: 0.39966875314712524
[+] Loading VOC 2012 dataset...
[++] Loading validation dataset...
[++] Ready !
[+] Ready !
[+] Starting validation ...
Creating index...
index created!
Accumulating evaluation results...
DONE (t=1.22s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.21695
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.38755
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.22149
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.00860
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.11326
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.28423
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.28542
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.34727
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.34801
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.01210
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.16494
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.44495
[+] Ready, the validation phase took: 0:06:13.857505
```

### Neck: CA - OPT: AdamW - Last Epoch: 97
```
root@9ba0fdc560d5:/thesis/ca-bifpn# python val_A1.py --summary --batch_size 4 --path_checkpoint /thesis/checkpoint/base/20230624_2341_A1_ca_internimage_s_1k_224_fpn_5l_97.pth
[+] Loading checkpoint...
[+] Ready !
[+] Preparing base configs...
[+] Ready !
[+] backbone used: internimage_s_1k_224 - bifpn used: fpn_5l 
using core type: DCNv3
using activation layer: GELU
using main norm layer: LN
using dpr: linear, 0.3
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
│    │    └─StemLayer: 3-1                                                  (30,240)
│    │    └─Dropout: 3-2                                                    --
│    │    └─ModuleList: 3-3                                                 (48,472,320)
│    └─Sequential: 2-2                                                      --
│    │    └─CABiFPN: 3-4                                                    511,616
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
Total params: 57,419,465
Trainable params: 8,916,905
Non-trainable params: 48,502,560
====================================================================================================
[+] Loading checkpoint...
[++] All keys matched successfully
[+] Ready. last_epoch: 97 - last_loss: 0.3894036114215851
[+] Loading VOC 2012 dataset...
[++] Loading validation dataset...
[++] Ready !
[+] Ready !
[+] Starting validation ...
Creating index...
index created!
Accumulating evaluation results...
DONE (t=1.91s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.28098
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.50245
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.28546
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.07071
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.18423
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.33344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.32626
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.43975
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.44399
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.17006
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.34715
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.49451
[+] Ready, the validation phase took: 0:07:30.331691
```

### Neck: CA - OPT: SGD - Last Epoch: 170
```
root@9ba0fdc560d5:/thesis/ca-bifpn# python val_A1.py --summary --batch_size 4 --path_checkpoint /thesis/checkpoint/base/20230624_2341_A1_ca_internimage_s_1k_224_fpn_5l_97.pth
[+] Loading checkpoint...
[+] Ready !
[+] Preparing base configs...
[+] Ready !
[+] backbone used: internimage_s_1k_224 - bifpn used: fpn_5l 
using core type: DCNv3
using activation layer: GELU
using main norm layer: LN
using dpr: linear, 0.3
level2_post_norm: False
level2_post_norm_block_ids: None
res_post_norm: False
root@9ba0fdc560d5:/thesis/ca-bifpn# python val_A1.py --summary --batch_size 4 --path_checkpoint /thesis/checkpoint/A2/20230708_0538_A2_ca_internimage_s_1k_224_fpn_5l_170.pth
[+] Loading checkpoint...
[+] Ready !
[+] Preparing base configs...
[+] Ready !
[+] backbone used: internimage_s_1k_224 - bifpn used: fpn_5l 
using core type: DCNv3
using activation layer: GELU
using main norm layer: LN
using dpr: linear, 0.3
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
│    │    └─StemLayer: 3-1                                                  (30,240)
│    │    └─Dropout: 3-2                                                    --
│    │    └─ModuleList: 3-3                                                 (48,472,320)
│    └─Sequential: 2-2                                                      --
│    │    └─CABiFPN: 3-4                                                    511,616
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
Total params: 57,419,465
Trainable params: 8,916,905
Non-trainable params: 48,502,560
====================================================================================================
[+] Loading checkpoint...
[++] All keys matched successfully
[+] Ready. last_epoch: 170 - last_loss: 0.3533281087875366
[+] Loading VOC 2012 dataset...
[++] Loading validation dataset...
[++] Ready !
[+] Ready !
[+] Starting validation ...
Creating index...
index created!
Accumulating evaluation results...
DONE (t=1.71s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.28710
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.50048
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.29658
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.07155
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.18893
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.33744
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.33104
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.44121
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.44466
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.15733
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.34374
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.49586
[+] Ready, the validation phase took: 0:07:23.932380
```



## Config: internimage_b_1k_224

### Neck: BS - OPT: AdamW - Last Epoch: 98
```
root@9ba0fdc560d5:/thesis/ca-bifpn# python val_A1.py --summary --batch_size 4 --path_checkpoint /thesis/checkpoint/base/20230625_2332_A1_bs_internimage_b_1k_224_fpn_5l_98.pth
[+] Loading checkpoint...
[+] Ready !
[+] Preparing base configs...
[+] Ready !
[+] backbone used: internimage_b_1k_224 - bifpn used: fpn_5l 
using core type: DCNv3
using activation layer: GELU
using main norm layer: LN
using dpr: linear, 0.4
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
│    │    └─StemLayer: 3-1                                                  (58,464)
│    │    └─Dropout: 3-2                                                    --
│    │    └─ModuleList: 3-3                                                 (94,851,456)
│    └─Sequential: 2-2                                                      --
│    │    └─BiFPN: 3-4                                                      272,847
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
Total params: 102,416,148
Trainable params: 7,506,228
Non-trainable params: 94,909,920
====================================================================================================
[+] Loading checkpoint...
[++] All keys matched successfully
[+] Ready. last_epoch: 98 - last_loss: 0.4077000617980957
[+] Loading VOC 2012 dataset...
[++] Loading validation dataset...
[++] Ready !
[+] Ready !
[+] Starting validation ...
Creating index...
index created!
Accumulating evaluation results...
DONE (t=1.34s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.21051
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.38016
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.21265
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.03088
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.08223
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.27777
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.27412
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.33098
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.33246
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.04845
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.13714
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.42142
[+] Ready, the validation phase took: 0:08:24.668650
```
### Neck: BS - OPT: SGD - Last Epoch: 181
```
root@9ba0fdc560d5:/thesis/ca-bifpn# python val_A1.py --summary --batch_size 4 --path_checkpoint /thesis/checkpoint/A2/20230709_0834_A2_bs_internimage_b_1k_224_fpn_5l_181.pth
[+] Loading checkpoint...
[+] Ready !
[+] Preparing base configs...
[+] Ready !
[+] backbone used: internimage_b_1k_224 - bifpn used: fpn_5l 
using core type: DCNv3
using activation layer: GELU
using main norm layer: LN
using dpr: linear, 0.4
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
│    │    └─StemLayer: 3-1                                                  (58,464)
│    │    └─Dropout: 3-2                                                    --
│    │    └─ModuleList: 3-3                                                 (94,851,456)
│    └─Sequential: 2-2                                                      --
│    │    └─BiFPN: 3-4                                                      272,847
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
Total params: 102,416,148
Trainable params: 7,506,228
Non-trainable params: 94,909,920
====================================================================================================
[+] Loading checkpoint...
[++] All keys matched successfully
[+] Ready. last_epoch: 181 - last_loss: 0.3871271014213562
[+] Loading VOC 2012 dataset...
[++] Loading validation dataset...
[++] Ready !
[+] Ready !
[+] Starting validation ...
Creating index...
index created!
Accumulating evaluation results...
DONE (t=1.28s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.21474
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.38048
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.21850
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.03172
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.08340
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.28358
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.27957
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.33810
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.33994
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.05332
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.13642
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.42969
[+] Ready, the validation phase took: 0:08:03.967477
```

### Neck: CA - OPT: AdamW - Last Epoch: 100
```
root@9ba0fdc560d5:/thesis/ca-bifpn# python val_A1.py --summary --batch_size 4 --path_checkpoint /thesis/checkpoint/base/20230627_0650_A1_ca_internimage_b_1k_224_fpn_5l_100.pth
[+] Loading checkpoint...
[+] Ready !
[+] Preparing base configs...
[+] Ready !
[+] backbone used: internimage_b_1k_224 - bifpn used: fpn_5l 
using core type: DCNv3
using activation layer: GELU
using main norm layer: LN
using dpr: linear, 0.4
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
│    │    └─StemLayer: 3-1                                                  (58,464)
│    │    └─Dropout: 3-2                                                    --
│    │    └─ModuleList: 3-3                                                 (94,851,456)
│    └─Sequential: 2-2                                                      --
│    │    └─CABiFPN: 3-4                                                    565,376
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
Total params: 103,880,585
Trainable params: 8,970,665
Non-trainable params: 94,909,920
====================================================================================================
[+] Loading checkpoint...
[++] All keys matched successfully
[+] Ready. last_epoch: 100 - last_loss: 0.37833109498023987
[+] Loading VOC 2012 dataset...
[++] Loading validation dataset...
[++] Ready !
[+] Ready !
[+] Starting validation ...
Creating index...
index created!
Accumulating evaluation results...
DONE (t=1.49s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.28219
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.50102
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.28744
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.05861
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.17594
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.34402
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.32872
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.43089
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.43302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.13885
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.31810
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.49509
[+] Ready, the validation phase took: 0:09:14.618652
```
### Neck: CA - OPT: SGD - Last Epoch: 191
```
root@9ba0fdc560d5:/thesis/ca-bifpn# python val_A1.py --summary --batch_size 4 --path_checkpoint /thesis/checkpoint/A2/20230710_1329_A2_ca_internimage_b_1k_224_fpn_5l_191.pth
[+] Loading checkpoint...
[+] Ready !
[+] Preparing base configs...
[+] Ready !
[+] backbone used: internimage_b_1k_224 - bifpn used: fpn_5l 
using core type: DCNv3
using activation layer: GELU
using main norm layer: LN
using dpr: linear, 0.4
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
│    │    └─StemLayer: 3-1                                                  (58,464)
│    │    └─Dropout: 3-2                                                    --
│    │    └─ModuleList: 3-3                                                 (94,851,456)
│    └─Sequential: 2-2                                                      --
│    │    └─CABiFPN: 3-4                                                    565,376
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
Total params: 103,880,585
Trainable params: 8,970,665
Non-trainable params: 94,909,920
====================================================================================================
[+] Loading checkpoint...
[++] All keys matched successfully
[+] Ready. last_epoch: 191 - last_loss: 0.346662700176239
[+] Loading VOC 2012 dataset...
[++] Loading validation dataset...
[++] Ready !
[+] Ready !
[+] Starting validation ...
Creating index...
index created!
Accumulating evaluation results...
DONE (t=1.49s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.29127
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.50435
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.30093
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.06087
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.19019
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.34934
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.33488
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.44157
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.44442
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.13748
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.33195
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.50442
[+] Ready, the validation phase took: 0:09:14.701547
```

## Config: internimage_l_22k_192to384

### Neck: BS - OPT: AdamW - Last Epoch: 98
```
root@9ba0fdc560d5:/thesis/ca-bifpn# python val_A1.py --summary --batch_size 4 --path_checkpoint /thesis/checkpoint/base/20230702_0830_A1_bs_internimage_l_22k_192to384_fpn_5l_98.pth
[+] Loading checkpoint...
[+] Ready !
[+] Preparing base configs...
[+] Ready !
[+] backbone used: internimage_l_22k_192to384 - bifpn used: fpn_5l 
using core type: DCNv3
using activation layer: GELU
using main norm layer: LN
using dpr: linear, 0.4
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
│    │    └─StemLayer: 3-1                                                  (118,080)
│    │    └─Dropout: 3-2                                                    --
│    │    └─ModuleList: 3-3                                                 (218,834,290)
│    └─Sequential: 2-2                                                      --
│    │    └─BiFPN: 3-4                                                      1,029,135
│    │    └─BiFPN: 3-5                                                      411,663
│    │    └─BiFPN: 3-6                                                      411,663
│    │    └─BiFPN: 3-7                                                      411,663
│    │    └─BiFPN: 3-8                                                      411,663
├─RegionProposalNetwork: 1-3                                                --
│    └─AnchorGenerator: 2-3                                                 --
│    └─RPNHead: 2-4                                                         --
│    │    └─Sequential: 3-9                                                 590,080
│    │    └─Conv2d: 3-10                                                    4,112
│    │    └─Conv2d: 3-11                                                    16,448
├─RoIHeads: 1-4                                                             --
│    └─MultiScaleRoIAlign: 2-5                                              --
│    └─TwoMLPHead: 2-6                                                      --
│    │    └─Linear: 3-12                                                    12,846,080
│    │    └─Linear: 3-13                                                    1,049,600
│    └─FastRCNNPredictor: 2-7                                               --
│    │    └─Linear: 3-14                                                    21,525
│    │    └─Linear: 3-15                                                    86,100
====================================================================================================
Total params: 236,242,102
Trainable params: 17,289,732
Non-trainable params: 218,952,370
====================================================================================================
[+] Loading checkpoint...
[++] All keys matched successfully
[+] Ready. last_epoch: 98 - last_loss: 0.38783296942710876
[+] Loading VOC 2012 dataset...
[++] Loading validation dataset...
[++] Ready !
[+] Ready !
[+] Starting validation ...
Creating index...
index created!
Accumulating evaluation results...
DONE (t=1.88s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.24410
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.43406
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.24679
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.06425
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.15347
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.29479
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.30494
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.39516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.39720
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.15670
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.28237
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.44836
[+] Ready, the validation phase took: 0:14:57.352857
```
### Neck: BS - OPT: SGD - Last Epoch: 196
```
root@9ba0fdc560d5:/thesis/ca-bifpn# python val_A1.py --path_checkpoint /thesis/checkpoint/A2/20230713_0603_A2_bs_internimage_l_22k_192to384_fpn_5l_196.pth --summary --batch_size 4
[+] Loading checkpoint...
[+] Ready !
[+] Preparing base configs...
[+] Ready !
[+] backbone used: internimage_l_22k_192to384 - bifpn used: fpn_5l 
using core type: DCNv3
using activation layer: GELU
using main norm layer: LN
using dpr: linear, 0.4
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
│    │    └─StemLayer: 3-1                                                  (118,080)
│    │    └─Dropout: 3-2                                                    --
│    │    └─ModuleList: 3-3                                                 (218,834,290)
│    └─Sequential: 2-2                                                      --
│    │    └─BiFPN: 3-4                                                      1,029,135
│    │    └─BiFPN: 3-5                                                      411,663
│    │    └─BiFPN: 3-6                                                      411,663
│    │    └─BiFPN: 3-7                                                      411,663
│    │    └─BiFPN: 3-8                                                      411,663
├─RegionProposalNetwork: 1-3                                                --
│    └─AnchorGenerator: 2-3                                                 --
│    └─RPNHead: 2-4                                                         --
│    │    └─Sequential: 3-9                                                 590,080
│    │    └─Conv2d: 3-10                                                    4,112
│    │    └─Conv2d: 3-11                                                    16,448
├─RoIHeads: 1-4                                                             --
│    └─MultiScaleRoIAlign: 2-5                                              --
│    └─TwoMLPHead: 2-6                                                      --
│    │    └─Linear: 3-12                                                    12,846,080
│    │    └─Linear: 3-13                                                    1,049,600
│    └─FastRCNNPredictor: 2-7                                               --
│    │    └─Linear: 3-14                                                    21,525
│    │    └─Linear: 3-15                                                    86,100
====================================================================================================
Total params: 236,242,102
Trainable params: 17,289,732
Non-trainable params: 218,952,370
====================================================================================================
[+] Loading checkpoint...
[++] All keys matched successfully
[+] Ready. last_epoch: 196 - last_loss: 0.3548715114593506
[+] Loading VOC 2012 dataset...
[++] Loading validation dataset...
[++] Ready !
[+] Ready !
[+] Starting validation ...
Creating index...
index created!
Accumulating evaluation results...
DONE (t=1.65s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.27058
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.46173
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.28070
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.06972
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.17048
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.32655
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.32244
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.42244
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.42408
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.15481
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.30896
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.47969
[+] Ready, the validation phase took: 0:15:06.002116
```

### Neck: CA - OPT: AdamW - Last Epoch: 99
```
root@9ba0fdc560d5:/thesis/ca-bifpn# python val_A1.py --summary --batch_size 4 --path_checkpoint /thesis/checkpoint/base/20230704_2037_A1_ca_internimage_l_22k_192to384_fpn_5l_99.pth
[+] Loading checkpoint...
[+] Ready !
[+] Preparing base configs...
[+] Ready !
[+] backbone used: internimage_l_22k_192to384 - bifpn used: fpn_5l 
using core type: DCNv3
using activation layer: GELU
using main norm layer: LN
using dpr: linear, 0.4
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
│    │    └─StemLayer: 3-1                                                  (118,080)
│    │    └─Dropout: 3-2                                                    --
│    │    └─ModuleList: 3-3                                                 (218,834,290)
│    └─Sequential: 2-2                                                      --
│    │    └─CABiFPN: 3-4                                                    2,508,800
│    │    └─CABiFPN: 3-5                                                    1,892,352
│    │    └─CABiFPN: 3-6                                                    1,892,352
│    │    └─CABiFPN: 3-7                                                    1,892,352
│    │    └─CABiFPN: 3-8                                                    1,892,352
├─RegionProposalNetwork: 1-3                                                --
│    └─AnchorGenerator: 2-3                                                 --
│    └─RPNHead: 2-4                                                         --
│    │    └─Sequential: 3-9                                                 590,080
│    │    └─Conv2d: 3-10                                                    4,112
│    │    └─Conv2d: 3-11                                                    16,448
├─RoIHeads: 1-4                                                             --
│    └─MultiScaleRoIAlign: 2-5                                              --
│    └─TwoMLPHead: 2-6                                                      --
│    │    └─Linear: 3-12                                                    12,846,080
│    │    └─Linear: 3-13                                                    1,049,600
│    └─FastRCNNPredictor: 2-7                                               --
│    │    └─Linear: 3-14                                                    21,525
│    │    └─Linear: 3-15                                                    86,100
====================================================================================================
Total params: 243,644,523
Trainable params: 24,692,153
Non-trainable params: 218,952,370
====================================================================================================
[+] Loading checkpoint...
[++] All keys matched successfully
[+] Ready. last_epoch: 99 - last_loss: 0.38425159454345703
[+] Loading VOC 2012 dataset...
[++] Loading validation dataset...
[++] Ready !
[+] Ready !
[+] Starting validation ...
Creating index...
index created!
Accumulating evaluation results...
DONE (t=1.81s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.25312
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.46568
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.24903
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.07004
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.15992
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.29955
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.30626
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.39323
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.39641
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.15453
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.29680
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.43595
[+] Ready, the validation phase took: 0:17:30.134941
```
### Neck: CA - OPT: SGD - Last Epoch: 

















