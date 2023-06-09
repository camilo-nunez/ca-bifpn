### Original code extracted from:
# Title: Yet Another EfficientDet Pytorch
# Author: https://github.com/zylo117
# Retrieved Date: june-2023
# Last commit retrieved: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/commit/c533bc2de65135a6fe1d25ca437765c630943afb
# Availability: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
###
import torch
import torch.nn as nn

from .utils import SeparableConvBlock, MaxPool2dStaticSamePadding, Swish, Conv2dStaticSamePadding

class BiFPN(nn.Module):
    
    def __init__(self, num_channels, conv_channels, 
                 first_time=False, 
                 epsilon=1e-4, 
                 attention=True,):
        """
        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
        """
        super(BiFPN, self).__init__()
        self.epsilon = epsilon


        # Conv layers
        self.conv3_up = SeparableConvBlock(num_channels)
        self.conv2_up = SeparableConvBlock(num_channels)
        self.conv1_up = SeparableConvBlock(num_channels)
        
        self.conv2_down = SeparableConvBlock(num_channels)
        self.conv3_down = SeparableConvBlock(num_channels)
        self.conv4_down = SeparableConvBlock(num_channels)


        # Feature scaling layers
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p1_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p2_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p3_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)


        self.swish = Swish()

        self.first_time = first_time
        if self.first_time:
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[3], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p2_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p1_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # Weight
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
        self.p2_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p2_w1_relu = nn.ReLU()
        self.p1_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p1_w1_relu = nn.ReLU()

        self.p2_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p2_w2_relu = nn.ReLU()
        self.p3_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_w2_relu = nn.ReLU()
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit for InterImage backbone
            P4_in --------------------------> P4_out -------->
               |-------------|                ↑
                             ↓                |
            P3_in ---------> P3_up ---------> P3_out -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P2_in ---------> P2_up ---------> P2_out -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P1_in --------------------------> P1_out -------->
        """

        if self.first_time:
            p1, p2, p3, p4 = inputs
            
            p4_in = self.p4_down_channel(p4)
            p3_in = self.p3_down_channel(p3)
            p2_in = self.p2_down_channel(p2)
            p1_in = self.p1_down_channel(p1)

        else:
            p1_in, p2_in, p3_in, p4_in = inputs

        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_up = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_in)))

        p2_w1 = self.p2_w1_relu(self.p2_w1)
        weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        p2_up = self.conv2_up(self.swish(weight[0] * p2_in + weight[1] * self.p2_upsample(p3_up)))

        p1_w1 = self.p1_w1_relu(self.p1_w1)
        weight = p1_w1 / (torch.sum(p1_w1, dim=0) + self.epsilon)
        p1_out = self.conv3_up(self.swish(weight[0] * p1_in + weight[1] * self.p1_upsample(p2_up)))

        p2_w2 = self.p2_w2_relu(self.p2_w2)
        weight = p2_w2 / (torch.sum(p2_w2, dim=0) + self.epsilon)
        p2_out = self.conv2_down(
            self.swish(weight[0] * p2_in + weight[1] * p2_up + weight[2] * self.p2_downsample(p1_out)))

        p3_w2 = self.p3_w2_relu(self.p3_w2)
        weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        p3_out = self.conv3_down(
            self.swish(weight[0] * p3_in + weight[1] * p3_up + weight[2] * self.p3_downsample(p2_out)))

        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.conv4_down(self.swish(weight[0] * p4_in + weight[1] * self.p4_downsample(p3_out)))

        return p1_out, p2_out, p3_out, p4_out