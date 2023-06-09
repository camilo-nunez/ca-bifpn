import torch
import torch.nn as nn

from .utils import MaxPool2dStaticSamePadding, Conv2dStaticSamePadding

# Context Agregators
class LocalContextAggregator(nn.Module):
    
    def __init__(self,
                 channels,
                 r = 4,
                ):
        super().__init__()
        inter_channels = int(channels // r)
        
        self.conv = nn.Sequential(nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False),
                                  nn.InstanceNorm2d(inter_channels, affine=True),
                                  nn.GELU(),
                                  nn.Conv2d(inter_channels, channels, kernel_size=1, bias=False),
                                  nn.InstanceNorm2d(channels, affine=True),
                                  )
        self.sigmoid  = nn.Sigmoid()
        
    def forward(self, x):
        x_hat = self.conv(x)
        s =  self.sigmoid(x_hat)

        return torch.mul(x, s)

class GlobalContextAggregator(nn.Module):
    
    def __init__(self,
                 channels,
                 r = 4,
                ):
        super().__init__()
        inter_channels = int(channels // r)
        
        self.conv = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(inter_channels),
                                  nn.GELU(),
                                  nn.Conv2d(inter_channels, channels, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(channels),
                                 )        
        self.sigmoid  = nn.Sigmoid()
        
    def forward(self, x):
        x_hat = self.conv(x)
        s =  self.sigmoid(x_hat)

        return torch.mul(x, s)

## Fused-MBConv with Context Agregators
class FMBConvCA(nn.Module):
    def __init__(self,
                 channels,
                 ca_local = True,
                ):
        super().__init__()
        
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
                                  nn.InstanceNorm2d(channels, affine=True),
                                  nn.GELU(),
                                  LocalContextAggregator(channels) if ca_local else GlobalContextAggregator(channels),
                                  nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.InstanceNorm2d(channels, affine=True),
                                 )
                
    def forward(self, x):
        x_hat = self.conv(x)
        
        return x + x_hat

## CA + BiFPN
class CABiFPN(nn.Module):
    
    def __init__(self, 
                 channels,
                 conv_channels, 
                 first_time=False):
        
        super().__init__()
        
        
        # Convs blocks
        ## Inner Nodes
        ### P31
        self.p31_gca_p30 = FMBConvCA(channels, ca_local=False)
        self.p31_lca_p40 = FMBConvCA(channels)
        ### P21
        self.p21_gca_p20 = FMBConvCA(channels, ca_local=False)
        self.p21_lca_p31 = FMBConvCA(channels)
        
        ## Outer Nodes
        ### P12
        self.p12_gca_p10 = FMBConvCA(channels, ca_local=False)
        self.p12_lca_p21 = FMBConvCA(channels)
        
        ### P22
        self.p22_lca_p20 = FMBConvCA(channels)
        self.p22_lca_P21 = FMBConvCA(channels)
        self.p22_gca_p12 = FMBConvCA(channels, ca_local = False)
        
        ### P32
        self.p32_lca_p30 = FMBConvCA(channels)
        self.p32_lca_P31 = FMBConvCA(channels)
        self.p32_gca_p22 = FMBConvCA(channels, ca_local=False)
        
        ### P42
        self.p42_lca_p40 = FMBConvCA(channels)
        self.p42_gca_p32 = FMBConvCA(channels, ca_local=False)
        

        # Feature scaling layers
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p1_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p2_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p3_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.first_time = first_time
        if self.first_time:
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[3], channels, 1, bias=False),
#                 nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3),
                nn.InstanceNorm2d(channels, affine=True),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], channels, 1, bias=False),
#                 nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3),
                nn.InstanceNorm2d(channels, affine=True),
            )
            self.p2_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], channels, 1, bias=False),
#                 nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3),
                nn.InstanceNorm2d(channels, affine=True),
            )
            self.p1_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], channels, 1, bias=False),
#                 nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3),
                nn.InstanceNorm2d(channels, affine=True),
            )


    def forward(self, inputs):
        """
        illustration of a minimal CABiFPN unit
            P4_0 --------------------------> P4_2 -------->
               |-------------|                ↑
                             ↓                |
            P3_0 ---------> P3_1 ----------> P3_2 -------->
               |-------------|---------------↑ ↑
                             ↓                 |
            P2_0 ---------> P2_1 ----------> P2_2 -------->
               |-------------|---------------↑ ↑
                             |---------------↓ |
            P1_0 --------------------------> P1_2 -------->
        """
        if self.first_time:
            p1, p2, p3, p4 = inputs

            p4_0 = self.p4_down_channel(p4)
            p3_0 = self.p3_down_channel(p3)
            p2_0 = self.p2_down_channel(p2)
            p1_0 = self.p1_down_channel(p1)

        else:
            p1_0, p2_0, p3_0, p4_0 = inputs


        ## Nodes
        ### Inner Nodes
        p3_1 = self.p31_gca_p30(p3_0) + self.p31_lca_p40(self.p3_upsample(p4_0))
        p2_1 = self.p21_gca_p20(p2_0) + self.p21_lca_p31(self.p2_upsample(p3_1))
        
        ### Outer Nodes
        p1_2 = self.p12_gca_p10(p1_0) + self.p12_lca_p21(self.p1_upsample(p2_1))
        p2_2 = self.p22_lca_p20(p2_0) + self.p22_lca_P21(p2_1) + self.p22_gca_p12(self.p2_downsample(p1_2))
        p3_2 = self.p32_lca_p30(p3_0) + self.p32_lca_P31(p3_1) + self.p32_gca_p22(self.p3_downsample(p2_2))
        p4_2 = self.p42_lca_p40(p4_0) + self.p42_gca_p32(self.p4_downsample(p3_2))

        return p1_2, p2_2, p3_2, p4_2