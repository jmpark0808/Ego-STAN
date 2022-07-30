import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Any, Optional, List
from torch import Tensor
import math
import torchvision

class HeatMap(nn.Module):
    def __init__(self, num_classes=16):
        super(HeatMap, self).__init__()
        # Resnet 101 without last average pooling and fully connected layers
        self.resnet101 = torchvision.models.resnet101(pretrained=False)
        # First Deconvolution to obtain 2D heatmap
        self.heatmap_deconv = nn.Sequential(*[nn.ConvTranspose2d(2048, 1024, kernel_size=3,
                                                                 stride=2, dilation=1, padding=1),
                                              nn.BatchNorm2d(1024),
                                              nn.ReLU(),
                                              nn.ConvTranspose2d(1024, num_classes, kernel_size=3,
                                                                 stride=2, dilation=1, padding=0),
                                                                 ])


    def update_resnet101(self):
        self.resnet101 = nn.Sequential(*[l for ind, l in enumerate(self.resnet101.children()) if ind < 8])

    def forward(self, x):
        # x = 3 x 368 x 368

        x = self.resnet101(x)
        # x = 2048 x 12 x 12

        heatmap = self.heatmap_deconv(x)
        # heatmap = 15 x 47 x 47

        return heatmap


class HeatMapDist(nn.Module):
    def __init__(self):
        super(HeatMapDist, self).__init__()
        # Resnet 101 without last average pooling and fully connected layers
        self.resnet_2d = torchvision.models.resnet101(pretrained=False)
        self.resnet_1d = torchvision.models.resnet101(pretrained=False)
        # First Deconvolution to obtain 2D heatmap
        self.heatmap_deconv = nn.Sequential(*[nn.ConvTranspose2d(2048, 1024, kernel_size=3,
                                                                 stride=2, dilation=1, padding=1),
                                              nn.ConvTranspose2d(1024, 16, kernel_size=3,
                                                                 stride=2, dilation=1, padding=0)])
        self.heatmap_spatial_linear = nn.Linear(12*12, 200)
        self.heatmap_channel_linear = nn.Conv1d(2048, 16, 1)

    def update_resnet(self):
        self.resnet_2d = nn.Sequential(*[l for ind, l in enumerate(self.resnet_2d.children()) if ind < 8])
        self.resnet_1d = nn.Sequential(*[l for ind, l in enumerate(self.resnet_1d.children()) if ind < 8])

    def forward(self, x):
        # x = 3 x 368 x 368

        x_2d = self.resnet_2d(x)
        x_1d = self.resnet_1d(x)
        # x = 2048 x 12 x 12
        
        heatmap_2d = self.heatmap_deconv(x_2d)
        # heatmap_2d = 16 x 47 x 47

        x_1d = x_1d.reshape(x_1d.size(0), x_1d.size(1), -1)
        heatmap_1d = self.heatmap_spatial_linear(x_1d)
        heatmap_1d = self.heatmap_channel_linear(heatmap_1d)
        # heatmap_1d = 16 x 30


        return heatmap_2d, heatmap_1d

# -> Edit of HeatMap class, except returns a feature map as well -> Why is it in net and not in blocks?

class FeatureHeatMaps(nn.Module):
    def __init__(self):
        super(FeatureHeatMaps, self).__init__()
        # Resnet 101 without last average pooling and fully connected layers
        self.resnet101 = torchvision.models.resnet101(pretrained=False)
        # Convolutions to feature map pool
        self.featuremap_deconv = nn.Sequential(*[nn.ConvTranspose2d(2048, 1024, kernel_size=3,
                                                                 stride=2, dilation=1, padding=1),
                                              nn.ConvTranspose2d(1024, 15, kernel_size=3,
                                                                 stride=2, dilation=1, padding=0)])
        # Identical Upconvolutions -> Might experiment with different features
        self.heatmap_deconv = nn.Sequential(*[nn.ConvTranspose2d(2048, 1024, kernel_size=3,
                                                         stride=2, dilation=1, padding=1),
                                      nn.ConvTranspose2d(1024, 15, kernel_size=3,
                                                         stride=2, dilation=1, padding=0)])

    def update_resnet101(self):
        self.resnet101 = nn.Sequential(*[l for ind, l in enumerate(self.resnet101.children()) if ind < 8])

    def forward(self, x):
        # x = 3 x 368 x 368

        x = self.resnet101(x)
        # x = 2048 x 12 x 12

        heatmap = self.heatmap_deconv(x)
        # heatmap = 15 x 47 x 47

        depthmap = self.featuremap_deconv(x)
        # depthmap = 15 x 47 x 47

        return heatmap, depthmap



class Encoder(nn.Module):
    def __init__(self, num_classes=16, heatmap_resolution=47):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, 64, kernel_size=4, stride=2, padding=2)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(128, 512, kernel_size=4, stride=2, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.2)

        self.linear1 = nn.Linear(512*math.ceil(heatmap_resolution/8)**2, 2048)
        self.lrelu4 = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(2048, 512)
        self.lrelu5 = nn.LeakyReLU(0.2)
        self.linear3 = nn.Linear(512, 20)
        self.lrelu6 = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.lrelu3(x)
        x = x.reshape(x.size(0), -1) # flatten
        x = self.linear1(x)
        x = self.lrelu4(x)
        x = self.linear2(x)
        x = self.lrelu5(x)
        x = self.linear3(x)
        x = self.lrelu6(x)
        return x

# -> Variation of the Encoder class that concatenates the heatmap and feature map
# -> Instead of going from 15 -> 64 feature maps in the original, concatenates feature map and heatmap into 30 maps
# -> Main change is goes from concat(15, 15) -> 30 -> conv(30) -> 64 *instead of 15 -> conv(15) -> 64 

class FeatureConcatEncoder(nn.Module):
    def __init__(self):
        super(FeatureConcatEncoder, self).__init__()
        self.conv1 = nn.Conv2d(30, 64, kernel_size=4, stride=2, padding=2)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(128, 512, kernel_size=4, stride=2, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.2)

        self.linear1 = nn.Linear(18432, 2048)
        self.lrelu4 = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(2048, 512)
        self.lrelu5 = nn.LeakyReLU(0.2)
        self.linear3 = nn.Linear(512, 20)
        self.lrelu6 = nn.LeakyReLU(0.2)

    def forward(self, x, y):
        x = torch.concat([x, y], 1)
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.lrelu3(x)
        x = x.reshape(x.size(0), -1) # flatten
        x = self.linear1(x)
        x = self.lrelu4(x)
        x = self.linear2(x)
        x = self.lrelu5(x)
        x = self.linear3(x)
        x = self.lrelu6(x)
        return x

# -> Variation of the FeatureReEncoder class that reconstructs a heatmap-like feature map (15x47x47)
# -> Instead of going from 30 maps to 64, goes from 30 maps to 15 to 64

class FeatureReEncoder(nn.Module):
    def __init__(self):
        super(FeatureReEncoder, self).__init__()
        self.deconv_1 = nn.ConvTranspose2d(30, 15, kernel_size=4, stride=1, padding=2)
        self.conv1 = nn.Conv2d(15, 64, kernel_size=4, stride=2, padding=2)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(128, 512, kernel_size=4, stride=2, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.2)

        self.linear1 = nn.Linear(18432, 2048)
        self.lrelu4 = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(2048, 512)
        self.lrelu5 = nn.LeakyReLU(0.2)
        self.linear3 = nn.Linear(512, 20)
        self.lrelu6 = nn.LeakyReLU(0.2)

    def forward(self, x, y):
        x = torch.concat([x, y], 1)
        x = self.deconv_1(x)
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.lrelu3(x)
        x = x.reshape(x.size(0), -1) # flatten
        x = self.linear1(x)
        x = self.lrelu4(x)
        x = self.linear2(x)
        x = self.lrelu5(x)
        x = self.linear3(x)
        x = self.lrelu6(x)
        return x

# -> Variation of the Encoder class that concatenates the features extracted from Feature maps and HeatMaps.
# -> Instead of going from 512 maps -> 18432-D vector in the original, goes from 2 512 maps to 36864-D vector -> 18432-D vector
# -> Main change is goes from 2 512 maps (one for each vector)  -? 36864-D vector instead of 512 directly do 18432-D vector

class FeatureBranchEncoder(nn.Module):
    def __init__(self):
        super(FeatureBranchEncoder, self).__init__()
        self.convs_hm = nn.Sequential(*[
            nn.Conv2d(15, 64, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
            ])

        self.convs_dm = nn.Sequential(*[
            nn.Conv2d(15, 64, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
            ])

        self.linear1 = nn.Linear(36864, 18432)
        self.lrelu4 = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(18432, 2048)
        self.lrelu5 = nn.LeakyReLU(0.2)
        self.linear3 = nn.Linear(2048, 512)
        self.lrelu6 = nn.LeakyReLU(0.2)
        self.linear4 = nn.Linear(512, 20)
        self.lrelu7 = nn.LeakyReLU(0.2)

    def forward(self, x, y):
        x_hm = self.convs_hm(x)
        x_dm = self.convs_dm(y)
        x = torch.cat(tensors=[x_hm, x_dm], dim=1)
        x = x.reshape(x.size(0), -1) # flatten
        x = self.linear1(x)
        x = self.lrelu4(x)
        x = self.linear2(x)
        x = self.lrelu5(x)
        x = self.linear3(x)
        x = self.lrelu6(x)
        x = self.linear4(x)
        x = self.lrelu7(x)
        return x

class PoseDecoder(nn.Module):
    def __init__(self, initial_dim=20, num_classes=16):
        super(PoseDecoder, self).__init__()
        self.linear1 = nn.Linear(initial_dim, 32)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(32, 32)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.linear3 = nn.Linear(32, num_classes*3)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.linear1(x)
        x = self.lrelu1(x)
        x = self.linear2(x)
        x = self.lrelu2(x)
        x = self.linear3(x)
        x = x.reshape(x.size(0), self.num_classes, 3)
        return x

class HeatmapDecoder(nn.Module):
    def __init__(self, num_classes=16, heatmap_resolution=47):
        super(HeatmapDecoder, self).__init__()
        self.linear1 = nn.Linear(20, 512)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(512, 2048)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.spatial_resolution = math.ceil(heatmap_resolution/8.)
        self.linear3 = nn.Linear(2048, self.spatial_resolution**2*512)
        self.lrelu3 = nn.LeakyReLU(0.2)
        self.deconv1 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, num_classes, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        x = self.linear1(x)
        x = self.lrelu1(x)
        x = self.linear2(x)
        x = self.lrelu2(x)
        x = self.linear3(x)
        x = self.lrelu3(x)
        x = x.reshape(x.size(0), 512, self.spatial_resolution, self.spatial_resolution)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x

class HM2Pose(nn.Module):
    def __init__(self, num_class=16, heatmap_resolution=47, dropout=0.0):
        super(HM2Pose, self).__init__()
        self.num_class = num_class
        self.conv1 = nn.Conv2d(num_class, 64, kernel_size=4, stride=2, padding=2)
        self.lrelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.lrelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(128, 512, kernel_size=4, stride=2, padding=1)
        self.lrelu3 = nn.PReLU()
        self.spatial_resolution = math.ceil(heatmap_resolution/8.)
        self.linear1 = nn.Linear(self.spatial_resolution**2*512, 2048)
        self.lrelu4 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(2048, 512)
        self.lrelu5 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(512, num_class*3)
 

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu1(x)

        x = self.conv2(x)
        x = self.lrelu2(x)

        x = self.conv3(x)
        x = self.lrelu3(x)

        x = x.reshape(x.size(0), -1) # flatten
        x = self.linear1(x)
        x = self.lrelu4(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.lrelu5(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = x.reshape(x.size(0), -1, 3)
        return x

# class HM2Pose(nn.Module):
#     def __init__(self, num_class=16):
#         super(HM2Pose, self).__init__()
#         self.num_class = num_class
#         self.conv1 = nn.Conv2d(num_class, 32, kernel_size=4, stride=2, padding=2)
#         self.lrelu1 = nn.PReLU()
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
#         self.lrelu2 = nn.PReLU()
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
#         self.lrelu3 = nn.PReLU()

#         self.linear1 = nn.Linear(4608, num_class*3)

 

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.lrelu1(x)
#         x = self.conv2(x)
#         x = self.lrelu2(x)
#         x = self.conv3(x)
#         x = self.lrelu3(x)
#         x = x.reshape(x.size(0), -1) # flatten
#         x = self.linear1(x)
#         x = x.reshape(x.size(0), -1, 3)
#         return x

class HM2PoseDist(nn.Module):
    def __init__(self):
        super(HM2PoseDist, self).__init__()
        self.conv1_2d = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=2)
        self.relu1_2d = nn.PReLU()
        self.conv2_2d = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.relu2_2d = nn.PReLU()
        self.conv3_2d = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.relu3_2d = nn.PReLU()

        self.conv1_1d = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu1_1d = nn.PReLU()
        self.conv2_1d = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2_1d = nn.PReLU()
        self.conv3_1d = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu3_1d = nn.PReLU()

        self.linear1 = nn.Linear(7808, 2048)
        self.relu4 = nn.PReLU()
        self.linear2 = nn.Linear(2048, 512)
        self.relu5 = nn.PReLU()
        self.linear3 = nn.Linear(512, 48)
 

    def forward(self, heatmap_2d, heatmap_1d):
        heatmap_2d = self.conv1_2d(heatmap_2d)
        heatmap_2d = self.relu1_2d(heatmap_2d)
        heatmap_2d = self.conv2_2d(heatmap_2d)
        heatmap_2d = self.relu2_2d(heatmap_2d)
        heatmap_2d = self.conv3_2d(heatmap_2d)
        heatmap_2d = self.relu3_2d(heatmap_2d)
        heatmap_2d = heatmap_2d.reshape(heatmap_2d.size(0), -1) # flatten

        heatmap_1d = self.conv1_1d(heatmap_1d)
        heatmap_1d = self.relu1_1d(heatmap_1d)
        heatmap_1d = self.conv2_1d(heatmap_1d)
        heatmap_1d = self.relu2_1d(heatmap_1d)
        heatmap_1d = self.conv3_1d(heatmap_1d)
        heatmap_1d = self.relu3_1d(heatmap_1d)
        heatmap_1d = heatmap_1d.reshape(heatmap_1d.size(0), -1) # flatten


        heatmap = torch.cat((heatmap_2d, heatmap_1d), dim=1)
        pose = self.linear1(heatmap)
        pose = self.relu4(pose)
        pose = self.linear2(pose)
        pose = self.relu5(pose)
        pose = self.linear3(pose)
        pose = pose.reshape(pose.size(0), 16, 3)
        return pose

class UNet(nn.Module):
    def __init__(self, load_resnet=None):
        super(UNet, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=False)
        if load_resnet:
            self.resnet18.load_state_dict(torch.load(load_resnet))

        self.encoder1 = nn.Sequential(*[l for ind, l in enumerate(self.resnet18.children()) if ind < 3])
        self.encoder2 = nn.Sequential(*[l for ind, l in enumerate(self.resnet18.children()) if 3 <= ind <= 4])
        self.encoder3 = nn.Sequential(*[l for ind, l in enumerate(self.resnet18.children()) if ind == 5])
        self.encoder4 = nn.Sequential(*[l for ind, l in enumerate(self.resnet18.children()) if ind == 6])
        self.encoder5 = nn.Sequential(*[l for ind, l in enumerate(self.resnet18.children()) if ind == 7])

        self.up1 = Up(512, 256, False)
        self.up2 = Up(256, 128, False )
        self.up3 = Up(128, 64, False)
        self.up4 = Up(64, 64, False)
        self.outc = nn.Conv2d(64, 16, 1)

    def forward(self, x):

        encoder1 = self.encoder1(x)
        # 64 x 192 x 192
        encoder2 = self.encoder2(encoder1)
        # 64 x 96 x 96
        encoder3 = self.encoder3(encoder2)
        # 128 x 48 x 48
        encoder4 = self.encoder4(encoder3)
        # 256 x 24 x 24
        encoder5 = self.encoder5(encoder4)
        # 512 x 12 x 12
        x = self.up1(encoder5, encoder4)
        x = self.up2(x, encoder3)
        x = self.up3(x, encoder2)
        x = self.up4(x, encoder1)
        x = self.outc(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels//2 + out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class HM2PoseUNet(nn.Module):
    def __init__(self):
        super(HM2PoseUNet, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.lrelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.lrelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.lrelu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.lrelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.lrelu5 = nn.PReLU()

        self.linear1 = nn.Linear(9216, 2048)
        self.lrelu4 = nn.PReLU()
        self.linear2 = nn.Linear(2048, 512)
        self.lrelu5 = nn.PReLU()
        self.linear3 = nn.Linear(512, 48)
 

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.lrelu3(x)
        x = self.conv4(x)
        x = self.lrelu4(x)
        x = self.conv5(x)
        x = self.lrelu5(x)
        x = x.reshape(x.size(0), -1) # flatten

        x = self.linear1(x)
        x = self.lrelu4(x)
        x = self.linear2(x)
        x = self.lrelu5(x)
        x = self.linear3(x)
        x = x.reshape(x.size(0), 16, 3)
        return x


class HM2PoseLinear(nn.Module):
    def __init__(self, num_class=16):
        super().__init__()
        self.num_class = num_class
        self.linear = nn.Linear(16*47*47, num_class*3)
 

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        x = x.reshape(x.size(0), -1, 3)
        return x

class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
  

    def forward(self, x):
        y = self.w1(x)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class LinearModel(nn.Module):
    def __init__(self,
                 linear_size=1024,
                 num_stage=2,
                 p_dropout=0.5):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size =  17 * 2
        # 3d joints
        self.output_size = 17 * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)


        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        x = x.reshape(x.size(0), -1)
        y = self.w1(x)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)
        y = y.reshape(y.size(0), 17, 3)
        return y