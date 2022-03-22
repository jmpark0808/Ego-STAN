import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Any, Optional, List
from torch import Tensor
import math
import torchvision

class HeatMap(nn.Module):
    def __init__(self):
        super(HeatMap, self).__init__()
        # Resnet 101 without last average pooling and fully connected layers
        self.resnet101 = torchvision.models.resnet101(pretrained=False)
        # First Deconvolution to obtain 2D heatmap
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

        return heatmap


class HeatMapDist(nn.Module):
    def __init__(self):
        super(HeatMapDist, self).__init__()
        # Resnet 101 without last average pooling and fully connected layers
        self.resnet_2d_heatmap = torchvision.models.resnet50(pretrained=False)
        self.resnet_1d_heatmap = torchvision.models.resnet50(pretrained=False)
        # First Deconvolution to obtain 2D heatmap
        self.heatmap_deconv = nn.Sequential(*[nn.ConvTranspose2d(2048, 1024, kernel_size=3,
                                                                 stride=2, dilation=1, padding=1),
                                              nn.ConvTranspose2d(1024, 16, kernel_size=3,
                                                                 stride=2, dilation=1, padding=0)])
        self.heatmap_linear = nn.Linear(2048, 480)



    def update_resnet50(self):
        self.resnet_2d_heatmap = nn.Sequential(*[l for ind, l in enumerate(self.resnet_2d_heatmap.children()) if ind < 8])
        self.resnet_1d_heatmap = nn.Sequential(*[l for ind, l in enumerate(self.resnet_1d_heatmap.children()) if ind < 9])

    def forward(self, x):
        # x = 3 x 368 x 368

        x_2d = self.resnet_2d_heatmap(x)
        # x_2d = 2048 x 12 x 12
        
        x_1d = self.resnet_1d_heatmap(x)
        x_1d = x_1d.reshape(x_1d.size(0), -1)
        # x_1d = 2048
 
        heatmap_2d = self.heatmap_deconv(x_2d)
        # heatmap_2d = 16 x 47 x 47
        heatmap_1d = self.heatmap_linear(x_1d)
        heatmap_1d = heatmap_1d.reshape(heatmap_1d.size(0), 16, 30)
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
    def __init__(self):
        super(Encoder, self).__init__()
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
    def __init__(self, initial_dim=20):
        super(PoseDecoder, self).__init__()
        self.linear1 = nn.Linear(initial_dim, 32)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(32, 32)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.linear3 = nn.Linear(32, 48)

    def forward(self, x):
        x = self.linear1(x)
        x = self.lrelu1(x)
        x = self.linear2(x)
        x = self.lrelu2(x)
        x = self.linear3(x)
        x = x.reshape(x.size(0), 16, 3)
        return x

class HeatmapDecoder(nn.Module):
    def __init__(self):
        super(HeatmapDecoder, self).__init__()
        self.linear1 = nn.Linear(20, 512)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(512, 2048)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.linear3 = nn.Linear(2048, 18432)
        self.lrelu3 = nn.LeakyReLU(0.2)
        self.deconv1 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 15, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        x = self.linear1(x)
        x = self.lrelu1(x)
        x = self.linear2(x)
        x = self.lrelu2(x)
        x = self.linear3(x)
        x = self.lrelu3(x)
        x = x.reshape(x.size(0), 512, 6, 6)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x

class HM2Pose(nn.Module):
    def __init__(self):
        super(HM2Pose, self).__init__()
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
        self.linear3 = nn.Linear(512, 48)
 

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
        x = x.reshape(x.size(0), 16, 3)
        return x

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

        self.linear1 = nn.Linear(5120, 2048)
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