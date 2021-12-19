# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import *
import torchvision
import time
import math
from torchsummary import summary

class xREgoPose(nn.Module):
    def __init__(self):
        super(xREgoPose, self).__init__()
        # Resnet 101 without last average pooling and fully connected layers
        self.resnet101 = nn.Sequential(*[l for ind, l in enumerate(torchvision.models.resnet101(pretrained=False).children()) if ind < 8])
        # First Deconvolution to obtain 2D heatmap
        self.heatmap_deconv = nn.Sequential(*[nn.ConvTranspose2d(2048, 1024, kernel_size=3,
                                                                 stride=2, dilation=1, padding=1),
                                              nn.ConvTranspose2d(1024, 15, kernel_size=3,
                                                                 stride=2, dilation=1, padding=0)])
        # Encoder that takes 2D heatmap and transforms to latent vector Z
        self.encoder = Encoder()
        # Pose decoder that takes latent vector Z and transforms to 3D pose coordinates
        self.pose_decoder = PoseDecoder()
        # Heatmap decoder that takes latent vector Z and generates the original 2D heatmap
        self.heatmap_decoder = HeatmapDecoder()


    def forward(self, x):
        # x = 3 x 368 x 368

        x = self.resnet101(x)
        # x = 2048 x 12 x 12

        heatmap = self.heatmap_deconv(x)
        # heatmap = 15 x 47 x 47

        z = self.encoder(heatmap)
        # z = 20

        pose = self.pose_decoder(z)
        # pose = 16 x 3

        generated_heatmaps = self.heatmap_decoder(z)
        # generated_heatmaps = 15 x 47 x 47

        return heatmap, pose, generated_heatmaps


