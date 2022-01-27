# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import *
import torchvision
import time
import math
from torchsummary import summary

class HeatMap(nn.Module):
    def __init__(self):
        super(HeatMap, self).__init__()
        # Resnet 101 without last average pooling and fully connected layers
        self.resnet101 = nn.Sequential(*[l for ind, l in enumerate(torchvision.models.resnet101(pretrained=False).children()) if ind < 8])
        # First Deconvolution to obtain 2D heatmap
        self.heatmap_deconv = nn.Sequential(*[nn.ConvTranspose2d(2048, 1024, kernel_size=3,
                                                                 stride=2, dilation=1, padding=1),
                                              nn.ConvTranspose2d(1024, 15, kernel_size=3,
                                                                 stride=2, dilation=1, padding=0)])


    def forward(self, x):
        # x = 3 x 368 x 368

        x = self.resnet101(x)
        # x = 2048 x 12 x 12

        heatmap = self.heatmap_deconv(x)
        # heatmap = 15 x 47 x 47

        return heatmap


class PoseEstimator(nn.Module):
    def __init__(self):
        super(PoseEstimator, self).__init__()
        # Encoder that takes 2D heatmap and transforms to latent vector Z
        self.encoder = Encoder()
        # Pose decoder that takes latent vector Z and transforms to 3D pose coordinates
        self.pose_decoder = PoseDecoder()
        # Heatmap decoder that takes latent vector Z and generates the original 2D heatmap
        self.heatmap_decoder = HeatmapDecoder()


    def forward(self, x):
        z = self.encoder(x)
        # z = 20

        pose = self.pose_decoder(z)
        # pose = 16 x 3

        generated_heatmaps = self.heatmap_decoder(z)
        # generated_heatmaps = 15 x 47 x 47

        return generated_heatmaps, pose

class SequenceEmbedder(nn.Module):
    def __init__(self):
        super(SequenceEmbedder, self).__init__()
        # Generator that produces the HeatMap
        self.heatmap = HeatMap()
        # Encoder that takes 2D heatmap and transforms to latent vector Z
        self.encoder = Encoder()


    def forward(self, x):
        # Flattening first two dimensions

        dim = x.shape 
        #shape -> batch_size x len_seq x 3 x 368 x 368

        imgs = torch.reshape(x, (dim[0]*dim[1], dim[2], dim[3], dim[4]))
        # imgs = # (batch_size*len_seq) x 3 x 368 x 368

        hms = self.heatmap(imgs)
        # hms = (batch_size*len_seq) x 15 x 47 x 47

        z_all = self.encoder(hms)
        # z_all = (batch_size*len_seq) x 20

        zs = torch.reshape(z_all, (dim[0], dim[1], z_all.shape[-1]))
        # zs = batch_size x len_seq x 20

        return zs
