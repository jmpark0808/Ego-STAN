# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import *
import torchvision
import time
import math
from torchsummary import summary
from net.transformer import PoseTransformer

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

# -> Edit of HeatMap class, except returns a feature map as well -> Why is it in net and not in blocks?

class FeatureHeatMaps(nn.Module):
    def __init__(self):
        super(FeatureHeatMaps, self).__init__()
        # Resnet 101 without last average pooling and fully connected layers
        self.resnet101 = torchvision.models.resnet101(pretrained=False)
        # Upsampling to get 47x47 feature maps
        self.feature_upsample = nn.Upsample(scale_factor=(47/12), mode= 'nearest')
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
    def __init__(self, seq_len):
        super(SequenceEmbedder, self).__init__()
        # Generator that produces the HeatMap
        self.heatmap = HeatMap()
        # Encoder that takes 2D heatmap and transforms to latent vector Z
        self.encoder = Encoder()
        # Transformer that takes sequence of latent vector Z and outputs a single Z vector
        self.seq_transformer = PoseTransformer(seq_len=seq_len, dim=256, depth=3, heads=8, mlp_dim=512)
        # Pose decoder that takes latent vector Z and transforms to 3D pose coordinates
        self.pose_decoder = PoseDecoder()
        # Heatmap decoder that takes latent vector Z and generates the original 2D heatmap
        self.heatmap_decoder = HeatmapDecoder()


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

        z = self.seq_transformer(zs)
        # z = batch_size x 20

        p3d = self.pose_decoder(z)
        # p3d = batch_size x 16 x 3

        p2d = self.heatmap_decoder(z_all)
        # p2d = (batch_size*len_seq) x 15 x 47 x 47

        return hms, p3d, p2d

class xREgoPose(nn.Module):
    def __init__(self):
        super(xREgoPose, self).__init__()
        # Generator that produces the HeatMap
        self.heatmap = HeatMap()
        # Encoder that takes 2D heatmap and transforms to latent vector Z
        self.encoder = Encoder()
        # Pose decoder that takes latent vector Z and transforms to 3D pose coordinates
        self.pose_decoder = PoseDecoder()
        # Heatmap decoder that takes latent vector Z and generates the original 2D heatmap
        self.heatmap_decoder = HeatmapDecoder()


    def forward(self, x):
        # x = 3 x 368 x 368

        heatmap = self.heatmap(x)
        # heatmap = 15 x 47 x 47
        
        z = self.encoder(heatmap)
        # z = 20

        pose = self.pose_decoder(z)
        # pose = 16 x 3

        generated_heatmaps = self.heatmap_decoder(z)
        # generated_heatmaps = 15 x 47 x 47

        return heatmap, pose, generated_heatmaps
        
class FeatureConcatEgoPose(nn.Module):

    encoder_dict = {
        'map_concat': FeatureConcatEncoder(),
        'branch_concat': FeatureBranchEncoder(),
        'concat_reconstruct': FeatureReEncoder()
    }

    def __init__(self, encoder_type = 'map_concat'):
        super(FeatureConcatEgoPose, self).__init__()
        # Generator that produces both the HeatMap and Feature Map (increasing uncertainty)
        self.feature_heatmaps = FeatureHeatMaps()
        # Encoder that takes 2D heatmap as well as Feature Maps and transforms to latent vector Z
        self.encoder = self.encoder_dict[encoder_type]
        # Pose decoder that takes latent vector Z and transforms to 3D pose coordinates
        self.pose_decoder = PoseDecoder()
        # Heatmap decoder that takes latent vector Z and generates the original 2D heatmap
        self.heatmap_decoder = HeatmapDecoder()


    def forward(self, x):
        # x = 3 x 368 x 368

        heatmap, depthmap = self.feature_heatmaps(x)
        # heatmap = 15 x 47 x 47, depthmap = 15 x 47 x 47
        
        z = self.encoder(heatmap, depthmap)
        # z = 20

        pose = self.pose_decoder(z)
        # pose = 16 x 3

        generated_heatmaps = self.heatmap_decoder(z)
        # generated_heatmaps = 15 x 47 x 47

        return heatmap, pose, generated_heatmaps
