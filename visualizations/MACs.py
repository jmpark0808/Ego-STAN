import sys
sys.path.insert(0, '/home/eddie/waterloo/xREgoPose')

from ptflops import get_model_complexity_info
from net.blocks import HeatMap, Encoder, PoseDecoder, HeatmapDecoder, LinearModel, HM2Pose
from net.transformer import ResNetTransformerCls
import torchvision
import torch.nn as nn


heatmap = HeatMap(16)
heatmap.update_resnet101()
# Encoder that takes 2D heatmap and transforms to latent vector Z
encoder = Encoder(16, 47)
# Pose decoder that takes latent vector Z and transforms to 3D pose coordinates
pose_decoder = PoseDecoder(num_classes = 16)
# Heatmap decoder that takes latent vector Z and generates the original 2D heatmap
heatmap_decoder = HeatmapDecoder(16, 47)
linear = LinearModel()

resnet = torchvision.models.resnet101(pretrained=False)
resnet101 = nn.Sequential(*[l for ind, l in enumerate(resnet.children()) if ind < 8])
heatmap_deconv = nn.Sequential(*[nn.ConvTranspose2d(2048, 1024, kernel_size=3,
                                                                 stride=2, dilation=1, padding=1),
                                              nn.ConvTranspose2d(1024, 16, kernel_size=3,
                                                                 stride=2, dilation=1, padding=0)])

resnet_transformer = ResNetTransformerCls(in_dim=2048, spatial_dim= 12*12, seq_len=5*12*12, dim=512, depth=3, heads=8, mlp_dim=1024, dim_head=64, dropout=0.5)
hm2pose = HM2Pose(16, 47, 0.5)







heatmap_flops, params = get_model_complexity_info(heatmap, input_res=(3, 368, 368), as_strings=False)
encoder_flops, params = get_model_complexity_info(encoder, input_res=(16, 47, 47), as_strings=False)
pose_decoder_flops, _ = get_model_complexity_info(pose_decoder, input_res=(20,), as_strings=False)
heatmap_decoder_flops, _ = get_model_complexity_info(heatmap_decoder, input_res=(20,), as_strings=False)
linear_flops, _ = get_model_complexity_info(linear, input_res=(17*2,), as_strings=False)


ego_resnet_flops, _ = get_model_complexity_info(heatmap, input_res=(3, 368, 368), as_strings=False)
tfm_flops, _ = get_model_complexity_info(resnet_transformer, input_res=(5*12*12, 2048), as_strings=False)
heatmap_deconv_flops, _ = get_model_complexity_info(heatmap_deconv, input_res=(2048, 12, 12), as_strings=False)
hm2pose_flops, _ = get_model_complexity_info(hm2pose, input_res=(16, 47, 47), as_strings=False)



xregopose_flops = heatmap_flops + encoder_flops + pose_decoder_flops + heatmap_decoder_flops
martinez_flops = heatmap_flops + linear_flops
egostan_flops = ego_resnet_flops*1 + tfm_flops + heatmap_decoder_flops + hm2pose_flops
print(xregopose_flops, martinez_flops, egostan_flops)
