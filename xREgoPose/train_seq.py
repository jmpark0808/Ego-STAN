import datetime
import os
import numpy as np
import argparse
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
from loss import mse, auto_encoder_loss
from network import *
from dataset.mocap import Mocap
from dataset.mocap_transformer import MocapTransformer
from utils import config
from base import SetType
import dataset.transform as trsf
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--load_hm",
                        help="Directory of pre-trained model for heatmap,  \n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model")
    parser.add_argument("--load_pose",
                        help="Directory of pre-trained model for pose estimator,  \n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model")
    parser.add_argument('--dataset', help='Directory of your Dataset', required=True, default=None)
    parser.add_argument('--cuda', help="'cuda' for cuda, 'cpu' for cpu, default = cuda",
                        default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', help="batchsize, default = 1", default=1, type=int)
    parser.add_argument('--epoch', help='# of epochs. default = 20', default=20, type=int)
    parser.add_argument('--sequence_length', help="# of images/frames input into sequential model, default = 5",
                        default='5', type=int)
        
                        
    args = parser.parse_args()
    device = torch.device(args.cuda)
    batch_size = args.batch_size
    epoch = args.epoch
    seq_len = args.sequence_length

    data_transform = transforms.Compose([
        trsf.ImageTrsf(),
        trsf.Joints3DTrsf(),
        trsf.ToTensor()])

    data = MocapTransformer(
        args.dataset,
        SetType.TRAIN,
        transform=data_transform,
        sequence_length = seq_len)
    dataloader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True)

    load_hm = args.load_hm
    load_pose = args.load_pose
    start_iter = 0
    model_hm = HeatMap().to(device=args.cuda)
    model_pose = PoseEstimator().to(device=args.cuda)
    sequence_embedder = SequenceEmbedder().to(device=args.cuda)

    # Xavier Initialization
    def weight_init(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model_hm.resnet101.apply(weight_init)
    sequence_embedder.heatmap.resnet101.apply(weight_init)
    model_hm.heatmap_deconv.apply(weight_init)
    sequence_embedder.heatmap.heatmap_deconv.apply(weight_init)

    model_pose.encoder.apply(weight_init)
    sequence_embedder.encoder.apply(weight_init)
    model_pose.pose_decoder.apply(weight_init)
    model_pose.heatmap_decoder.apply(weight_init)

    now = datetime.datetime.now()
    start_epo = 0


    if load_hm is not None:
        state_dict_hm = torch.load(load_hm, map_location=args.cuda)


        start_iter = int(load_hm.split('epo_')[1].strip('step.ckpt'))
        start_epo = int(load_hm.split('/')[-1].split('epo')[0])
        now = datetime.datetime.strptime(load_hm.split('/')[-2], '%m%d%H%M')

        print("Loading Model from {}".format(load_hm))
        print("Start_iter : {}".format(start_iter))
        print("now : {}".format(now.strftime('%m%d%H%M')))
        model_hm.load_state_dict(state_dict_hm)
        sequence_embedder.heatmap.load_state_dict(model_hm.state_dict())
        print('Loading_Complete')

    if load_pose is not None:
        state_dict_pose = torch.load(load_pose, map_location=args.cuda)

        start_iter = int(load_pose.split('epo_')[1].strip('step.ckpt'))
        start_epo = int(load_pose.split('/')[-1].split('epo')[0])
        now = datetime.datetime.strptime(load_pose.split('/')[-2], '%m%d%H%M')

        print("Loading Model from {}".format(load_pose))
        print("Start_iter : {}".format(start_iter))
        print("now : {}".format(now.strftime('%m%d%H%M')))
        model_pose.load_state_dict(state_dict_pose)
        sequence_embedder.encoder.load_state_dict(model_pose.encoder.state_dict())
        print('Loading_Complete')

    # Freezing the embedder
    sequence_embedder.eval()
    for embedder_param in sequence_embedder.parameters():
        embedder_param.requires_grad = False

    for epo in range(start_epo, epoch):
        print("\nEpoch : {}".format(epo))
        for i, batch in enumerate(tqdm(dataloader)):
            sequence_imgs, p2d, p3d, action = batch
            sequence_imgs = sequence_imgs.cuda()
            p2d = p2d.cuda()
            p3d = p3d.cuda()
            embeddings = sequence_embedder(sequence_imgs)
