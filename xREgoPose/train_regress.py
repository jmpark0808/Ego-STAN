import argparse
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from matplotlib.pyplot import MultipleLocator
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import dataset.transform as trsf
from base import SetType
from dataset.mocap import Mocap
from loss import auto_encoder_loss, mse
from utils import config, evaluate
from xREgoPose.net.DirectRegression import DirectRegression

# Deterministic
pl.seed_everything(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--load",
                        help="Directory of pre-trained model,  \n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model")
    parser.add_argument('--dataset_tr', help='Directory of your train Dataset', required=True, default=None)
    parser.add_argument('--dataset_val', help='Directory of your validation Dataset', required=True, default=None)
    parser.add_argument('--cuda', help="'cuda' for cuda, 'cpu' for cpu, default = cuda",
                        default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', help="batchsize, default = 1", default=1, type=int)
    parser.add_argument('--epoch', help='# of epochs. default = 20', default=20, type=int)
    parser.add_argument('--model_save_freq', help='How often to save model weights, in batch units', default=64, type=int)
    parser.add_argument('--val_freq', help='How often to run validation set, in batch units', default=64, type=int)
    parser.add_argument('--es_patience', help='Max # of consecutive validation runs w/o improvment', default=5, type=int)
    parser.add_argument('--logdir', help='logdir for models and losses. default = .', default='./', type=str)
    parser.add_argument('--lr', help='learning_rate for pose. default = 0.001', default=0.001, type=float)
    parser.add_argument('--lr_decay', help='Learning rate decrease by lr_decay time per decay_step, default = 0.1',
                        default=0.1, type=float)
    parser.add_argument('--decay_step', help='Learning rate decrease by lr_decay time per decay_step,  default = 7000',
                        default=1E100, type=int)
    parser.add_argument('--display_freq', help='Frequency to display result image on Tensorboard, in batch units',
                        default=64, type=int)
    parser.add_argument('--load_resnet', help='Directory of ResNet 101 weights', default=None)
    parser.add_argument('--hm_train_steps', help='Number of steps to pre-train heatmap predictor', default=100000, type=int)

    args = parser.parse_args()
    dict_args = vars(args)

    # Initialize logging paths
    now = datetime.datetime.now()
    os.makedirs(os.path.join('log', now.strftime('%m%d%H%M')), exist_ok=True)
    weight_save_dir = os.path.join(dict_args["logdir"], os.path.join('models', 'state_dict', now.strftime('%m%d%H%M')))
    os.makedirs(os.path.join(weight_save_dir), exist_ok=True)

    # Initialize model to train
    model = DirectRegression().to(device=dict_args['cuda'])

    # Callback: early stopping parameters
    early_stopping_callback = EarlyStopping(
        monitor="val_mpjpe_full_body",
        mode="min",
        verbose=True,
        patience=dict_args["es_patience"],
    )
    
    # Callback: monitor learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Callback: model checkpoint strategy
    checkpoint_callback = ModelCheckpoint(
        dirpath=weight_save_dir, save_top_k=5, verbose=True, monitor="val_mpjpe_full_body", mode="min"
    )

    # Data: data transformation strategy
    data_transform = transforms.Compose(
        [trsf.ImageTrsf(), trsf.Joints3DTrsf(), trsf.ToTensor()]
    )

    # Data: create train dataloader
    data_train = Mocap(dict_args["dataset_tr"], SetType.TRAIN, transform=data_transform)
    dataloader_train = DataLoader(
        data_train, batch_size=dict_args["batch_size"], shuffle=True, pin_memory=True
    )

    # Data: create validation dataloader
    data_val = Mocap(dict_args["dataset_val"], SetType.VAL, transform=data_transform)
    dataloader_val = DataLoader(data_val, batch_size=dict_args["batch_size"], pin_memory=True)

