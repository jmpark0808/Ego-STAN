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
from utils import config
from base import SetType
import dataset.transform as trsf
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt

if __name__ == "__main__":
    batch_size = 1024
    epochs = 1
    data_path = "/home/s42hossa/projects/def-pfieguth/xREgoPose/xR-EgoPose/data/Dataset/ValSet"

    data_transform = transforms.Compose([
        trsf.ImageTrsf(),
        trsf.Joints3DTrsf(),
        trsf.ToTensor()])

    data = Mocap(
        data_path,
        SetType.TRAIN,
        transform=data_transform)
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True)
	
	for i, batch in enumerate(tqdm(dataloader)):
		if(i%100 == 0):
			print("Batch Number: {}".format(i)) 
		
