import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Any, Optional, List
from torch import Tensor
import math


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

class PoseDecoder(nn.Module):
    def __init__(self):
        super(PoseDecoder, self).__init__()
        self.linear1 = nn.Linear(20, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 48)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = x.reshape(x.size(0), 16, 3)
        return x

class HeatmapDecoder(nn.Module):
    def __init__(self):
        super(HeatmapDecoder, self).__init__()
        self.linear1 = nn.Linear(20, 512)
        self.linear2 = nn.Linear(512, 2048)
        self.linear3 = nn.Linear(2048, 18432)
        self.deconv1 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 15, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = x.reshape(x.size(0), 512, 6, 6)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x