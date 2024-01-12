# ----------------------------------------------------------- #
#  This is code confidential, for peer-review purposes only   #
#  and protected under conference code of ethics              #
# ----------------------------------------------------------- #

# Code adapted from https://github.com/facebookresearch/xR-EgoPose authored by Denis Tome

"""
Transformation to apply to the data

Adapted from original

"""
import torch
import numpy as np
from base import BaseTransform
from utils import config
import matplotlib.pyplot as plt

class ImageTrsf(BaseTransform):
    """Image Transform"""

    def __init__(self, mean=0.5, std=0.5):

        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """Perform transformation

        Arguments:
            data {dict} -- frame data
        """

        if 'image' not in list(data.keys()):
            return data

        # get image from all data
        img = data['image']

        # channel last to channel first
        img = np.transpose(img, [2, 0, 1])

        # normalization
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        img[0, :, :] = (img[0, :, :]-mean[0])/std[0]
        img[1, :, :] = (img[1, :, :]-mean[1])/std[1]
        img[2, :, :] = (img[2, :, :]-mean[2])/std[2]
        data.update({'image': img})

        return data


class Joints3DTrsf(BaseTransform):
    """Joint Transform"""

    def __init__(self, jid_to_zero = None):

        super().__init__()
        joint_zeroed = config.transforms.norm

        # Added a parameter to manually specify which joint to subtract from p3d

        if jid_to_zero is None:
            assert joint_zeroed in config.skel.keys()
            self.jid_zeroed = config.skel[joint_zeroed].jid
        else:
            self.jid_zeroed = jid_to_zero

    def __call__(self, data):
        """Perform transformation

        Arguments:
            data {dict} -- frame data
        """

        if 'joints3D' not in list(data.keys()):
            return data

        p3d = data['joints3D']
        joint_zeroed = p3d[self.jid_zeroed][np.newaxis]

        # update p3d
        p3d -= joint_zeroed
        data.update({'joints3D': p3d})

        return data

class HorizontalFlip(BaseTransform):
    """Image Transform"""

    def __init__(self, probability=0.5):

        super().__init__()
        self.prob = probability

    def __call__(self, data):
        """Perform transformation

        Arguments:
            data {dict} -- frame data
        """
        random_dice = data['random_dice'][0]
        if 'image' in list(data.keys()):
            img = data['image']
            if random_dice<=self.prob:
                img = np.flip(img, axis=2).copy()
            data.update({'image': img})
        elif 'joints2D_heatmap' in list(data.keys()):
            p2d = data['joints2D_heatmap']
            if random_dice<=self.prob:
                p2d = np.flip(p2d, axis=2).copy()
            data.update({'joints2D_heatmap': p2d})
        elif 'joints3D' in list(data.keys()):
            p3d = data['joints3D']
            if random_dice<=self.prob:
                left = [2, 3, 4, 8, 9, 10]
                right = [5, 6, 7, 11, 12, 13]
                p3d[:, 0] *= -1
                p3d[left+right] = p3d[right+left]
            data.update({'joints3D': p3d})
        else:
            raise('Wrong keys')
        # get image from all data
               
        return data

class ToTensor(BaseTransform):
    """Convert ndarrays to Tensors."""

    def __call__(self, data):
        """Perform transformation

        Arguments:
            data {dict} -- frame data
        """

        keys = list(data.keys())
        for k in keys:
            pytorch_data = torch.from_numpy(data[k]).float()
            data.update({k: pytorch_data})

        return data
