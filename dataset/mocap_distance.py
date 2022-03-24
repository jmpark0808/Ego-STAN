# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Data processing where only Images and associated 3D
joint positions are loaded.

@author: Denis Tome'

"""
import os
import pytorch_lightning as pl
from skimage import io as sio
from skimage.transform import resize
import numpy as np
import torch
from base import BaseDataset
from utils import io, config
from base import SetType
import dataset.transform as trsf
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.mocap import generate_heatmap, generate_heatmap_distance
import matplotlib.pyplot as plt

def generate_heatmap_1d(distances, heatmap_sigma):
    """
    :param distances:  [nof_joints]
    :return: target, target_weight(1: visible, 0: invisible)
    """
    heatmap_size = 200
    num_joints = 16
    target = np.zeros((num_joints,
                       heatmap_size),
                      dtype=np.float32)
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    tmp_size = heatmap_sigma * 3

    for joint_id in range(num_joints):
        mu_x = int(distances[joint_id])
        if mu_x < 0:
            target_weight[joint_id] = 0
            continue
        # Check that any part of the gaussian is in-bounds
        l = int(mu_x - tmp_size)
        r = int(mu_x + tmp_size + 1)
        if l >= heatmap_size or r < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        x0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2) / (2 * heatmap_sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -l), min(r, heatmap_size) - l

        # Image range
        img_x = max(0, l), min(r, heatmap_size)


        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_x[0]:img_x[1]] = \
                g[g_x[0]:g_x[1]]


    return target



class MocapDistance(BaseDataset):
    """Mocap Dataset loader"""

    ROOT_DIRS = ['rgba', 'json']
    CM_TO_M = 100

    def __init__(self, *args, heatmap_type='baseline', **kwargs):
        """Init class, to allow variable sequence length, inherits from Base
        Keyword Arguments:
            sequence_length -- length of image sequence (default: {5})
        """

        self.heatmap_type = heatmap_type
        super().__init__(*args, **kwargs)

    def index_db(self):

        return self._index_dir(self.path)

    def _index_dir(self, path):
        """Recursively add paths to the set of
        indexed files

        Arguments:
            path {str} -- folder path

        Returns:
            dict -- indexed files per root dir
        """

        indexed_paths = dict()
        sub_dirs, _ = io.get_subdirs(path)
        if set(self.ROOT_DIRS) <= set(sub_dirs):

            # get files from subdirs
            n_frames = -1

            # let's extract the rgba and json data per frame
            for sub_dir in self.ROOT_DIRS:
                d_path = os.path.join(path, sub_dir)
                _, paths = io.get_files(d_path)

                if n_frames < 0:
                    n_frames = len(paths)
                else:
                    if len(paths) != n_frames:
                        self.logger.error(
                            'Frames info in {} not matching other passes'.format(d_path))

                encoded = [p.encode('utf8') for p in paths]
                indexed_paths.update({sub_dir: encoded})

            return indexed_paths

        # initialize indexed_paths
        for sub_dir in self.ROOT_DIRS:
            indexed_paths.update({sub_dir: []})

        # check subdirs of path and merge info
        for sub_dir in sub_dirs:
            indexed = self._index_dir(os.path.join(path, sub_dir))

            for r_dir in self.ROOT_DIRS:
                indexed_paths[r_dir].extend(indexed[r_dir])

        return indexed_paths

    def _process_points(self, data):
        """Filter joints to select only a sub-set for
        training/evaluation

        Arguments:
            data {dict} -- data dictionary with frame info

        Returns:
            np.ndarray -- 2D joint positions, format (J x 2)
            np.ndarray -- 3D joint positions, format (J x 3)
        """

        p2d_orig = np.array(data['pts2d_fisheye']).T
        p3d_orig = np.array(data['pts3d_fisheye']).T
        joint_names = {j['name'].replace('mixamorig:', ''): jid
                       for jid, j in enumerate(data['joints'])}

        # ------------------- Filter joints -------------------

        p2d = np.empty([len(config.skel), 2], dtype=p2d_orig.dtype)
        p3d = np.empty([len(config.skel), 3], dtype=p2d_orig.dtype)

        for jid, j in enumerate(config.skel.keys()):
            p2d[jid] = p2d_orig[joint_names[j]]
            p3d[jid] = p3d_orig[joint_names[j]]

        p3d /= self.CM_TO_M

        return p2d, p3d

    def __getitem__(self, index):

        # load image

        img_path = self.index['rgba'][index].decode('utf8')
        img = sio.imread(img_path).astype(np.float32)
        img /= 255.0
        img = img[:, 180:1120, :] #crop
        img = resize(img, (368, 368))



        # read joint positions
        json_path = self.index['json'][index].decode('utf8')

        data = io.read_json(json_path)

        p2d, p3d = self._process_points(data)
        p2d[:, 0] = p2d[:, 0]-180 # Translate p2d coordinates by 180 pixels to the left

        
        
        if self.heatmap_type == 'baseline':
            p2d_heatmap = generate_heatmap(p2d, 3) # exclude head
        elif self.heatmap_type == 'distance':
            distances = np.sqrt(np.sum(p3d**2, axis=1))[1:]
            p2d_heatmap = generate_heatmap_distance(p2d[1:, :], distances) # exclude head
        else:
            self.logger.error('Unrecognized heatmap type')
        
        # get action name
        action = data['action']

        if self.transform:
            img = self.transform({'image': img})['image']
            p3d = self.transform({'joints3D': p3d})['joints3D']
            p2d = self.transform({'joints2D': p2d})['joints2D']

        dist = torch.sqrt(torch.sum(torch.pow(p3d, 2), dim=1)).cpu().numpy()*100
        p2d_dist = generate_heatmap_1d(dist, 21) 
        if self.transform:
            p2d_dist = self.transform({'joints2d_dist': p2d_dist})['joints2d_dist']

        return img, p2d_heatmap, p2d_dist, p3d, action

    def __len__(self):

        return len(self.index[self.ROOT_DIRS[0]])

class MocapDistanceDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.train_dir = kwargs.get('dataset_tr')
        self.val_dir = kwargs.get('dataset_val')
        self.test_dir = kwargs.get('dataset_test')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers', 0)
        self.heatmap_type = kwargs.get('heatmap_type')

        # Data: data transformation strategy
        self.data_transform = transforms.Compose(
            [trsf.ImageTrsf(), trsf.Joints3DTrsf(), trsf.ToTensor()]
        )
        
    def train_dataloader(self):
        data_train = MocapDistance(self.train_dir, SetType.TRAIN, transform=self.data_transform, heatmap_type=self.heatmap_type)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        data_val = MocapDistance(self.val_dir, SetType.VAL, transform=self.data_transform, heatmap_type=self.heatmap_type)
        return DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        data_test = MocapDistance(self.test_dir, SetType.TEST, transform=self.data_transform, heatmap_type=self.heatmap_type)
        return DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)

