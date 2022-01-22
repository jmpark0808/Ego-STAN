# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Data processing where only Images and associated 3D
joint positions are loaded.
@author: Denis Tome'
"""
import os
import torch
from skimage import io as sio
from skimage.transform import resize
import numpy as np
from base import BaseDataset
from utils import io, config
import matplotlib.pyplot as plt

def generate_heatmap(joints, heatmap_sigma):
    """
    :param joints:  [nof_joints, 2]
    :return: target, target_weight(1: visible, 0: invisible)
    """
    heatmap_size = [47, 47]
    num_joints = 15
    target = np.zeros((num_joints,
                       heatmap_size[0],
                       heatmap_size[1]),
                      dtype=np.float32)
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    tmp_size = heatmap_sigma * 3

    for joint_id in range(num_joints):
        feat_stride = np.asarray([940, 800]) / np.asarray([heatmap_size[0], heatmap_size[1]])
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        if mu_x < 0 or mu_y < 0:
            target_weight[joint_id] = 0
            continue
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * heatmap_sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]


    return target


class MocapTransformer(BaseDataset):
    """Mocap Dataset loader"""

    ROOT_DIRS = ['rgba', 'json']
    CM_TO_M = 100
    SEQUENCE_LENGTH = 5

    def __init__(self, *args, sequence_length=5, **kwargs):
        """Init class, to allow variable sequence length, inherits from Base
        Keyword Arguments:
            sequence_length -- length of image sequence (default: {5})
        """

        super().__init__(*args, **kwargs)
        self.SEQUENCE_LENGTH = sequence_length

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
                
                # get the data in sequences -> assuming no missing frames

                encoded = []

                len_seq = self.SEQUENCE_LENGTH

                if sub_dir == 'rgba':
                    for p in paths:
                        encoded_sequence = []
                        frame_idx = int(p[-10:-4])

                        for i in range(len_seq):
                            if os.path.exists(p[0:-10] + "{0:06}.png".format(frame_idx+i)):
                                frame_path = p[0:-10] + "{0:06}.png".format(frame_idx+i)
                                encoded_sequence.append(frame_path.encode('utf8'))
                        
                        if(len(encoded_sequence) == len_seq):
                            encoded.append(encoded_sequence)

                elif sub_dir == 'json':
                    for p in paths:
                        isSequence = True
                        frame_idx = int(p[-11:-5])

                        for i in range(len_seq):
                            if not os.path.exists(p[0:-11] + "{0:06}.json".format(frame_idx+i)):
                                isSequence = False

                        if(isSequence):
                            head_path = p[0:-11] + "{0:06}.json".format(frame_idx+len_seq-1)
                            encoded.append(head_path.encode('utf8'))

                else: 
                    self.logger.error(
                        "No case for handling {} sub-directory".format(sub_dir))
                    
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

        # load image sequence

        img_paths = [path.decode('utf8') for path in self.index['rgba'][index]]

        # checking for correct sequence
        for i in range(len(img_paths)-1):
            if int(img_paths[i][-10:-4]) != int(img_paths[i+1][-10:-4]) -1:
                self.logger.error(
                    '{} \n is not the correct frame after \n {}'.format(
                                                    img_paths[i+1], img_paths[i])
                                                                                )

        imgs = [sio.imread(img_path).astype(np.float32) for img_path in img_paths]
        imgs = [img / 255.0 for img in imgs]
        imgs = [img[:, 180:1120, :] for img in imgs]
        imgs = np.array([resize(img, (368, 368)) for img in imgs])

        # read joint positions
        json_path = self.index['json'][index].decode('utf8')
        
        # checking if json path corresponds to sequence path

        if int(json_path[-11:-5]) != int(img_paths[-1][-10:-4]):
            self.logger.error(
                '{} \n json path does not match last frame: \n {}'.format(
                                                        json_path, img_paths[-1])
                                                                                )

        data = io.read_json(json_path)

        p2d, p3d = self._process_points(data)
        p2d[:, 0] = p2d[:, 0]-180 # Translate p2d coordinates by 180 pixels to the left


        p2d_heatmap = generate_heatmap(p2d[1:, :], 3) # exclude head

        # get action name
        action = data['action']

        if self.transform:
            imgs = np.array(
                [self.transform({'image': img})['image'].numpy() for img in imgs])
            p3d = self.transform({'joints3D': p3d})['joints3D']
            p2d = self.transform({'joints2D': p2d})['joints2D']

        return torch.tensor(imgs), p2d_heatmap, p3d, action

    def __len__(self):

        return len(self.index[self.ROOT_DIRS[0]])
