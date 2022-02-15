# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Data processing where only Images and associated 3D
joint positions are loaded.
@author: Denis Tome'
"""
import os
import torch
import pytorch_lightning as pl
from skimage import io as sio
from skimage.transform import resize
import numpy as np
from base import BaseDataset
from utils import io, config
from base import SetType
import dataset.transform as trsf
from dataset.mocap import generate_heatmap
from torch.utils.data import DataLoader
from torchvision import transforms

class MocapTransformer(BaseDataset):
    """Mocap Dataset loader"""

    ROOT_DIRS = ['rgba', 'json']
    CM_TO_M = 100

    def __init__(self, *args, sequence_length=5, skip =0, **kwargs):
        """Init class, to allow variable sequence length, inherits from Base
        Keyword Arguments:
            sequence_length -- length of image sequence (default: {5})
        """

        self.sequence_length = sequence_length
        self.skip = skip
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
                
                # get the data in sequences -> checks for missing frames

                encoded = []

                len_seq = self.sequence_length
                m = self.skip 

                if sub_dir == 'rgba':
                    for p in paths:
                        encoded_sequence = []
                        frame_idx = int(p[-10:-4])

                        for i in range(len_seq):
                            if os.path.exists(p[0:-10] + "{0:06}.png".format(frame_idx+i+i*m)):
                                frame_path = p[0:-10] + "{0:06}.png".format(frame_idx+i+i*m)
                                encoded_sequence.append(frame_path.encode('utf8'))
                        
                        if(len(encoded_sequence) == len_seq):
                            encoded.append(encoded_sequence)

                elif sub_dir == 'json':
                    for p in paths:
                        isSequence = True
                        encoded_sequence = []
                        frame_idx = int(p[-11:-5])

                        for i in range(len_seq):
                            if not os.path.exists(p[0:-11] + "{0:06}.json".format(frame_idx+i+i*m)):
                                isSequence = False
                            else:
                                json_path = p[0:-11] + "{0:06}.json".format(frame_idx+i+i*m)
                                encoded_sequence.append(json_path.encode('utf8'))

                        if(isSequence) and (len(encoded_sequence) == len_seq):
                            encoded.append(encoded_sequence)

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

        # checking for correct sequence of rgba/image files
        for i in range(len(img_paths)-1):
            if int(img_paths[i][-10:-4]) != int(img_paths[i+1][-10:-4]) - (self.skip + 1):
                self.logger.error(
                    '{} \n is not the correct frame after \n {}'.format(
                                                    img_paths[i+1], img_paths[i])
                                                                                )

        imgs = [sio.imread(img_path).astype(np.float32) for img_path in img_paths]
        imgs = [img / 255.0 for img in imgs]
        imgs = [img[:, 180:1120, :] for img in imgs]
        imgs = np.array([resize(img, (368, 368)) for img in imgs])

        # read joint positions
        json_paths = [path.decode('utf8') for path in self.index['json'][index]]

        # checking if json path corresponds to the path of the last rgba frame in the sequence
        # checking for correct sequence of rgba/image files
        for i in range(len(json_paths)-1):
            if int(json_paths[i][-11:-5]) != int(json_paths[i+1][-11:-5]) - (self.skip + 1):
                self.logger.error(
                    '{} \n is not the correct frame after \n {}'.format(
                                                    json_paths[i+1], json_paths[i])
                                                                                )
            if int(json_paths[i][-11:-5]) != int(img_paths[i][-10:-4]):
                self.logger.error(
                    '{} \n does not match \n {}'.format(
                                                    img_paths[i], json_paths[i])
                                                                                )

        all_p2d_heatmap = []
        for json_path in json_paths:
            data = io.read_json(json_path)

            p2d, p3d = self._process_points(data)
            p2d[:, 0] = p2d[:, 0]-180 # Translate p2d coordinates by 180 pixels to the left


            p2d_heatmap = generate_heatmap(p2d[1:, :], 3) # exclude head
            all_p2d_heatmap.append(p2d_heatmap)
            # get action name
            action = data['action']

        if self.transform:
            imgs = np.array(
                [self.transform({'image': img})['image'].numpy() for img in imgs])
            p3d = self.transform({'joints3D': p3d})['joints3D']
            p2d = np.array([self.transform({'joints2D': p2d})['joints2D'].numpy() for p2d in all_p2d_heatmap])

        return torch.tensor(imgs), torch.tensor(p2d), p3d, action

    def __len__(self):

        return len(self.index[self.ROOT_DIRS[0]])


class MocapSeqDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.train_dir = kwargs['dataset_tr']
        self.val_dir = kwargs['dataset_val']
        self.test_dir = kwargs['dataset_test']
        self.batch_size = kwargs['batch_size']
        self.num_workers = kwargs['num_workers']
        self.seq_len = kwargs['seq_len']
        self.skip = kwargs['skip']

        # Data: data transformation strategy
        self.data_transform = transforms.Compose(
            [trsf.ImageTrsf(), trsf.Joints3DTrsf(), trsf.ToTensor()]
        )
        
    def train_dataloader(self):
        data_train = MocapTransformer(
            self.train_dir,
            SetType.TRAIN,
            transform=self.data_transform,
            sequence_length = self.seq_len,
            skip = self.skip)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        data_val =  MocapTransformer(
            self.val_dir,
            SetType.VAL,
            transform=self.data_transform,
            sequence_length = self.seq_len,
            skip = self.skip)
        return DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        data_test =  MocapTransformer(
            self.test_dir,
            SetType.TEST,
            transform=self.data_transform,
            sequence_length = self.seq_len,
            skip = self.skip)
        return DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)


