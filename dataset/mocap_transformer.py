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
from dataset.mocap import generate_heatmap, generate_heatmap_distance
from torch.utils.data import DataLoader
from torchvision import transforms

class MocapTransformer(BaseDataset):
    """Mocap Dataset loader"""

    ROOT_DIRS = ['rgba', 'json']
    CM_TO_M = 100

    def __init__(self, *args, sequence_length=5, skip =0, image_resolution=[368, 368], heatmap_resolution=[47, 47], heatmap_type='baseline', **kwargs):
        """Init class, to allow variable sequence length, inherits from Base
        Keyword Arguments:
            sequence_length -- length of image sequence (default: {5})
        """

        self.sequence_length = sequence_length
        self.skip = skip
        self.heatmap_type = heatmap_type
        self.heatmap_resolution = heatmap_resolution
        self.image_resolution = image_resolution

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

            rgba_d_path = os.path.join(path, self.ROOT_DIRS[0]) #rgba
            json_d_path = os.path.join(path, self.ROOT_DIRS[1]) #json
            _, rgba_paths = io.get_files(rgba_d_path)
            __, json_paths = io.get_files(json_d_path)

            if(len(rgba_paths) != len(json_paths)):
                self.logger.error("Json and Rgba Directories do not have equal sizes")

            # get files from subdirs
            n_frames = -1

            if n_frames < 0:
                n_frames = len(json_paths)
            else:
                if len(json_paths) != n_frames:
                    self.logger.error(
                        'Frames info in {} not matching other passes'.format(json_d_path))
                
            # get the data in sequences -> checks for missing frames
            encoded_json = []
            encoded_rgba = []
            len_seq = self.sequence_length
            m = self.skip 

            for json_path in json_paths:
                encoded_rgba_sequence = []
                encoded_json_sequence = []
                frame_idx = json_path.split('_')[-1].split('.json')[0]
                if len(frame_idx) == 6:
                    frame_idx = int(frame_idx)
                    rgba_path = json_path[0:-12].replace('json','rgba') + '.rgba.{0:06}.png'.format(frame_idx)
                    for i in range(len_seq):
                        if (
                            os.path.exists(rgba_path[0:-10] + "{0:06}.png".format(frame_idx+i+i*m)) and
                            os.path.exists(json_path[0:-11] + "{0:06}.json".format(frame_idx+i+i*m))
                            ):
                            rgba_frame_path = rgba_path[0:-10] + "{0:06}.png".format(frame_idx+i+i*m)
                            json_frame_path = json_path[0:-11] + "{0:06}.json".format(frame_idx+i+i*m)
                            encoded_rgba_sequence.append(rgba_frame_path.encode('utf8'))
                            encoded_json_sequence.append(json_frame_path.encode('utf8'))   
                elif len(frame_idx) == 4:
                    frame_idx = int(frame_idx)
                    rgba_path = json_path[0:-10].replace('json','rgba') + '.rgba.{0:04}.png'.format(frame_idx)
                    for i in range(len_seq):
                        if (
                            os.path.exists(rgba_path[0:-8] + "{0:04}.png".format(frame_idx+i+i*m)) and
                            os.path.exists(json_path[0:-9] + "{0:04}.json".format(frame_idx+i+i*m))
                            ):
                            rgba_frame_path = rgba_path[0:-8] + "{0:04}.png".format(frame_idx+i+i*m)
                            json_frame_path = json_path[0:-9] + "{0:04}.json".format(frame_idx+i+i*m)
                            encoded_rgba_sequence.append(rgba_frame_path.encode('utf8'))
                            encoded_json_sequence.append(json_frame_path.encode('utf8')) 
                else:
                    self.logger.error('Frame idx length is other than 6 or 4')

                if(
                    len(encoded_json_sequence) == len_seq and
                    len(encoded_rgba_sequence) == len_seq
                ):
                    encoded_json.append(encoded_json_sequence)
                    encoded_rgba.append(encoded_rgba_sequence)

            indexed_paths.update({'rgba': encoded_rgba})
            indexed_paths.update({'json': encoded_json})

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
            try:
                if int(img_paths[i][-10:-4]) != int(img_paths[i+1][-10:-4]) - (self.skip + 1):
                    self.logger.error(
                        '{} \n is not the correct frame after \n {}'.format(
                                                        img_paths[i+1], img_paths[i])
                                                                                )
            except:
                if int(img_paths[i][-8:-4]) != int(img_paths[i+1][-8:-4]) - (self.skip + 1):
                    self.logger.error(
                        '{} \n is not the correct frame after \n {}'.format(
                                                        img_paths[i+1], img_paths[i])
                                                                                )

        imgs = [sio.imread(img_path).astype(np.float32) for img_path in img_paths]
        imgs = [img / 255.0 for img in imgs]
        imgs = [img[:, 180:1120, :] for img in imgs]
        imgs = np.array([resize(img, (self.image_resolution[0], self.image_resolution[1])) for img in imgs])

        # read joint positions
        json_paths = [path.decode('utf8') for path in self.index['json'][index]]
       
        # checking if json path corresponds to the path of the last rgba frame in the sequence
        # checking for correct sequence of rgba/image files
        for i in range(len(json_paths)-1):
            try:
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
            except:
                if int(json_paths[i][-9:-5]) != int(json_paths[i+1][-9:-5]) - (self.skip + 1):
                    self.logger.error(
                        '{} \n is not the correct frame after \n {}'.format(
                                                        json_paths[i+1], json_paths[i])
                                                                                    )
                if int(json_paths[i][-9:-5]) != int(img_paths[i][-8:-4]):
                    self.logger.error(
                        '{} \n does not match \n {}'.format(
                                                        img_paths[i], json_paths[i])
                                                                                    )

        all_p2d_heatmap = []
        all_p3d = []
        all_raw_p2d = []
        for json_path in json_paths:
            data = io.read_json(json_path)

            p2d, p3d = self._process_points(data)
            p2d[:, 0] = p2d[:, 0]-180 # Translate p2d coordinates by 180 pixels to the left
            all_raw_p2d.append(p2d)
            if self.heatmap_type == 'baseline':
                p2d_heatmap = generate_heatmap(p2d, 3, self.heatmap_resolution) # exclude head
            elif self.heatmap_type == 'distance':
                distances = np.sqrt(np.sum(p3d**2, axis=1))
                p2d_heatmap = generate_heatmap_distance(p2d, distances) # exclude head
            else:
                self.logger.error('Unrecognized heatmap type')

            all_p2d_heatmap.append(p2d_heatmap)
            all_p3d.append(p3d)
            # get action name
            action = data['action']

        if self.transform:
            imgs = np.array(
                [self.transform({'image': img})['image'].numpy() for img in imgs])
            p3d = np.array([self.transform({'joints3D': p3d})['joints3D'].numpy() for p3d in all_p3d])
            p2d = np.array([self.transform({'joints2D': p2d})['joints2D'].numpy() for p2d in all_p2d_heatmap])

        return torch.tensor(imgs), torch.tensor(p2d), torch.tensor(p3d), action, img_paths[-1]#, all_raw_p2d

    def __len__(self):

        return len(self.index[self.ROOT_DIRS[0]])


class MocapSeqDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.train_dir = kwargs.get('dataset_tr')
        self.val_dir = kwargs.get('dataset_val')
        self.test_dir = kwargs.get('dataset_test')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers', 0)
        self.heatmap_type = kwargs.get('heatmap_type')
        self.seq_len = kwargs.get('seq_len')
        self.skip = kwargs.get('skip')
        self.heatmap_resolution = kwargs.get('heatmap_resolution')
        self.image_resolution = kwargs.get('image_resolution')


        # Data: data transformation strategy
        self.data_transform = transforms.Compose(
            [trsf.ImageTrsf(), trsf.Joints3DTrsf(), trsf.ToTensor()]
        )
        
    def train_dataloader(self):
        data_train = MocapTransformer(
            self.train_dir,
            SetType.TRAIN,
            transform=self.data_transform,
            image_resolution=self.image_resolution,
            heatmap_resolution=self.heatmap_resolution,
            sequence_length = self.seq_len,
            skip = self.skip,
            heatmap_type=self.heatmap_type)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=False)

    def val_dataloader(self):
        data_val =  MocapTransformer(
            self.val_dir,
            SetType.VAL,
            transform=self.data_transform,
            sequence_length = self.seq_len,
            image_resolution=self.image_resolution,
            heatmap_resolution=self.heatmap_resolution,
            skip = self.skip,
            heatmap_type=self.heatmap_type)
        return DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)

    def test_dataloader(self):
        data_test =  MocapTransformer(
            self.test_dir,
            SetType.TEST,
            transform=self.data_transform,
            sequence_length = self.seq_len,
            image_resolution=self.image_resolution,
            heatmap_resolution=self.heatmap_resolution,
            skip = self.skip,
            heatmap_type=self.heatmap_type)
        return DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)


