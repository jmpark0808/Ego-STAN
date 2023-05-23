# Code based off of that of Dennis Tome's xREgoPose repo but with many changes to accomodate human3.6m

import os
import pytorch_lightning as pl
from skimage import io as sio
from skimage.transform import resize
import numpy as np
from base import BaseDataset
from utils import io, config
from base import SetType
import dataset.transform as trsf
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import copy 
from dataset.common import *
from PIL import Image


class MocapH36MCrop(BaseDataset):
    """Mocap Dataset loader"""

    ROOT_DIRS = ['rgba', 'json']
    MM_TO_M = 1000

    subject_sets = {
        'p2_train': ['S1', 'S5', 'S6', 'S7', 'S8', 'S9'],
        'p2_test' : ['S11'],
        'p1_train' : ['S1', 'S5', 'S6', 'S7', 'S8'],
        'p1_test' : ['S9', 'S11'],
        'val' : ['S8'],
    }
    # subject_sets = {
    #     'p2_train': ['S1', 'S5', 'S6', 'S7', 'S8', 'S9'],
    #     'p2_test' : ['S11'],
    #     'p1_train' : ['S5'],
    #     'p1_test' : ['S5'],
    #     'val' : ['S5'],
    # }

    def __init__(self, *args, heatmap_type='baseline', heatmap_resolution=[47, 47], image_resolution=[368, 368], sigma = 3, 
                                protocol = 'p1_train', w2c=True, sr=1, **kwargs):
        """Init class, to allow variable sequence length, inherits from Base
        Keyword Arguments:
            sequence_length -- length of image sequence (default: {5})
        """

        self.heatmap_type = heatmap_type
        self.heatmap_resolution = heatmap_resolution
        self.image_resolution = image_resolution
        self.protocol = protocol
        self._cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
        self.w2c = w2c
        self.sigma = sigma
        self.sr = sr
        self.aug_transform = trsf.Compose([
            trsf.RandomAffineRotation(degrees=25, shear=10, translate=0.05, scale=1),
            trsf._ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            trsf.GaussianBlur(high=0.1),
            ])
        super().__init__(*args, **kwargs)

    def _load_index(self):
        """Overloading this to allow handling for more 
        protocols. Get indexed set. If the set has already been
        indexed, load the file, otherwise index it and save cache.
        Returns:
            dict -- index set
        """

        if self.protocol == 'val':
            idx_path = os.path.join(self.path, 'index_val.h5')
        elif self.protocol.lower() == 'p1_test' or self.protocol.lower() == 'p2_test':
            idx_path = os.path.join(self.path, 'index_test.h5')
        else:
            idx_path = os.path.join(self.path, 'index_train.h5')

        if io.exists(idx_path):
            return io.read_h5(idx_path)

        index = self.index_db()
        io.write_h5(idx_path, index)
        return index

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

                if path.split(os.path.sep)[-3] in self.subject_sets[self.protocol]:
                    if self.protocol.split('_')[-1] in ['train', 'val'] :
                        # encoded = [p.encode('utf8') for p in paths]
                        # indexed_paths.update({sub_dir: encoded})
                        encoded = []
                        for p in paths:
                            frame_idx = p.split('_')[-1].split('.')[0]
                            if int(frame_idx)%self.sr == 0:
                                encoded.append(p.encode('utf8'))
                        indexed_paths.update({sub_dir: encoded})
                    elif self.protocol.split('_')[-1] in ['test']:
                        encoded = []
                        for p in paths:
                            frame_idx = p.split('_')[-1].split('.')[0]
                            if int(frame_idx)%64 == 0:
                                encoded.append(p.encode('utf8'))
                        indexed_paths.update({sub_dir: encoded})
                    else:
                        self.logger.error('Un-recognized protocol type')
                else:
                    indexed_paths.update({'rgba':[]})
                    indexed_paths.update({'json':[]})

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

        p2d = np.zeros((17, 2))
        p3d = np.zeros((17, 3))

        joint_names = []
        for key in data['joints'].keys():
            joint_names.append(key)
        
        for jid, joint_name in enumerate(joint_names):
            p2d[jid][0] = data['joints'][joint_name]['2d'][0]
            p2d[jid][1] = data['joints'][joint_name]['2d'][1]
            p3d[jid][0] = data['joints'][joint_name]['3d'][0]
            p3d[jid][1] = data['joints'][joint_name]['3d'][1]
            p3d[jid][2] = data['joints'][joint_name]['3d'][2]

        

        # World to camera
        if self.w2c:
            p3d = np.expand_dims(p3d, 0)
            subject = data['subject']
            camera = data['camera']
            orientation = np.array(self._cameras[f'S{subject}'][camera]['orientation'])
            translation = np.array(self._cameras[f'S{subject}'][camera]['translation'])/1000.
            p3d = world_to_camera(p3d, orientation, translation)
            p3d = np.squeeze(p3d)
        # else:
        #     p3d /= self.MM_TO_M

        # Normalize
        # p3d[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16], :] -= p3d[14, :]

        #p3d[0, :] = p3d[1, :] # Set artifical head value to neck value
        

        return p2d, p3d

    def __getitem__(self, index):

        # load image

        img_path = self.index['rgba'][index].decode('utf8')
        img = sio.imread(img_path).astype(np.float32)
        img /= 255.0
        h, w, c = img.shape
        orig_img = img
        img = resize(img, (self.image_resolution[0], self.image_resolution[1]))

        # read joint positions
        json_path = self.index['json'][index].decode('utf8')

        data = io.read_json(json_path)

        p2d, p3d = self._process_points(data)
        Z_c = p3d[:, 2:3]
        # cropping

        coords = p2d

        max_x = max(coords[:, 0])
        min_x = min(coords[:, 0])
        max_y = max(coords[:, 1])
        min_y = min(coords[:, 1])

        x_diff = max_x - min_x
        y_diff = max_y - min_y

        max_xc = int(min([w, max_x + 0.25*x_diff]))
        min_xc = int(max([0, min_x - 0.25*x_diff]))
        max_yc = int(min([h, max_y + 0.25*y_diff]))
        min_yc = int(max([0, min_y - 0.25*x_diff]))

        a_diff = ((max_xc - min_xc) - (max_yc - min_yc))/2

        # to maintain aspect ratio

        if a_diff <= 0:

            # the height (y) is more than the width (x)

            max_xn = max_xc + int(abs(a_diff)) 
            min_xn = min_xc - int(abs(a_diff))
            max_yn = max_yc
            min_yn = min_yc

            d_max_xn = w - max_xn
            d_min_xn = min_xn - 0

            if d_max_xn < 0:
                max_xn = w
                min_xn = int(min_xn + d_max_xn)

            if d_min_xn < 0:
                min_xn = 0
                max_xn = int(max_xn - d_min_xn)
        else:

            # the width (x) is more than the height (y)

            max_yn = max_yc + int(a_diff) 
            min_yn = min_yc - int(a_diff)
            max_xn = max_xc
            min_xn = min_xc

            d_max_yn = h - max_yn
            d_min_yn = min_yn - 0

            if d_max_yn < 0:
                max_yn = h
                min_yn = int(min_yn + d_max_yn)

            if d_min_yn < 0:
                min_yn = 0
                max_yn = int(max_yn - d_min_yn)

        maintain_aspect = True
        if maintain_aspect == True:
            cropped_img = orig_img[min_yn:max_yn, min_xn:max_xn]
            p2d_crop = np.array([(coord[0] - min_xn, coord[1] - min_yn) for coord in coords])
        else:
            cropped_img = orig_img[min_yc:max_yc, min_xc:max_xc]
            p2d_crop = np.array([(coord[0] - min_xc, coord[1] - min_yc) for coord in coords])

        w_crop, h_crop, c = cropped_img.shape

        cropped_img = resize(cropped_img, (self.image_resolution[0], self.image_resolution[1]))
        img_for_aug = Image.fromarray((255*cropped_img).astype(np.uint8))

        intrinsic_matrix = np.zeros((3,3))
        intrinsic_matrix[0, 0] = camera2int[data['camera']]['focal_length'][0]
        intrinsic_matrix[1, 1] = camera2int[data['camera']]['focal_length'][1]
        intrinsic_matrix[0, 2] = camera2int[data['camera']]['center'][0]
        intrinsic_matrix[1, 2] = camera2int[data['camera']]['center'][1]
        intrinsic_matrix[2, 2] = 1

        aug_img, aug_data = self.aug_transform(img_for_aug, keypoint2d=p2d_crop, intrinsic_matrix=intrinsic_matrix)
        p2d_crop = aug_data['keypoint2d']
        intrinsic_matrix_aug = aug_data['intrinsic_matrix']
        #aug_param = aug_data['aug_param']
        aug_img = np.array(aug_img)/255.0

        uv1 = np.concatenate([np.copy(p2d_crop), np.ones((17, 1))], axis=1)
        uv1 = uv1 * Z_c

        p3d = np.matmul(np.linalg.inv(intrinsic_matrix_aug), uv1.T).T
        p3d  = (p3d - p3d[14:15, :])/1000.
        # p3d[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16], :] -= p3d[14, :]

        # p2d_heatmap = np.squeeze(normalize_screen_coordinates(np.expand_dims(p2d, 0), w=w, h=h))

        if self.heatmap_type == 'baseline':
            p2d_heatmap = generate_heatmap(p2d, self.sigma, resolution=self.heatmap_resolution, h=h, w=w) # exclude head
            p2d_hm_crop = generate_heatmap(p2d_crop, self.sigma, 
                            resolution=self.heatmap_resolution, h=h_crop, w=w_crop)
        elif self.heatmap_type == 'distance':
            distances = np.sqrt(np.sum(p3d**2, axis=1))
            p2d_heatmap = generate_heatmap_distance(p2d, distances, h, w) # exclude head
            p2d_hm_crop = generate_heatmap_distance(p2d_crop, distances, h_crop, w_crop) # exclude head
        else:
            self.logger.error('Unrecognized heatmap type')

        # get action name
        action = data['action']
        if self.transform:
            random_dice = np.random.uniform(0, 1, [1])
            img = self.transform({'image': img, 'random_dice': random_dice})['image']
            cropped_img = self.transform({'image': cropped_img, 'random_dice': random_dice})['image']
            aug_img = self.transform({'image': aug_img, 'random_dice': random_dice})['image']
            p3d = self.transform({'joints3D': p3d, 'random_dice': random_dice})['joints3D']
            p2d_heatmap = self.transform({'joints2D_heatmap': p2d_heatmap, 'random_dice': random_dice})['joints2D_heatmap']
            p2d_heatmap_crop = self.transform({'joints2D_heatmap': p2d_hm_crop, 'random_dice': random_dice})['joints2D_heatmap']

        return aug_img, p2d_heatmap_crop, p3d, action

    def __len__(self):

        return len(self.index[self.ROOT_DIRS[0]])

class MocapH36MCropDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.train_dir = kwargs.get('dataset_tr')
        self.val_dir = kwargs.get('dataset_val')
        self.test_dir = kwargs.get('dataset_test')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers', 0)
        self.heatmap_type = kwargs.get('heatmap_type')
        self.heatmap_resolution = kwargs.get('heatmap_resolution')
        self.image_resolution = kwargs.get('image_resolution')
        self.protocol = kwargs.get('protocol')
        self.w2c = kwargs.get('w2c')
        self.sigma = kwargs.get('sigma')
        self.sr = kwargs.get('h36m_sample_rate')
        self.p_train = f'{self.protocol}_train'
        self.p_test = f'{self.protocol}_test'

        # Data: data transformation strategy
        self.data_transform_train = transforms.Compose(
            [trsf.ImageTrsf(), trsf.ToTensor()]
        )
        self.data_transform_test = transforms.Compose(
            [trsf.ImageTrsf(), trsf.ToTensor()]
        )
        
    def train_dataloader(self):
        data_train = MocapH36MCrop(self.train_dir, SetType.TRAIN, transform=self.data_transform_train,
         heatmap_type=self.heatmap_type, heatmap_resolution=self.heatmap_resolution,
          image_resolution=self.image_resolution, sigma=self.sigma, protocol=self.p_train, w2c=self.w2c, sr=self.sr)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        data_val = MocapH36MCrop(self.val_dir, SetType.VAL, transform=self.data_transform_test,
         heatmap_type=self.heatmap_type, heatmap_resolution=self.heatmap_resolution,
          image_resolution=self.image_resolution, sigma=self.sigma, protocol=self.p_test, w2c=self.w2c, sr=self.sr)
        return DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        data_test = MocapH36MCrop(self.test_dir, SetType.TEST, transform=self.data_transform_test,
         heatmap_type=self.heatmap_type, heatmap_resolution=self.heatmap_resolution,
          image_resolution=self.image_resolution, sigma=self.sigma, protocol=self.p_test, w2c=self.w2c, sr=self.sr)
        return DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)
