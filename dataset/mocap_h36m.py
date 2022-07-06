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

def generate_heatmap_distance(joints, heatmap_sigma, h, w):
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
    heatmap_sigma += 2
    sigma_size = heatmap_sigma * 3

    for joint_id in range(num_joints):
        tmp_size = sigma_size[joint_id]
        feat_stride = np.asarray([h, w]) / np.asarray([heatmap_size[0], heatmap_size[1]])
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
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * heatmap_sigma[joint_id] ** 2))

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

def generate_heatmap(joints, heatmap_sigma, resolution=[47, 47], h=940, w=800):
    """
    :param joints:  [nof_joints, 2]
    :return: target, target_weight(1: visible, 0: invisible)
    """
    heatmap_size = resolution
    num_joints = joints.shape[0]
    target = np.zeros((num_joints,
                       heatmap_size[0],
                       heatmap_size[1]),
                      dtype=np.float32)
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    tmp_size = heatmap_sigma * 3 

    for joint_id in range(num_joints):
        feat_stride = np.asarray([h, w]) / np.asarray([heatmap_size[0], heatmap_size[1]])
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


class MocapH36M(BaseDataset):
    """Mocap Dataset loader"""

    ROOT_DIRS = ['rgba', 'json']
    MM_TO_M = 1000

    subject_sets = {
        'p2_train': ['S1', 'S5', 'S6', 'S7', 'S9'],
        'p2_test' : ['S11'],
        'p1_train' : ['S1', 'S5', 'S6', 'S7'],
        'p1_test' : ['S9', 'S11'],
        'val' : ['S8'],
    }

    def __init__(self, *args, heatmap_type='baseline', heatmap_resolution=[47, 47], image_resolution=[368, 368], protocol = 'p1_train', **kwargs):
        """Init class, to allow variable sequence length, inherits from Base
        Keyword Arguments:
            sequence_length -- length of image sequence (default: {5})
        """

        self.heatmap_type = heatmap_type
        self.heatmap_resolution = heatmap_resolution
        self.image_resolution = image_resolution
        self.protocol = protocol
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
                        encoded = [p.encode('utf8') for p in paths]
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

        #p3d[0, :] = p3d[1, :] # Set artifical head value to neck value
        p3d /= self.MM_TO_M

        return p2d, p3d

    def __getitem__(self, index):

        # load image

        img_path = self.index['rgba'][index].decode('utf8')
        img = sio.imread(img_path).astype(np.float32)
        img /= 255.0
        h, w, c = img.shape
        img = resize(img, (self.image_resolution[0], self.image_resolution[1]))

        # read joint positions
        json_path = self.index['json'][index].decode('utf8')

        data = io.read_json(json_path)

        p2d, p3d = self._process_points(data)
        
        if self.heatmap_type == 'baseline':
            p2d_heatmap = generate_heatmap(p2d, int(1*self.heatmap_resolution[0]/47.), resolution=self.heatmap_resolution, h=h, w=w) # exclude head
        elif self.heatmap_type == 'distance':
            distances = np.sqrt(np.sum(p3d**2, axis=1))
            p2d_heatmap = generate_heatmap_distance(p2d, distances, h, w) # exclude head
        else:
            self.logger.error('Unrecognized heatmap type')
        
        # get action name
        action = data['action']

        if self.transform:
            img = self.transform({'image': img})['image']
            p3d = self.transform({'joints3D': p3d})['joints3D']
            p2d = self.transform({'joints2D': p2d})['joints2D']

        return img, p2d_heatmap, p3d, action

    def __len__(self):

        return len(self.index[self.ROOT_DIRS[0]])

class MocapH36MDataModule(pl.LightningDataModule):

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

        self.p_train = f'{self.protocol}_train'
        self.p_test = f'{self.protocol}_test'

        # Data: data transformation strategy
        self.data_transform = transforms.Compose(
            [trsf.ImageTrsf(), trsf.ToTensor()]
        )
        
    def train_dataloader(self):
        data_train = MocapH36M(self.train_dir, SetType.TRAIN, transform=self.data_transform, heatmap_type=self.heatmap_type, heatmap_resolution=self.heatmap_resolution, image_resolution=self.image_resolution, protocol=self.p_train)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        data_val = MocapH36M(self.val_dir, SetType.VAL, transform=self.data_transform, heatmap_type=self.heatmap_type, heatmap_resolution=self.heatmap_resolution, image_resolution=self.image_resolution, protocol='val')
        return DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        data_test = MocapH36M(self.test_dir, SetType.TEST, transform=self.data_transform, heatmap_type=self.heatmap_type, heatmap_resolution=self.heatmap_resolution, image_resolution=self.image_resolution, protocol=self.p_test)
        return DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)
