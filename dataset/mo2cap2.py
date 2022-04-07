from ntpath import join
import os
import h5py
import torch
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

def generate_heatmap_distance(joints, heatmap_sigma):
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
        feat_stride = np.asarray([256, 256]) / np.asarray([heatmap_size[0], heatmap_size[1]])
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

def generate_heatmap(joints, heatmap_sigma):
    """
    :param joints:  [nof_joints, 2]
    :return: target, target_weight(1: visible, 0: invisible)
    """
    heatmap_size = [47, 47]
    num_joints = joints.shape[0]
    target = np.zeros((num_joints,
                       heatmap_size[0],
                       heatmap_size[1]),
                      dtype=np.float32)
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    tmp_size = heatmap_sigma * 3

    for joint_id in range(num_joints):
        feat_stride = np.asarray([256, 256]) / np.asarray([heatmap_size[0], heatmap_size[1]])
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

class Mo2Cap2(BaseDataset):
    """Mo2Cap2 Dataset loader"""

    ROOT_DIRS = ['rgba', 'json']
    MM_TO_M = 1000

    def __init__(self, *args, heatmap_type='baseline', **kwargs):
        """Init class, to allow variable sequence length, inherits from Base
        Keyword Arguments:
            sequence_length -- length of image sequence (default: {5})
        """

        self.heatmap_type = heatmap_type
        super().__init__(*args, **kwargs)

    def index_db(self):

        indexed_paths = dict()
        for sub_dir in self.ROOT_DIRS:
            indexed_paths.update({sub_dir : []})

        for dir in os.listdir(self.path):
            _, rgba_paths = io.get_files(os.path.join(self.path, dir, 'rgba'))
            _, json_paths = io.get_files(os.path.join(self.path, dir, 'json'))
            if len(rgba_paths) != len(json_paths):
                self.logger.error(
                    "Image and JSON file numbers are not matching for {}".format(dir)
                )
            encoded_rgba = [p.encode('utf8') for p in rgba_paths]
            encoded_json = [p.encode('utf8') for p in json_paths]
            indexed_paths.update({'rgba' : indexed_paths['rgba'] + encoded_rgba})
            indexed_paths.update({'json' : indexed_paths['json'] + encoded_json})

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

        p2d = np.zeros((16, 2))
        p3d = np.zeros((16, 3))

        joint_names = []
        for key in data.keys():
            if key not in ['action']:
                joint_names.append(key)

        for jid, joint_name in enumerate(joint_names):
            p2d[jid][0] = data[joint_name]['2d'][0]
            p2d[jid][1] = data[joint_name]['2d'][1]
            p3d[jid][0] = data[joint_name]['3d'][0]
            p3d[jid][1] = data[joint_name]['3d'][1]
            p3d[jid][2] = data[joint_name]['3d'][2]

        p3d[0, :] = p3d[1, :] # Set artifical head value to neck value
        p3d /= self.MM_TO_M

        return p2d, p3d

    def __getitem__(self, index):

        # load image
        img_path = self.index['rgba'][index].decode('utf8')
        img = sio.imread(img_path).astype(np.float32)
        img /= 255.0
        #img = img[:, 180:1120, :] # no-crop
        img = resize(img, (368, 368))

        # read joint positions
        json_path = self.index['json'][index].decode('utf8')
        data = io.read_json(json_path)
        p2d, p3d = self._process_points(data)
        #p3d = p3d[1:, :] # Remove artificial head
        #p2d = p2d[1:, :] # Remove artificial head
        p2d[:, 0] = p2d[:, 0]-33 # Translate p2d coordinates by 33 pixels to the left

        if self.heatmap_type == 'baseline':
            p2d_heatmap = generate_heatmap(p2d[1:, :], 3) # exclude head
        elif self.heatmap_type == 'distance':
            distances = np.sqrt(np.sum(p3d**2, axis=1))[1:]
            p2d_heatmap = generate_heatmap_distance(p2d[1:, :], distances) # exclude head
        else:
            self.logger.error('Unrecognized heatmap type')

        if 'action' not in data.keys():
            action = "unknown"
        else:
            action = data['action']

        if self.transform:
            img = self.transform({'image': img})['image']
            p3d = self.transform({'joints3D': p3d})['joints3D']
            p2d = self.transform({'joints2D': p2d})['joints2D']

        return img, p2d_heatmap, p3d, action

    def __len__(self):

        return len(self.index['rgba'])


class Mo2Cap2DataModule(pl.LightningDataModule):

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
        data_train = Mo2Cap2(self.train_dir, SetType.TRAIN, transform=self.data_transform, heatmap_type=self.heatmap_type)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        data_val = Mo2Cap2(self.val_dir, SetType.VAL, transform=self.data_transform, heatmap_type=self.heatmap_type)
        return DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        data_test = Mo2Cap2(self.test_dir, SetType.TEST, transform=self.data_transform, heatmap_type=self.heatmap_type)
        return DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)

