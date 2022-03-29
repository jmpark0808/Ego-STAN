import os
import h5py
import torch
import pytorch_lightning as pl
from skimage import io as sio
from skimage.transform import resize
import numpy as np
from dataset.mocap import generate_heatmap
from base import BaseDataset
from utils import io, config
from base import SetType
import dataset.transform as trsf
from torch.utils.data import DataLoader
from torchvision import transforms

class Mo2Cap2(BaseDataset):
    """Mocap Dataset loader"""

    def __init__(self, *args, heatmap_type='baseline', **kwargs):
        """Init class, to allow variable sequence length, inherits from Base
        Keyword Arguments:
            sequence_length -- length of image sequence (default: {5})
        """

        self.heatmap_type = heatmap_type
        super().__init__(*args, **kwargs)

    def index_db(self):

        frame_tracks = [] #these are not file paths, but the path to the chunk and index

        for file in os.listdir(self.path):
            with h5py.File(os.path.join(self.path, file), "r") as chunk:
                if len(chunk["Images"]) != len(chunk["Heatmaps"]):
                    self.logger.error("Mismatch in Image-Label Size in Chunk {}".format(file))
                for i in range(len(chunk["Images"])):
                    frame_tracks.append("{0}-frame_{1:05}".format(file, i).encode('utf8'))

        return {'tracks' : frame_tracks}

    def __getitem__(self, index):

        # load image

        img_track = self.index['tracks'][index].decode('utf8')
        chunk_path = img_track[:-12]
        frame_num = img_track[-5:]

        chunk = h5py.File(os.path.join(self.path, chunk_path), "r")
        img = chunk["Images"][int(frame_num)]
        img = resize(img, (3, 368, 368))
        img = torch.Tensor(img).type(torch.FloatTensor)
        p2d = chunk["Annot2D"][int(frame_num)]
        p2d[:, 0] = p2d[:, 0] # Translate p2d coordinates by 180 pixels to the left
        p2d_heatmap = generate_heatmap(p2d, 3) # no head in mocap dataset
        p2d_heatmap = torch.Tensor(p2d_heatmap).type(torch.FloatTensor)
        p3d = chunk["Annot3D"][int(frame_num)]
        p3d = np.insert(p3d, 0, np.array([0.0,0.0,0.0]), 0)
        p3d = torch.Tensor(p3d).type(torch.FloatTensor)
        action = "unknown" #placeholder for now
        chunk.close()

        return img, p2d_heatmap, p3d, action

    def __len__(self):

        return len(self.index['tracks'])


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

