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
from dataset.mo2cap2 import generate_heatmap, generate_heatmap_distance

class Mo2Cap2Transformer(BaseDataset):
    """Mo2Cap2 Sequential Dataset loader"""

    ROOT_DIRS = ['rgba', 'json']
    MM_TO_M = 1000

    def __init__(self, *args, sequence_length=5, skip =0, heatmap_type='baseline', **kwargs):
        """Init class, to allow variable sequence length, inherits from Base
        Keyword Arguments:
            sequence_length -- length of image sequence (default: {5})
        """
        
        self.sequence_length = sequence_length
        self.skip = skip
        self.heatmap_type = heatmap_type
        super().__init__(*args, **kwargs)

    def index_db(self):
        
        indexed_paths = dict()
        for sub_dir in self.ROOT_DIRS:
            indexed_paths.update({sub_dir : []})

        encoded_json = []
        encoded_rgba = []
        len_seq = self.sequence_length
        m = self.skip 

        for dir in os.listdir(self.path):
            _, rgba_paths = io.get_files(os.path.join(self.path, dir, 'rgba'))
            _, json_paths = io.get_files(os.path.join(self.path, dir, 'json'))
            if len(rgba_paths) != len(json_paths):
                self.logger.error(
                    "Image and JSON file numbers are not matching for {}".format(dir)
                )

            # Handling the cases for sets 

            for json_path in json_paths:
                encoded_rgba_sequence = []
                encoded_json_sequence = []
                if 'olek_outdoor' in dir: #Exception to the rules we have
                    frame_idx = json_path.split('-')[-1].split('.json')[0]
                    rgba_path = json_path[0:-5].replace('json','rgba') + '.jpg'
                else:
                    frame_idx = json_path.split('_')[-1].split('.json')[0]
                    rgba_path = json_path[0:-5].replace('json','rgba') + '.png'
                if len(frame_idx) == 6:
                    for i in range(len_seq):
                        frame_idx = int(frame_idx)
                        if (
                            os.path.exists(rgba_path[0:-10] + "{0:06}.png".format(frame_idx+i+i*m)) and
                            os.path.exists(json_path[0:-11] + "{0:06}.json".format(frame_idx+i+i*m))
                            ):
                            rgba_frame_path = rgba_path[0:-10] + "{0:06}.png".format(frame_idx+i+i*m)
                            json_frame_path = json_path[0:-11] + "{0:06}.json".format(frame_idx+i+i*m)
                            encoded_rgba_sequence.append(rgba_frame_path.encode('utf8'))
                            encoded_json_sequence.append(json_frame_path.encode('utf8'))   
                elif len(frame_idx) == 4:
                    for i in range(len_seq):
                        frame_idx = int(frame_idx)
                        if (
                            (os.path.exists(rgba_path[0:-8] + "{0:04}.png".format(frame_idx+i+i*m)) or
                             os.path.exists(rgba_path[0:-8] + "{0:04}.jpg".format(frame_idx+i+i*m))) and
                             os.path.exists(json_path[0:-9] + "{0:04}.json".format(frame_idx+i+i*m))
                            ):
                            if 'olek_outdoor' in rgba_path:
                                rgba_frame_path = rgba_path[0:-8] + "{0:04}.jpg".format(frame_idx+i+i*m)
                            else:
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
                
        indexed_paths.update({'rgba' : encoded_rgba})
        indexed_paths.update({'json' : encoded_json})

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

        p2d = np.zeros((15, 2))
        p3d = np.zeros((15, 3))

        joint_names = []
        for key in data.keys():
            if key not in ['action', 'Head']: # keys to skip from json
                joint_names.append(key)

        for jid, joint_name in enumerate(joint_names):
            p2d[jid][0] = data[joint_name]['2d'][0]
            p2d[jid][1] = data[joint_name]['2d'][1]
            p3d[jid][0] = data[joint_name]['3d'][0]
            p3d[jid][1] = data[joint_name]['3d'][1]
            p3d[jid][2] = data[joint_name]['3d'][2]

        #p3d[0, :] = p3d[1, :] # Set artifical head value to neck value
        p3d /= self.MM_TO_M

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
        #imgs = [img[:, 180:1120, :] for img in imgs] #no-crop
        imgs = np.array([resize(img, (368, 368)) for img in imgs])

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
        for json_path in json_paths:
            data = io.read_json(json_path)

            p2d, p3d = self._process_points(data)
            p2d[:, 0] = p2d[:, 0]-33 # Translate p2d coordinates by 33 pixels to the left

            if self.heatmap_type == 'baseline':
                p2d_heatmap = generate_heatmap(p2d, 1.3) # exclude head
            elif self.heatmap_type == 'distance':
                distances = np.sqrt(np.sum(p3d**2, axis=1))
                p2d_heatmap = generate_heatmap_distance(p2d, distances) # exclude head
            else:
                self.logger.error('Unrecognized heatmap type')

            all_p2d_heatmap.append(p2d_heatmap)
            all_p3d.append(p3d)

            # get action name
            if 'action' not in data.keys():
                action = "unknown"
            else:
                action = data['action']

        if self.transform:
            imgs = np.array(
                [self.transform({'image': img})['image'].numpy() for img in imgs])
            p3d = np.array([self.transform({'joints3D': p3d})['joints3D'].numpy() for p3d in all_p3d])
            p2d = np.array([self.transform({'joints2D': p2d})['joints2D'].numpy() for p2d in all_p2d_heatmap])

        return torch.tensor(imgs), torch.tensor(p2d), torch.tensor(p3d), action

    def __len__(self):

        return len(self.index['rgba'])


class Mo2Cap2SeqDataModule(pl.LightningDataModule):

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


        # Data: data transformation strategy
        self.data_transform = transforms.Compose(
            [trsf.ImageTrsf(), trsf.Joints3DTrsf(jid_to_zero = 0), trsf.ToTensor()]
        )
        
    def train_dataloader(self):
        data_train = Mo2Cap2Transformer(
            self.train_dir,
            SetType.TRAIN,
            transform=self.data_transform,
            sequence_length = self.seq_len,
            skip = self.skip,
            heatmap_type=self.heatmap_type)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        data_val =  Mo2Cap2Transformer(
            self.val_dir,
            SetType.VAL,
            transform=self.data_transform,
            sequence_length = self.seq_len,
            skip = self.skip,
            heatmap_type=self.heatmap_type)
        return DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        data_test =  Mo2Cap2Transformer(
            self.test_dir,
            SetType.TEST,
            transform=self.data_transform,
            sequence_length = self.seq_len,
            skip = self.skip,
            heatmap_type=self.heatmap_type)
        return DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)