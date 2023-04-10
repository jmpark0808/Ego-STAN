# Code based off of that of Dennis Tome's xREgoPose repo but with many changes to accomodate human3.6m

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
from dataset.mocap_h36m import generate_heatmap, generate_heatmap_distance, world_to_camera, h36m_cameras_extrinsic_params, h36m_cameras_intrinsic_params, normalize_screen_coordinates
from torch.utils.data import DataLoader
from torchvision import transforms
import copy 
from dataset.common import *


class MocapH36MCropTransformer(BaseDataset):
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
    #     'p1_train': ['S5'],
    #     'p1_test' : ['S5'],
    #     'p2_train' : ['S1', 'S5', 'S6', 'S7'],
    #     'p2_test' : ['S9', 'S11'],
    #     'val' : ['S5'],
    # }

    def __init__(self, *args, sequence_length=5, skip =0, heatmap_type='baseline', heatmap_resolution=[47, 47], 
     image_resolution=[368, 368], sigma = 3, protocol = 'p1_train', w2c=True, sr=1, **kwargs):
        """Init class, to allow variable sequence length, inherits from Base
        Keyword Arguments:
            sequence_length -- length of image sequence (default: {5})
            protocol -- one of protocol 1, 2, or neither (training is default)
        """

        self.sequence_length = sequence_length
        self.skip = skip
        self.heatmap_type = heatmap_type
        self.heatmap_resolution = heatmap_resolution
        self.image_resolution = image_resolution
        self.protocol = protocol
        self.w2c = w2c
        self.sigma = sigma
        self.sr = sr
        self._cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
 
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

            if path.split(os.path.sep)[-3] not in self.subject_sets[self.protocol]:
                indexed_paths.update({'rgba':[]})
                indexed_paths.update({'json':[]})
                return indexed_paths

            for json_path in json_paths:
                encoded_rgba_sequence = []
                encoded_json_sequence = []
                frame_idx = json_path.split('_')[-1].split('.json')[0]
                if len(frame_idx) == 6:
                    frame_idx = int(frame_idx)
                    rgba_path = json_path[0:-16].replace('json','rgba') + 'img_{0:06}.jpg'.format(frame_idx)
                    for i in range(len_seq):
                        if (
                            os.path.exists(rgba_path[0:-10] + "{0:06}.jpg".format(frame_idx+i+i*m)) and
                            os.path.exists(json_path[0:-11] + "{0:06}.json".format(frame_idx+i+i*m))
                            ):
                            rgba_frame_path = rgba_path[0:-10] + "{0:06}.jpg".format(frame_idx+i+i*m)
                            json_frame_path = json_path[0:-11] + "{0:06}.json".format(frame_idx+i+i*m)
                            encoded_rgba_sequence.append(rgba_frame_path.encode('utf8'))
                            encoded_json_sequence.append(json_frame_path.encode('utf8'))   
                else:
                    self.logger.error('Frame idx length is other than 6')

                if(
                    len(encoded_json_sequence) == len_seq and
                    len(encoded_rgba_sequence) == len_seq
                ):
                    if self.protocol.split('_')[-1] in ['train', 'val'] :
                        last_frame = encoded_json_sequence[-1]
                        last_frame_idx = last_frame.decode('utf8').split('_')[-1].split('.json')[0]
                        if int(last_frame_idx)%self.sr == 0:
                            encoded_json.append(encoded_json_sequence)
                            encoded_rgba.append(encoded_rgba_sequence)
                        # encoded_json.append(encoded_json_sequence)
                        # encoded_rgba.append(encoded_rgba_sequence)
                    elif self.protocol.split('_')[-1] in ['test']:
                        last_frame = encoded_json_sequence[-1]
                        last_frame_idx = last_frame.decode('utf8').split('_')[-1].split('.json')[0]
                        if int(last_frame_idx)%4 == 0:
                            encoded_json.append(encoded_json_sequence)
                            encoded_rgba.append(encoded_rgba_sequence)
                    else:
                        self.logger.error('Un-recognized protocol type')

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
        orig_imgs = imgs
        #imgs = [img[:, 180:1120, :] for img in imgs]
        img_shapes = [img.shape for img in imgs]
        imgs = np.array([resize(img, (self.image_resolution[0], self.image_resolution[1])) for img in imgs])

        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(10, 2))
        # columns = 5
        # rows = 1
        # for i in range(1, columns*rows +1):
        #     img = imgs[i-1]
        #     fig.add_subplot(rows, columns, i)
        #     plt.imshow(img)
        # plt.show()
        # assert(0)
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

        # all_p2d_heatmap = []
        all_p3d = []
        all_p2d = []

        for i, json_path in enumerate(json_paths):
            h, w, c = img_shapes[i]
            data = io.read_json(json_path)

            p2d, p3d = self._process_points(data)

            # if self.heatmap_type == 'baseline':
            #     p2d_heatmap = generate_heatmap(p2d, self.sigma, resolution=self.heatmap_resolution, h=h, w=w)
            # elif self.heatmap_type == 'distance':
            #     distances = np.sqrt(np.sum(p3d**2, axis=1))
            #     p2d_heatmap = generate_heatmap_distance(p2d, distances, h, w) # exclude head
            # else:
            #     self.logger.error('Unrecognized heatmap type')
  
            # all_p2d_heatmap.append(p2d_heatmap)
            all_p3d.append(p3d)
            all_p2d.append(p2d)
            # get action name
            action = data['action']

        cropped_imgs = []
        cropped_hms = []
        cropped_p3ds = []
        for i, cimg in enumerate(orig_imgs):
            coords = all_p2d[i]
            p3d = all_p3d[i]
            Z_c = p3d[:, 2:3]
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

            maintain_aspect = True # set to false if we do not want this

            if maintain_aspect == True:
                cropped_img = cimg[min_yn:max_yn, min_xn:max_xn]
                p2d_crop = np.array([(coord[0] - min_xn, coord[1] - min_yn) for coord in coords])
            else:
                cropped_img = cimg[min_yc:max_yc, min_xc:max_xc]
                p2d_crop = np.array([(coord[0] - min_xc, coord[1] - min_yc) for coord in coords])

            w_crop, h_crop, c = cropped_img.shape

            cropped_img = resize(cropped_img, (self.image_resolution[0], self.image_resolution[1]))
            uv1 = np.concatenate([np.copy(p2d_crop), np.ones((17, 1))], axis=1)
            uv1 = uv1 * Z_c
            intrinsic_matrix = np.zeros((3,3))
            intrinsic_matrix[0, 0] = camera2int[data['camera']]['focal_length'][0]
            intrinsic_matrix[1, 1] = camera2int[data['camera']]['focal_length'][1]
            intrinsic_matrix[0, 2] = camera2int[data['camera']]['center'][0]
            intrinsic_matrix[1, 2] = camera2int[data['camera']]['center'][1]
            intrinsic_matrix[2, 2] = 1
            p3d = np.matmul(np.linalg.inv(intrinsic_matrix), uv1.T).T
            p3d  = (p3d - p3d[14:15, :])/1000.
            cropped_p3ds.append(p3d)
            cropped_img = resize(cropped_img, (self.image_resolution[0], self.image_resolution[1]))
            cropped_imgs.append(cropped_img)

            if self.heatmap_type == 'baseline':
                p2d_hm_crop = generate_heatmap(p2d_crop, self.sigma, 
                                        resolution=self.heatmap_resolution, h=h_crop, w=w_crop)
            elif self.heatmap_type == 'distance':
                distances = np.sqrt(np.sum(p3d**2, axis=1))
                p2d_hm_crop = generate_heatmap_distance(p2d_crop, distances, h_crop, w_crop) # exclude head
            else:
                self.logger.error('Unrecognized heatmap type')

            cropped_hms.append(p2d_hm_crop)

        if self.transform:
            imgs = np.array(
                [self.transform({'image': img})['image'].numpy() for img in imgs])
            cropped_imgs = np.array(
                [self.transform({'image': c_img})['image'].numpy() for c_img in cropped_imgs])
            p3d = np.array([self.transform({'joints3D': p3d})['joints3D'].numpy() for p3d in cropped_p3ds])
            # p2d = np.array([self.transform({'joints2D': p2d})['joints2D'].numpy() for p2d in all_p2d_heatmap])
            p2d_crop = np.array([self.transform({'joints2D': p2d})['joints2D'].numpy() for p2d in cropped_hms])

        return torch.tensor(cropped_imgs), torch.tensor(p2d_crop), torch.tensor(p3d), action

    def __len__(self):

        return len(self.index[self.ROOT_DIRS[0]])


class MocapH36MCropSeqDataModule(pl.LightningDataModule):

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
        self.protocol = kwargs.get('protocol')
        self.sigma = kwargs.get('sigma')
        self.w2c = kwargs.get('w2c')
        self.sr = kwargs.get('h36m_sample_rate')
        self.p_train = f'{self.protocol}_train'
        self.p_test = f'{self.protocol}_test'

        # Data: data transformation strategy
        self.data_transform = transforms.Compose(
            [trsf.ImageTrsf(), trsf.ToTensor()]
        )
        
    def train_dataloader(self):
        data_train = MocapH36MCropTransformer(
            self.train_dir,
            SetType.TRAIN,
            transform=self.data_transform,
            sequence_length = self.seq_len,
            skip = self.skip,
            heatmap_type=self.heatmap_type,
            heatmap_resolution=self.heatmap_resolution,
            image_resolution=self.image_resolution, sigma=self.sigma,
            protocol=self.p_train, w2c=self.w2c, sr=self.sr)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=False)

    def val_dataloader(self):
        data_val =  MocapH36MCropTransformer(
            self.val_dir,
            SetType.VAL,
            transform=self.data_transform,
            sequence_length = self.seq_len,
            skip = self.skip,
            heatmap_type=self.heatmap_type,
            heatmap_resolution=self.heatmap_resolution,
            image_resolution=self.image_resolution, sigma=self.sigma,
            protocol=self.p_test, w2c=self.w2c, sr=self.sr)
        return DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)

    def test_dataloader(self):
        data_test =  MocapH36MCropTransformer(
            self.test_dir,
            SetType.TEST,
            transform=self.data_transform,
            sequence_length = self.seq_len,
            skip = self.skip,
            heatmap_type=self.heatmap_type,
            heatmap_resolution=self.heatmap_resolution,
            image_resolution=self.image_resolution, sigma=self.sigma,
            protocol=self.p_test, w2c=self.w2c, sr=self.sr)
        return DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)

