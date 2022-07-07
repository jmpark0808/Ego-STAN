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

h36m_cameras_intrinsic_params = [
    {
        'id': '54138969',
        'center': [512.54150390625, 515.4514770507812],
        'focal_length': [1145.0494384765625, 1143.7811279296875],
        'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],
        'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': 70, # Only used for visualization
    },
    {
        'id': '55011271',
        'center': [508.8486328125, 508.0649108886719],
        'focal_length': [1149.6756591796875, 1147.5916748046875],
        'radial_distortion': [-0.1942136287689209, 0.2404085397720337, 0.006819975562393665],
        'tangential_distortion': [-0.0016190266469493508, -0.0027408944442868233],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': -70, # Only used for visualization
    },
    {
        'id': '58860488',
        'center': [519.8158569335938, 501.40264892578125],
        'focal_length': [1149.1407470703125, 1148.7989501953125],
        'radial_distortion': [-0.2083381861448288, 0.25548800826072693, -0.0024604974314570427],
        'tangential_distortion': [0.0014843869721516967, -0.0007599993259645998],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': 110, # Only used for visualization
    },
    {
        'id': '60457274',
        'center': [514.9682006835938, 501.88201904296875],
        'focal_length': [1145.5113525390625, 1144.77392578125],
        'radial_distortion': [-0.198384091258049, 0.21832367777824402, -0.008947807364165783],
        'tangential_distortion': [-0.0005872055771760643, -0.0018133620033040643],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': -110, # Only used for visualization
    },
]

h36m_cameras_extrinsic_params = {
    'S1': {'54138969':
        {
            'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
            'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        },
        '55011271':
        {
            'orientation': [0.6157187819480896, -0.764836311340332, -0.14833825826644897, 0.11794740706682205],
            'translation': [1761.278564453125, -5078.0068359375, 1606.2650146484375],
        },
        '58860488':
        {
            'orientation': [0.14651472866535187, -0.14647851884365082, 0.7653023600578308, -0.6094175577163696],
            'translation': [-1846.7777099609375, 5215.04638671875, 1491.972412109375],
        },
        '60457274':
        {
            'orientation': [0.5834008455276489, -0.7853162288665771, 0.14548823237419128, -0.14749594032764435],
            'translation': [-1794.7896728515625, -3722.698974609375, 1574.8927001953125],
        },
    },
    'S2': [
        {},
        {},
        {},
        {},
    ],
    'S3': [
        {},
        {},
        {},
        {},
    ],
    'S4': [
        {},
        {},
        {},
        {},
    ],
    'S5': {'54138969':
        {
            'orientation': [0.1467377245426178, -0.162370964884758, -0.7551892995834351, 0.6178938746452332],
            'translation': [2097.3916015625, 4880.94482421875, 1605.732421875],
        },
        '55011271':
        {
            'orientation': [0.6159758567810059, -0.7626792192459106, -0.15728192031383514, 0.1189815029501915],
            'translation': [2031.7008056640625, -5167.93310546875, 1612.923095703125],
        },
        '58860488':
        {
            'orientation': [0.14291371405124664, -0.12907841801643372, 0.7678384780883789, -0.6110143065452576],
            'translation': [-1620.5948486328125, 5171.65869140625, 1496.43701171875],
        },
        '60457274':
        {
            'orientation': [0.5920479893684387, -0.7814217805862427, 0.1274748593568802, -0.15036417543888092],
            'translation': [-1637.1737060546875, -3867.3173828125, 1547.033203125],
        },
    },
    'S6': {'54138969':
        {
            'orientation': [0.1337897777557373, -0.15692396461963654, -0.7571090459823608, 0.6198879480361938],
            'translation': [1935.4517822265625, 4950.24560546875, 1618.0838623046875],
        },
        '55011271':
        {
            'orientation': [0.6147197484970093, -0.7628812789916992, -0.16174767911434174, 0.11819244921207428],
            'translation': [1969.803955078125, -5128.73876953125, 1632.77880859375],
        },
        '58860488':
        {
            'orientation': [0.1529948115348816, -0.13529130816459656, 0.7646096348762512, -0.6112781167030334],
            'translation': [-1769.596435546875, 5185.361328125, 1476.993408203125],
        }
        ,'60457274':
        {
            'orientation': [0.5916101336479187, -0.7804774045944214, 0.12832270562648773, -0.1561593860387802],
            'translation': [-1721.668701171875, -3884.13134765625, 1540.4879150390625],
        },
    },
    'S7': {'54138969':
        {
            'orientation': [0.1435241848230362, -0.1631336808204651, -0.7548328638076782, 0.6188824772834778],
            'translation': [1974.512939453125, 4926.3544921875, 1597.8326416015625],
        },
        '55011271':
        {
            'orientation': [0.6141672730445862, -0.7638262510299683, -0.1596645563840866, 0.1177929937839508],
            'translation': [1937.0584716796875, -5119.7900390625, 1631.5665283203125],
        },
        '58860488':
        {
            'orientation': [0.14550060033798218, -0.12874816358089447, 0.7660516500473022, -0.6127139329910278],
            'translation': [-1741.8111572265625, 5208.24951171875, 1464.8245849609375],
        },
        '60457274':
        {
            'orientation': [0.5912848114967346, -0.7821764349937439, 0.12445473670959473, -0.15196487307548523],
            'translation': [-1734.7105712890625, -3832.42138671875, 1548.5830078125],
        },
    },
    'S8': {'54138969':
        {
            'orientation': [0.14110587537288666, -0.15589867532253265, -0.7561917304992676, 0.619644045829773],
            'translation': [2150.65185546875, 4896.1611328125, 1611.9046630859375],
        },
        '55011271':
        {
            'orientation': [0.6169601678848267, -0.7647668123245239, -0.14846350252628326, 0.11158157885074615],
            'translation': [2219.965576171875, -5148.453125, 1613.0440673828125],
        },
        '58860488':
        {
            'orientation': [0.1471444070339203, -0.13377119600772858, 0.7670128345489502, -0.6100369691848755],
            'translation': [-1571.2215576171875, 5137.0185546875, 1498.1761474609375],
        },
        '60457274':
        {
            'orientation': [0.5927824378013611, -0.7825870513916016, 0.12147816270589828, -0.14631995558738708],
            'translation': [-1476.913330078125, -3896.7412109375, 1547.97216796875],
        },
    },
    'S9': {'54138969':
        {
            'orientation': [0.15540587902069092, -0.15548215806484222, -0.7532095313072205, 0.6199594736099243],
            'translation': [2044.45849609375, 4935.1171875, 1481.2275390625],
        },
        '55011271':
        {
            'orientation': [0.618784487247467, -0.7634735107421875, -0.14132238924503326, 0.11933968216180801],
            'translation': [1990.959716796875, -5123.810546875, 1568.8048095703125],
        },
        '58860488':
        {
            'orientation': [0.13357827067375183, -0.1367100477218628, 0.7689454555511475, -0.6100738644599915],
            'translation': [-1670.9921875, 5211.98583984375, 1528.387939453125],
        },
        '60457274':
        {
            'orientation': [0.5879399180412292, -0.7823407053947449, 0.1427614390850067, -0.14794869720935822],
            'translation': [-1696.04345703125, -3827.099853515625, 1591.4127197265625],
        },
    },
    'S11': {'54138969':
        {
            'orientation': [0.15232472121715546, -0.15442320704460144, -0.7547563314437866, 0.6191070079803467],
            'translation': [2098.440185546875, 4926.5546875, 1500.278564453125],
        },
        '55011271':
        {
            'orientation': [0.6189449429512024, -0.7600917220115662, -0.15300633013248444, 0.1255258321762085],
            'translation': [2083.182373046875, -4912.1728515625, 1561.07861328125],
        },
        '58860488':
        {
            'orientation': [0.14943228662014008, -0.15650227665901184, 0.7681233882904053, -0.6026304364204407],
            'translation': [-1609.8153076171875, 5177.3359375, 1537.896728515625],
        },
        '60457274':
        {
            'orientation': [0.5894251465797424, -0.7818877100944519, 0.13991211354732513, -0.14715361595153809],
            'translation': [-1590.738037109375, -3854.1689453125, 1578.017578125],
        },
    },
}

def world_to_camera(X, R, t): 
    Rt = wrap(qinverse, R) # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) # Rotate and translate

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape)-1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape)-1)
    return (v + 2 * (q[..., :1] * uv + uuv))
    
    
def qinverse(q, inplace=False):
    # We assume the quaternion to be normalized
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape)-1)

def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result

def normalize_screen_coordinates(X, w, h): 
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X/w*2 - [1, h/w]

class MocapH36M(BaseDataset):
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

    def __init__(self, *args, heatmap_type='baseline', heatmap_resolution=[47, 47], image_resolution=[368, 368], protocol = 'p1_train', **kwargs):
        """Init class, to allow variable sequence length, inherits from Base
        Keyword Arguments:
            sequence_length -- length of image sequence (default: {5})
        """

        self.heatmap_type = heatmap_type
        self.heatmap_resolution = heatmap_resolution
        self.image_resolution = image_resolution
        self.protocol = protocol
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

        p3d /= self.MM_TO_M
        # World to camera
        p3d = np.expand_dims(p3d, 0)
        subject = data['subject']
        camera = data['camera']
        orientation = self._cameras[f'S{subject}'][camera]['orientation']
        translation = self._cameras[f'S{subject}'][camera]['translation']/1000.
        p3d = world_to_camera(p3d, orientation, translation)
        p3d = np.squeeze(p3d)
        # Normalize
        p3d[:, ~14] -= p3d[:, 14]
        #p3d[0, :] = p3d[1, :] # Set artifical head value to neck value
        

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
        data_val = MocapH36M(self.val_dir, SetType.VAL, transform=self.data_transform, heatmap_type=self.heatmap_type, heatmap_resolution=self.heatmap_resolution, image_resolution=self.image_resolution, protocol=self.p_test)
        return DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        data_test = MocapH36M(self.test_dir, SetType.TEST, transform=self.data_transform, heatmap_type=self.heatmap_type, heatmap_resolution=self.heatmap_resolution, image_resolution=self.image_resolution, protocol=self.p_test)
        return DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)
