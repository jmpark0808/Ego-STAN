# ----------------------------------------------------------- #
#  This is code confidential, for peer-review purposes only   #
#  and protected under conference code of ethics              #
# ----------------------------------------------------------- #

import os
import h5py
import json
import scipy.io
import shutil
import argparse
import numpy as np
import io 
from PIL import Image

def write_json(path, data):
    """Save data into a json file
    Arguments:
        path {str} -- path where to save the file
        data {serializable} -- data to be stored
    """

    assert isinstance(path, str)
    with open(path, 'w') as out_file:
        json.dump(data, out_file, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--original_dir', help='Original Directory of Dataset', required=True, default=None)
    parser.add_argument('--destination_dir', help='Destination Directory', required=True, default=None)

    args = parser.parse_args()
    dict_args = vars(args)

    dataset_dir = dict_args["original_dir"]
    destination_dir = dict_args["destination_dir"]

    rgba_dir = os.path.join(destination_dir, 'rgba')
    json_dir = os.path.join(destination_dir, 'json')

    # Make the rgba and json directories for test and train
    if(not os.path.exists(rgba_dir)):
        os.mkdir(rgba_dir)
    if(not os.path.exists(json_dir)):
        os.mkdir(json_dir)

    img_dir = os.path.join(dataset_dir, 'images')
    metadata_path = os.path.join(dataset_dir, 'joints.mat')

    # Load in the metadata for the MPII dataset
    metadata_lsp = np.transpose(scipy.io.loadmat(metadata_path)['joints'], (2, 1, 0))
    num_raw_imgs = len(os.listdir(img_dir))

    for i in range(num_raw_imgs):
        fname = "im{0:04}.jpg".format(i+1)
        fpath = os.path.join(img_dir, fname)
        json_path = os.path.join(json_dir, "{}.json".format(fname[:-4]))
        if(os.path.exists(json_path)):
            continue
        img = Image.open(fpath)
        width, height = img.size

        joint_coords = [[float(metadata_lsp[i][n][0]), 
                         float(metadata_lsp[i][n][1]), 
                         float(metadata_lsp[i][n][2])] for n in range(14)]

        frame_action = 'sports'

        dict_json_info = {}
        dict_json_info.update({'img_width': int(width), 
                               'img_height': int(height),
                               'action': frame_action})

        dict_person_info = {
            'RightAnkle': joint_coords[0],
            'RightKnee': joint_coords[1],
            'RightHip': joint_coords[2],
            'LeftHip': joint_coords[3],
            'LeftKnee': joint_coords[4],
            'LeftAnkle': joint_coords[5],
            'RightWrist': joint_coords[6],
            'RightElbow': joint_coords[7],
            'RightShoulder': joint_coords[8],
            'LeftShoulder': joint_coords[9],
            'LeftElbow': joint_coords[10],
            'LeftWrist': joint_coords[11],
            'UpperNeck': joint_coords[12],
            'Head': joint_coords[13],
            'Pelvis': [-1., -1., -1.],
            'Thorax': [-1., -1., -1.],
        }
        dict_json_info.update({0: dict_person_info})

        write_json(json_path, dict_json_info)
        shutil.copy(fpath, rgba_dir)
        print(json_path)
