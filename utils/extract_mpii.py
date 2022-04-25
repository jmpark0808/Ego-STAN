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
    metadata_path = os.path.join(dataset_dir, 'mpii_human_pose_v1_u12_1.mat')

    # Load in the metadata for the MPII dataset
    metadata_mpii = scipy.io.loadmat(metadata_path)
    num_raw_imgs = len(metadata_mpii['RELEASE']['annolist'][0][0]['image'][0])

    for i in range(num_raw_imgs):
        if metadata_mpii['RELEASE']['annolist'][0][0]['annorect'][0][i].dtype.names is not None:
            if 'annopoints' in metadata_mpii['RELEASE']['annolist'][0][0]['annorect'][0][i].dtype.names:
                fname = metadata_mpii['RELEASE']['annolist'][0][0]['image'][0][i]['name'][0][0][0]
                fpath = os.path.join(img_dir, fname)
                json_path = os.path.join(json_dir, "{}.json".format(fname[:-4]))
                if(os.path.exists(json_path)):
                    continue
                img = Image.open(fpath)
                width, height = img.size

                person_idxs = []
                joint_coords_all = []
                for j, elem in  enumerate(metadata_mpii['RELEASE']['annolist'][0][0]['annorect'][0][i]['annopoints'][0]):
                    if len(metadata_mpii['RELEASE']['annolist'][0][0]['annorect'][0][i]['annopoints'][0][j]) > 0:
                        person_idxs.append(j)

                for person_idx in person_idxs:
                    joint_coords = [[-1, -1] for num in range(16)]
                    group_metadata = metadata_mpii['RELEASE']['annolist'][0][0]['annorect'][0][i]['annopoints'][0][person_idx]['point'][0][0]
                    for n, elem in enumerate(group_metadata[0]):
                        joint_idx = int(group_metadata['id'][0][n][0][0])
                        joint_coords[joint_idx][0] = float(group_metadata['x'][0][n][0][0])
                        joint_coords[joint_idx][1] = float(group_metadata['y'][0][n][0][0])
                    joint_coords_all.append(joint_coords)

                if(len(metadata_mpii['RELEASE']['act'][0][0][i]) > 0):
                    if(len(metadata_mpii['RELEASE']['act'][0][0][i][0][0]) > 0):
                        frame_action = metadata_mpii['RELEASE']['act'][0][0][i][0][0][0]
                    else:
                        frame_action = 'unknown'
                else:
                    frame_action = 'unknown'

                dict_json_info = {}
                dict_json_info.update({'img_width': int(width), 
                                       'img_height': int(height),
                                       'action': frame_action})
                for person_idx in range(len(person_idxs)):
                    dict_person_info = {
                        'RightAnkle': joint_coords_all[person_idx][0],
                        'RightKnee': joint_coords_all[person_idx][1],
                        'RightHip': joint_coords_all[person_idx][2],
                        'LeftHip': joint_coords_all[person_idx][3],
                        'LeftKnee': joint_coords_all[person_idx][4],
                        'LeftAnkle': joint_coords_all[person_idx][5],
                        'RightWrist': joint_coords_all[person_idx][10],
                        'RightElbow': joint_coords_all[person_idx][11],
                        'RightShoulder': joint_coords_all[person_idx][12],
                        'LeftShoulder': joint_coords_all[person_idx][13],
                        'LeftElbow': joint_coords_all[person_idx][14],
                        'LeftWrist': joint_coords_all[person_idx][15],
                        'UpperNeck': joint_coords_all[person_idx][8],
                        'Head': joint_coords_all[person_idx][9],
                        'Pelvis': joint_coords_all[person_idx][6],
                        'Thorax': joint_coords_all[person_idx][7],
                    }
                    dict_json_info.update({person_idx: dict_person_info})

                write_json(json_path, dict_json_info)
                shutil.copy(fpath, rgba_dir)
                print(json_path)
