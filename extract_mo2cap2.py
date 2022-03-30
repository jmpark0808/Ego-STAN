import os
import h5py
import json
import argparse
import numpy as np
from utils import io
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--original_dir', help='Original Directory of Dataset', required=True, default=None)
    parser.add_argument('--destination_dir', help='Destination Directory', required=True, default=None)
    parser.add_argument('--dataset_type', help='One of TRAIN, VAL, TEST', required=True, default=None)

    args = parser.parse_args()
    dict_args = vars(args)

    dataset_dir = dict_args["original_dir"]
    destination_dir = dict_args["destination_dir"]

    if(dict_args["dataset_type"] == "TRAIN"):
        dataset_dir = os.path.join(dataset_dir, "TrainSet")
        destination_dir = os.path.join(destination_dir, "TrainSet")
    elif(dict_args["dataset_type"] == "TEST"):
        dataset_dir = os.path.join(dataset_dir, "TestSet")
        destination_dir = os.path.join(destination_dir, "TestSet")
    elif(dict_args["dataset_type"] == "VAL"):
        dataset_dir = os.path.join(dataset_dir, "ValSet")
        destination_dir = os.path.join(destination_dir, "ValSet")
    else:
        print("DataSet directory not in options")
        quit()

    os.mkdir(destination_dir)

    for chunk_path in os.listdir(dataset_dir):
        os.mkdir(os.path.join(destination_dir, chunk_path[:-5]))
        os.mkdir(os.path.join(destination_dir, chunk_path[:-5], 'rgba'))
        os.mkdir(os.path.join(destination_dir, chunk_path[:-5], 'json'))
        with h5py.File(os.path.join(dataset_dir, chunk_path), 'r') as chunk:
            for i in range(len(chunk['Images'])):
                img = Image.fromarray(chunk['Images'][i].transpose(1, 2, 0))
                p2d = chunk["Annot2D"][i]
                p3d = chunk["Annot3D"][i]
                dict_json_info = {
                    'Head': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(np.array([0.0, 0.0, 0.0], np.float64))},
                    'Neck': {'2d': list(p2d[0]), '3d': list(p3d[0])},
                    'LeftArm': {'2d': list(p2d[1]), '3d': list(p3d[1])},
                    'LeftForeArm': {'2d': list(p2d[2]), '3d': list(p3d[2])},
                    'LeftHand': {'2d': list(p2d[3]), '3d': list(p3d[3])},
                    'RightArm':  {'2d': list(p2d[4]), '3d': list(p3d[4])},
                    'RightForeArm':  {'2d': list(p2d[5]), '3d': list(p3d[5])},
                    'RightHand':  {'2d': list(p2d[6]), '3d': list(p3d[6])},
                    'LeftUpLeg': {'2d': list(p2d[7]), '3d': list(p3d[7])},
                    'LeftLeg': {'2d': list(p2d[8]), '3d': list(p3d[8])},
                    'LeftFoot': {'2d': list(p2d[9]), '3d': list(p3d[9])},
                    'LeftToeBase': {'2d': list(p2d[10]), '3d': list(p3d[10])},
                    'RightUpLeg': {'2d': list(p2d[11]), '3d': list(p3d[11])},
                    'RightLeg': {'2d': list(p2d[12]), '3d': list(p3d[12])},
                    'RightFoot': {'2d': list(p2d[13]), '3d': list(p3d[13])},
                    'RightToeBase': {'2d': list(p2d[14]), '3d': list(p3d[14])},
                }
                img.save(os.path.join(destination_dir, chunk_path[:-5], 'rgba', '{0}_{1:06}.png'.format(chunk_path[:-5], i)))
                print(os.path.join(destination_dir, chunk_path[:-5], 'rgba', '{0}_{1:06}.png'.format(chunk_path[:-5], i)))
                io.write_json(os.path.join(destination_dir, chunk_path[:-5], 'json', '{0}_{1:06}.json'.format(chunk_path[:-5], i)), dict_json_info)
                print(os.path.join(destination_dir, chunk_path[:-5], 'json', '{0}_{1:06}.json'.format(chunk_path[:-5], i)))