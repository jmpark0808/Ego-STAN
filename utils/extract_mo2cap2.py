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

    if(not os.path.exists(destination_dir)):
        os.mkdir(destination_dir)

    if(dict_args["dataset_type"] == "TEST"):
        
        # Converting the TestSet

        # Creating Segments for frames

        actions = ['walking','sitting','crawling','crouching', 'boxing', 'dancing', 'stretching', 'waving', 'total']

        olek = [np.arange(158,818),
                np.arange(1017,1213),
                np.arange(2432,2841),
                np.arange(818,1017), 
                np.arange(1469,1639),
                np.arange(1639,2184),
                np.arange(2184,2432),
                np.arange(1213,1469),
                np.arange(158,2841)]

        weipeng = [np.concatenate([np.arange(387, 654), np.arange(1086, 1461), np.arange(1867, 2040)], axis=0),
                   np.concatenate([np.arange(654,877), np.arange(1535,1867)], axis=0),
                   np.concatenate([np.arange(877,1086), np.arange(3019,3168)], axis=0),
                   np.arange(2883,3019),
                   np.concatenate([np.arange(1461,1535), np.arange(2040,2215)], axis=0),
                   np.arange(2215,2741),
                   np.arange(2741,2883),
                   np.arange(3168,3289),
                   np.arange(387,3289)]

        # Creating the directories

        orig_dir_path = dataset_dir
        destination_path = destination_dir

        olek_dir_path = os.path.join(orig_dir_path, 'olek_outdoor')
        weipeng_dir_path = os.path.join(orig_dir_path, 'weipeng_studio')
        olek_dest_path = os.path.join(destination_path, 'olek_outdoor')
        weipeng_dest_path = os.path.join(destination_path, 'weipeng_studio')
        olek_rgba_path = os.path.join(olek_dest_path, 'rgba')
        olek_json_path = os.path.join(olek_dest_path, 'json')
        weipeng_rgba_path = os.path.join(weipeng_dest_path, 'rgba')
        weipeng_json_path = os.path.join(weipeng_dest_path, 'json')

        if(not os.path.exists(olek_dir_path)):
            os.mkdir(olek_dir_path)
        if(not os.path.exists(weipeng_dir_path)):
            os.mkdir(weipeng_dir_path)
        if(not os.path.exists(olek_dest_path)):
            os.mkdir(olek_dest_path)
        if(not os.path.exists(weipeng_dest_path)):
            os.mkdir(weipeng_dest_path)
        if(not os.path.exists(olek_rgba_path)):
            os.mkdir(olek_rgba_path)
        if(not os.path.exists(olek_json_path)):
            os.mkdir(olek_json_path)
        if(not os.path.exists(weipeng_rgba_path)):
            os.mkdir(weipeng_rgba_path)
        if(not os.path.exists(weipeng_json_path)):
            os.mkdir(weipeng_json_path)

        # Loading the SciPy files

        oleks = scipy.io.loadmat(os.path.join(orig_dir_path, 'olek_outdoor_gt.mat'))
        weipengs = scipy.io.loadmat(os.path.join(orig_dir_path, 'weipeng_studio_gt.mat'))

        oleks_p3d = oleks['pose_gt']
        weipengs_p3d = weipengs['pose_gt']

        dict_olek = {}
        dict_weipeng = {}

        # Matching the actions to frames

        for i, segment in enumerate(olek):
            if actions[i] != 'total':
                for j in segment:
                    dict_olek.update({j : actions[i]})

        for i, segment in enumerate(weipeng):
            if actions[i] != 'total':
                for j in segment:
                    dict_weipeng.update({j : actions[i]})

        # Creating and Copying the JSON files

        for fpath in os.listdir(olek_dir_path):
            frame_idx = int(fpath[-8:-4])
            if frame_idx in dict_olek.keys():
                shutil.copy(os.path.join(olek_dir_path, fpath), olek_rgba_path)
                p3d = oleks_p3d[frame_idx-158] # Frame offset when reading from the .mat file
                dict_json_info = {
                    #'Head': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(np.array([0.0, 0.0, 0.0], np.float64))},
                    'Neck': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[0])},
                    'LeftArm': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[1])},
                    'LeftForeArm': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[2])},
                    'LeftHand': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[3])},
                    'RightArm':  {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[4])},
                    'RightForeArm':  {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[5])},
                    'RightHand':  {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[6])},
                    'LeftUpLeg': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[7])},
                    'LeftLeg': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[8])},
                    'LeftFoot': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[9])},
                    'LeftToeBase': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[10])},
                    'RightUpLeg': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[11])},
                    'RightLeg': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[12])},
                    'RightFoot': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[13])},
                    'RightToeBase': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[14])},
                    'action': dict_olek[frame_idx]
                    }
                write_json(os.path.join(olek_json_path, "{0}{1:04}.json".format(fpath[:-8], frame_idx)), dict_json_info)
                print(os.path.join(olek_json_path, "{0}{1:04}.json".format(fpath[:-8], frame_idx)))

        for fpath in os.listdir(weipeng_dir_path):
            frame_idx = int(fpath[-8:-4])
            if frame_idx in dict_weipeng.keys():
                shutil.copy(os.path.join(weipeng_dir_path, fpath), weipeng_rgba_path)
                p3d = weipengs_p3d[frame_idx-387] # Frame offset when reading from the .mat file
                dict_json_info = {
                    #'Head': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(np.array([0.0, 0.0, 0.0], np.float64))},
                    'Neck': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[0])},
                    'LeftArm': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[1])},
                    'LeftForeArm': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[2])},
                    'LeftHand': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[3])},
                    'RightArm':  {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[4])},
                    'RightForeArm':  {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[5])},
                    'RightHand':  {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[6])},
                    'LeftUpLeg': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[7])},
                    'LeftLeg': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[8])},
                    'LeftFoot': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[9])},
                    'LeftToeBase': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[10])},
                    'RightUpLeg': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[11])},
                    'RightLeg': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[12])},
                    'RightFoot': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[13])},
                    'RightToeBase': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(p3d[14])},
                    'action': dict_weipeng[frame_idx]
                    }
                write_json(os.path.join(weipeng_json_path, "{0}{1:04}.json".format(fpath[:-8], frame_idx)), dict_json_info)
                print(os.path.join(weipeng_json_path, "{0}{1:04}.json".format(fpath[:-8], frame_idx)))
    else:

        # Converting the TrainSet

        for chunk_path in os.listdir(dataset_dir):

            # Making the directories

            chunk_dir = os.path.join(destination_dir, chunk_path[:-5])
            if(os.path.exists(f"{chunk_dir}.tar.gz")):
                continue
            chunk_rgba_dir = os.path.join(destination_dir, chunk_path[:-5], 'rgba')
            chunk_json_dir = os.path.join(destination_dir, chunk_path[:-5], 'json')

            if(not os.path.exists(chunk_dir)):
                os.mkdir(chunk_dir)
            if(not os.path.exists(chunk_rgba_dir)):
                os.mkdir(chunk_rgba_dir)
            if(not os.path.exists(chunk_json_dir)):
                os.mkdir(chunk_json_dir)

            # Opening the h5py file to make the images and json files (if not full)

            if len(os.listdir(chunk_rgba_dir)) != 1000 and len(os.listdir(chunk_json_dir)) != 1000:
                with h5py.File(os.path.join(dataset_dir, chunk_path), 'r') as chunk:
                    print(os.path.join(dataset_dir, chunk_path))
                    for i in range(len(chunk['Images'])):
                        json_d_path = os.path.join(chunk_json_dir, '{0}_{1:06}.json'.format(chunk_path[:-5], i))
                        rgba_d_path = os.path.join(chunk_rgba_dir, '{0}_{1:06}.png'.format(chunk_path[:-5], i))
                        if(not os.path.exists(json_d_path) and not os.path.exists(rgba_d_path)):
                            img = Image.fromarray(chunk['Images'][i].transpose(1, 2, 0))
                            p2d = chunk["Annot2D"][i]
                            p3d = chunk["Annot3D"][i]
                            dict_json_info = {
                                #'Head': {'2d': list(np.array([0.0, 0.0], np.float64)), '3d': list(np.array([0.0, 0.0, 0.0], np.float64))},
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
                            if(not os.path.exists(rgba_d_path)):
                                img.save(rgba_d_path)
                                print(rgba_d_path)
                            if(not os.path.exists(json_d_path)):
                                write_json(json_d_path, dict_json_info)
                                print(json_d_path)
