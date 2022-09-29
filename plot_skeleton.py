# ----------------------------------------------------------- #
#  This is code confidential, for peer-review purposes only   #
#  and protected under conference code of ethics              #
# ----------------------------------------------------------- #

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from skimage import io 
from skimage.transform import resize
from utils.evaluate import highest_differences
import json

"""
    Plot, overlay and save joint skeletons for comparison

    @param poses (list):
        A list of different skeletons to overlay and plot.
        This allows for multiple pose inferences to be
        viewed concurrently. Each item in the list is
        required to have the following information:
        [
            {
                "legend_name": "...",
                "plot_color": "...",
                "pose": < num_joints by 3 array >
            },
            ...
        ]

    @param output_file (str):
        The file path and name to save the figure

    """
def read_json(path):
    """Read data from json file
    Arguments:
        path {str} -- file path
    Returns:
        dict -- data
    """

    with open(path, 'r') as in_file:
        data = json.load(in_file)

    return data

hp_seq_results = pd.read_pickle(r'/mnt/hdd/waterloo/lightning_logs/3d_plots/xregopose_seq_hm_direct_05_07_21_57_52/raw_results_xregopose_seq_hm_direct_05_07_21_57_52.pkl')
hp_seq_results_files = pd.read_pickle(r'/mnt/hdd/waterloo/lightning_logs/3d_plots/xregopose_seq_hm_direct_05_07_21_57_52/results_xregopose_seq_hm_direct_05_07_21_57_52')
hp_baseline_results = pd.read_pickle(r'/mnt/hdd/waterloo/lightning_logs/3d_plots/xregopose_05_07_21_47_14/raw_results_xregopose_05_07_21_47_14.pkl')
hp_baseline_results_files = pd.read_pickle(r'/mnt/hdd/waterloo/lightning_logs/3d_plots/xregopose_05_07_21_47_14/results_xregopose_05_07_21_47_14')
h2d_results = pd.read_pickle(r'/mnt/hdd/waterloo/models/xregopose_seq_hm_direct_05_08_16_21_05/handpicked_results_xregopose_seq_hm_direct_05_08_16_21_05.pkl')

seq_filenames = np.array(hp_seq_results_files['Filenames'])
baseline_filenames = np.array(hp_baseline_results_files['Filenames'])


joints = {
        # 'Hips': {'parent': None, 'col': 0},
        # 'Spine': {'parent': 'Hips', 'col': 0},
        # 'Spine1': {'parent': 'Spine', 'col': 0},
        # 'Spine2': {'parent': 'Spine1', 'col': 0},
        'Head': {'parent': 'Neck', 'col': 5},
        'Neck': {'parent': 'Spine2', 'col': 5},
        # 'LeftShoulder': {'parent': 'Spine2', 'col': 0},
        'LeftArm': {'parent': 'LeftShoulder', 'col': 3},
        'LeftForeArm': {'parent': 'LeftArm', 'col': 3},
        'LeftHand': {'parent': 'LeftForeArm', 'col': 4},
        # 'RightShoulder': {'parent': 'Spine2', 'col': 0},
        'RightArm':  {'parent': 'RightShoulder', 'col': 1},
        'RightForeArm':  {'parent': 'RightArm', 'col': 1},
        'RightHand':  {'parent': 'RightForeArm', 'col': 2},
        'LeftUpLeg': {'parent': 'Hips', 'col': 8},
        'LeftLeg': {'parent': 'LeftUpLeg', 'col': 8},
        'LeftFoot': {'parent': 'LeftLeg', 'col': 9},
        'LeftToeBase': {'parent': 'LeftFoot', 'col': 9},
        'RightUpLeg': {'parent': 'Hips', 'col': 6},
        'RightLeg': {'parent': 'RightUpLeg', 'col': 6},
        'RightFoot': {'parent': 'RightLeg', 'col': 7},
        'RightToeBase': {'parent': 'RightFoot', 'col': 7},
    }

for jid, v in enumerate(joints.values()):
    v.update({'jid': jid})



output_file = '/home/eddie/skeleton/'
highest_differences = set(highest_differences)
diff_dict = {}
for idx, file in enumerate(seq_filenames):
    gt_pose = hp_seq_results['gts'][idx]
    seq_pose = hp_seq_results['preds'][idx]

    idx_baseline = np.argwhere(baseline_filenames == file)[0, 0]
    baseline_pose = hp_baseline_results['preds'][idx_baseline]

    seq_mpjpe = np.sqrt(np.sum(np.power(seq_pose - gt_pose, 2), axis=1))
    baseline_mpjpe = np.sqrt(np.sum(np.power(baseline_pose - gt_pose, 2), axis=1))
    diff = baseline_mpjpe - seq_mpjpe
    diff = np.sum(diff[[4, 7, 11, 15]])

    diff_dict[file] = diff

    if file in highest_differences:
        plot_img = True
        for azim in range(0, 100, 10):
            for elev in range(0, 100, 10):
                gt_pose = hp_seq_results['gts'][idx]
                seq_pose = hp_seq_results['preds'][idx]

                idx_baseline = np.argwhere(baseline_filenames == file)[0, 0]
                baseline_pose = hp_baseline_results['preds'][idx_baseline]
                

                subject = file.split('_rgba')[0]
                number = file.split('_rgba_')[-1]
                
                
                # img = h2d_results[file]['img']

                # mean=[0.485, 0.456, 0.406]
                # std=[0.229, 0.224, 0.225]
                # img = img[-1]

                # img[0] = img[0]*std[0]+mean[0]
                # img[1] = img[1]*std[1]+mean[1]
                # img[2] = img[2]*std[2]+mean[2]
                if plot_img:
                    try:
                        img = io.imread(os.path.join('/home/eddie/TestSet/', subject, 'env_001', 'cam_down', 'rgba', subject+'.rgba.'+number+'.png')).astype(np.float32)
                        img /= 255.0
                        img = img[:, 180:1120, :] #crop
                        img = resize(img, (368, 368))

                        data = read_json(os.path.join('/home/eddie/TestSet/', subject, 'env_001', 'cam_down', 'json', subject+'_'+number+'.json'))

                        p2d_orig = np.array(data['pts2d_fisheye']).T
                        joint_names = {j['name'].replace('mixamorig:', ''): jid
                                    for jid, j in enumerate(data['joints'])}

                        # ------------------- Filter joints -------------------

                        p2d = np.empty([len(joints), 2], dtype=p2d_orig.dtype)
    

                        for jid, j in enumerate(joints.keys()):
                            p2d[jid] = p2d_orig[joint_names[j]]
        
                        p2d[:, 0] = p2d[:, 0]-180
                        h2d = p2d
                        # h2d = h2d_results[file]['p2d'] # 16 x 2
                        fig = plt.figure(num=1, clear=True, figsize=(9, 9))
                        ax = plt.axes()
                        ax.imshow(img) #.transpose((1, 2, 0)))
                        BONE_LINKS = [
                            [0, 1],
                            [1, 2],
                            [1, 5],
                            [2, 3],
                            [3, 4],
                            [2, 8],
                            [8, 9],
                            [9, 10],
                            [10, 11],
                            [8, 12],
                            [5, 12],
                            [5, 6],
                            [6, 7],
                            [12, 13],
                            [13, 14],
                            [14, 15],
                        ]

                        
                        xs = h2d[:, 0]*368./940.
                        ys = h2d[:, 1]*368./800.
                        # draw bones
                        for ind, bone in enumerate(BONE_LINKS):
                            if (file == 'female_010_a_a_rgba_003845' or file == 'male_002_a_a_rgba_1447') and (ind == 4 or ind ==12):
                                continue

                            if file == 'female_010_a_a_rgba_004332' and (ind == 4 or ind == 12 or ind == 3 or ind == 11):
                                continue

                            index1, index2 = bone[0], bone[1]

                            ax.plot(
                                [xs[index1], xs[index2]],
                                [ys[index1], ys[index2]],
                                linewidth=5,
                                color="goldenrod",
                                zorder=1
                            )
                        if file == 'female_010_a_a_rgba_003845' or file == 'male_002_a_a_rgba_1447':
                            ignore_hands = [0, 1, 2, 3,  5, 6,  8, 9, 10, 11, 12, 13, 14, 15]
                            ax.scatter(xs[ignore_hands], ys[ignore_hands], color="goldenrod", marker='o', s=100, zorder=2)
                        elif file == 'female_010_a_a_rgba_004332':
                            ignore_hands = [0, 1, 2,  5,  8, 9, 10, 11, 12, 13, 14, 15]
                            ax.scatter(xs[ignore_hands], ys[ignore_hands], color="goldenrod", marker='o', s=100, zorder=2)
                        else:
                            ax.scatter(xs, ys, color="goldenrod", marker='o', s=100, zorder=2)
                        ax.axis('off')

                        fig.tight_layout()
                        # os.makedirs(output_file, exist_ok=True)

                        fig.savefig(output_file+f'{file}_img.jpg')

                        plot_img = False
                    except:
                        pass


                poses = [{"legend_name": "Dual-branch",
                        "plot_color": "red", 
                        "pose": baseline_pose},
                        {"legend_name": "Ego-STAN",
                        "plot_color": "green", 
                        "pose": seq_pose},
                        {"legend_name": "Ground Truth",
                        "plot_color": "goldenrod", 
                        "pose": gt_pose}]


                fig = plt.figure(num=1, clear=True, figsize=(16, 9))
                ax = plt.axes(projection="3d")

                BONE_LINKS = [
                    [0, 1],
                    [1, 2],
                    [1, 5],
                    [2, 3],
                    [3, 4],
                    [2, 8],
                    [8, 9],
                    [9, 10],
                    [10, 11],
                    [8, 12],
                    [5, 12],
                    [5, 6],
                    [6, 7],
                    [12, 13],
                    [13, 14],
                    [14, 15],
                ]

                for item in poses:
                    xs = item["pose"][:, 0]
                    ys = item["pose"][:, 1]
                    zs = -item["pose"][:, 2]
                    # draw bones
                    for ind, bone in enumerate(BONE_LINKS):
                        index1, index2 = bone[0], bone[1]
                        if ind == 0:

                            ax.plot3D(
                                [xs[index1], xs[index2]],
                                [ys[index1], ys[index2]],
                                [zs[index1], zs[index2]],
                                linewidth=3,
                                color=item["plot_color"],
                                label=item["legend_name"],
                            )
                        else:
                            ax.plot3D(
                                [xs[index1], xs[index2]],
                                [ys[index1], ys[index2]],
                                [zs[index1], zs[index2]],
                                linewidth=3,
                                color=item["plot_color"]
                            )
                    # draw joints
                    ax.scatter(xs, ys, zs, color=item["plot_color"], s=20)

                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
                ax.legend()
                ax.title.set_text(f"{output_file}")
                plt.axis("off")
                fig.tight_layout()
                ax.view_init(elev=elev, azim=azim)

                # os.makedirs(output_file, exist_ok=True)
                fig.savefig(output_file+f'{file}_{azim}_{elev}.jpg')
        highest_differences.remove(file)      



sorted_files = [k for k, v in sorted(diff_dict.items(), key=lambda item: item[1])]
sorted_values = [v for k, v in sorted(diff_dict.items(), key=lambda item: item[1])]

print(sorted_files[:100])