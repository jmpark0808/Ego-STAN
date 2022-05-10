import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from utils.evaluate import highest_differences
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


hp_seq_results = pd.read_pickle(r'/home/eddie/waterloo/lightning_logs/3d_plots/xregopose_seq_hm_direct_05_07_21_57_52/raw_results_xregopose_seq_hm_direct_05_07_21_57_52.pkl')
hp_seq_results_files = pd.read_pickle(r'/home/eddie/waterloo/lightning_logs/3d_plots/xregopose_seq_hm_direct_05_07_21_57_52/results_xregopose_seq_hm_direct_05_07_21_57_52')
hp_baseline_results = pd.read_pickle(r'/home/eddie/waterloo/lightning_logs/3d_plots/xregopose_05_07_21_47_14/raw_results_xregopose_05_07_21_47_14.pkl')
hp_baseline_results_files = pd.read_pickle(r'/home/eddie/waterloo/lightning_logs/3d_plots/xregopose_05_07_21_47_14/results_xregopose_05_07_21_47_14')
h2d_results = pd.read_pickle(r'/home/eddie/waterloo/models/xregopose_seq_hm_direct_05_08_16_21_05/handpicked_results_xregopose_seq_hm_direct_05_08_16_21_05.pkl')


seq_filenames = np.array(hp_seq_results_files['Filenames'])
baseline_filenames = np.array(hp_baseline_results_files['Filenames'])

output_file = '/home/eddie/waterloo/lightning_logs/3d_plots/skeleton_comparison'
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
                img = h2d_results[file]['img']

                mean=[0.485, 0.456, 0.406]
                std=[0.229, 0.224, 0.225]
                img = img[-1]

                img[0] = img[0]*std[0]+mean[0]
                img[1] = img[1]*std[1]+mean[1]
                img[2] = img[2]*std[2]+mean[2]
                if plot_img:
                    h2d = h2d_results[file]['p2d'] # 16 x 2
                    fig = plt.figure(num=1, clear=True, figsize=(9, 9))
                    ax = plt.axes()
                    ax.imshow(img.transpose((1, 2, 0)))
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

                        index1, index2 = bone[0], bone[1]

                        ax.plot(
                            [xs[index1], xs[index2]],
                            [ys[index1], ys[index2]],
                            linewidth=5,
                            color="gray",
                            zorder=1
                        )
                    if file == 'female_010_a_a_rgba_003845' or file == 'male_002_a_a_rgba_1447':
                        ignore_hands = [0, 1, 2, 3,  5, 6,  8, 9, 10, 11, 12, 13, 14, 15]
                        ax.scatter(xs[ignore_hands], ys[ignore_hands], color="gray", marker='o', s=100, zorder=2)
                    else:
                        ax.scatter(xs, ys, color="green", marker='o', s=100, zorder=2)
                    ax.axis('off')

                    fig.tight_layout()
                    # os.makedirs(output_file, exist_ok=True)
  
                    fig.savefig(output_file+f'{file}_img.jpg')

                    plot_img = False


                poses = [{"legend_name": "Dual-branch",
                        "plot_color": "red", 
                        "pose": baseline_pose},
                        {"legend_name": "Ego-STAN",
                        "plot_color": "green", 
                        "pose": seq_pose},
                        {"legend_name": "Ground Truth",
                        "plot_color": "gray", 
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
                                linewidth=1,
                                color=item["plot_color"],
                                label=item["legend_name"],
                            )
                        else:
                            ax.plot3D(
                                [xs[index1], xs[index2]],
                                [ys[index1], ys[index2]],
                                [zs[index1], zs[index2]],
                                linewidth=1,
                                color=item["plot_color"]
                            )
                    # draw joints
                    ax.scatter(xs, ys, zs, color=item["plot_color"])

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