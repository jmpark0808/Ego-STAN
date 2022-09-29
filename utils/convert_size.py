# ----------------------------------------------------------- #
#  This is code confidential, for peer-review purposes only   #
#  and protected under conference code of ethics              #
# ----------------------------------------------------------- #

import glob
import os
import shutil
from skimage import io as sio
from skimage.transform import resize
import json

root_dir = '/path/to/h36m/'
dest_dir = '/path/to/make_video/'

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import cv2

def save_skeleton(
    gt_pose: np.ndarray,
    img_filename: str,
    output_directory: str,
):
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection="3d")

    bone_links = [
                [0, 1], # Head -> Neck
                [1, 2], # Neck -> LShoulder
                [1, 5], # Neck -> RShoulder
                [2, 3], # LShoulder -> LElbow
                [3, 4], # LElbow -> LWrist
                [8, 9], # LHip -> LKnee
                [9, 10], # LKnee -> LFoot
                [5, 6], # RShoulder -> RElbow
                [6, 7], # REblow -> RWrist
                [11, 12], # RHip -> RKnee
                [12, 13], # RKnee -> RFoot
                [14, 11], # Hip -> RHip
                [14, 8], # Hip -> LHip
                [1, 15], # Neck -> Thorax
                [15, 16], # Thorax -> Spine
                [16, 14] # Spine -> Hip
            ]
    skeletons = [
        {"pose": gt_pose, "color": "blue"},
    ]

    for item in skeletons:
        pose = item["pose"]
        color = item["color"]
        xs = pose[:, 0]
        ys = pose[:, 1]
        zs = -pose[:, 2]

        # draw bones
        for bone in bone_links:
            index1, index2 = bone[0], bone[1]
            ax.plot3D(
                [xs[index1], xs[index2]],
                [ys[index1], ys[index2]],
                [zs[index1], zs[index2]],
                linewidth=1,
                color=color,
            )
        # draw joints
        ax.scatter(xs, ys, zs, color=color)

    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)
    ax.title.set_text(f"{img_filename}")
    plt.axis("off")
    fig.tight_layout()
    ax.view_init(elev=27.0, azim=41.0)
    frame_file_path = os.path.join(
        output_directory, f"{img_filename}_3d.png"
    )
    fig.savefig(frame_file_path)
    plt.close(fig)

    # update video name set for later video creation



def create_videos(input_frame_dir, output_video_dir):
    img_array = []
    file_pattern = os.path.join(input_frame_dir, f"*.png")
    file_list = sorted(glob.glob(file_pattern))
 
    size = None
    for filename in file_list:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    # create video object
    video_path = os.path.join(output_video_dir, f"video.avi")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"DIVX"), 15, size)

    # write frames to video
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

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

count = 0
for r, d, f in os.walk(root_dir):
    if len(f) == 0:
        continue
    elif f[0].endswith('.json'):
        continue
    elif f[0].endswith('.jpg'):
        jpgs = f
        root = r.split('h36m/')[-1]
        
        for file in jpgs:
            img = sio.imread(os.path.join(r, file))
            img = resize(img, (368, 368))
            sio.imsave(os.path.join(r, file), img)

    else:
        continue


