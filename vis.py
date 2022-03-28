import argparse
import glob
import os
import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from train import DATALOADER_DIRECTORY, MODEL_DIRECTORY

VIDEO_LIST = set()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataloader", required=True)
    parser.add_argument("--dataset_val", required=True, type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--output_directory", required=True, type=str)
    parser.add_argument("--heatmap_type", required=True)
    parser.add_argument("--num_workers", type=int, required=True)

    dict_args = vars(parser.parse_args())

    # Create output directory
    img_dir = os.path.join(dict_args["output_directory"], "frames")
    vid_dir = os.path.join(dict_args["output_directory"], "videos")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)

    # Data: load validation dataloader
    print("[p] getting val_dataloader")
    assert dict_args["dataloader"] in DATALOADER_DIRECTORY
    data_module = DATALOADER_DIRECTORY[dict_args["dataloader"]](**dict_args)
    val_dataloader = data_module.val_dataloader()

    # Iterate through each batch to generate visuals
    print("[p] processing batches")
    for batch in tqdm(val_dataloader):
        img, p2d, p3d, action, img_path = batch

        p3d = p3d.cpu().numpy()
        pose = p3d

        print("[p] rendering skeletons")
        for idx in range(p3d.shape[0]):
            filename = pathlib.Path(img_path[idx]).stem
            # Remove periods in filename
            filename = str(filename).replace(".", "_")
            save_skeleton(
                p3d[idx],
                pose[idx],
                filename,
                action[idx],
                img_dir,
            )

    create_videos(input_frame_dir=img_dir, output_video_dir=vid_dir)


def save_skeleton(
    gt_pose: np.ndarray,
    pred_pose: np.ndarray,
    img_filename: str,
    action: str,
    output_directory: str,
):
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection="3d")

    bone_links = [
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
    skeletons = [
        {"pose": gt_pose, "color": "blue"},
        {"pose": pred_pose, "color": "green"},
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

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.title.set_text(f"{img_filename}")
    plt.axis("off")
    fig.tight_layout()
    ax.view_init(elev=27.0, azim=41.0)
    frame_count = img_filename.split("_")[-1]
    video_name = "_".join(img_filename.split("_")[:-1]) + f"_{action}"
    frame_file_path = os.path.join(
        output_directory, f"{video_name}_{frame_count}_3d.png"
    )
    fig.savefig(frame_file_path)
    plt.close()

    # update video name set for later video creation
    VIDEO_LIST.add(video_name)


def create_videos(input_frame_dir, output_video_dir):
    for count, video_name in enumerate(VIDEO_LIST):
        img_array = []
        file_pattern = os.path.join(input_frame_dir, f"{video_name}*.png")
        file_list = sorted(glob.glob(file_pattern))
        print(
            f"[p][{count+1}/{len(VIDEO_LIST)}] creating '{video_name}' video with {len(file_list)} files"
        )

        size = None
        for filename in file_list:
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        # create video object
        video_path = os.path.join(output_video_dir, f"{video_name}.avi")
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"DIVX"), 15, size)

        # write frames to video
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


if __name__ == "__main__":
    main()
    print(f"[p] VIDEO_LIST = {VIDEO_LIST}")
