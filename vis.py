import argparse
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from train import DATALOADER_DIRECTORY, MODEL_DIRECTORY


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--model_checkpoint_file", required=True, type=str)
    parser.add_argument("--dataloader", required=True)
    parser.add_argument("--dataset_test", required=True, type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--output_directory", required=True, type=str)
    parser.add_argument("--cuda", default="cuda", choices=["cuda", "cpu"], type=str)
    parser.add_argument("--heatmap_type", required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument(
        "--skip",
        help="# of images/frames to skip in between frames",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--seq_len",
        help="# of images/frames input into sequential model",
        default=5,
        type=int,
    )

    dict_args = vars(parser.parse_args())

    # Create output directory
    img_dir = os.path.join(dict_args["output_directory"], "frames")
    os.makedirs(img_dir, exist_ok=True)

    # Data: load validation dataloader
    assert dict_args["dataloader"] in DATALOADER_DIRECTORY
    data_module = DATALOADER_DIRECTORY[dict_args["dataloader"]](**dict_args)
    test_dataloader = data_module.test_dataloader()

    # Initialize model to test
    assert dict_args["model"] in MODEL_DIRECTORY
    model = MODEL_DIRECTORY[dict_args["model"]](**dict_args)
    model = model.load_from_checkpoint(
        checkpoint_path=dict_args["model_checkpoint_file"],
        map_location=dict_args["cuda"],
    )
    model.eval()

    # Iterate through each batch to generate visuals
    print("[p] processing batches")
    for batch in tqdm(test_dataloader):
        img, p2d, p3d, action, img_path = batch

        p3d = p3d.cpu().numpy()
        pose = model(img).detach().numpy()

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
    file_destination = os.path.join(output_directory, video_name)
    os.makedirs(file_destination, exist_ok=True)

    frame_file_path = os.path.join(
        file_destination, f"{video_name}_{frame_count}_3d.png"
    )
    fig.savefig(frame_file_path)
    plt.close()


if __name__ == "__main__":
    main()
