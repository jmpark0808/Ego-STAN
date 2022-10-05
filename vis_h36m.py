import argparse
import glob
import os
import pathlib
from utils.evaluate import get_p3ds_t
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib
from utils.pck import get_max_preds

matplotlib.use('Agg')
from train import DATALOADER_DIRECTORY, MODEL_DIRECTORY

VIDEO_LIST = set()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model_static", required=True, type=str)
    parser.add_argument("--model_static_checkpoint_file", required=True, type=str)
    parser.add_argument("--model_seq", required=True, type=str)
    parser.add_argument("--model_seq_checkpoint_file", required=True, type=str)

    parser.add_argument("--dataloader", required=True)
    parser.add_argument("--dataset_val", required=True, type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--output_directory", required=True, type=str)
    parser.add_argument("--cuda", default="cuda", choices=["cuda", "cpu"], type=str)
    parser.add_argument("--heatmap_type", default='baseline')
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
    parser.add_argument("--heatmap_resolution",  help='2D heatmap resolution', nargs="*", type=int, default=[47, 47])
    parser.add_argument("--image_resolution",  help='Image resolution', nargs="*", type=int, default=[368, 368])
    parser.add_argument('--sigma', help='Sigma for heatmap generation', type=int, default=3)
    parser.add_argument('--protocol', help='Protocol for H36M, p1 for protocol 1 and p2 for protocol 2', type=str, default='p2')


    dict_args = vars(parser.parse_args())
    dict_args.update({"dropout":0})

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

    # Initialize model to test
    assert dict_args["model_static"] in MODEL_DIRECTORY
    model_static = MODEL_DIRECTORY[dict_args["model_static"]](**dict_args)
    model_static = model_static.load_from_checkpoint(
        checkpoint_path=dict_args["model_static_checkpoint_file"],
        map_location=dict_args["cuda"],
    )
    model_static = model_static.cuda()
    model_static.eval()

    assert dict_args["model_seq"] in MODEL_DIRECTORY
    model_seq = MODEL_DIRECTORY[dict_args["model_seq"]](**dict_args)
    model_seq = model_seq.load_from_checkpoint(
        checkpoint_path=dict_args["model_seq_checkpoint_file"],
        map_location=dict_args["cuda"],
    )
    model_seq = model_seq.cuda()
    model_seq.eval()

    # Iterate through each batch to generate visuals
    print("[p] processing batches")
    sx = 0
    num_skips = 0
    for i, batch in enumerate(tqdm(val_dataloader)):
        if i>num_skips:
            img, p2d, p3d, action = batch

            img = img.cuda()
            p2d = p2d.cuda()
            p2d = p2d[:, -1, :, :, :]

            pred_static = model_static(img[:, -1, :, :, :].cuda()).detach().cpu().numpy()
            pred_seq = model_seq(img).detach().cpu().numpy()


            print("[p] rendering skeletons")

            for idx in range(p2d.shape[0]):

                save_skeleton(
                    p2d[idx].detach().cpu().numpy(),
                    pred_static[idx],
                    pred_seq[idx],
                    img[idx, -1, :, :, :].detach().cpu().numpy(),
                    f'{sx}_{idx}',
                    img_dir,
                    dict_args
                )
            sx += 1
            if sx > 50:
                break
    create_videos(input_frame_dir=img_dir, output_video_dir=vid_dir)


def save_skeleton(
    gt_pose: np.ndarray,
    pred_pose_static: np.ndarray,
    pred_pose_seq: np.ndarray,
    img: np.ndarray,
    tag: str,
    output_directory: str,
    dict_args: dict,
):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img_plot = np.copy(img)

    img_plot[0, :, :] = img_plot[0, :, :]*std[0]+mean[0]
    img_plot[1, :, :] = img_plot[1, :, :]*std[1]+mean[1]
    img_plot[2, :, :] = img_plot[2, :, :]*std[2]+mean[2]
    ax[0].imshow(np.transpose(img_plot, axes=[1, 2, 0]))
    ax[1].imshow(np.transpose(img_plot, axes=[1, 2, 0]))

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
        {"pose": get_max_preds(np.expand_dims(pred_pose_static, 0))[0][0], "color": "red", 'legend': 'HRNet', 'ind': 0},
        {"pose": get_max_preds(np.expand_dims(pred_pose_seq, 0))[0][0], "color": "green", 'legend': 'Ego-STAN', 'ind': 1},
        {"pose": get_max_preds(np.expand_dims(gt_pose, 0))[0][0], "color": "goldenrod", 'legend': 'Ground Truth', 'ind': 2},
        
    ]

    for item in skeletons:
        pose = item["pose"]
        color = item["color"]
        ind = item['ind']
 
        xs = pose[:, 0]/dict_args["heatmap_resolution"][0]*dict_args["image_resolution"][0]
        ys = pose[:, 1]/dict_args["heatmap_resolution"][1]*dict_args["image_resolution"][1]
    
        if ind == 0:
            for bone in bone_links:
                index1, index2 = bone[0], bone[1]
                ax[0].plot(
                    [xs[index1], xs[index2]],
                    [ys[index1], ys[index2]],
                    linewidth=1,
                    color=color,
                )
            # draw joints
            ax[0].scatter(xs, ys, color=color, label=item['legend'])
        elif ind == 2:
            for bone in bone_links:
                index1, index2 = bone[0], bone[1]
                ax[0].plot(
                    [xs[index1], xs[index2]],
                    [ys[index1], ys[index2]],
                    linewidth=1,
                    color=color,
                )
                ax[1].plot(
                    [xs[index1], xs[index2]],
                    [ys[index1], ys[index2]],
                    linewidth=1,
                    color=color,
                )
            # draw joints
            ax[0].scatter(xs, ys, color=color, label=item['legend'])
            ax[1].scatter(xs, ys, color=color, label=item['legend'])
        else:
            for bone in bone_links:
                index1, index2 = bone[0], bone[1]
                ax[1].plot(
                    [xs[index1], xs[index2]],
                    [ys[index1], ys[index2]],
                    linewidth=1,
                    color=color,
                )
            # draw joints
            ax[1].scatter(xs, ys, color=color, label=item['legend'])
            

    # ax[0].legend()
    # ax[1].legend()
    
    ax[0].axis("off")
    ax[1].axis("off")
    fig.tight_layout()
    
    frame_count = int(tag.split("_")[0])*dict_args["batch_size"] + int(tag.split("_")[1])
    frame_file_path = os.path.join(
        output_directory, '{0:06}.png'.format(frame_count)
    )
    fig.savefig(frame_file_path)
    

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
    video_path = os.path.join(output_video_dir, f"h36m_video.avi")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"DIVX"), 15, size)

    # write frames to video
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == "__main__":
    main()
    print(f"[p] VIDEO_LIST = {VIDEO_LIST}")
