import argparse
import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import dataset.transform as trsf
from base import SetType
from dataset import Mocap
from network import *
from utils import ConsoleLogger, config, evaluate, io

LOGGER = ConsoleLogger("Main")


def main():
    """Main"""

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--load_hm",
        help="Directory of pre-trained model for heatmap,  \n"
        "None --> Do not use pre-trained model. Training will start from random initialized model",
    )
    parser.add_argument(
        "--load_pose",
        help="Directory of pre-trained model for pose estimator,  \n"
        "None --> Do not use pre-trained model. Training will start from random initialized model",
    )
    parser.add_argument(
        "--dataset", help="Directory of your Dataset", required=True, default=None
    )
    parser.add_argument(
        "--cuda",
        help="'cuda' for cuda, 'cpu' for cpu, default = cuda",
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--batch_size", help="batchsize, default = 1", default=1, type=int
    )
    parser.add_argument(
        "--output_file",
        help="output file to store evaluation. default = ./eval_results.json",
        default="./eval_results.json",
        type=str,
    )

    LOGGER.info("Starting evalution...")

    args = parser.parse_args()
    device = torch.device(args.cuda)
    batch_size = args.batch_size

    # ------------------- Setup data loader

    data_transform = transforms.Compose(
        [
            trsf.ImageTrsf(),
            trsf.Joints3DTrsf(),
            trsf.ToTensor(),
        ]
    )

    # let's load data from validation set as example
    data = Mocap(args.dataset, SetType.VAL, transform=data_transform)
    data_loader = DataLoader(data, batch_size=args.batch_size)

    # ------------------- Setup evalution pipeline

    eval_body = evaluate.EvalBody()
    eval_upper = evaluate.EvalUpperBody()
    eval_lower = evaluate.EvalUpperBody()

    # ------------------- Load models
    load_hm = args.load_hm
    load_pose = args.load_pose
    start_iter = 0
    model_hm = HeatMap().to(device=args.cuda)
    # TODO: model_hm.eval()
    model_hm.eval()  # was model_hm.train()
    model_pose = PoseEstimator().to(device=args.cuda)

    # Xavier Initialization
    def weight_init(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model_hm.resnet101.apply(weight_init)
    model_hm.heatmap_deconv.apply(weight_init)

    model_pose.encoder.apply(weight_init)
    model_pose.pose_decoder.apply(weight_init)
    model_pose.heatmap_decoder.apply(weight_init)

    now = datetime.datetime.now()
    start_epo = 0

    if load_hm is not None:
        state_dict_hm = torch.load(load_hm, map_location=args.cuda)

        start_iter = int(load_hm.split("epo_")[1].strip("step.ckpt"))
        start_epo = int(load_hm.split("/")[-1].split("epo")[0])
        now = datetime.datetime.strptime(load_hm.split("/")[-2], "%m%d%H%M")

        LOGGER.info("Loading Model from {}".format(load_hm))
        LOGGER.info("Start_iter : {}".format(start_iter))
        LOGGER.info("now : {}".format(now.strftime("%m%d%H%M")))
        model_hm.load_state_dict(state_dict_hm)
        LOGGER.info("Loading_Complete")

    if load_pose is not None:
        state_dict_pose = torch.load(load_pose, map_location=args.cuda)

        start_iter = int(load_pose.split("epo_")[1].strip("step.ckpt"))
        start_epo = int(load_pose.split("/")[-1].split("epo")[0])
        now = datetime.datetime.strptime(load_pose.split("/")[-2], "%m%d%H%M")

        LOGGER.info("Loading Model from {}".format(load_pose))
        LOGGER.info("Start_iter : {}".format(start_iter))
        LOGGER.info("now : {}".format(now.strftime("%m%d%H%M")))
        model_pose.load_state_dict(state_dict_pose)
        LOGGER.info("Loading_Complete")

    model_pose.eval()
    # ------------------- Read dataset frames -------------------

    with torch.no_grad():
        for it, (img, p2d, p3d, action) in enumerate(data_loader):

            LOGGER.info("Iteration: {}".format(it))
            LOGGER.info("Images: {}".format(img.shape))
            LOGGER.info("p2ds: {}".format(p2d.shape))
            LOGGER.info("p3ds: {}".format(p3d.shape))
            LOGGER.info("Actions: {}".format(action))

            # -----------------------------------------------------------
            # ------------------- Run your model here -------------------
            # -----------------------------------------------------------
            img = img.cuda()
            p3d = p3d.cuda()
            heatmap = model_hm(img)
            heatmap = torch.sigmoid(heatmap)
            heatmap = heatmap.detach()
            generated_heatmap, p3d_hat = model_pose(heatmap)

            # Evaluate results using different evaluation metrices
            y_output = p3d_hat.data.cpu().numpy()
            y_target = p3d.data.cpu().numpy()

            eval_body.eval(y_output, y_target, action)
            eval_upper.eval(y_output, y_target, action)
            eval_lower.eval(y_output, y_target, action)

    # ------------------- Save results -------------------

    LOGGER.info("Saving evaluation results...")
    res = {
        "FullBody": eval_body.get_results(),
        "UpperBody": eval_upper.get_results(),
        "LowerBody": eval_lower.get_results(),
    }

    io.write_json(args.output_file, res)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
