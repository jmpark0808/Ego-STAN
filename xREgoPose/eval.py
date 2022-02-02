import argparse
import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import dataset.transform as trsf
from base import SetType
from dataset import Mocap
from net.xRNet import *
from utils import ConsoleLogger, config, evaluate, io

LOGGER = ConsoleLogger("Main")


def main():
    """Main"""

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--load", help="Directory of pre-trained model", required=True
    )
    parser.add_argument(
        "--dataset",
        help="Directory of your Dataset",
        required=True,
        default=None,
    )
    parser.add_argument(
        "--cuda",
        help="'cuda' for cuda, 'cpu' for cpu, default = cuda",
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--batch_size",
        help="batchsize, default = 1",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--output_file",
        help="output file to store evaluation. default = ./eval_results.json",
        default="./eval_results.json",
        type=str,
    )

    LOGGER.info("Starting evalution...")

    args = parser.parse_args()

    # Load evaluation dataset
    data_transform = transforms.Compose(
        [trsf.ImageTrsf(), trsf.Joints3DTrsf(), trsf.ToTensor()]
    )
    data = Mocap(args.dataset, SetType.TEST, transform=data_transform)
    data_loader = DataLoader(data, batch_size=args.batch_size)

    # Initialize evaluation pipline
    eval_body = evaluate.EvalBody()
    eval_upper = evaluate.EvalUpperBody()
    eval_lower = evaluate.EvalLowerBody()

    # Load model
    assert args.load is not None
    load = args.load
    model = xREgoPose().to(device=args.cuda)
    model.eval()

    state_dict_hm = torch.load(load, map_location=args.cuda)
    model.load_state_dict(state_dict_hm)
    LOGGER.info("Loading_Complete for {}".format(load))



    # Run evaluations on dataset
    with torch.no_grad():
        for iteration, (img, p2d, p3d, action) in enumerate(data_loader):

            LOGGER.info("Iteration: {}".format(iteration))
            LOGGER.info("Images: {}".format(img.shape))
            LOGGER.info("p2ds: {}".format(p2d.shape))
            LOGGER.info("p3ds: {}".format(p3d.shape))
            LOGGER.info("Actions: {}".format(action))

            # Forward pass on batch
            img = img.cuda()
            p3d = p3d.cuda()
            heatmap, p3d_hat, _ = model(img)
            heatmap = torch.sigmoid(heatmap)
            heatmap = heatmap.detach()


            # Evaluate results using different evaluation metrices
            y_output = p3d_hat.data.cpu().numpy()
            y_target = p3d.data.cpu().numpy()

            eval_body.eval(y_output, y_target, action)
            eval_upper.eval(y_output, y_target, action)
            eval_lower.eval(y_output, y_target, action)

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
