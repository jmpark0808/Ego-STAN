import argparse
import datetime
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from train import DATALOADER_DIRECTORY, MODEL_DIRECTORY
from utils.evaluate import create_results_csv


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--model_checkpoint_file", required=True, type=str)
    parser.add_argument("--dataloader", required=True, default=None)
    parser.add_argument("--dataset_test", required=True, type=str)
    parser.add_argument("--cuda", default="cuda", choices=["cuda", "cpu"], type=str)
    parser.add_argument("--gpus", help="Number of gpus to use", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--output_directory", required=True, type=str)
    parser.add_argument(
        "--num_workers",
        help="# of dataloader cpu process",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--skip",
        help="# of images/frames to skip in between frames",
        default="0",
        type=int,
    )
    parser.add_argument(
        "--seq_len",
        help="# of images/frames input into sequential model",
        default="5",
        type=int,
    )
    parser.add_argument('--heatmap_type', help='Type of 2D ground truth heatmap, Defaults to "baseline"', 
                        default= 'baseline')



    dict_args = vars(parser.parse_args())

    # Create output directory
    os.makedirs(dict_args["output_directory"], exist_ok=True)

    # Deterministic
    pl.seed_everything(dict_args["seed"])

    # Initialize model to test
    assert dict_args["model"] in MODEL_DIRECTORY
    model = MODEL_DIRECTORY[dict_args["model"]](**dict_args)
    model = model.load_from_checkpoint(
        checkpoint_path=dict_args["model_checkpoint_file"],
        map_location=dict_args["cuda"],
    )

    # Data: load data module
    assert dict_args["dataloader"] in DATALOADER_DIRECTORY
    data_module = DATALOADER_DIRECTORY[dict_args["dataloader"]](**dict_args)
    logger = TensorBoardLogger(save_dir=dict_args['output_directory'], name='lightning_logs', log_graph=True)
    # Trainer: initialize training behaviour
    trainer = pl.Trainer(
        gpus=dict_args["gpus"],
        deterministic=True,
        logger=logger
    )

    trainer.test(model, datamodule=data_module)

    # Save: store test output results
    now = datetime.datetime.now().strftime("%m%d%H%M")
    test_mpjpe_dict = model.test_results
    print(test_mpjpe_dict)
    mpjpe_csv_path = os.path.join(
        dict_args["output_directory"],
        f"{now}_eval.csv",
    )
    create_results_csv(test_mpjpe_dict, mpjpe_csv_path)


if __name__ == "__main__":
    main()