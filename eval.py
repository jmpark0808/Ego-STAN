import argparse
import csv
import datetime
import os

import pytorch_lightning as pl

from train import DATALOADER_DIRECTORY, MODEL_DIRECTORY


def create_results_csv(mpjpe_dict: dict, csv_path: str):
    """
    Save a csv of mpjpe evalutions stored in a dict.
    Refer to the `test_results` dict in DirectRegression.test_epoch_end
    for the expected structure for `mpjpe_dict`.
    """

    m_to_mm = 1000

    # get csv column names
    action_list = list(mpjpe_dict["Full Body"].keys())
    action_list.sort()
    columns = ["Evalution Error [mm]"]
    columns.extend(action_list)
    print(f"[print] columns: {columns}")

    with open(csv_path, mode="w") as f:
        mpjpe_writer = csv.writer(f)
        mpjpe_writer.writerow(columns)
        for body_split, action_dict in mpjpe_dict.items():
            # the first column is the body split (e.g. "Full Body")
            row = [body_split]
            row_std = [body_split + " Error STD"]
            # store mpjpe in order of sorted 'action_list'
            for action in action_list:
                row.append(action_dict[action]["mpjpe"] * m_to_mm)
                row_std.append(action_dict[action]["std_mpjpe"] * m_to_mm)

            mpjpe_writer.writerow(row)
            mpjpe_writer.writerow(row_std)


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

    # Trainer: initialize training behaviour
    trainer = pl.Trainer(
        gpus=dict_args["gpus"],
        deterministic=True,
    )

    trainer.test(model, datamodule=data_module)

    # Save: store test output results
    now = datetime.datetime.now().strftime("%m%d%H%M")
    test_mpjpe_dict = model.test_results
    print(test_mpjpe_dict)
    mpjpe_csv_path = os.path.join(
        dict_args["output_directory"],
        f"test_mpjpe_{dict_args['model']}_{now}.csv",
    )
    create_results_csv(test_mpjpe_dict, mpjpe_csv_path)


if __name__ == "__main__":
    main()
