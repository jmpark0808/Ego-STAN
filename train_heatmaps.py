import argparse
import datetime
import os
import random
import time
from re import X

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger

from dataset.mpii import MPIIDataModule
from dataset.lsp import LSPDataModule
from net.Mo2Cap2Heatmap import Mo2Cap2Heatmap

MODEL_DIRECTORY = {
    "mo2cap2_heatmap": Mo2Cap2Heatmap,
}
DATALOADER_DIRECTORY = {
    'mpii': MPIIDataModule,
    'lsp': LSPDataModule,
} 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='Model name to train', required=True, default=None)
    parser.add_argument('--dataloader', help="Type of dataloader", required=True, default=None)
    parser.add_argument("--load",
                        help="Directory of pre-trained model,  \n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model")
    parser.add_argument('--dataset_tr', help='Directory of your train Dataset', required=True, default=None)
    parser.add_argument('--dataset_val', help='Directory of your validation Dataset', required=True, default=None)
    parser.add_argument('--dataset_test', help='Directory of your test Dataset', default=None)
    parser.add_argument('--cuda', help="'cuda' for cuda, 'cpu' for cpu, default = cuda",
                        default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--gpus', help="Number of gpus to use for training", default=1, type=int)
    parser.add_argument('--batch_size', help="batchsize, default = 1", default=1, type=int)
    parser.add_argument('--epoch', help='# of epochs. default = 20', default=20, type=int)
    parser.add_argument('--num_workers', help="# of dataloader cpu process", default=0, type=int)
    parser.add_argument('--es_patience', help='Max # of consecutive validation runs w/o improvment', default=5, type=int)
    parser.add_argument('--logdir', help='logdir for models and losses. default = .', default='./', type=str)
    parser.add_argument('--lr', help='learning_rate for pose. default = 0.001', default=0.001, type=float)
    parser.add_argument('--display_freq', help='Frequency to display result image on Tensorboard, in batch units',
                        default=64, type=int)
    parser.add_argument('--load_resnet', help='Directory of ResNet 101 weights', default=None)
    parser.add_argument('--seq_len', help="# of images/frames input into sequential model, default = 5",
                        default='5', type=int)
    parser.add_argument('--skip', help="# of images/frames to skip in between frames, default = 0",
                        default='0', type=int)
    parser.add_argument("--heatmap_resolution",  help='2D heatmap resolution', nargs="*", type=int, default=[47, 47])
    parser.add_argument("--image_resolution",  help='Image resolution', nargs="*", type=int, default=[368, 368])
    parser.add_argument('--seed', help='Seed for reproduceability', 
                        default=42, type=int)
    parser.add_argument('--clip_grad_norm', help='Clipping gradient norm, 0 means no clipping', type=float, default=0.)
    parser.add_argument('--dropout', help='Dropout for transformer', type=float, default=0.)

    args = parser.parse_args()
    dict_args = vars(args)

    pl.seed_everything(dict_args['seed'])
    # Initialize model to train
    assert dict_args['model'] in MODEL_DIRECTORY
    model = MODEL_DIRECTORY[dict_args['model']](**dict_args)

    # Initialize logging paths
    random_sec = random.randint(1, 20)
    time.sleep(random_sec)
    now = datetime.datetime.now().strftime('%m%d%H%M%S')
    weight_save_dir = os.path.join(dict_args["logdir"], os.path.join('models', 'state_dict', now))
    while os.path.exists(weight_save_dir):
        random_sec = random.randint(1, 20)
        time.sleep(random_sec)
        now = datetime.datetime.now().strftime('%m%d%H%M%S')
        weight_save_dir = os.path.join(dict_args["logdir"], os.path.join('models', 'state_dict', now))

    os.makedirs(weight_save_dir, exist_ok=True)


    # Callback: early stopping parameters
    early_stopping_callback = EarlyStopping(
        monitor="Total HM loss",
        mode="min",
        verbose=True,
        patience=dict_args["es_patience"],
    )

    # Callback: model checkpoint strategy
    checkpoint_callback = ModelCheckpoint(
        dirpath=weight_save_dir, save_top_k=5, verbose=True, monitor="Total HM loss", mode="min"
    )

    # Data: load data module
    assert dict_args['dataloader'] in DATALOADER_DIRECTORY
    data_module = DATALOADER_DIRECTORY[dict_args['dataloader']](**dict_args)

    # Trainer: initialize training behaviour
    profiler = SimpleProfiler()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(save_dir=dict_args['logdir'], version=now, name='lightning_logs', log_graph=True)
    trainer = pl.Trainer(
        callbacks=[early_stopping_callback, checkpoint_callback, lr_monitor],
        deterministic=True,
        gpus=dict_args['gpus'],
        profiler=profiler,
        logger=logger,
        max_epochs=dict_args["epoch"],
        log_every_n_steps=10,
        gradient_clip_val=dict_args['clip_grad_norm']
    )

    # Trainer: train model
    trainer.fit(model, data_module)
