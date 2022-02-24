import argparse
import datetime
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger


from dataset.mocap import MocapDataModule
from dataset.mocap_transformer import MocapSeqDataModule
from net.DirectRegression import DirectRegression
from net.xRNetSeq import xREgoPoseSeq
from net.xRNetBaseLine import xREgoPose
from net.xRNetConcat import xRNetConcat
from net.xRNetHeatmap import xREgoPoseHeatMap
from net.xRNetSeqHM import xREgoPoseSeqHM

# Deterministic


MODEL_DIRECTORY = {
    "direct_regression": DirectRegression,
    "xregopose": xREgoPose,
    "xregopose_seq": xREgoPoseSeq,
    "xregopose_concat":xRNetConcat,
    "xregopose_heatmap": xREgoPoseHeatMap,
    "xregopose_seq_hm": xREgoPoseSeqHM
}
DATALOADER_DIRECTORY = {
    'baseline': MocapDataModule,
    'sequential': MocapSeqDataModule
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
    parser.add_argument('--val_freq', help='How often to run validation set within a training epoch, i.e. 0.25 will run 4 validation runs in 1 training epoch', default=0.1, type=float)
    parser.add_argument('--es_patience', help='Max # of consecutive validation runs w/o improvment', default=5, type=int)
    parser.add_argument('--logdir', help='logdir for models and losses. default = .', default='./', type=str)
    parser.add_argument('--lr', help='learning_rate for pose. default = 0.001', default=0.001, type=float)
    parser.add_argument('--lr_decay', help='Learning rate decrease by lr_decay time per decay_step, default = 0.1',
                        default=0.1, type=float)
    parser.add_argument('--decay_step', help='Learning rate decrease by lr_decay time per decay_step,  default = 7000',
                        default=1E100, type=int)
    parser.add_argument('--display_freq', help='Frequency to display result image on Tensorboard, in batch units',
                        default=64, type=int)
    parser.add_argument('--load_resnet', help='Directory of ResNet 101 weights', default=None)
    parser.add_argument('--hm_train_steps', help='Number of steps to pre-train heatmap predictor', default=100000, type=int)
    parser.add_argument('--seq_len', help="# of images/frames input into sequential model, default = 5",
                        default='5', type=int)
    parser.add_argument('--skip', help="# of images/frames to skip in between frames, default = 0",
                        default='0', type=int)
    parser.add_argument('--encoder_type', help='Type of encoder for concatenation, Defaults to "branch_concat"', 
                        default= 'branch_concat')
    parser.add_argument('--heatmap_type', help='Type of 2D ground truth heatmap, Defaults to "baseline"', 
                        default= 'baseline')
    parser.add_argument('--seed', help='Seed for reproduceability', 
                        default=42, type=int)

    args = parser.parse_args()
    dict_args = vars(args)
    
    pl.seed_everything(dict_args['seed'])
    # Initialize model to train
    assert dict_args['model'] in MODEL_DIRECTORY
    model = MODEL_DIRECTORY[dict_args['model']](**dict_args)

    # Callback: early stopping parameters
    early_stopping_callback = EarlyStopping(
        monitor="val_mpjpe_full_body",
        mode="min",
        verbose=True,
        patience=dict_args["es_patience"],
    )

    # Callback: model checkpoint strategy
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5, verbose=True, monitor="val_mpjpe_full_body", mode="min"
    )

    # Data: load data module
    assert dict_args['dataloader'] in DATALOADER_DIRECTORY
    data_module = DATALOADER_DIRECTORY[dict_args['dataloader']](**dict_args)

    # Trainer: initialize training behaviour
    profiler = SimpleProfiler()
    logger = TensorBoardLogger(save_dir=dict_args['logdir'], name='lightning_logs', log_graph=True)
    trainer = pl.Trainer(
        callbacks=[early_stopping_callback, checkpoint_callback],
        val_check_interval=dict_args['val_freq'],
        deterministic=True,
        gpus=dict_args['gpus'],
        profiler=profiler,
        logger=logger,
        max_epochs=dict_args["epoch"],
        log_every_n_steps=10
    )

    # Trainer: train model
    trainer.fit(model, data_module) 
