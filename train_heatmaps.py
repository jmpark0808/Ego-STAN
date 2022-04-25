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

DATALOADER_DIRECTORY = {
    'mpii': MPIIDataModule,
    'lsp': MSPDataModule,
} 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dataloader', help="Type of dataloader", required=True, default=None)
    parser.add_argument('--dataset_tr', help='Directory of your train Dataset', required=True, default=None)
    parser.add_argument('--dataset_val', help='Directory of your validation Dataset', required=True, default=None)
    parser.add_argument('--dataset_test', help='Directory of your test Dataset', default=None)

    args = parser.parse_args()
    dict_args = vars(args)

    # Data: load data module
    assert dict_args['dataloader'] in DATALOADER_DIRECTORY
    data_module = DATALOADER_DIRECTORY[dict_args['dataloader']](**dict_args)