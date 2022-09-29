# ----------------------------------------------------------- #
#  This is code confidential, for peer-review purposes only   #
#  and protected under conference code of ethics              #
# ----------------------------------------------------------- #

import argparse
import datetime
import os
import random
import time
from re import X
from xxlimited import Str

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger

from dataset.mo2cap2 import Mo2Cap2DataModule
from dataset.mocap import MocapDataModule
from dataset.mo2cap2 import Mo2Cap2DataModule
from dataset.mocap_distance import MocapDistanceDataModule
from dataset.mocap_h36m_cropped_hm import MocapH36MCropHMDataModule
from dataset.mocap_h36m_hm import MocapH36MHMDataModule
from dataset.mocap_transformer import MocapSeqDataModule
from dataset.mo2cap2_transformer import Mo2Cap2SeqDataModule
from dataset.mocap_h36m import MocapH36MDataModule
from dataset.mocap_h36m_cropped import MocapH36MCropDataModule
from dataset.mocap_h36m_transformer import MocapH36MSeqDataModule
from dataset.mocap_h36m_cropped_transformer import MocapH36MCropSeqDataModule
from dataset.mocap_h36m_2d import Mocap2DH36MDataModule

from net.DirectRegression import DirectRegression
from net.HRNetBaseline import HRNetBaseline
from net.HRNetEgo import HRNetEgoSTAN
from net.Mo2Cap2BaselineL1 import Mo2Cap2BaselineL1
from net.Mo2Cap2Direct import Mo2Cap2Direct
from net.Mo2Cap2GlobalTrans import Mo2Cap2GlobalTrans
from net.Mo2Cap2Seq import Mo2Cap2Seq
from net.Mo2Cap2SeqHMDirect import Mo2Cap2SeqHMDirect
from net.Mo2Cap2SeqHMDirectAvg import Mo2Cap2SeqHMDirectAvg
from net.Mo2Cap2SeqHMDirectSlice import Mo2Cap2SeqHMDirectSlice
from net.xRNetBaseLine2D import xREgoPose2D
from net.xRNetBaseLineL1 import xREgoPoseL1
from net.xRNetDirect import xREgoPoseDirect
from net.Mo2Cap2Baseline import Mo2Cap2Baseline
from net.xRNetPosterior2D import xREgoPosePosterior2D
from net.xRNetPosteriorLinear import xREgoPosePosteriorLinear
from net.xRNetSeq import xREgoPoseSeq
from net.xRNetBaseLine import xREgoPose
from net.xRNetConcat import xRNetConcat
from net.xRNetHeatmap import xREgoPoseHeatMap
from net.xRNetSeqHM import xREgoPoseSeqHM
from net.xRNetPosterior import xREgoPosePosterior
from net.xRNetPosteriorDist import xREgoPosePosteriorDist
from net.xRNetSeqDirect import xREgoPoseSeqDirect
from net.xRNetSeqHMDirect import xREgoPoseSeqHMDirect
from net.xRNetGlobalTrans import xREgoPoseGlobalTrans
from net.xRNetDist import xREgoPoseDist
from net.xRNetSeqHMDirectAvg import xREgoPoseSeqHMDirectAvg
from net.xRNetSeqHMDirectED import xREgoPoseSeqHMDirectED
from net.xRNetSeqHMDirectEDExp import xREgoPoseSeqHMDirectEDExp
from net.xRNetSeqHMDirectRevPos import xREgoPoseSeqHMDirectRevPos
from net.xRNetSeqHMDirectSlice import xREgoPoseSeqHMDirectSlice
from net.xRNetUNet import xREgoPoseUNet
from utils.evaluate import create_results_csv

# Deterministic

MODEL_DIRECTORY = {
    "direct_regression": DirectRegression,
    "xregopose": xREgoPose,
    "xregopose_l1": xREgoPoseL1,
    "xregopose_seq": xREgoPoseSeq,
    "xregopose_concat":xRNetConcat,
    "xregopose_heatmap": xREgoPoseHeatMap,
    "xregopose_seq_hm": xREgoPoseSeqHM,
    "xregopose_posterior": xREgoPosePosterior,
    "xregopose_posterior_2d": xREgoPosePosterior2D,
    "xregopose_posterior_dist": xREgoPosePosteriorDist,
    "xregopose_posterior_linear": xREgoPosePosteriorLinear,
    "xregopose_seq_hm_direct": xREgoPoseSeqHMDirect,
    "xregopose_seq_hm_direct_ed": xREgoPoseSeqHMDirectED,
    "xregopose_seq_hm_direct_ed_exp": xREgoPoseSeqHMDirectEDExp,
    "xregopose_seq_direct": xREgoPoseSeqDirect,
    "xregopose_global_trans": xREgoPoseGlobalTrans,
    "xregopose_dist": xREgoPoseDist,
    "xregopose_unet": xREgoPoseUNet,
    "xregopose_direct": xREgoPoseDirect,
    "xregopose_seq_hm_direct_rev_pos": xREgoPoseSeqHMDirectRevPos,
    "xregopose_seq_hm_direct_avg": xREgoPoseSeqHMDirectAvg,
    "xregopose_seq_hm_direct_slice": xREgoPoseSeqHMDirectSlice,
    "mo2cap2": Mo2Cap2Baseline,
    "mo2cap2_l1": Mo2Cap2BaselineL1,
    "mo2cap2_direct": Mo2Cap2Direct,
    "mo2cap2_global_trans": Mo2Cap2GlobalTrans,
    "mo2cap2_seq": Mo2Cap2Seq,
    "mo2cap2_slice": Mo2Cap2SeqHMDirectSlice,
    "mo2cap2_avg": Mo2Cap2SeqHMDirectAvg,
    "mo2cap2_ego": Mo2Cap2SeqHMDirect,
    "xregopose_2d": xREgoPose2D,
    "HRNetBaseline": HRNetBaseline,
    "HRNetEgoSTAN": HRNetEgoSTAN


}
DATALOADER_DIRECTORY = {
    'baseline': MocapDataModule,
    'sequential': MocapSeqDataModule,
    'distance': MocapDistanceDataModule,
    'mo2cap2': Mo2Cap2DataModule,
    'mo2cap2_seq': Mo2Cap2SeqDataModule,
    'h36m_static': MocapH36MDataModule,
    'h36m_seq' : MocapH36MSeqDataModule,
    'h36m_2d' : Mocap2DH36MDataModule,
    'h36m_hm': MocapH36MHMDataModule,
    'h36m_crop': MocapH36MCropDataModule,
    'h36m_crop_hm': MocapH36MCropHMDataModule,
    'h36m_seq_crop': MocapH36MCropSeqDataModule,
} 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='Model name to train', required=True, default=None)
    parser.add_argument('--eval', help='Whether to test model on the best iteration after training'
                        , action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--dataloader', help="Type of dataloader", required=True, default=None)
    parser.add_argument("--load",
                        help="Directory of pre-trained model weights only,  \n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model", default=None)
    parser.add_argument("--resume_from_checkpoint",
                        help="Directory of pre-trained checkpoint including hyperparams,  \n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model", default=None)
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
    parser.add_argument("--heatmap_resolution",  help='2D heatmap resolution', nargs="*", type=int, default=[47, 47])
    parser.add_argument("--image_resolution",  help='Image resolution', nargs="*", type=int, default=[368, 368])
    parser.add_argument('--seed', help='Seed for reproduceability', 
                        default=42, type=int)
    parser.add_argument('--clip_grad_norm', help='Clipping gradient norm, 0 means no clipping', type=float, default=0.)
    parser.add_argument('--dropout', help='Dropout for transformer', type=float, default=0.)
    parser.add_argument('--dropout_linear', help='Dropout for linear layers in 2D to 3D', type=float, default=0.)
    parser.add_argument('--protocol', help='Protocol for H36M, p1 for protocol 1 and p2 for protocol 2', type=str, default='p2')
    parser.add_argument('--w2c', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--weight_regularization', help='Weight regularization hyperparameter', type=float, default=0.01)
    parser.add_argument('--monitor_metric', help='Which metric to monitor for early stopping', type=str, default='val_mpjpe_full_body')
    parser.add_argument('--sigma', help='Sigma for heatmap generation', type=int, default=3)
    parser.add_argument('--h36m_sample_rate', help='Sample rate for h36m', type=int, default=1)
    parser.add_argument('--csv_mode', help='CSV results mode, 2D or 3D', type=str, default='3D')
    args = parser.parse_args()
    dict_args = vars(args)

    pl.seed_everything(dict_args['seed'])
    # Initialize model to train
    assert dict_args['model'] in MODEL_DIRECTORY
    if dict_args['load'] is not None:
        model = MODEL_DIRECTORY[dict_args['model']].load_from_checkpoint(dict_args['load'], **dict_args)
    else:
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
        monitor=dict_args['monitor_metric'],
        mode="min",
        verbose=True,
        patience=dict_args["es_patience"],
    )

    # Callback: model checkpoint strategy
    checkpoint_callback = ModelCheckpoint(
        dirpath=weight_save_dir, save_top_k=5, verbose=True, monitor=dict_args['monitor_metric'], mode="min"
    )

    # Data: load data module
    assert dict_args['dataloader'] in DATALOADER_DIRECTORY
    data_module = DATALOADER_DIRECTORY[dict_args['dataloader']](**dict_args)

    # Trainer: initialize training behaviour
    profiler = SimpleProfiler()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(save_dir=dict_args['logdir'], version=now, name='lightning_logs', log_graph=True)
    if dict_args['gpus'] > 1:
        accelerator = 'dp'
    elif dict_args['gpus'] == 1:
        accelerator = 'gpu'
    elif dict_args['gpus'] == 0:
        accelerator = 'cpu'
    trainer = pl.Trainer(
        callbacks=[early_stopping_callback, checkpoint_callback, lr_monitor],
        val_check_interval=dict_args['val_freq'],
        deterministic=True,
        gpus=dict_args['gpus'],
        profiler=profiler,
        logger=logger,
        max_epochs=dict_args["epoch"],
        log_every_n_steps=10,
        gradient_clip_val=dict_args['clip_grad_norm'],
        resume_from_checkpoint=dict_args['resume_from_checkpoint'],
        accelerator=accelerator
    )

    # Trainer: train model
    trainer.fit(model, data_module)

    # Evaluate model on best ckpt (defined in 'ModelCheckpoint' callback)
    if dict_args['eval'] and dict_args['dataset_test']:
        trainer.test(model, ckpt_path='best', datamodule=data_module)
        test_mpjpe_dict = model.test_results
        mpjpe_csv_path = os.path.join(weight_save_dir, f'{now}_eval.csv')
        # Store mpjpe test results as a csv
        create_results_csv(test_mpjpe_dict, mpjpe_csv_path, dict_args['dataloader'], dict_args['csv_mode'])
    else:
        print("Evaluation skipped")
