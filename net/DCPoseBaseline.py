# -*- coding: utf-8 -*-

import pytorch_lightning as pl
import torch
import torch.nn as nn
from utils import evaluate
from pose_models.hrnet import get_pose_net
from utils.hrnet_config import cfg
from utils.meter import AverageMeterList
from utils.pck import accuracy
import numpy as np
import os
from utils import evaluate
import torch.nn.functional as F
from net.blocks import *
from pose_models.DCPose_main.posetimation.zoo.build import build_model
from pose_models.DCPose_main.posetimation.config.defaults import _C

class DCPose(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # parameters
        self.batch_size = kwargs.get("batch_size")
        self.lr = kwargs.get("lr")
        self.es_patience = kwargs.get("es_patience")
        self.which_data = kwargs.get('dataloader')
        self.heatmap_resolution = kwargs.get('heatmap_resolution')
        self.image_resolution = kwargs.get('image_resolution')
        self.hm_train_steps = kwargs.get("hm_train_steps")
        self.seq_len = kwargs.get('seq_len')
        self.dropout = kwargs.get('dropout')
        self.protocol = kwargs.get('protocol')
        self.weight_regularization = kwargs.get('weight_regularization')
        self.dropout_linear = kwargs.get('dropout_linear')
        if self.which_data in ['baseline', 'sequential'] :
            num_class = 16
        elif self.which_data == 'mo2cap2':
            num_class = 15
        elif self.which_data.startswith('h36m'):
            num_class = 17
        

        # must be defined for logging computational graph
        # self.example_input_array = torch.rand((1, self.seq_len*3, self.image_resolution[0], self.image_resolution[1]))
        cfg = _C.clone()
        cfg.merge_from_file('pose_models/DCPose_main/configs/posetimation/DcPose/posetrack18/model_RSN.yaml')

        # Generator that produces the HeatMap
        self.model = build_model(cfg, phase='train')
        self.regression = HM2Pose(num_class, 47, self.dropout_linear)

        # Initialize the mpjpe evaluation pipeline
        self.val_loss_3d_pose_total = torch.tensor(0., device=self.device)
        self.val_loss_hm = torch.tensor(0., device=self.device)
        self.iteration = 0
        self.test_iteration = 0

        def weight_init(m):
            """
            Xavier Initialization
            """
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        # Initialize weights
        self.apply(weight_init)
        self.save_hyperparameters()
        self.test_results = {}

    def mse(self, pred, label):
        pred = pred.reshape(pred.size(0), -1)
        label = label.reshape(label.size(0), -1)
        return torch.sum(torch.mean(torch.pow(pred-label, 2), dim=1))

    def auto_encoder_loss(self, pose_pred, pose_label):
        """
        Defining the loss funcition:
        """
        lambda_p = 0.1
        lambda_theta = -0.01
        lambda_L = 0.5
        pose_l2norm = torch.sqrt(torch.sum(torch.sum(torch.pow(pose_pred-pose_label, 2), dim=2), dim=1))
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        cosine_similarity_error = torch.sum(cos(pose_pred, pose_label), dim=1)
        limb_length_error = torch.sum(torch.sum(torch.abs(pose_pred-pose_label), dim=2), dim=1)
        LAE_pose = lambda_p*(pose_l2norm + lambda_theta*cosine_similarity_error + lambda_L*limb_length_error)
        return torch.mean(LAE_pose)

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        
        optimizer = torch.optim.AdamW(
        self.parameters(), lr=self.lr 
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=self.es_patience-3,
            min_lr=1e-8,
            verbose=True)
        
        return optimizer
    
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # skip the first 500 steps
        if self.trainer.global_step < int(self.hm_train_steps/self.batch_size):
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / int(self.hm_train_steps/self.batch_size))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
      

    def forward(self, x, margin):
        """
        Forward pass through model
        :param x: Input image
        :return: 2D heatmap, 16x3 joint inferences, 2D reconstructed heatmap
        """
        # x = 3 x 368 x 368
        _, pose_2d = self.model(x, margin=margin)
        pose_2d_int = F.interpolate(pose_2d, (47, 47))
        pose_3d = self.regression(pose_2d_int)
        return pose_2d_int, pose_3d

    def training_step(self, batch, batch_idx):
        """
        Compute and return the training loss
        logging resources:
        https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
        """
        tensorboard = self.logger.experiment
        img, p2d, p3d, action, img_path = batch
        img = img.cuda()
        p2d = p2d.cuda()
        p2d = p2d[:, -2, :, :, :]
        p3d = p3d.cuda()
        p3d = p3d[:, -2, :, :]
        
        img_p = img[:, 0, :, :, :]
        img_c = img[:, 1, :, :, :]
        img_n = img[:, 2, :, :, :]

        concat_input = torch.cat((img_c, img_p, img_n), 1).cuda()
        margin = torch.stack([torch.tensor(1).unsqueeze(0), torch.tensor(1).unsqueeze(0)], dim=1).cuda()
        # forward pass
        pred_hm, pred_3d = self.forward(concat_input, margin)


        if self.iteration <= self.hm_train_steps:
            loss = self.mse(pred_hm, p2d)
            self.log('Total HM loss', loss.item())
        else:
            hm_loss = self.mse(pred_hm, p2d)
            loss_3d_pose = self.auto_encoder_loss(pred_3d, p3d)
            loss = hm_loss + loss_3d_pose
            self.log('Total HM loss', hm_loss.item())
            self.log('Total 3D loss', loss_3d_pose.item())

     
        mpjpe = torch.mean(torch.sqrt(torch.sum(torch.pow(p3d - pred_3d, 2), dim=2)))
        mpjpe_std = torch.std(torch.sqrt(torch.sum(torch.pow(p3d - pred_3d, 2), dim=2)))
        self.log("train_mpjpe_full_body", mpjpe)
        self.log("train_mpjpe_std", mpjpe_std)
        self.iteration += 1
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Compute the metrics for validation batch
        validation loop: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
        """
 
        tensorboard = self.logger.experiment
        img, p2d, p3d, action, img_path = batch
        img = img.cuda()
        p2d = p2d.cuda()
        p2d = p2d[:, -2, :, :, :]
        p3d = p3d.cuda()
        p3d = p3d[:, -2, :, :]
        
        img_p = img[:, 0, :, :, :]
        img_c = img[:, 1, :, :, :]
        img_n = img[:, 2, :, :, :]

        concat_input = torch.cat((img_c, img_p, img_n), 1).cuda()
        margin = torch.stack([torch.tensor(1).unsqueeze(0), torch.tensor(1).unsqueeze(0)], dim=1).cuda()


        pred_hm, pred_3d = self.forward(concat_input, margin)
        val_hm_loss = self.mse(pred_hm, p2d)
        val_loss_3d_pose = self.auto_encoder_loss(pred_3d, p3d)

        # update 3d pose loss
        self.val_loss_hm += val_hm_loss
        self.val_loss_3d_pose_total += val_loss_3d_pose

        # Evaluate mpjpe
        y_output = pred_3d.data.cpu().numpy()
        y_target = p3d.data.cpu().numpy()
        self.eval_body.eval(y_output, y_target, action)

        self.num_batches += 1
        return val_loss_3d_pose

    def on_validation_start(self):
        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody(mode=self.which_data, protocol=self.protocol)
        # self.eval_upper = evaluate.EvalUpperBody(mode=self.which_data, protocol=self.protocol)
        # self.eval_lower = evaluate.EvalLowerBody(mode=self.which_data, protocol=self.protocol)

        # Initialize total validation pose loss
        self.val_loss_3d_pose_total = torch.tensor(0., device=self.device)
        self.val_loss_hm = torch.tensor(0., device=self.device)
        self.num_batches = 0

    def validation_epoch_end(self, validation_step_outputs):
        val_mpjpe = self.eval_body.get_results()
        # val_mpjpe_upper = self.eval_upper.get_results()
        # val_mpjpe_lower = self.eval_lower.get_results()
        if self.iteration >= self.hm_train_steps:
            self.log("val_mpjpe_full_body", val_mpjpe["All"]["mpjpe"])
            self.log("val_mpjpe_full_body_std", val_mpjpe["All"]["std_mpjpe"])
            # self.log("val_mpjpe_upper_body", val_mpjpe_upper["All"]["mpjpe"])
            # self.log("val_mpjpe_lower_body", val_mpjpe_lower["All"]["mpjpe"])
            self.log("val_loss_3d", self.val_loss_3d_pose_total/self.num_batches)
            self.log("val_loss_2d", self.val_loss_hm/self.num_batches)
            self.scheduler.step(val_mpjpe["All"]["mpjpe"])
        else:
            self.log("val_mpjpe_full_body", 0.3-0.01*(self.iteration/self.hm_train_steps))
            self.scheduler.step(0.3-0.01*(self.iteration/self.hm_train_steps))

    def on_test_start(self):
        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody(mode=self.which_data, protocol=self.protocol)
        self.eval_upper = evaluate.EvalUpperBody(mode=self.which_data, protocol=self.protocol)
        self.eval_lower = evaluate.EvalLowerBody(mode=self.which_data, protocol=self.protocol)
        self.eval_per_joint = evaluate.EvalPerJoint(mode=self.which_data, protocol=self.protocol)
        # self.eval_samples = evaluate.EvalSamples()
        self.filenames = []

    def test_step(self, batch, batch_idx):

        img, p2d, p3d, action, img_path = batch
        img = img.cuda()
        p2d = p2d.cuda()
        p2d = p2d[:, -2, :, :, :]
        p3d = p3d.cuda()
        p3d = p3d[:, -2, :, :]
        
        img_p = img[:, 0, :, :, :]
        img_c = img[:, 1, :, :, :]
        img_n = img[:, 2, :, :, :]

        concat_input = torch.cat((img_c, img_p, img_n), 1).cuda()
        margin = torch.stack([torch.tensor(1).unsqueeze(0), torch.tensor(1).unsqueeze(0)], dim=1).cuda()
        
        pred_hm, pred_3d = self.forward(concat_input, margin)

        y_output = pred_3d.data.cpu().numpy()
        y_target = p3d.data.cpu().numpy()
        self.eval_body.eval(y_output, y_target, action)
        self.eval_upper.eval(y_output, y_target, action)
        self.eval_lower.eval(y_output, y_target, action)
        self.eval_per_joint.eval(y_output, y_target)
        self.test_iteration += img.size(0)

    def test_epoch_end(self, test_step_outputs):
        test_mpjpe = self.eval_body.get_results()
        test_mpjpe_upper = self.eval_upper.get_results()
        test_mpjpe_lower = self.eval_lower.get_results()
        self.test_raw_p2ds = {'preds': self.eval_per_joint.pds, 'gts': self.eval_per_joint.gts}
        test_mpjpe_per_joint = self.eval_per_joint.get_results()
        # self.test_mpjpe_samples = self.eval_samples.error
        self.test_results = {
            "Full Body": test_mpjpe,
            "Upper Body": test_mpjpe_upper,
            "Lower Body": test_mpjpe_lower,
            "Per Joint": test_mpjpe_per_joint,
        }  



if __name__ == "__main__":
    pass
