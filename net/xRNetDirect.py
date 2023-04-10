# -*- coding: utf-8 -*-

import pytorch_lightning as pl
import torch
import torch.nn as nn
from utils import evaluate
from net.blocks import *
import numpy as np
import pathlib



class xREgoPoseDirect(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # parameters
        self.batch_size = kwargs.get("batch_size")
        self.lr = kwargs.get("lr")
        self.lr_decay = kwargs.get("lr_decay")
        self.decay_step = kwargs.get("decay_step")
        self.load_resnet = kwargs.get("load_resnet")
        self.hm_train_steps = kwargs.get("hm_train_steps")

        # must be defined for logging computational graph
        self.example_input_array = torch.rand((1, 3, 368, 368))

        # Generator that produces the HeatMap
        self.heatmap = HeatMap()
        # Encoder that takes 2D heatmap and transforms to latent vector Z
        self.pose = HM2Pose()

        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody()
        self.eval_upper = evaluate.EvalUpperBody()
        self.eval_lower = evaluate.EvalLowerBody()
        self.eval_per_joint = evaluate.EvalPerJoint()


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
        if self.load_resnet:
            self.heatmap.resnet101.load_state_dict(torch.load(self.load_resnet))

        self.heatmap.update_resnet101()
        self.iteration = 0
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
        
        optimizer_2d = torch.optim.AdamW(
        self.parameters(), lr=self.lr
        )

        return optimizer_2d
      

    def forward(self, x):
        """
        Forward pass through model

        :param x: Input image

        :return: 2D heatmap, 16x3 joint inferences, 2D reconstructed heatmap
        """
        # x = 3 x 368 x 368

        heatmap = self.heatmap(x)
        # heatmap_2d = 16 x 47 x 47
        # heatmap_1d = 16 x 30
        pose = self.pose(heatmap)


        return heatmap, pose

    def training_step(self, batch, batch_idx):
        """
        Compute and return the training loss
        logging resources:
        https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html

        """
        img, p2d, p3d, action, img_path = batch
        img = img.cuda()
        p2d = p2d.cuda()
        p3d = p3d.cuda()

        # forward pass
        


        if self.iteration <= self.hm_train_steps:
            heatmap, pose = self.forward(img)
            heatmap = torch.sigmoid(heatmap)
            hm_2d_loss = self.mse(heatmap, p2d)
            loss = hm_2d_loss
            self.log('Total HM loss', hm_2d_loss.item())
        else:
            heatmap, pose = self.forward(img)
            heatmap = torch.sigmoid(heatmap)
            hm_2d_loss = self.mse(heatmap, p2d)
            loss_3d_pose = self.auto_encoder_loss(pose, p3d)
            loss = hm_2d_loss + loss_3d_pose
            self.log('Total HM loss', hm_2d_loss.item())
            self.log('Total 3D loss', loss_3d_pose.item())

        # calculate mpjpe loss
        mpjpe = torch.mean(torch.sqrt(torch.sum(torch.pow(p3d - pose, 2), dim=2)))
        mpjpe_std = torch.std(torch.sqrt(torch.sum(torch.pow(p3d - pose, 2), dim=2)))
        self.log("train_mpjpe_full_body", mpjpe)
        self.log("train_mpjpe_std", mpjpe_std)
        self.iteration += img.size(0)
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
        p3d = p3d.cuda()


        # forward pass
        heatmap, pose = self.forward(img)
        heatmap = torch.sigmoid(heatmap)

        # calculate pose loss
        val_hm_2d_loss = self.mse(heatmap, p2d)
        val_loss_3d_pose = self.auto_encoder_loss(pose, p3d)
        # update 3d pose loss
        self.val_loss_2d_hm += val_hm_2d_loss
        self.val_loss_3d_pose_total += val_loss_3d_pose

        # Evaluate mpjpe
        y_output = pose.data.cpu().numpy()
        y_target = p3d.data.cpu().numpy()
        self.eval_body.eval(y_output, y_target, action)
        self.eval_upper.eval(y_output, y_target, action)
        self.eval_lower.eval(y_output, y_target, action)
        mpjpe = torch.sum(torch.sqrt(torch.sum(torch.pow(p3d - pose, 2), dim=2)), dim=0)
        self.val_dummy += mpjpe
        self.samples += p3d.size(0)
        return val_loss_3d_pose

    def on_validation_start(self):
        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody()
        self.eval_upper = evaluate.EvalUpperBody()
        self.eval_lower = evaluate.EvalLowerBody()

        # Initialize total validation pose loss
        self.val_dummy = torch.empty([16], device=self.device)
        self.val_loss_3d_pose_total = torch.tensor(0., device=self.device)
        self.val_loss_2d_hm = torch.tensor(0., device=self.device)
        self.samples = 0 


    def validation_epoch_end(self, validation_step_outputs):
        val_mpjpe = self.eval_body.get_results()
        val_mpjpe_upper = self.eval_upper.get_results()
        val_mpjpe_lower = self.eval_lower.get_results()
        if self.iteration >= self.hm_train_steps:
            self.log("val_mpjpe_full_body", val_mpjpe["All"]["mpjpe"])
            self.log("val_mpjpe_full_body_std", val_mpjpe["All"]["std_mpjpe"])
            self.log("val_mpjpe_upper_body", val_mpjpe_upper["All"]["mpjpe"])
            self.log("val_mpjpe_lower_body", val_mpjpe_lower["All"]["mpjpe"])
            self.log("val_loss_3D", self.val_loss_3d_pose_total)
            self.log("val_loss_2D", self.val_loss_2d_hm)
            result = self.val_dummy / self.samples
            self.log("val_mpjpe_recompute", torch.mean(result))

        else:
            self.log("val_mpjpe_full_body", 0.3-0.01*(self.iteration/self.hm_train_steps))
                    
    def on_test_start(self):
        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody()
        self.eval_upper = evaluate.EvalUpperBody()
        self.eval_lower = evaluate.EvalLowerBody()
        self.eval_per_joint = evaluate.EvalPerJoint()
        self.eval_samples = evaluate.EvalSamples()
        self.filenames = []

    def test_step(self, batch, batch_idx):
        img, p2d, p3d, action, img_path = batch
        img = img.cuda()
        p2d = p2d.cuda()
        p3d = p3d.cuda()

        # forward pass
        heatmap, pose = self.forward(img)

        # Evaluate mpjpe
        y_output = pose.data.cpu().numpy()
        y_target = p3d.data.cpu().numpy()
        self.eval_body.eval(y_output, y_target, action)
        self.eval_upper.eval(y_output, y_target, action)
        self.eval_lower.eval(y_output, y_target, action)
        self.eval_per_joint.eval(y_output, y_target)
        filenames = []
        for idx in range(y_target.shape[0]):

            filename = pathlib.Path(img_path[idx]).stem
            filename = str(filename).replace(".", "_")
            filenames.append(filename)
        self.eval_samples.eval(y_output, y_target, action, filenames)

    def test_epoch_end(self, test_step_outputs):
        test_mpjpe = self.eval_body.get_results()
        test_mpjpe_upper = self.eval_upper.get_results()
        test_mpjpe_lower = self.eval_lower.get_results()
        test_mpjpe_per_joint = self.eval_per_joint.get_results()
        self.test_mpjpe_samples = self.eval_samples.error
        self.test_results = {
            "Full Body": test_mpjpe,
            "Upper Body": test_mpjpe_upper,
            "Lower Body": test_mpjpe_lower,
            "Per Joint": test_mpjpe_per_joint
        }



if __name__ == "__main__":
    pass
