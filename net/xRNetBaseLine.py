# -*- coding: utf-8 -*-

import pytorch_lightning as pl
import torch
import torch.nn as nn
from utils import evaluate
from net.blocks import *
import numpy as np
import pathlib



class xREgoPose(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # parameters
        self.batch_size = kwargs.get("batch_size")
        self.lr = kwargs.get("lr")
        self.lr_decay = kwargs.get("lr_decay")
        self.decay_step = kwargs.get("decay_step")
        self.load_resnet = kwargs.get("load_resnet")
        self.hm_train_steps = kwargs.get("hm_train_steps")
        self.es_patience = kwargs.get("es_patience")
        self.which_data = kwargs.get('dataloader')
        self.protocol = kwargs.get('protocol')
        if self.which_data in ['baseline', 'sequential'] :
            num_class = 16
        elif self.which_data == 'mo2cap2':
            num_class = 15
        elif self.which_data in ['h36m_static', 'h36m_seq']:
            num_class = 17

        # must be defined for logging computational graph
        self.example_input_array = torch.rand((1, 3, 368, 368))

        # Generator that produces the HeatMap
        self.heatmap = HeatMap(num_class)
        # Encoder that takes 2D heatmap and transforms to latent vector Z
        self.encoder = Encoder(num_class)
        # Pose decoder that takes latent vector Z and transforms to 3D pose coordinates
        self.pose_decoder = PoseDecoder(num_classes = num_class)
        # Heatmap decoder that takes latent vector Z and generates the original 2D heatmap
        self.heatmap_decoder = HeatmapDecoder(num_class)

        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody(mode=self.which_data, protocol=self.protocol)
        # self.eval_upper = evaluate.EvalUpperBody(mode=self.which_data, protocol=self.protocol)
        # self.eval_lower = evaluate.EvalLowerBody(mode=self.which_data, protocol=self.protocol)
        # self.eval_per_joint = evaluate.EvalPerJoint(mode=self.which_data, protocol=self.protocol)

        # Initialize total validation pose loss
        self.val_loss_3d_pose_total = torch.tensor(0., device=self.device)
        self.val_loss_hm = torch.tensor(0., device=self.device)

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

    def auto_encoder_loss(self, pose_pred, pose_label, hm_decoder, hm_resnet):
        """
        Defining the loss funcition:
        """
        lambda_p = 0.1
        lambda_theta = -0.01
        lambda_L = 0.5
        lambda_hm = 0.001
        pose_l2norm = torch.sum(torch.sum(torch.pow(pose_pred-pose_label, 2), dim=2), dim=1)
        # pose_l2norm = torch.sqrt(torch.sum(torch.sum(torch.pow(pose_pred-pose_label, 2), dim=2), dim=1))
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        cosine_similarity_error = torch.sum(cos(pose_pred, pose_label), dim=1)
        limb_length_error = torch.sum(torch.sqrt(torch.sum(torch.pow(pose_pred-pose_label, 2), dim=2)), dim=1)
        heatmap_error = torch.sum(torch.pow(hm_resnet.view(hm_resnet.size(0), -1) - hm_decoder.view(hm_decoder.size(0), -1), 2), dim=1)
        # limb_length_error = torch.sum(torch.sum(torch.abs(pose_pred-pose_label), dim=2), dim=1)
        # heatmap_error = torch.sqrt(torch.sum(torch.pow(hm_resnet.reshape(hm_resnet.size(0), -1) - hm_decoder.reshape(hm_decoder.size(0), -1), 2), dim=1))
        LAE_pose = lambda_p*(pose_l2norm + lambda_theta*cosine_similarity_error + lambda_L*limb_length_error)
        LAE_hm = lambda_hm*heatmap_error
        return torch.mean(LAE_pose), torch.mean(LAE_hm)

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        
        optimizer = torch.optim.SGD(
        self.parameters(), lr=self.lr, momentum=0.9, nesterov=True
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=self.es_patience-3,
            min_lr=1e-8,
            verbose=True)
        
        return optimizer
      

    def forward(self, x, gt_heatmap=None):
        """
        Forward pass through model
        :param x: Input image
        :return: 2D heatmap, 16x3 joint inferences, 2D reconstructed heatmap
        """
        # x = 3 x 368 x 368

        heatmap = self.heatmap(x)

        # heatmap = 15 x 47 x 47
        
        if gt_heatmap is not None:
            z = self.encoder(gt_heatmap)
        else:
            z = self.encoder(heatmap)
        # z = 20

        pose = self.pose_decoder(z)
        # pose = 16 x 3

        generated_heatmaps = self.heatmap_decoder(z)

        # generated_heatmaps = 15 x 47 x 47

        return heatmap, pose, generated_heatmaps

    def training_step(self, batch, batch_idx):
        """
        Compute and return the training loss
        logging resources:
        https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
        """
        tensorboard = self.logger.experiment
        img, p2d, p3d, action = batch
        img = img.cuda()
        p2d = p2d.cuda()
        p3d = p3d.cuda()
        if self.which_data in ['h36m_static', 'h36m_seq']:
            p3d[:, 14, :] = 0
        # forward pass
        


        if self.iteration <= self.hm_train_steps:
            heatmap, pose, generated_heatmap = self.forward(img)
            heatmap = torch.sigmoid(heatmap)
            hm_loss = self.mse(heatmap, p2d)
            loss = hm_loss
            self.log('Total HM loss', hm_loss.item())
        else:
            heatmap, pose, generated_heatmap = self.forward(img)
            heatmap = torch.sigmoid(heatmap)
            generated_heatmap = torch.sigmoid(generated_heatmap)
            hm_loss = self.mse(heatmap, p2d)
            loss_3d_pose, loss_2d_ghm = self.auto_encoder_loss(pose, p3d, generated_heatmap, heatmap)
            ae_loss = loss_2d_ghm + loss_3d_pose
            loss = hm_loss + ae_loss
            self.log('Total HM loss', hm_loss.item())
            self.log('Total 3D loss', loss_3d_pose.item())
            self.log('Total GHM loss', loss_2d_ghm.item())
     
        # calculate mpjpe loss
        mpjpe = torch.mean(torch.sqrt(torch.sum(torch.pow(p3d - pose, 2), dim=2)))
        mpjpe_std = torch.std(torch.sqrt(torch.sum(torch.pow(p3d - pose, 2), dim=2)))
        self.log("train_mpjpe_full_body", mpjpe)
        self.log("train_mpjpe_std", mpjpe_std)
        self.iteration += 1
        if batch_idx % 2500 == 0:
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            img_plot = img.clone()
            img_plot[:, 0, :, :] = img_plot[:, 0, :, :]*std[0]+mean[0]
            img_plot[:, 1, :, :] = img_plot[:, 1, :, :]*std[1]+mean[1]
            img_plot[:, 2, :, :] = img_plot[:, 2, :, :]*std[2]+mean[2]
            tensorboard.add_images('TR Images', img_plot, self.iteration)
            tensorboard.add_images('TR Ground Truth 2D Heatmap', torch.clip(torch.sum(p2d, dim=1, keepdim=True), 0, 1), self.iteration)
            tensorboard.add_images('TR Predicted 2D Heatmap', torch.clip(torch.sum(heatmap, dim=1, keepdim=True), 0, 1), self.iteration)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Compute the metrics for validation batch
        validation loop: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
        """
 
        tensorboard = self.logger.experiment
        img, p2d, p3d, action = batch
        img = img.cuda()
        p2d = p2d.cuda()
        p3d = p3d.cuda()
        if self.which_data in ['h36m_static', 'h36m_seq']:
            p3d[:, 14, :] = 0
        # forward pass  
        heatmap, pose, generated_heatmap = self.forward(img)
        heatmap = torch.sigmoid(heatmap)
        generated_heatmap = torch.sigmoid(generated_heatmap)
        # calculate pose loss
        val_hm_loss = self.mse(heatmap, p2d)
        val_loss_3d_pose, _ = self.auto_encoder_loss(pose, p3d, generated_heatmap, heatmap)
        # update 3d pose loss
        self.val_loss_hm += val_hm_loss
        self.val_loss_3d_pose_total += val_loss_3d_pose

        # Evaluate mpjpe
        y_output = pose.data.cpu().numpy()
        y_target = p3d.data.cpu().numpy()
        self.eval_body.eval(y_output, y_target, action)
        # self.eval_upper.eval(y_output, y_target, action)
        # self.eval_lower.eval(y_output, y_target, action)

        # Log images if needed
        if batch_idx == 0:
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            img_plot = img.clone()
            img_plot[:, 0, :, :] = img_plot[:, 0, :, :]*std[0]+mean[0]
            img_plot[:, 1, :, :] = img_plot[:, 1, :, :]*std[1]+mean[1]
            img_plot[:, 2, :, :] = img_plot[:, 2, :, :]*std[2]+mean[2]
            tensorboard.add_images('Val Images', img_plot, self.iteration)
            tensorboard.add_images('Val Ground Truth 2D Heatmap', torch.clip(torch.sum(p2d, dim=1, keepdim=True), 0, 1), self.iteration)
            tensorboard.add_images('Val Predicted 2D Heatmap', torch.clip(torch.sum(heatmap, dim=1, keepdim=True), 0, 1), self.iteration)
 

        return val_loss_3d_pose

    def on_validation_start(self):
        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody(mode=self.which_data, protocol=self.protocol)
        # self.eval_upper = evaluate.EvalUpperBody(mode=self.which_data, protocol=self.protocol)
        # self.eval_lower = evaluate.EvalLowerBody(mode=self.which_data, protocol=self.protocol)

        # Initialize total validation pose loss
        self.val_loss_3d_pose_total = torch.tensor(0., device=self.device)
        self.val_loss_hm = torch.tensor(0., device=self.device)

    def validation_epoch_end(self, validation_step_outputs):
        val_mpjpe = self.eval_body.get_results()
        # val_mpjpe_upper = self.eval_upper.get_results()
        # val_mpjpe_lower = self.eval_lower.get_results()
        if self.iteration >= self.hm_train_steps:
            self.log("val_mpjpe_full_body", val_mpjpe["All"]["mpjpe"])
            self.log("val_mpjpe_full_body_std", val_mpjpe["All"]["std_mpjpe"])
            # self.log("val_mpjpe_upper_body", val_mpjpe_upper["All"]["mpjpe"])
            # self.log("val_mpjpe_lower_body", val_mpjpe_lower["All"]["mpjpe"])
            self.log("val_loss", self.val_loss_3d_pose_total)
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
        img, p2d, p3d, action = batch
        img = img.cuda()
        p2d = p2d.cuda()
        p3d = p3d.cuda()
        if self.which_data in ['h36m_static', 'h36m_seq']:
            p3d[:, 14, :] = 0
        # forward pass
        heatmap, pose, generated_heatmap = self.forward(img)
   
        # Evaluate mpjpe
        y_output = pose.data.cpu().numpy()
        y_target = p3d.data.cpu().numpy()
        self.eval_body.eval(y_output, y_target, action)
        self.eval_upper.eval(y_output, y_target, action)
        self.eval_lower.eval(y_output, y_target, action)
        self.eval_per_joint.eval(y_output, y_target)
        # filenames = []
        # for idx in range(y_target.shape[0]):

        #     filename = pathlib.Path(img_path[idx]).stem
        #     filename = str(filename).replace(".", "_")
        #     filenames.append(filename)
        # self.eval_samples.eval(y_output, y_target, action, filenames)

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
            "Per Joint": test_mpjpe_per_joint
        }



if __name__ == "__main__":
    pass
