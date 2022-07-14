# -*- coding: utf-8 -*-

import pytorch_lightning as pl
import torch
import torch.nn as nn
from utils import evaluate
from net.blocks import *
import os


class xREgoPosePosterior(pl.LightningModule):
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
        self.example_input_array = torch.rand((1, 17, 2))

        # Generator that produces the HeatMap
        self.hm2pose = HM2Pose(num_class)

        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody(mode=self.which_data, protocol=self.protocol)
        # self.eval_upper = evaluate.EvalUpperBody()
        # self.eval_lower = evaluate.EvalLowerBody()

        # Initialize total validation pose loss
        self.val_loss_3d_pose_total = torch.tensor(0., device=self.device)

        # def weight_init(m):
        #     """
        #     Xavier Initialization
        #     """
        #     if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             torch.nn.init.zeros_(m.bias)
    
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)

        # Initialize weights
        self.apply(weight_init)

        self.iteration = 0
        self.save_hyperparameters()
        self.test_results = {}
    
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
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=self.es_patience-3,
            min_lr=1e-8,
            verbose=True)
        return optimizer
      

    def forward(self, x):
        """
        Forward pass through model

        :param x: Input image

        :return: 2D heatmap, 16x3 joint inferences, 2D reconstructed heatmap
        """
        # x = 15, 47, 47

        pose = self.hm2pose(x)
        # pose = 16 x 3

        return pose


    def training_step(self, batch, batch_idx):
        """
        Compute and return the training loss
        logging resources:
        https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html

        """
        tensorboard = self.logger.experiment
        img, p2d, p3d, action = batch
        p2d = p2d.cuda()
        p3d = p3d.cuda()
        if self.which_data in ['h36m_static', 'h36m_seq']:
            p3d[:, 14, :] = 0
        # forward pass
        pose = self.forward(p2d)

        loss_3d_pose = self.auto_encoder_loss(pose, p3d)
        self.log('Total 3D loss', loss_3d_pose.item())
     
        # calculate mpjpe loss
        mpjpe = torch.mean(torch.sqrt(torch.sum(torch.pow(p3d - pose, 2), dim=2)))
        mpjpe_std = torch.std(torch.sqrt(torch.sum(torch.pow(p3d - pose, 2), dim=2)))
        self.log("train_mpjpe_full_body", mpjpe)
        self.log("train_mpjpe_std", mpjpe_std)
        self.iteration += 1

        if self.iteration % 2500 == 0 and self.protocol in ['p1', 'p2'] \
        and self.which_data in ['h36m_static', 'h36m_seq']:
            y_output = pose.data.cpu().numpy()
            y_target = p3d.data.cpu().numpy()
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            img_plot = img.clone()
            img_plot[:, 0, :, :] = img_plot[:, 0, :, :]*std[0]+mean[0]
            img_plot[:, 1, :, :] = img_plot[:, 1, :, :]*std[1]+mean[1]
            img_plot[:, 2, :, :] = img_plot[:, 2, :, :]*std[2]+mean[2]
            skel_dir = os.path.join(self.logger.log_dir, 'skel_plots')
            if not os.path.exists(skel_dir):
                os.mkdir(skel_dir)

            # Get the procrustes aligned 3D Pose and log
            if self.protocol == 'p1':
                fig_compare_preds = evaluate.plot_skels_compare( p3ds_1 = y_output, p3ds_2 = y_target,
                                label_1 = 'Pred Raw', label_2 = 'Ground Truth', 
                                savepath = os.path.join(skel_dir, 'train_pred_raw_vs_GT.png'), dataset='h36m')
            elif self.protocol == 'p2':
                y_output = evaluate.p_mpjpe(y_output, y_target, False)
                fig_compare_preds = evaluate.plot_skels_compare( p3ds_1 = y_output, p3ds_2 = y_target,
                                label_1 = 'Pred PA', label_2 = 'Ground Truth', 
                                savepath = os.path.join(skel_dir, 'train_pred_PA_vs_GT.png'), dataset='h36m')
            else:
                raise('Not a valid protocol')
            

            # Tensorboard log images
            tensorboard.add_images('TR Image', img_plot, self.iteration)
            tensorboard.add_figure('TR GT 3D Skeleton vs Predicted 3D Skeleton', fig_compare_preds, global_step = self.iteration)




        return loss_3d_pose

    def validation_step(self, batch, batch_idx):
        """
        Compute the metrics for validation batch
        validation loop: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
        """
        tensorboard = self.logger.experiment
        img, p2d, p3d, action = batch
        p2d = p2d.cuda()
        p3d = p3d.cuda()
        if self.which_data in ['h36m_static', 'h36m_seq']:
            p3d[:, 14, :] = 0
        # forward pass
        pose = self.forward(p2d)

        loss_3d_pose = self.auto_encoder_loss(pose, p3d)
        self.val_loss_3d_pose_total += loss_3d_pose

        # Evaluate mpjpe
        y_output = pose.data.cpu().numpy()
        y_target = p3d.data.cpu().numpy()
        self.eval_body.eval(y_output, y_target, action)
        # self.eval_upper.eval(y_output, y_target, action)
        # self.eval_lower.eval(y_output, y_target, action)

        if batch_idx == 0 and self.protocol in ['p1', 'p2'] \
        and self.which_data in ['h36m_static', 'h36m_seq']:
            y_output = pose.data.cpu().numpy()
            y_target = p3d.data.cpu().numpy()
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            img_plot = img.clone()
            img_plot[:, 0, :, :] = img_plot[:, 0, :, :]*std[0]+mean[0]
            img_plot[:, 1, :, :] = img_plot[:, 1, :, :]*std[1]+mean[1]
            img_plot[:, 2, :, :] = img_plot[:, 2, :, :]*std[2]+mean[2]
            skel_dir = os.path.join(self.logger.log_dir, 'skel_plots')
            if not os.path.exists(skel_dir):
                os.mkdir(skel_dir)

            # Get the procrustes aligned 3D Pose and log
            if self.protocol == 'p1':
                fig_compare_preds = evaluate.plot_skels_compare( p3ds_1 = y_output, p3ds_2 = y_target,
                                label_1 = 'Pred Raw', label_2 = 'Ground Truth', 
                                savepath = os.path.join(skel_dir, 'val_pred_raw_vs_GT.png'), dataset='h36m')
            elif self.protocol == 'p2':
                y_output = evaluate.p_mpjpe(y_output, y_target, False)
                fig_compare_preds = evaluate.plot_skels_compare( p3ds_1 = y_output, p3ds_2 = y_target,
                                label_1 = 'Pred PA', label_2 = 'Ground Truth', 
                                savepath = os.path.join(skel_dir, 'val_pred_PA_vs_GT.png'), dataset='h36m')
            else:
                raise('Not a valid protocol')
            

            # Tensorboard log images
            tensorboard.add_images('Val Image', img_plot, self.iteration)
            tensorboard.add_figure('Val GT 3D Skeleton vs Predicted 3D Skeleton', fig_compare_preds, global_step = self.iteration)



        return loss_3d_pose

    def on_validation_start(self):
        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody(mode=self.which_data, protocol=self.protocol)
        # self.eval_upper = evaluate.EvalUpperBody()
        # self.eval_lower = evaluate.EvalLowerBody()

        # Initialize total validation pose loss
        self.val_loss_3d_pose_total = torch.tensor(0., device=self.device)

    def validation_epoch_end(self, validation_step_outputs):
        val_mpjpe = self.eval_body.get_results()
        # val_mpjpe_upper = self.eval_upper.get_results()
        # val_mpjpe_lower = self.eval_lower.get_results()

        self.log("val_mpjpe_full_body", val_mpjpe["All"]["mpjpe"])
        self.log("val_mpjpe_full_body_std", val_mpjpe["All"]["std_mpjpe"])
        # self.log("val_mpjpe_upper_body", val_mpjpe_upper["All"]["mpjpe"])
        # self.log("val_mpjpe_lower_body", val_mpjpe_lower["All"]["mpjpe"])
        self.log("val_loss", self.val_loss_3d_pose_total)
        self.scheduler.step(val_mpjpe["All"]["mpjpe"])

    def on_test_start(self):
        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody(mode=self.which_data, protocol=self.protocol)
        self.eval_upper = evaluate.EvalUpperBody(mode=self.which_data, protocol=self.protocol)
        self.eval_lower = evaluate.EvalLowerBody(mode=self.which_data, protocol=self.protocol)
        self.eval_per_joint = evaluate.EvalPerJoint(mode=self.which_data, protocol=self.protocol)
        
    def test_step(self, batch, batch_idx):
        img, p2d, p3d, action = batch
        p2d = p2d.cuda()
        p3d = p3d.cuda()
        if self.which_data in ['h36m_static', 'h36m_seq']:
            p3d[:, 14, :] = 0
        # forward pass
        pose = self.forward(p2d)
   
        # Evaluate mpjpe
        y_output = pose.data.cpu().numpy()
        y_target = p3d.data.cpu().numpy()
        self.eval_body.eval(y_output, y_target, action)
        self.eval_upper.eval(y_output, y_target, action)
        self.eval_lower.eval(y_output, y_target, action)
        self.eval_per_joint.eval(y_output, y_target)


    def test_epoch_end(self, test_step_outputs):
        test_mpjpe = self.eval_body.get_results()
        test_mpjpe_upper = self.eval_upper.get_results()
        test_mpjpe_lower = self.eval_lower.get_results()
        test_mpjpe_per_joint = self.eval_per_joint.get_results()

        self.test_results = {
            "Full Body": test_mpjpe,
            "Upper Body": test_mpjpe_upper,
            "Lower Body": test_mpjpe_lower,
            "Per Joint": test_mpjpe_per_joint
        }
  
                    

if __name__ == "__main__":
    pass
