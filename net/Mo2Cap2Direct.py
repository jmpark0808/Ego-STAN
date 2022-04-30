# -*- coding: utf-8 -*-

import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
from utils import evaluate
from net.blocks import *



class Mo2Cap2Direct(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # parameters
        self.batch_size = kwargs.get("batch_size")
        self.lr = kwargs.get("lr")
        self.lr_decay = kwargs.get("lr_decay")
        self.decay_step = kwargs.get("decay_step")
        self.load_resnet = kwargs.get("load_resnet")
        self.hm_train_steps = kwargs.get("hm_train_steps")
        self.es_patience = kwargs.get('es_patience')
        self.which_data = kwargs.get('dataloader')
        # must be defined for logging computational graph
        self.example_input_array = torch.rand((1, 3, 368, 368))
        num_class = 15
        # Generator that produces the HeatMap
        self.heatmap = HeatMap(num_class)
        # Encoder that takes 2D heatmap and transforms to latent vector Z
        self.pose = HM2Pose(num_class)

        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody(mode=self.which_data)
        self.eval_upper = evaluate.EvalUpperBody(mode=self.which_data)
        self.eval_lower = evaluate.EvalLowerBody(mode=self.which_data)
        self.eval_per_joint = evaluate.EvalPerJoint(mode=self.which_data)


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
        self.heatmap.update_resnet101()

        if self.load_resnet:
            pretrained_dict = torch.load(self.load_resnet)
            model_dict = self.heatmap.resnet101.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            self.heatmap.resnet101.load_state_dict(model_dict)


        
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

        layer_names = []
        for idx, (name, param) in enumerate(self.heatmap.resnet101.named_parameters()):
            layer_names.append(name)
            
        grouped_parameters = []


        # store params & learning rates
        threshold = False
        for idx, name in enumerate(layer_names):
            # append layer parameters
            if 'layer3.15' in name:
                threshold = True

            if not threshold:
                grouped_parameters += [{'params': [p for n, p in self.heatmap.resnet101.named_parameters() if n == name and p.requires_grad],
                                'lr': self.lr*0.001}]
            else:
                grouped_parameters += [{'params': [p for n, p in self.heatmap.resnet101.named_parameters() if n == name and p.requires_grad],
                                'lr': self.lr}]
            

        grouped_parameters += [
            {"params": self.heatmap.heatmap_deconv.parameters()},
            {"params": self.pose.parameters()},
        ]

        optimizer = torch.optim.AdamW(
        grouped_parameters, lr=self.lr
        )
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
        # x = 3 x 368 x 368

        heatmap = self.heatmap(x)
        pose = self.pose(heatmap)


        return heatmap, pose

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
            loss = loss_3d_pose + hm_2d_loss
            self.log('Total HM loss', hm_2d_loss.item())
            self.log('Total 3D loss', loss_3d_pose.item())

        # calculate mpjpe loss
        mpjpe = torch.mean(torch.sqrt(torch.sum(torch.pow(p3d - pose, 2), dim=2)))
        mpjpe_std = torch.std(torch.sqrt(torch.sum(torch.pow(p3d - pose, 2), dim=2)))
        self.log("train_mpjpe_full_body", mpjpe)
        self.log("train_mpjpe_std", mpjpe_std)
        self.iteration += img.size(0)

        # Log images and skeletons every 3k batches
        if batch_idx%3000 == 0:
            tensorboard.add_images('TR Images', img, self.iteration)
            tensorboard.add_images('TR Ground Truth 2D Heatmap', torch.clip(torch.sum(p2d, dim=1, keepdim=True), 0, 1), self.iteration)
            tensorboard.add_images('TR Predicted 2D Heatmap', torch.clip(torch.sum(heatmap, dim=1, keepdim=True), 0, 1), self.iteration)  

            # Plotting the skeletons
            skel_dir = os.path.join(self.logger.log_dir, 'skel_plots')
            if not os.path.exists(skel_dir):
                os.mkdir(skel_dir) 

            y_output = pose.data.cpu().numpy()
            y_target = p3d.data.cpu().numpy()
            fig_p3d_pred = evaluate.plot_skels(y_output, os.path.join(skel_dir, 'train_p3d_pred.png'))
            fig_p3d_gt = evaluate.plot_skels(y_target, os.path.join(skel_dir, 'train_p3d_gt.png'))
            fig_compare = evaluate.plot_skels_compare( p3ds_1 = y_target, p3ds_2 = y_output,
                                            label_1 = 'GT', label_2 = 'Pred', 
                                            savepath = os.path.join(skel_dir, 'train_gt_vs_pred.png'))

            tensorboard.add_figure('TR Ground Truth 3D Skeleton', fig_p3d_gt, global_step = self.iteration)
            tensorboard.add_figure('TR Predicted 3D Skeleton', fig_p3d_pred, global_step = self.iteration)
            tensorboard.add_figure('TR Pred vs GT 3D Skeleton', fig_compare, global_step = self.iteration)

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
        if batch_idx == 0:

            skel_dir = os.path.join(self.logger.log_dir, 'skel_plots')
            if not os.path.exists(skel_dir):
                os.mkdir(skel_dir)

            # Get the procrustes aligned 3D Pose and log
            p3d_pred_t, p3d_gt_rot_t = evaluate.get_p3ds_t(y_output, y_target)
            fig_p3d_pred = evaluate.plot_skels(y_output, os.path.join(skel_dir, 'val_p3d_pred.png'))
            fig_p3d_pred_t = evaluate.plot_skels(p3d_pred_t, os.path.join(skel_dir, 'val_p3d_pred_t.png'))
            fig_p3d_gt = evaluate.plot_skels(y_target, os.path.join(skel_dir, 'val_p3d_gt.png'))
            fig_p3d_gt_rot = evaluate.plot_skels(p3d_gt_rot_t, os.path.join(skel_dir, 'val_p3d_gt_rot.png'))
            fig_compare_rescale = evaluate.plot_skels_compare( p3ds_1 = p3d_gt_rot_t, p3ds_2 = p3d_pred_t,
                                label_1 = 'GT Rot + Rescale', label_2 = 'Pred Rescale', 
                                savepath = os.path.join(skel_dir, 'val_gt_rescale_vs_pred_rescale.png'))
            fig_compare_preds = evaluate.plot_skels_compare( p3ds_1 = y_output, p3ds_2 = p3d_pred_t,
                                label_1 = 'Pred Raw', label_2 = 'Pred Rescale', 
                                savepath = os.path.join(skel_dir, 'val_pred_vs_pred_rescale.png'))

            # Tensorboard log images
            tensorboard.add_images('Val Ground Truth 2D Heatmap', torch.clip(torch.sum(p2d, dim=1, keepdim=True), 0, 1), self.iteration)
            tensorboard.add_figure('Val Ground Truth 3D Skeleton', fig_p3d_gt, global_step = self.iteration)
            tensorboard.add_figure('Val Aligned Ground Truth 3D Skeleton + Rescaling', fig_p3d_gt_rot, global_step = self.iteration)
            tensorboard.add_images('Val Predicted 2D Heatmap', torch.clip(torch.sum(heatmap, dim=1, keepdim=True), 0, 1), self.iteration)
            tensorboard.add_figure('Val Predicted 3D Skeleton', fig_p3d_pred, global_step = self.iteration)
            tensorboard.add_figure('Val Predicted 3D Skeleton + Rescaling', fig_p3d_pred_t, global_step = self.iteration)
            tensorboard.add_figure('Val GT 3D Skeleton vs Predicted 3D Skeleton (Rescaled)', fig_compare_rescale, global_step = self.iteration)
            tensorboard.add_figure('Val Comparing Predicted 3D Skeleton (Raw vs Rescaled)', fig_compare_preds, global_step = self.iteration)

        return val_loss_3d_pose

    def on_validation_start(self):
        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody(mode=self.which_data)
        self.eval_upper = evaluate.EvalUpperBody(mode=self.which_data)
        self.eval_lower = evaluate.EvalLowerBody(mode=self.which_data)

        # Initialize total validation pose loss
        self.val_loss_3d_pose_total = torch.tensor(0., device=self.device)
        self.val_loss_2d_hm = torch.tensor(0., device=self.device)


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
            self.scheduler.step(val_mpjpe["All"]["mpjpe"])
        else:
            self.log("val_mpjpe_full_body", 0.3-0.01*(self.iteration/self.hm_train_steps))
            self.scheduler.step(0.3-0.01*(self.iteration/self.hm_train_steps))
   
                    
    def on_test_start(self):
        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody(mode=self.which_data)
        self.eval_upper = evaluate.EvalUpperBody(mode=self.which_data)
        self.eval_lower = evaluate.EvalLowerBody(mode=self.which_data)
        self.eval_per_joint = evaluate.EvalPerJoint(mode=self.which_data)

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
