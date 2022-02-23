# -*- coding: utf-8 -*-

import pytorch_lightning as pl
import torch
import torch.nn as nn
from utils import evaluate

from net.blocks import HeatMap


class DirectRegression(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # parameters
        self.batch_size = kwargs["batch_size"]
        self.lr = kwargs["lr"]
        self.lr_decay = kwargs["lr_decay"]
        self.decay_step = kwargs["decay_step"]
        self.load_resnet = kwargs["load_resnet"]

        # must be defined for logging computational graph
        self.example_input_array = torch.rand((1, 3, 368, 368))

        self.heatmap = HeatMap()
        self.l1 = nn.Linear(33135, 690)
        self.l2 = nn.Linear(690, 48)

        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody()
        self.eval_upper = evaluate.EvalUpperBody()
        self.eval_lower = evaluate.EvalLowerBody()

        # Initialize total validation pose loss
        self.val_loss_3d_pose_total = torch.tensor(0, device=self.device)

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
        self.save_hyperparameters()

    def loss(self, pose_pred, pose_label):
        """
        Defining the loss funcition:
        loss = f(pose_l2norm, cosine_similarity_error, limb_length_error)
        """
        lambda_p = 0.1
        lambda_theta = -0.01
        lambda_L = 0.5

        pose_l2norm = torch.sqrt(
            torch.sum(torch.sum(torch.pow(pose_pred - pose_label, 2), dim=2), dim=1)
        )
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        cosine_similarity_error = torch.sum(cos(pose_pred, pose_label), dim=1)
        limb_length_error = torch.sum(
            torch.sum(torch.abs(pose_pred - pose_label), dim=2), dim=1
        )
        LAE_pose = lambda_p * (
            pose_l2norm
            + lambda_theta * cosine_similarity_error
            + lambda_L * limb_length_error
        )
        return torch.mean(LAE_pose)

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        """

        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=0.9, nesterov=True
        )

        return optimizer

    def forward(self, x):
        """
        Forward pass through model

        :param x: Input image

        :return: 16x3 joint inferences
        """
        # x = 3 x 368 x 368
        x = self.heatmap(x)
        # x = 15 x 47 x 47
        x = x.reshape(x.size(0), -1)
        # x = 33135
        x = self.l1(x)
        x = self.l2(x)
        x = x.reshape(x.size(0), 16, 3)
        return x

    def training_step(self, batch, batch_idx):
        """
        Compute and return the training loss
        logging resources:
        https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html

        """
        img, p2d, p3d, action = batch
        img = img.cuda()
        p2d = p2d.cuda()
        p3d = p3d.cuda()

        # forward pass
        pose = self.forward(img)

        # calculate pose loss
        pose_loss = self.loss(p3d, pose)
        self.log("train_loss", pose_loss)

        # calculate mpjpe loss
        mpjpe = torch.mean(torch.sqrt(torch.sum(torch.pow(p3d - pose, 2), dim=2)))
        mpjpe_std = torch.std(torch.sqrt(torch.sum(torch.pow(p3d - pose, 2), dim=2)))
        self.log("train_mpjpe_full_body", mpjpe)
        self.log("train_mpjpe_std", mpjpe_std)

        return pose_loss

    def validation_step(self, batch, batch_idx):
        """
        Compute the metrics for validation batch
        validation loop: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
        """
        img, p2d, p3d, action = batch
        img = img.cuda()
        p2d = p2d.cuda()
        p3d = p3d.cuda()

        # forward pass
        pose = self.forward(img)

        # calculate pose loss
        pose_loss = self.loss(p3d, pose)
        # update 3d pose loss
        self.val_loss_3d_pose_total += pose_loss

        # Evaluate mpjpe
        y_output = pose.data.cpu().numpy()
        y_target = p3d.data.cpu().numpy()
        self.eval_body.eval(y_output, y_target, action)
        self.eval_upper.eval(y_output, y_target, action)
        self.eval_lower.eval(y_output, y_target, action)

        return pose_loss

    def on_validation_start(self):
        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody()
        self.eval_upper = evaluate.EvalUpperBody()
        self.eval_lower = evaluate.EvalLowerBody()

        # Initialize total validation pose loss
        self.val_loss_3d_pose_total = torch.tensor(0.0, device=self.device)

    def validation_epoch_end(self, validation_step_outputs):
        val_mpjpe = self.eval_body.get_results()
        val_mpjpe_upper = self.eval_upper.get_results()
        val_mpjpe_lower = self.eval_lower.get_results()

        self.log("val_mpjpe_full_body", val_mpjpe["All"]["mpjpe"])
        self.log("val_mpjpe_full_body_std", val_mpjpe["All"]["std_mpjpe"])
        self.log("val_mpjpe_upper_body", val_mpjpe_upper["All"]["mpjpe"])
        self.log("val_mpjpe_lower_body", val_mpjpe_lower["All"]["mpjpe"])
        self.log("val_loss", self.val_loss_3d_pose_total)


if __name__ == "__main__":
    pass
