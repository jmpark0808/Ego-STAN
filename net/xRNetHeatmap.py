# -*- coding: utf-8 -*-

import pytorch_lightning as pl
import torch
import torch.nn as nn
from utils import evaluate
from net.blocks import *



class xREgoPoseHeatMap(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # parameters
        self.batch_size = kwargs["batch_size"]
        self.lr = kwargs["lr"]
        self.lr_decay = kwargs["lr_decay"]
        self.decay_step = kwargs["decay_step"]
        self.load_resnet = kwargs["load_resnet"]
        self.hm_train_steps = kwargs["hm_train_steps"]

        # must be defined for logging computational graph
        self.example_input_array = torch.rand((1, 3, 368, 368))

        # Generator that produces the HeatMap
        self.heatmap = HeatMap()

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
    

    def mse(self, pred, label):
        pred = pred.reshape(pred.size(0), -1)
        label = label.reshape(label.size(0), -1)
        return torch.sum(torch.mean(torch.pow(pred-label, 2), dim=1))

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

        :return: 2D heatmap, 16x3 joint inferences, 2D reconstructed heatmap
        """
        # x = 3 x 368 x 368

        heatmap = self.heatmap(x)
        # heatmap = 15 x 47 x 47


        return heatmap

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

        # forward pass
        heatmap = self.forward(img)



        heatmap = torch.sigmoid(heatmap)
        loss = self.mse(heatmap, p2d)
        self.log('Total HM loss', loss.item())
     
        tensorboard.add_images('Ground Truth 2D Heatmap', torch.clip(torch.sum(p2d, dim=1, keepdim=True), 0, 1), self.iteration)
        tensorboard.add_images('Predicted 2D Heatmap', torch.clip(torch.sum(heatmap, dim=1, keepdim=True), 0, 1), self.iteration)
        self.iteration += img.size(0)
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

        # forward pass
        heatmap = self.forward(img)
        heatmap = torch.sigmoid(heatmap)
   
        # calculate pose loss
        val_hm_loss = self.mse(heatmap, p2d)

        # update 3d pose loss
        self.val_loss_hm += val_hm_loss

        # Evaluate mpjpe
        tensorboard.add_images('Val Ground Truth 2D Heatmap', torch.clip(torch.sum(p2d, dim=1, keepdim=True), 0, 1), self.iteration)
        tensorboard.add_images('Val Predicted 2D Heatmap', torch.clip(torch.sum(heatmap, dim=1, keepdim=True), 0, 1), self.iteration)
        return val_hm_loss

    def on_validation_start(self):
        self.val_loss_hm = torch.tensor(0., device=self.device)

    def validation_epoch_end(self, validation_step_outputs):
        self.log("val_mpjpe_full_body", self.val_loss_hm)
                    

if __name__ == "__main__":
    pass
