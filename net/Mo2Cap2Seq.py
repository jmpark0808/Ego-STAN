# -*- coding: utf-8 -*-

from mimetypes import init
import pytorch_lightning as pl
import torch
import numpy as np
from utils import evaluate
from net.blocks import *
from net.transformer import PoseTransformer
import matplotlib
import pathlib


class Mo2Cap2Seq(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # parameters
        self.batch_size = kwargs.get("batch_size")
        self.lr = kwargs.get("lr")
        self.lr_decay = kwargs.get("lr_decay")
        self.decay_step = kwargs.get("decay_step")
        self.load_resnet = kwargs.get("load_resnet")
        self.hm_train_steps = kwargs.get("hm_train_steps")
        self.seq_len = kwargs.get('seq_len')
        self.es_patience = kwargs.get('es_patience')
        self.dropout = kwargs.get('dropout')

        # must be defined for logging computational graph
        self.example_input_array = torch.rand((1, self.seq_len, 3, 368, 368))

        # Generator that produces the HeatMap
        self.heatmap = HeatMap(15)
        # Encoder that takes 2D heatmap and transforms to latent vector Z
        self.encoder = Encoder(15)
        # Transformer that takes sequence of latent vector Z and outputs a single Z vector
        self.seq_transformer = PoseTransformer(seq_len=self.seq_len, dim=20, depth=3, heads=1, mlp_dim=40, dropout=self.dropout)
        # Pose decoder that takes latent vector Z and transforms to 3D pose coordinates
        self.pose_decoder = PoseDecoder(initial_dim=20)
        # Heatmap decoder that takes latent vector Z and generates the original 2D heatmap
        self.heatmap_decoder = HeatmapDecoder(15)

        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody(mode='mo2cap2')
        self.eval_upper = evaluate.EvalUpperBody(mode='mo2cap2')
        self.eval_lower = evaluate.EvalLowerBody(mode='mo2cap2')
        self.eval_per_joint = evaluate.EvalPerJoint(mode='mo2cap2')

        # Initialize total validation pose loss
        self.val_loss_3d_pose_total = torch.tensor(0., device=self.device)
        self.val_loss_hm = torch.tensor(0., device=self.device)
        self.iteration = 0
        self.test_iteration = 0
        self.image_limit = 100
        self.save_hyperparameters()

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
        pose_l2norm = torch.sqrt(torch.sum(torch.sum(torch.pow(pose_pred-pose_label, 2), dim=2), dim=1))
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        cosine_similarity_error = torch.sum(cos(pose_pred, pose_label), dim=1)
        limb_length_error = torch.sum(torch.sum(torch.abs(pose_pred-pose_label), dim=2), dim=1)
        heatmap_error = torch.sqrt(torch.sum(torch.pow(hm_resnet.view(hm_resnet.size(0), -1) - hm_decoder.view(hm_decoder.size(0), -1), 2), dim=1))
        LAE_pose = lambda_p*(pose_l2norm + lambda_theta*cosine_similarity_error + lambda_L*limb_length_error)
        LAE_hm = lambda_hm*heatmap_error
        return torch.mean(LAE_pose), torch.mean(LAE_hm)

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
        
        # scheduler = {'scheduler': torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.00000001, end_factor=1.0, total_iters=int(self.hm_train_steps/self.batch_size)),
        #                 'name': 'learning_rate',
        #                 'interval':'step',
        #                 'frequency': 1}
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

    def forward(self, x):
        """
        Forward pass through model

        :param x: Input sequence of image

        :return: 2D heatmap, 16x3 joint inferences, 2D reconstructed heatmap
        """
        # Flattening first two dimensions
        dim = x.shape 
        #shape -> batch_size x len_seq x 3 x 368 x 368

        imgs = torch.reshape(x, (dim[0]*dim[1], dim[2], dim[3], dim[4]))
        # imgs = # (batch_size*len_seq) x 3 x 368 x 368

        hms = self.heatmap(imgs)
        # hms = (batch_size*len_seq) x 15 x 47 x 47

        z_all = self.encoder(hms)
        # z_all = (batch_size*len_seq) x 20

        zs = torch.reshape(z_all, (dim[0], dim[1], z_all.shape[-1]))
        # zs = batch_size x len_seq x 20

        z, atts = self.seq_transformer(zs)
        # z = batch_size x 20

        p3d = self.pose_decoder(z)
        # p3d = batch_size x 16 x 3

        p2d = self.heatmap_decoder(z_all)
        # p2d = (batch_size*len_seq) x 15 x 47 x 47

        return hms, p3d, p2d, atts

    def training_step(self, batch, batch_idx):
        """
        Compute and return the training loss
        logging resources:
        https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html

        """
        
        sequence_imgs, p2d, p3d, action, img_path = batch
        sequence_imgs = sequence_imgs.cuda()
        p2d = p2d.cuda()
        p2d = p2d.reshape(-1, 16, 47, 47)
        p3d = p3d.cuda()
        p3d = p3d[:, -1, :, :]
        
        # forward pass
        pred_hm, pred_3d, gen_hm, atts = self.forward(sequence_imgs)


        if self.iteration <= self.hm_train_steps:
            pred_hm = torch.sigmoid(pred_hm)
            loss = self.mse(pred_hm, p2d)
            self.log('Total HM loss', loss.item())
        else:
            pred_hm = torch.sigmoid(pred_hm)
            gen_hm = torch.sigmoid(gen_hm)
            hm_loss = self.mse(pred_hm, p2d)
            loss_3d_pose, loss_2d_ghm = self.auto_encoder_loss(pred_3d, p3d, gen_hm, pred_hm)
            ae_loss = loss_2d_ghm + loss_3d_pose
            loss = hm_loss + ae_loss
            self.log('Total HM loss', hm_loss.item())
            self.log('Total 3D loss', loss_3d_pose.item())
            self.log('Total GHM loss', loss_2d_ghm.item())
     
        # calculate mpjpe loss
        mpjpe = torch.mean(torch.sqrt(torch.sum(torch.pow(p3d - pred_3d, 2), dim=2)))
        mpjpe_std = torch.std(torch.sqrt(torch.sum(torch.pow(p3d - pred_3d, 2), dim=2)))
        self.log("train_mpjpe_full_body", mpjpe)
        self.log("train_mpjpe_std", mpjpe_std)
        self.iteration += sequence_imgs.size(0)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Compute the metrics for validation batch
        validation loop: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
        """
        tensorboard = self.logger.experiment
        sequence_imgs, p2d, p3d, action, img_path = batch
        sequence_imgs = sequence_imgs.cuda()
        p2d = p2d.cuda()
        p2d = p2d.reshape(-1, 16, 47, 47)
        p3d = p3d.cuda()
        p3d = p3d[:, -1, :, :]

        # forward pass
        heatmap, pose, generated_heatmap, atts = self.forward(sequence_imgs)
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
        self.eval_upper.eval(y_output, y_target, action)
        self.eval_lower.eval(y_output, y_target, action)

        return val_loss_3d_pose

    def on_validation_start(self):
        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody(mode='mo2cap2')
        self.eval_upper = evaluate.EvalUpperBody(mode='mo2cap2')
        self.eval_lower = evaluate.EvalLowerBody(mode='mo2cap2')

        # Initialize total validation pose loss
        self.val_loss_3d_pose_total = torch.tensor(0., device=self.device)
        self.val_loss_hm = torch.tensor(0., device=self.device)

    def validation_epoch_end(self, validation_step_outputs):
        val_mpjpe = self.eval_body.get_results()
        val_mpjpe_upper = self.eval_upper.get_results()
        val_mpjpe_lower = self.eval_lower.get_results()
        if self.iteration >= self.hm_train_steps:
            self.log("val_mpjpe_full_body", val_mpjpe["All"]["mpjpe"])
            self.log("val_mpjpe_full_body_std", val_mpjpe["All"]["std_mpjpe"])
            self.log("val_mpjpe_upper_body", val_mpjpe_upper["All"]["mpjpe"])
            self.log("val_mpjpe_lower_body", val_mpjpe_lower["All"]["mpjpe"])
            self.log("val_loss", self.val_loss_3d_pose_total)
            self.scheduler.step(val_mpjpe["All"]["mpjpe"])
        else:
            self.log("val_mpjpe_full_body", 0.3-0.01*(self.iteration/self.hm_train_steps))
            self.scheduler.step(0.3-0.01*(self.iteration/self.hm_train_steps))
                    
    def on_test_start(self):
        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody(mode='mo2cap2')
        self.eval_upper = evaluate.EvalUpperBody(mode='mo2cap2')
        self.eval_lower = evaluate.EvalLowerBody(mode='mo2cap2')
        self.eval_per_joint = evaluate.EvalPerJoint(mode='mo2cap2')
        self.handpicked_results = {}
        self.results = {}
        self.baseeval = evaluate.ActionMap()

    def test_step(self, batch, batch_idx):
        sequence_imgs, p2d, p3d, action, img_path = batch
        sequence_imgs = sequence_imgs.cuda()
        p2d = p2d.cuda()
        p2d = p2d.reshape(-1, 16, 47, 47)
        p3d = p3d.cuda()
        p3d = p3d[:, -1, :, :]

        # forward pass
        heatmap, pose, generated_heatmap, atts = self.forward(sequence_imgs)
        heatmap = torch.sigmoid(heatmap)
        generated_heatmap = torch.sigmoid(generated_heatmap)

        # Evaluate mpjpe
        y_output = pose.data.cpu().numpy()
        y_target = p3d.data.cpu().numpy()
        self.eval_body.eval(y_output, y_target, action)
        self.eval_upper.eval(y_output, y_target, action)
        self.eval_lower.eval(y_output, y_target, action)
        self.eval_per_joint.eval(y_output, y_target)
        errors = np.mean(np.sqrt(np.sum(np.power(y_target - y_output, 2), axis=2)), axis=1)
        for idx in range(y_target.shape[0]):

            filename = pathlib.Path(img_path[-1][idx]).stem
            filename = str(filename).replace(".", "_")
            if filename in evaluate.highest_differences:
                self.handpicked_results.update(
                {
                    filename: {
                        "gt_pose": y_target[idx],
                        "pred_pose": y_output[idx],
                        "img": sequence_imgs[idx, -1].cpu().numpy()
                    }
                }
            )
            self.results.update(
                {
                    filename: {
                        "action": self.baseeval.eval(None, None, action[idx]),
                        "full_mpjpe": errors[idx],
                    }
                }
            )

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
