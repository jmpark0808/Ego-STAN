# -*- coding: utf-8 -*-

import pytorch_lightning as pl
import torch
import numpy as np
import os
from utils import evaluate
from net.blocks import *
from net.transformer import ResNetTransformerCls
import matplotlib
import pathlib


class xREgoPoseSeqHMDirectED(pl.LightningModule):
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
        self.which_data = kwargs.get('dataloader')
        self.protocol = kwargs.get('protocol')
        self.heatmap_resolution = kwargs.get('heatmap_resolution')
        self.weight_regularization = kwargs.get('weight_regularization')
        self.dropout_linear = kwargs.get('dropout_linear')

        if self.which_data in ['baseline', 'sequential'] :
            num_class = 16

        elif self.which_data == 'mo2cap2':
            num_class = 15

        elif self.which_data in ['h36m_static', 'h36m_seq']:
            num_class = 17

        # must be defined for logging computational graph
        self.example_input_array = torch.rand((1, self.seq_len, 3, 368, 368))

        # Resnet 101 without last average pooling and fully connected layers
        self.resnet101 = torchvision.models.resnet101(pretrained=False)
        # First Deconvolution to obtain 2D heatmap
        self.heatmap_deconv = nn.Sequential(*[nn.ConvTranspose2d(2048, 1024, kernel_size=3,
                                                                 stride=2, dilation=1, padding=1),
                                              nn.ConvTranspose2d(1024, num_class, kernel_size=3,
                                                                 stride=2, dilation=1, padding=0)])
        # Transformer that takes sequence of heatmaps and outputs a sequence of heatmaps
        self.resnet_transformer = ResNetTransformerCls(seq_len=self.seq_len*12*12, dim=512, depth=3, heads=8, mlp_dim=1024, dim_head=64, dropout=self.dropout)
        # Direct regression from heatmap
        self.hm2pose = HM2Pose(num_class, self.heatmap_resolution[0], self.dropout_linear)

        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody(mode=self.which_data, protocol=self.protocol)
        # self.eval_upper = evaluate.EvalUpperBody(mode=self.which_data, protocol=self.protocol)
        # self.eval_lower = evaluate.EvalLowerBody(mode=self.which_data, protocol=self.protocol)
        # self.eval_per_joint = evaluate.EvalPerJoint(mode=self.which_data, protocol=self.protocol)

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
            self.resnet101.load_state_dict(torch.load(self.load_resnet))
        self.resnet101 = nn.Sequential(*[l for ind, l in enumerate(self.resnet101.children()) if ind < 8])
        
        

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
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_regularization)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.98,
            verbose=True)
        
        # scheduler = {'scheduler': torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.00000001, end_factor=1.0, total_iters=int(self.hm_train_steps/self.batch_size)),
        #                 'name': 'learning_rate',
        #                 'interval':'step',
        #                 'frequency': 1}
        return optimizer

    # learning rate warm-up
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

        resnet = self.resnet101(imgs)
        # resnet = batch_size*len_seq x 2048 x 12 x 12
        resnet = resnet.reshape(dim[0], dim[1], 2048, 12, 12)
        # resnet = batch_size x len_seq x 2048 x 12 x 12
        resnet = resnet.permute(0, 1, 3, 4, 2)
        resnet = resnet.reshape(dim[0], -1, 2048)
        # resnet = batch_size x len_seq*12*12 x 2048
        
        resnet, atts = self.resnet_transformer(resnet)
        # resnet = batch_size x 144 x 2048
        resnet = resnet.reshape(dim[0], 12, 12, 2048)
        resnet = resnet.permute(0, 3, 1, 2) 
        # resnet = batch_size x 2048 x 12 x 12

        hms = torch.sigmoid(self.heatmap_deconv(resnet))
        # hms = batch_size x 15 x 47 x 47

        p3d = self.hm2pose(hms)
        # p3d = batch_size x 16 x 3


        return hms, p3d, atts

    def training_step(self, batch, batch_idx):
        """
        Compute and return the training loss
        logging resources:
        https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html

        """
        tensorboard = self.logger.experiment
        sequence_imgs, p2d, p3d, action = batch
        sequence_imgs = sequence_imgs.cuda()
        p2d = p2d.cuda()
        p2d = p2d[:, -1, :, :, :]
        p3d = p3d.cuda()
        p3d = p3d[:, -1, :, :]
        if self.which_data in ['h36m_static', 'h36m_seq']:
            p3d[:, 14, :] = 0
        # forward pass
        pred_hm, pred_3d, atts = self.forward(sequence_imgs)


        if self.iteration <= self.hm_train_steps:
            # pred_hm = torch.sigmoid(pred_hm)
            loss = self.mse(pred_hm, p2d)
            self.log('Total HM loss', loss.item())
        else:
            # pred_hm = torch.sigmoid(pred_hm)
            hm_loss = self.mse(pred_hm, p2d)
            loss_3d_pose = self.auto_encoder_loss(pred_3d, p3d)
            loss = hm_loss + loss_3d_pose
            self.log('Total HM loss', hm_loss.item())
            self.log('Total 3D loss', loss_3d_pose.item())
     
        # calculate mpjpe loss
        mpjpe = torch.mean(torch.sqrt(torch.sum(torch.pow(p3d - pred_3d, 2), dim=2)))
        mpjpe_std = torch.std(torch.sqrt(torch.sum(torch.pow(p3d - pred_3d, 2), dim=2)))
        self.log("train_mpjpe_full_body", mpjpe)
        self.log("train_mpjpe_std", mpjpe_std)
        self.iteration += 1

        if batch_idx % 2500 == 0:
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            img_plot = sequence_imgs[:, -1, :, :, :].clone()
            img_plot[:, 0, :, :] = img_plot[:, 0, :, :]*std[0]+mean[0]
            img_plot[:, 1, :, :] = img_plot[:, 1, :, :]*std[1]+mean[1]
            img_plot[:, 2, :, :] = img_plot[:, 2, :, :]*std[2]+mean[2]
            tensorboard.add_images('TR Images', img_plot, self.iteration)
            tensorboard.add_images('TR Ground Truth 2D Heatmap', torch.clip(torch.sum(p2d, dim=1, keepdim=True), 0, 1), self.iteration)
            tensorboard.add_images('TR Predicted 2D Heatmap', torch.clip(torch.sum(pred_hm, dim=1, keepdim=True), 0, 1), self.iteration)
            y_output = pred_3d.data.cpu().numpy()
            y_target = p3d.data.cpu().numpy()
            if self.which_data in ['h36m_static', 'h36m_seq']:
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
                tensorboard.add_figure('TR GT 3D Skeleton vs Predicted 3D Skeleton', fig_compare_preds, global_step = self.iteration)

        return loss
    
    def on_train_epoch_end(self) -> None:
        self.scheduler.step()
        

    def validation_step(self, batch, batch_idx):
        """
        Compute the metrics for validation batch
        validation loop: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
        """
        tensorboard = self.logger.experiment
        sequence_imgs, p2d, p3d, action = batch
        sequence_imgs = sequence_imgs.cuda()
        p2d = p2d.cuda()
        p2d = p2d[:, -1, :, :, :]
        p3d = p3d.cuda()
        p3d = p3d[:, -1, :, :]
        if self.which_data in ['h36m_static', 'h36m_seq']:
            p3d[:, 14, :] = 0
        # forward pass
        heatmap, pose, atts = self.forward(sequence_imgs)
        # heatmap = torch.sigmoid(heatmap)

        # calculate pose loss
        val_hm_loss = self.mse(heatmap, p2d)
        val_loss_3d_pose = self.auto_encoder_loss(pose, p3d)

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
            img_plot = sequence_imgs[:, -1, :, :, :].clone()
            img_plot[:, 0, :, :] = img_plot[:, 0, :, :]*std[0]+mean[0]
            img_plot[:, 1, :, :] = img_plot[:, 1, :, :]*std[1]+mean[1]
            img_plot[:, 2, :, :] = img_plot[:, 2, :, :]*std[2]+mean[2]
            tensorboard.add_images('Val Images', img_plot, self.iteration)
            tensorboard.add_images('Val Ground Truth 2D Heatmap', torch.clip(torch.sum(p2d, dim=1, keepdim=True), 0, 1), self.iteration)
            tensorboard.add_images('Val Predicted 2D Heatmap', torch.clip(torch.sum(heatmap, dim=1, keepdim=True), 0, 1), self.iteration)
            if self.which_data in ['h36m_static', 'h36m_seq']:
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
                tensorboard.add_figure('Val GT 3D Skeleton vs Predicted 3D Skeleton', fig_compare_preds, global_step = self.iteration)


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
        else:
            self.log("val_mpjpe_full_body", 0.3-0.01*(self.iteration/self.hm_train_steps))

    def on_test_start(self):
        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody(mode=self.which_data, protocol=self.protocol)
        self.eval_upper = evaluate.EvalUpperBody(mode=self.which_data, protocol=self.protocol)
        self.eval_lower = evaluate.EvalLowerBody(mode=self.which_data, protocol=self.protocol)
        self.eval_per_joint = evaluate.EvalPerJoint(mode=self.which_data, protocol=self.protocol)
        # self.eval_samples = evaluate.EvalSamples()
        self.filenames = []

    def test_step(self, batch, batch_idx):
        logdir = self.logger.log_dir
        tensorboard = self.logger.experiment
        sequence_imgs, p2d, p3d, action = batch
        sequence_imgs = sequence_imgs.cuda()
        p2d = p2d.cuda()
        p2d = p2d[:, -1, :, :, :]
        p3d = p3d.cuda()
        p3d = p3d[:, -1, :, :]
        if self.which_data in ['h36m_static', 'h36m_seq']:
            p3d[:, 14, :] = 0
        # forward pass
        heatmap, pose, atts = self.forward(sequence_imgs)
        # heatmap = torch.sigmoid(heatmap)
        if self.image_limit > 0 and np.random.uniform() < 0.2:

            for level, att in enumerate(atts):
                for head in range(att.size(1)):
                    img = att[:, head, :, :].reshape(att.size(0), 1, att.size(2), att.size(3)) # batch, 1, (T+1)*12*12, (T+1)*12*12
                    img = img.detach().cpu().numpy()
                    np.save(os.path.join(logdir, f'{batch_idx}_{level}_{head}_att'), img[0, :, :, :])
                    cmap = matplotlib.cm.get_cmap('gist_heat')
                    rgba = np.transpose(np.squeeze(cmap(img), axis=1), (0, 3, 1, 2))[0, :3, :, :]
                    tensorboard.add_image(f'Level {level}, head {head}, Attention Map', rgba, global_step=self.test_iteration)
            
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            first_sample = sequence_imgs[0]
            first_sample[:, 0, :, :] = first_sample[:, 0, :, :]*std[0]+mean[0]
            first_sample[:, 1, :, :] = first_sample[:, 1, :, :]*std[1]+mean[1]
            first_sample[:, 2, :, :] = first_sample[:, 2, :, :]*std[2]+mean[2]
            first_sample_numpy = first_sample.detach().cpu().numpy()
            np.save(os.path.join(logdir, f'{batch_idx}_images'), first_sample_numpy)
            tensorboard.add_images('Test Images', first_sample, global_step=self.test_iteration)
            tensorboard.add_images('Test GT Heatmap', torch.clip(p2d[0], 0, 1).reshape(16, 1, 47, 47), global_step=self.test_iteration)
            tensorboard.add_images('Test Pred Heatmap', torch.clip(heatmap[0], 0, 1).reshape(16, 1, 47, 47), global_step=self.test_iteration)
            self.image_limit -= 1
        
        # Evaluate mpjpe
        y_output = pose.data.cpu().numpy()
        y_target = p3d.data.cpu().numpy()
        self.eval_body.eval(y_output, y_target, action)
        self.eval_upper.eval(y_output, y_target, action)
        self.eval_lower.eval(y_output, y_target, action)
        self.eval_per_joint.eval(y_output, y_target)
        self.test_iteration += sequence_imgs.size(0)
        # filenames = []
        # for idx in range(y_target.shape[0]):

        #     filename = pathlib.Path(img_path[-1][idx]).stem
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
            "Per Joint": test_mpjpe_per_joint,
        }          

if __name__ == "__main__":
    pass
