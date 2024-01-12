# ----------------------------------------------------------- #
#  This is code confidential, for peer-review purposes only   #
#  and protected under conference code of ethics              #
# ----------------------------------------------------------- #

# -*- coding: utf-8 -*-

import pytorch_lightning as pl
import torch
import numpy as np
from utils import evaluate
from net.blocks import *
from net.transformer import GlobalPixelTransformer
import matplotlib
import numpy as np
import pathlib
import os

class Mo2Cap2GlobalTrans(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # parameters
        self.batch_size = kwargs.get("batch_size")
        self.lr = kwargs.get("lr")
        self.lr_decay = kwargs.get("lr_decay")
        self.decay_step = kwargs.get("decay_step")
        self.load_resnet = kwargs.get("load_resnet")
        self.hm_train_steps = kwargs.get("hm_train_steps")
        self.dropout = kwargs.get("dropout")
        self.es_patience = kwargs.get('es_patience')

        # must be defined for logging computational graph
        self.example_input_array = torch.rand((1, 3, 368, 368))

        # Resnet 101 without last average pooling and fully connected layers
        self.resnet101 = torchvision.models.resnet101(pretrained=False)
        # First Deconvolution to obtain 2D heatmap
        self.heatmap_deconv = nn.Sequential(*[nn.ConvTranspose2d(2048, 1024, kernel_size=3,
                                                                 stride=2, dilation=1, padding=1),
                                              nn.ConvTranspose2d(1024, 15, kernel_size=3,
                                                                 stride=2, dilation=1, padding=0)])
        # Transformer that takes sequence of heatmaps and outputs a sequence of heatmaps
        self.resnet_transformer = GlobalPixelTransformer(dim=512, depth=3, heads=8, mlp_dim=1024, dim_head=64, dropout=self.dropout)
        # Direct regression from heatmap
        self.hm2pose = HM2Pose(15)

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
        self.automatic_optimization=False
        self.update_optimizer_flag = False
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


        self.resnet101 = nn.Sequential(*[l for ind, l in enumerate(self.resnet101.children()) if ind < 8])
        
        if self.load_resnet:
            pretrained_dict = torch.load(self.load_resnet)
            model_dict = self.resnet101.state_dict()
            # print([k.split('heatmap.')[-1] for k in pretrained_dict['state_dict'].keys()])
            # print(model_dict.keys())
            # assert(0)
            pretrained_dict = {k.split('heatmap.resnet101.')[-1]: v for k, v in pretrained_dict['state_dict'].items() if k.split('heatmap.resnet101.')[-1] in model_dict}
            model_dict.update(pretrained_dict) 
            self.resnet101.load_state_dict(pretrained_dict)

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
        
        if self.update_optimizer_flag:
            parameters = list(self.hm2pose.parameters())
            optimizer = torch.optim.AdamW(parameters, lr=self.lr)
        else:
            resnet_params = [(n, p) for n, p in self.named_parameters() if n.startswith('resnet101')]
            all_params = [p for n, p in self.named_parameters() if not n.startswith('resnet101')]

            length = len(resnet_params)
            threshold = int(13*length/15.)

            lowlevel_params = []

            for idx, (n, p) in enumerate(resnet_params):
                if idx < threshold:
                    lowlevel_params.append(p)
                else:
                    all_params.append(p)


            grouped_parameters = [
                {"params": lowlevel_params, 'lr': self.lr/50.},
                {"params": all_params, 'lr': self.lr},
            ]

            optimizer = torch.optim.AdamW(grouped_parameters, lr=self.lr)
  
        return optimizer

    def forward(self, x):
        """
        Forward pass through model

        :param x: Batch of images

        :return: 2D heatmap, 16x3 joint inferences
        """
        dim = x.shape 
        #shape -> batch_size x 3 x 368 x 368

        resnet = self.resnet101(x)
        # resnet = batch_size x 2048 x 12 x 12
        resnet = resnet.reshape(dim[0], 2048, -1)
        # resnet = batch_size x 2048 x 12*12
        resnet = resnet.permute(0, 2, 1)
        # resnet = batch_size x 12*12 x 2048
        
        resnet, atts = self.resnet_transformer(resnet)
        # resnet = batch_size x 144 x 2048
        resnet = resnet.reshape(dim[0], 12, 12, 2048)
        resnet = resnet.permute(0, 3, 1, 2) 
        # resnet = batch_size x 2048 x 12 x 12

        hms = self.heatmap_deconv(resnet)
        # hms = batch_size x 15 x 47 x 47

        p3d = self.hm2pose(hms)
        # p3d = batch_size x 15 x 3


        return hms, p3d, atts

    def training_step(self, batch, batch_idx):
        """
        Compute and return the training loss
        logging resources:
        https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html

        """
        
        imgs, p2d, p3d, action, img_path = batch
        imgs = imgs.cuda()
        p2d = p2d.cuda()
        p3d = p3d.cuda()

        if self.iteration > self.hm_train_steps and not self.update_optimizer_flag:
            self.update_optimizer_flag = True

        opt = self.configure_optimizers()
        opt.zero_grad()


        # forward pass
        pred_hm, pred_3d, atts = self.forward(imgs)



        if self.iteration <= self.hm_train_steps:
            pred_hm = torch.sigmoid(pred_hm)
            loss = self.mse(pred_hm, p2d)
            self.log('Total HM loss', loss.item())
        else:
            pred_hm = torch.sigmoid(pred_hm)
            hm_loss = self.mse(pred_hm, p2d)
            loss_3d_pose = self.auto_encoder_loss(pred_3d, p3d)
            loss = hm_loss + loss_3d_pose
            self.log('Total HM loss', hm_loss.item())
            self.log('Total 3D loss', loss_3d_pose.item())

        self.manual_backward(loss)
        opt.step()
        # calculate mpjpe loss
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
        imgs, p2d, p3d, action, img_path = batch
        imgs = imgs.cuda()
        p2d = p2d.cuda()
        p3d = p3d.cuda()

        # forward pass
        heatmap, pose, atts = self.forward(imgs)
        heatmap = torch.sigmoid(heatmap)

        # calculate pose loss
        val_hm_loss = self.mse(heatmap, p2d)
        val_loss_3d_pose = self.auto_encoder_loss(pose, p3d)

        # update 3d pose loss
        self.val_loss_hm += val_hm_loss
        self.val_loss_3d_pose_total += val_loss_3d_pose

        # Evaluate mpjpe
        y_output = pose.data.cpu().numpy()
        y_target = p3d.data.cpu().numpy()
        self.eval_body_1.eval(y_output, y_target, action)
        self.eval_body_2.eval(y_output, y_target, action)
        # self.eval_upper.eval(y_output, y_target, action)
        # self.eval_lower.eval(y_output, y_target, action)
        if batch_idx == 0:
            tensorboard.add_images('Val Image', imgs, self.iteration)
            tensorboard.add_images('Val Predicted 2D Heatmap', torch.clip(torch.sum(heatmap, dim=1, keepdim=True), 0, 1), self.iteration)

            skel_dir = os.path.join(self.logger.log_dir, 'skel_plots')
            if not os.path.exists(skel_dir):
                os.mkdir(skel_dir)

            y_output, y_target = evaluate.get_p3ds_t(y_output, y_target)
            fig_compare_preds = evaluate.plot_skels_compare( p3ds_1 = y_output, p3ds_2 = y_target,
                            label_1 = 'Pred Aligned', label_2 = 'Ground Truth', 
                            savepath = os.path.join(skel_dir, 'val_pred_aligned_vs_GT.png'), dataset='mo2cap2')

            tensorboard.add_figure('Val GT 3D Skeleton vs Predicted 3D Skeleton', fig_compare_preds, global_step = self.iteration)
        return val_loss_3d_pose

    def on_validation_start(self):
        # Initialize the mpjpe evaluation pipeline
        self.eval_body_1 = evaluate.EvalBody(mode='mo2cap2')
        self.eval_body_2 = evaluate.EvalBody(mode='h36m', protocol='p2')
        # self.eval_upper = evaluate.EvalUpperBody(mode='mo2cap2')
        # self.eval_lower = evaluate.EvalLowerBody(mode='mo2cap2')

        # Initialize total validation pose loss
        self.val_loss_3d_pose_total = torch.tensor(0., device=self.device)
        self.val_loss_hm = torch.tensor(0., device=self.device)

    def validation_epoch_end(self, validation_step_outputs):
        val_mpjpe_1 = self.eval_body_1.get_results()
        val_mpjpe_2 = self.eval_body_2.get_results()
        # val_mpjpe_upper = self.eval_upper.get_results()
        # val_mpjpe_lower = self.eval_lower.get_results()

        if self.iteration >= self.hm_train_steps:
            self.log("val_mpjpe_full_body", val_mpjpe_1["All"]["mpjpe"])
            self.log("val_mpjpe_full_body_h36m", val_mpjpe_2["All"]["mpjpe"])
            # self.log("val_mpjpe_full_body_std", val_mpjpe["All"]["std_mpjpe"])
            # self.log("val_mpjpe_upper_body", val_mpjpe_upper["All"]["mpjpe"])
            # self.log("val_mpjpe_lower_body", val_mpjpe_lower["All"]["mpjpe"])
            self.log("val_loss", self.val_loss_3d_pose_total)
            # self.scheduler.step(val_mpjpe["All"]["mpjpe"])
        else:
            self.log("val_mpjpe_full_body", 0.3-0.01*(self.iteration/self.hm_train_steps))
            # self.scheduler.step(0.3-0.01*(self.iteration/self.hm_train_steps))

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
        tensorboard = self.logger.experiment
        imgs, p2d, p3d, action, img_path = batch
        imgs = imgs.cuda()
        p2d = p2d.cuda()
        p3d = p3d.cuda()

        # forward pass
        heatmap, pose, atts = self.forward(imgs)
        heatmap = torch.sigmoid(heatmap)
        if self.image_limit > 0 and np.random.uniform() < 0.2:

            for level, att in enumerate(atts):
                for head in range(att.size(1)):
                    img = att[:, head, :, :].reshape(att.size(0), 1, att.size(2), att.size(3))
                    img = img.detach().cpu().numpy()
                    cmap = matplotlib.cm.get_cmap('gist_heat')
                    rgba = np.transpose(np.squeeze(cmap(img), axis=1), (0, 3, 1, 2))[0, :3, :, :]
                    tensorboard.add_image(f'Level {level}, head {head}, Attention Map', rgba, global_step=self.test_iteration)
            
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            first_sample = imgs[0]
            first_sample[0, :, :] = first_sample[0, :, :]*std[0]+mean[0]
            first_sample[1, :, :] = first_sample[1, :, :]*std[1]+mean[1]
            first_sample[2, :, :] = first_sample[2, :, :]*std[2]+mean[2]
            tensorboard.add_image('Test Image', first_sample, global_step=self.test_iteration)
            tensorboard.add_image('Test GT Heatmap', torch.clip(torch.sum(torch.squeeze(p2d[0]), dim=0, keepdim=True), 0, 1), global_step=self.test_iteration)
            tensorboard.add_image('Test Pred Heatmap', torch.clip(torch.sum(torch.squeeze(heatmap[0]), dim=0, keepdim=True), 0, 1), global_step=self.test_iteration)
            self.image_limit -= 1
        
        # Evaluate mpjpe
        y_output = pose.data.cpu().numpy()
        y_target = p3d.data.cpu().numpy()
        self.eval_body.eval(y_output, y_target, action)
        self.eval_upper.eval(y_output, y_target, action)
        self.eval_lower.eval(y_output, y_target, action)
        self.eval_per_joint.eval(y_output, y_target)
        self.test_iteration += imgs.size(0)
        
        errors = np.mean(np.sqrt(np.sum(np.power(y_target - y_output, 2), axis=2)), axis=1)
        for idx in range(y_target.shape[0]):

            filename = pathlib.Path(img_path[idx]).stem
            filename = str(filename).replace(".", "_")
            if filename in evaluate.highest_differences:
                self.handpicked_results.update(
                {
                    filename: {
                        "gt_pose": y_target[idx],
                        "pred_pose": y_output[idx],
                        "img": img.cpu().numpy()[idx]
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
