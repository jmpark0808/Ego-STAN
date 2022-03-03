# -*- coding: utf-8 -*-

import pytorch_lightning as pl
import torch
import numpy as np
from utils import evaluate
from net.blocks import *
from net.transformer import ResNetTransformerCls
import matplotlib

encoder_dict = {
    'map_concat': FeatureConcatEncoder(),
    'branch_concat': FeatureBranchEncoder(),
    'concat_reconstruct': FeatureReEncoder()
}


class xREgoPoseSeqConcat(pl.LightningModule):
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

        # must be defined for logging computational graph
        self.example_input_array = torch.rand((1, self.seq_len, 3, 368, 368))

        # Resnet 101 without last average pooling and fully connected layers
        self.resnet101 = torchvision.models.resnet101(pretrained=False)
        # First Deconvolution to obtain 2D heatmap
        self.heatmap_deconv = nn.Sequential(*[nn.ConvTranspose2d(2048, 1024, kernel_size=3,
                                                                 stride=2, dilation=1, padding=1),
                                              nn.ConvTranspose2d(1024, 15, kernel_size=3,
                                                                 stride=2, dilation=1, padding=0)])
        self.feature_heatmap_deconv = nn.Sequential(*[nn.ConvTranspose2d(2048, 1024, kernel_size=3,
                                                                 stride=2, dilation=1, padding=1),
                                              nn.ConvTranspose2d(1024, 15, kernel_size=3,
                                                                 stride=2, dilation=1, padding=0)])
        # Encoder that takes heatmap and featuremap and transforms to latent vector Z
        self.encoder = FeatureBranchEncoder()
        # Transformer that takes sequence of heatmaps and outputs a sequence of heatmaps
        self.resnet_transformer = ResNetTransformerCls(seq_len=self.seq_len*12*12, dim=512, depth=3, heads=8, mlp_dim=1024, dim_head=64, dropout=0.)
        # Pose decoder that takes latent vector Z and transforms to 3D pose coordinates
        self.pose_decoder = PoseDecoder()
        # Heatmap decoder that takes latent vector Z and generates the original 2D heatmap
        self.heatmap_decoder = HeatmapDecoder()

        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody()
        self.eval_upper = evaluate.EvalUpperBody()
        self.eval_lower = evaluate.EvalLowerBody()

        # Initialize total validation pose loss
        self.val_loss_3d_pose_total = torch.tensor(0., device=self.device)
        self.val_loss_hm = torch.tensor(0., device=self.device)
        self.iteration = 0
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
        heatmap_error = torch.sqrt(torch.sum(torch.pow(hm_resnet.reshape(hm_resnet.size(0), -1) - hm_decoder.reshape(hm_decoder.size(0), -1), 2), dim=1))
        LAE_pose = lambda_p*(pose_l2norm + lambda_theta*cosine_similarity_error + lambda_L*limb_length_error)
        LAE_hm = lambda_hm*heatmap_error
        return torch.mean(LAE_pose), torch.mean(LAE_hm)

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = {'scheduler': torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.00000001, end_factor=1.0, total_iters=int(self.hm_train_steps/self.batch_size)),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        return [optimizer], [scheduler]

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
        # resnet = batch_size x 12*12 x 2048
        resnet = resnet.reshape(dim[0], 12, 12, 2048)
        resnet = resnet.permute(0, 3, 1 ,2) 
        # resnet = batch_size x 2048 x 12 x 12
 
        hms = self.heatmap_deconv(resnet)
        # hms = batch_size x 15 x 47 x 47

        dms = self.feature_heatmap_deconv(resnet)
        # dms = batch_size x 15 x 47 x 47

        z = self.encoder(hms, dms)
        # z = batch_size x 20

        p3d = self.pose_decoder(z)
        # p3d = batch_size x 16 x 3

        p2d = self.heatmap_decoder(z)
        # p2d = batch_size x 15 x 47 x 47

        return hms, p3d, p2d, atts

    def training_step(self, batch, batch_idx):
        """
        Compute and return the training loss
        logging resources:
        https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html

        """
        
        sequence_imgs, p2d, p3d, action = batch
        sequence_imgs = sequence_imgs.cuda()
        p2d = p2d.cuda()
        p2d = p2d[:, -1, :, :, :]
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
        
        sequence_imgs, p2d, p3d, action = batch
        sequence_imgs = sequence_imgs.cuda()
        p2d = p2d.cuda()
        p2d = p2d[:, -1, :, :, :]
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
        self.eval_body = evaluate.EvalBody()
        self.eval_upper = evaluate.EvalUpperBody()
        self.eval_lower = evaluate.EvalLowerBody()

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
        else:
            self.log("val_mpjpe_full_body", 0.3-0.01*(self.iteration/self.hm_train_steps))

    def on_test_start(self):
        # Initialize the mpjpe evaluation pipeline
        self.eval_body = evaluate.EvalBody()
        self.eval_upper = evaluate.EvalUpperBody()
        self.eval_lower = evaluate.EvalLowerBody()

    def test_step(self, batch, batch_idx):
        tensorboard = self.logger.experiment
        sequence_imgs, p2d, p3d, action = batch
        sequence_imgs = sequence_imgs.cuda()
        p2d = p2d.cuda()
        p2d = p2d[:, -1, :, :, :]
        p3d = p3d.cuda()
        p3d = p3d[:, -1, :, :]

        # forward pass
        heatmap, pose, generated_heatmap, atts = self.forward(sequence_imgs)
        heatmap = torch.sigmoid(heatmap)
        generated_heatmap = torch.sigmoid(generated_heatmap)
        '''
        for level, att in enumerate(atts):
            for head in range(att.size(1)):
                img = att[:, head, :, :].reshape(att.size(0), 1, att.size(2), att.size(3))
                img = img.detach().cpu().numpy()
                cmap = matplotlib.cm.get_cmap('gist_heat')
                rgba = np.transpose(np.squeeze(cmap(img), axis=1), (0, 3, 1, 2))[:, :3, :, :]
                tensorboard.add_images(f'Level {level}, head {head}, Attention Map', rgba, global_step=self.iteration)
        '''
        # Evaluate mpjpe
        y_output = pose.data.cpu().numpy()
        y_target = p3d.data.cpu().numpy()
        self.eval_body.eval(y_output, y_target, action)
        self.eval_upper.eval(y_output, y_target, action)
        self.eval_lower.eval(y_output, y_target, action)
      

    def test_epoch_end(self, test_step_outputs):
        test_mpjpe = self.eval_body.get_results()
        test_mpjpe_upper = self.eval_upper.get_results()
        test_mpjpe_lower = self.eval_lower.get_results()

        self.test_results = {
            "Full Body": test_mpjpe,
            "Upper Body": test_mpjpe_upper,
            "Lower Body": test_mpjpe_lower,
        }          

if __name__ == "__main__":
    pass
