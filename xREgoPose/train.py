import datetime
import os
import numpy as np
import argparse
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
from loss import mse, auto_encoder_loss
from network import xREgoPose
from dataset.mocap import Mocap
from utils import config
from base import SetType
import dataset.transform as trsf
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt

if __name__ == '__main__':
    torch.cuda.manual_seed_all(1234)
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--load",
                        help="Directory of pre-trained model,  \n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model")
    parser.add_argument('--dataset', help='Directory of your Dataset', required=True, default=None)
    parser.add_argument('--cuda', help="'cuda' for cuda, 'cpu' for cpu, default = cuda",
                        default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', help="batchsize, default = 1", default=1, type=int)
    parser.add_argument('--epoch', help='# of epochs. default = 20', default=20, type=int)
    parser.add_argument('--logdir', help='logdir for models and losses. default = .', default='./', type=str)
    parser.add_argument('--lr', '--learning_rate', help='learning_rate. default = 0.001', default=0.001, type=float)
    parser.add_argument('--lr_decay', help='Learning rate decrease by lr_decay time per decay_step, default = 0.1',
                        default=0.1, type=float)
    parser.add_argument('--decay_step', help='Learning rate decrease by lr_decay time per decay_step,  default = 7000',
                        default=1E100, type=int)
    parser.add_argument('--display_freq', help='display_freq to display result image on Tensorboard',
                        default=1000, type=int)
    parser.add_argument('--e_batch_size', help="effective batchsize, default = 8", default=8, type=int)



    args = parser.parse_args()
    device = torch.device(args.cuda)
    batch_size = args.batch_size
    e_batch_size = args.e_batch_size
    epoch = args.epoch
    data_transform = transforms.Compose([
        trsf.ImageTrsf(),
        trsf.Joints3DTrsf(),
        trsf.ToTensor()])

    # let's load data from validation set as example
    data = Mocap(
        args.dataset,
        SetType.TRAIN,
        transform=data_transform)
    dataloader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True)

    load = args.load
    start_iter = 0
    model = xREgoPose().to(device=args.cuda)

    # Xavier Initialization
    def weight_init(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model.encoder.apply(weight_init)
    model.heatmap_deconv.apply(weight_init)
    model.pose_decoder.apply(weight_init)
    model.heatmap_decoder.apply(weight_init)
    now = datetime.datetime.now()
    start_epo = 0


    if load is not None:
        state_dict = torch.load(load, map_location=args.cuda)

        start_iter = int(load.split('epo_')[1].strip('step.ckpt'))
        start_epo = int(load.split('/')[-1].split('epo')[0])
        now = datetime.datetime.strptime(load.split('/')[-2], '%m%d%H%M')

        print("Loading Model from {}".format(load))
        print("Start_iter : {}".format(start_iter))
        print("now : {}".format(now.strftime('%m%d%H%M')))
        model.load_state_dict(state_dict)
        print('Loading_Complete')

    # Optimizer Setup
    learning_rate = args.lr
    lr_decay = args.lr_decay
    decay_step = args.decay_step
    learning_rate = learning_rate * (lr_decay ** (start_iter // decay_step))
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    # opt_front = torch.optim.SGD(list(model.resnet101.parameters()) + list(model.heatmap_deconv.parameters()),
    #                             lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    # opt_back = torch.optim.SGD(list(model.encoder.parameters())+list(model.pose_decoder.parameters())+list(model.heatmap_decoder.parameters()),
    #                            lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    # opt_front = torch.optim.Adam(model.parameters(),
    #                              lr=learning_rate)
    # opt_back = torch.optim.Adam(list(model.encoder.parameters()) + list(model.pose_decoder.parameters()) + list(
    #     model.heatmap_decoder.parameters()), lr=learning_rate)

    # Logger Setup
    os.makedirs(os.path.join('log', now.strftime('%m%d%H%M')), exist_ok=True)
    weight_save_dir = os.path.join(args.logdir, os.path.join('models', 'state_dict', now.strftime('%m%d%H%M')))
    plot_3d_dir = os.path.join(args.logdir, os.path.join('3d_plot', now.strftime('%m%d%H%M')))
    os.makedirs(os.path.join(weight_save_dir), exist_ok=True)
    os.makedirs(os.path.join(plot_3d_dir), exist_ok=True)
    writer = SummaryWriter(os.path.join(args.logdir, os.path.join('log', now.strftime('%m%d%H%M'))))
    iterate = start_iter


    for epo in range(start_epo, epoch):
        print("\nEpoch : {}".format(epo))
        for i, batch in enumerate(tqdm(dataloader)):
            img, p2d, p3d, action = batch
            img = img.cuda()
            p2d = p2d.cuda()
            p3d = p3d.cuda()
            # Freeze everything but resnet, and deconv
            # for param in model.resnet101.parameters():
            #     param.requires_grad = True
            #
            # for param in model.heatmap_deconv.parameters():
            #     param.requires_grad = True
            #
            # for param in model.heatmap_decoder.parameters():
            #     param.requires_grad = False
            #
            # for param in model.pose_decoder.parameters():
            #     param.requires_grad = False
            #
            # for param in model.encoder.parameters():
            #     param.requires_grad = False
            # opt.zero_grad()
            # heatmap = model(img)
            # heatmap = torch.sigmoid(heatmap)
            # loss_2d_hm = mse(heatmap, p2d)
            # loss_2d_hm.backward()
            # opt_front.step()
            #
            #
            # writer.add_scalar('Total HM loss', loss_2d_hm.item(), global_step=iterate)
            #
            # # Freeze resnet, deconv
            # for param in model.resnet101.parameters():
            #     param.requires_grad = False
            #
            # for param in model.heatmap_deconv.parameters():
            #     param.requires_grad = False
            #
            # for param in model.heatmap_decoder.parameters():
            #     param.requires_grad = True
            #
            # for param in model.pose_decoder.parameters():
            #     param.requires_grad = True
            #
            # for param in model.encoder.parameters():
            #     param.requires_grad = True
            #
            opt.zero_grad()
            heatmap, pose, generated_heatmaps = model(img)
            heatmap = torch.sigmoid(heatmap)
            generated_heatmaps = torch.sigmoid(generated_heatmaps)
            loss_2d_hm = mse(heatmap, p2d)
            loss_3d_pose, loss_2d_ghm = auto_encoder_loss(pose, p3d, generated_heatmaps, heatmap)
            loss = loss_3d_pose+loss_2d_ghm+loss_2d_hm
            loss.backward()
            opt.step()


            writer.add_scalar('3D loss', loss_3d_pose.item(), global_step=iterate)
            writer.add_scalar('Generated Heat Map loss', loss_2d_ghm.item(), global_step=iterate)
            writer.add_scalar('Total HM loss', loss_2d_hm.item(), global_step=iterate)

            writer.add_scalar('LR', learning_rate, global_step=iterate)
            with torch.no_grad():
                MPJPE = torch.mean(torch.pow(p3d-pose, 2))
            writer.add_scalar('Mean Per-Joint Position Error', MPJPE, global_step=iterate)
            if iterate % args.display_freq == 0:
                writer.add_image('Pred_heatmap', torch.clip(torch.sum(heatmap, dim=1, keepdim=True), 0, 1), global_step=iterate)
                writer.add_image('Generated_heatmap', torch.clip(torch.sum(generated_heatmaps, dim=1, keepdim=True), 0, 1), global_step=iterate)
                writer.add_image('GT_Heatmap', torch.clip(torch.sum(p2d, dim=1, keepdim=True), 0, 1), iterate)
                writer.add_image('GT_Image', img, iterate)
                #PLOT GT 3D pose, PRED 3D pose
                with torch.no_grad():
                    gt_pose = p3d.cpu().numpy()
                    pred_pose = pose.detach().cpu().numpy()
                    batch_dim = gt_pose.shape[0]
                    fig = plt.figure(figsize=(20, 10))
                    for batch_ind in range(batch_dim):
                        ax = fig.add_subplot(2, batch_dim, batch_ind+1, projection='3d')

                        xs = gt_pose[batch_ind, :, 0]
                        ys = gt_pose[batch_ind, :, 1]
                        zs = -gt_pose[batch_ind, :, 2]

                        def renderBones():
                            link = [[0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [2, 8]
                                , [8, 9], [9, 10], [10, 11], [8, 12]
                                , [5, 12], [5, 6], [6, 7], [12, 13], [13, 14], [14, 15]]
                            for l in link:
                                index1, index2 = l[0], l[1]
                                ax.plot([xs[index1], xs[index2]], [ys[index1], ys[index2]], [zs[index1], zs[index2]],
                                        linewidth=1)
                        renderBones()
                        ax.scatter(xs, ys, zs)

                        ax.set_xlim(-1, 1)
                        ax.set_ylim(-1, 1)
                        ax.set_zlim(-1, 1)
                        ax.title.set_text(f'Ground Truth {batch_ind}')
                    for batch_ind in range(batch_dim):
                        ax = fig.add_subplot(2, batch_dim, batch_dim+batch_ind+1, projection='3d')


                        xs = pred_pose[batch_ind, :, 0]
                        ys = pred_pose[batch_ind, :, 1]
                        zs = -pred_pose[batch_ind, :, 2]

                        def renderBones():
                            link = [[0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [2, 8]
                                , [8, 9], [9, 10], [10, 11], [8, 12]
                                , [5, 12], [5, 6], [6, 7], [12, 13], [13, 14], [14, 15]]
                            for l in link:
                                index1, index2 = l[0], l[1]
                                ax.plot([xs[index1], xs[index2]], [ys[index1], ys[index2]], [zs[index1], zs[index2]],
                                        linewidth=1, label=r'$z=y=x$')
                        renderBones()
                        ax.scatter(xs, ys, zs)

                        ax.set_xlim(-1, 1)
                        ax.set_ylim(-1, 1)
                        ax.set_zlim(-1, 1)
                        ax.title.set_text(f'Pred {batch_ind}')
                    fig.savefig(os.path.join(plot_3d_dir, '3D_plot'))
                    fig.clf()
            if iterate % (args.batch_size * (1000 // args.batch_size)) == 0:
                if i != 0:
                    torch.save(model.state_dict(),
                               os.path.join(weight_save_dir, '{}epo_{}step.ckpt'.format(epo, iterate)))
            if iterate % 1000 == 0 and i != 0:
                for file in weight_save_dir:
                    if '00' in file and '000' not in file:
                        os.remove(os.path.join(weight_save_dir, file))

            if iterate % (args.batch_size * (decay_step // args.batch_size)) == 0 and i != 0:
                learning_rate *= lr_decay
                opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
                # opt_front = torch.optim.Adam(
                #     list(model.resnet101.parameters()) + list(model.heatmap_deconv.parameters()), lr=learning_rate)
                # opt_back = torch.optim.Adam(
                #     list(model.encoder.parameters()) + list(model.pose_decoder.parameters()) + list(
                #         model.heatmap_decoder.parameters()),
                #     lr=learning_rate)
            iterate += args.batch_size

        start_iter = 0
