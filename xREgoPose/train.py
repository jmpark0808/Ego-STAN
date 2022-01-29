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
from network import *
from dataset.mocap import Mocap
from utils import config, evaluate
from base import SetType
import dataset.transform as trsf
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--load_hm",
                        help="Directory of pre-trained model for heatmap,  \n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model")
    parser.add_argument("--load_pose",
                        help="Directory of pre-trained model for pose estimator,  \n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model")
    parser.add_argument('--dataset_tr', help='Directory of your train Dataset', required=True, default=None)
    parser.add_argument('--dataset_val', help='Directory of your validation Dataset', required=True, default=None)
    parser.add_argument('--cuda', help="'cuda' for cuda, 'cpu' for cpu, default = cuda",
                        default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', help="batchsize, default = 1", default=1, type=int)
    parser.add_argument('--epoch', help='# of epochs. default = 20', default=20, type=int)
    parser.add_argument('--model_save_freq', help='How often to save model weights, in batch units', default=64, type=int)
    parser.add_argument('--val_freq', help='How often to run validation set, in batch units', default=64, type=int)
    parser.add_argument('--es_patience', help='Max # of consecutive validation runs w/o improvment', default=5, type=int)
    parser.add_argument('--logdir', help='logdir for models and losses. default = .', default='./', type=str)
    parser.add_argument('--lr_pose', help='learning_rate for pose. default = 0.001', default=0.001, type=float)
    parser.add_argument('--lr_hm', help='learning_rate for heat maps. default = 0.001', default=0.001, type=float)
    parser.add_argument('--lr_decay', help='Learning rate decrease by lr_decay time per decay_step, default = 0.1',
                        default=0.1, type=float)
    parser.add_argument('--decay_step', help='Learning rate decrease by lr_decay time per decay_step,  default = 7000',
                        default=1E100, type=int)
    parser.add_argument('--display_freq', help='Frequency to display result image on Tensorboard, in batch units',
                        default=64, type=int)


    args = parser.parse_args()
    device = torch.device(args.cuda)
    batch_size = args.batch_size
    epoch = args.epoch

    # initialize data transformation strategy
    data_transform = transforms.Compose([
        trsf.ImageTrsf(),
        trsf.Joints3DTrsf(),
        trsf.ToTensor()])

    # create train dataloader
    data_train = Mocap(
        args.dataset_tr,
        SetType.TRAIN,
        transform=data_transform)
    dataloader_train = DataLoader(
        data_train,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True)

    # create validation dataloader
    data_val = Mocap(
        args.dataset_val,
        SetType.VAL,
        transform=data_transform)
    dataloader_val = DataLoader(
        data_val,
        batch_size=args.batch_size,
        pin_memory=True)

    
    load_hm = args.load_hm
    load_pose = args.load_pose
    start_iter = 0
    model_hm = HeatMap().to(device=args.cuda)
    model_hm.train()
    model_pose = PoseEstimator().to(device=args.cuda)

    # Xavier Initialization
    def weight_init(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model_hm.resnet101.apply(weight_init)
    model_hm.heatmap_deconv.apply(weight_init)

    model_pose.encoder.apply(weight_init)
    model_pose.pose_decoder.apply(weight_init)
    model_pose.heatmap_decoder.apply(weight_init)

    now = datetime.datetime.now()
    start_epo = 0

    if load_hm is not None:
        state_dict_hm = torch.load(load_hm, map_location=args.cuda)


        start_iter = int(load_hm.split('epo_')[1].strip('step.ckpt'))
        start_epo = int(load_hm.split('/')[-1].split('epo')[0])
        now = datetime.datetime.strptime(load_hm.split('/')[-2], '%m%d%H%M')

        print("Loading Model from {}".format(load_hm))
        print("Start_iter : {}".format(start_iter))
        print("now : {}".format(now.strftime('%m%d%H%M')))
        model_hm.load_state_dict(state_dict_hm)
        print('Loading_Complete')

    if load_pose is not None:
        state_dict_pose = torch.load(load_pose, map_location=args.cuda)

        start_iter = int(load_pose.split('epo_')[1].strip('step.ckpt'))
        start_epo = int(load_pose.split('/')[-1].split('epo')[0])
        now = datetime.datetime.strptime(load_pose.split('/')[-2], '%m%d%H%M')

        print("Loading Model from {}".format(load_pose))
        print("Start_iter : {}".format(start_iter))
        print("now : {}".format(now.strftime('%m%d%H%M')))
        model_pose.load_state_dict(state_dict_pose)
        print('Loading_Complete')

    # Optimizer Setup
    learning_rate_hm = args.lr_hm
    learning_rate_pose = args.lr_pose
    lr_decay = args.lr_decay
    decay_step = args.decay_step
    learning_rate_hm = learning_rate_hm * (lr_decay ** (start_iter // decay_step))
    learning_rate_pose = learning_rate_pose * (lr_decay ** (start_iter // decay_step))
    opt_hm = torch.optim.AdamW(model_hm.parameters(), lr=learning_rate_hm, weight_decay=0.01)
    opt_pose = torch.optim.AdamW(model_pose.parameters(), lr=learning_rate_pose, weight_decay=0.01)

    # Logger Setup
    os.makedirs(os.path.join('log', now.strftime('%m%d%H%M')), exist_ok=True)
    weight_save_dir_hm = os.path.join(args.logdir, os.path.join('models/hm', 'state_dict', now.strftime('%m%d%H%M')))
    weight_save_dir_pose = os.path.join(args.logdir, os.path.join('models/pose', 'state_dict', now.strftime('%m%d%H%M')))
    val_weight_save_dir_hm = os.path.join(args.logdir, os.path.join('validation/hm', 'state_dict', now.strftime('%m%d%H%M')))
    val_weight_save_dir_pose = os.path.join(args.logdir, os.path.join('validation/pose', 'state_dict', now.strftime('%m%d%H%M')))
    plot_3d_dir = os.path.join(args.logdir, os.path.join('3d_plot', now.strftime('%m%d%H%M')))
    os.makedirs(os.path.join(weight_save_dir_hm), exist_ok=True)
    os.makedirs(os.path.join(weight_save_dir_pose), exist_ok=True)
    os.makedirs(os.path.join(val_weight_save_dir_hm), exist_ok=True)
    os.makedirs(os.path.join(val_weight_save_dir_pose), exist_ok=True)
    os.makedirs(os.path.join(plot_3d_dir), exist_ok=True)
    writer = SummaryWriter(os.path.join(args.logdir, os.path.join('log', now.strftime('%m%d%H%M'))))
    iterate = start_iter

    validation_metrics = {
            "best_mpjpe": None,
            "best_step": None,
            "current_patience": args.es_patience,
            }
    decay_max = iterate // (args.batch_size * (decay_step // args.batch_size))

    for epo in range(start_epo, epoch):
        print("\nEpoch : {}".format(epo))
        for batch_count, batch in enumerate(tqdm(dataloader_train)):
            img, p2d, p3d, action = batch
            img = img.cuda()
            p2d = p2d.cuda()
            p3d = p3d.cuda()

            model_hm.train()
            opt_hm.zero_grad()
            heatmap = model_hm(img)
            heatmap = torch.sigmoid(heatmap)
            loss = mse(heatmap, p2d)
            loss.backward()
            opt_hm.step()
            writer.add_scalar('Total HM loss', loss.item(), global_step=iterate)

            if loss.item() < 0.1:
                model_pose.train()
                opt_pose.zero_grad()
                heatmap = heatmap.detach()
                generated_heatmap, pose = model_pose(heatmap)
                generated_heatmap = torch.sigmoid(generated_heatmap)
                loss_3d_pose, loss_2d_ghm = auto_encoder_loss(pose, p3d, generated_heatmap, heatmap)
                loss = loss_2d_ghm+loss_3d_pose
                loss.backward()
                opt_pose.step()
                writer.add_scalar('Total 3D loss', loss_3d_pose.item(), global_step=iterate)
                writer.add_scalar('Total GHM loss', loss_2d_ghm.item(), global_step=iterate)
            else:
                model_pose.eval()
                with torch.no_grad():
                    heatmap = heatmap.detach()
                    generated_heatmap, pose = model_pose(heatmap)
                    generated_heatmap = torch.sigmoid(generated_heatmap)

            with torch.no_grad(): 
                l2_reg_hm = torch.tensor(0., device=device)
                l2_reg_pose = torch.tensor(0., device=device)
                for param in model_hm.parameters():
                    l2_reg_hm += torch.norm(param)
                for param in model_pose.parameters():
                    l2_reg_pose += torch.norm(param)

            writer.add_scalar('Regularization_HM', l2_reg_hm, global_step=iterate)
            writer.add_scalar('Regularization_Pose', l2_reg_pose, global_step=iterate)
            writer.add_scalar('LR_hm', learning_rate_hm, global_step=iterate)
            writer.add_scalar('LR_pose', learning_rate_pose, global_step=iterate)
            with torch.no_grad():
                MPJPE = torch.mean(torch.sqrt(torch.sum(torch.pow(p3d-pose, 2), dim=2)))
            writer.add_scalar('Mean Per-Joint Position Error', MPJPE, global_step=iterate)
            
            
            # TODO iterate is updated in increments of batch_size so it will skip
            # `iterate % ... == 0` checks
            if batch_count % args.val_freq == 0 and iterate != 0:
                # evaluate the validation set
                model_hm.eval()
                model_pose.eval()
                # Initialize evaluation pipline
                eval_body = evaluate.EvalBody()
                eval_upper = evaluate.EvalUpperBody()
                eval_lower = evaluate.EvalLowerBody()
                with torch.no_grad():
                    for img_val, p2d_val, p3d_val, action_val in tqdm(dataloader_val):
                        img_val = img_val.cuda()
                        p3d_val = p3d_val.cuda()
                        heatmap_val = model_hm(img_val)
                        heatmap_val = torch.sigmoid(heatmap_val)
                        heatmap_val = heatmap_val.detach()
                        _, pose_val = model_pose(heatmap_val)
                        # evaluate mpjpe for upper, lower and full body
                        # converting to numpy might cost time
                        y_output = pose_val.data.cpu().numpy()
                        y_target = p3d_val.data.cpu().numpy()

                        eval_body.eval(y_output, y_target, action_val)
                        eval_upper.eval(y_output, y_target, action_val)
                        eval_lower.eval(y_output, y_target, action_val)

                    val_mpjpe = eval_body.get_results()
                    val_mpjpe_upper = eval_upper.get_results()
                    val_mpjpe_lower = eval_lower.get_results()

                    writer.add_scalar("Validation MPJPE Fully Body", val_mpjpe['All']['mpjpe'], global_step=iterate)
                    writer.add_scalar("Validation MPJPE Upper Body", val_mpjpe_upper['All']['mpjpe'], global_step=iterate)
                    writer.add_scalar("Validation MPJPE Lower Body", val_mpjpe_lower['All']['mpjpe'], global_step=iterate)


                    if validation_metrics['best_mpjpe'] is None or validation_metrics['best_mpjpe'] > val_mpjpe['All']['mpjpe']:
                        validation_metrics['best_step'] = iterate
                        validation_metrics['best_mpjpe'] = val_mpjpe['All']['mpjpe']
                        # reset patience
                        validation_metrics['current_patience'] = args.es_patience

                        # list previously stored checkpoint
                        model_hm_paths = os.listdir(val_weight_save_dir_hm)
                        model_pose_paths = os.listdir(val_weight_save_dir_pose)

                        # remove the previous ckpt
                        if len(model_hm_paths) > 0:
                            for model_path in model_hm_paths:
                                os.remove(os.path.join(val_weight_save_dir_hm, model_path))
                        if len(model_pose_paths) > 0:
                            for model_path in model_pose_paths:
                                os.remove(os.path.join(val_weight_save_dir_pose, model_path))

                        # save model checkpoints
                        torch.save(model_hm.state_dict(),
                                   os.path.join(val_weight_save_dir_hm, '{}epo_{}step.ckpt'.format(epo, iterate)))
                        torch.save(model_pose.state_dict(),
                                   os.path.join(val_weight_save_dir_pose, '{}epo_{}step.ckpt'.format(epo, iterate)))
                    else:
                        validation_metrics['current_patience'] -= 1

                    # trigger early stopping if patience is up
                    if validation_metrics['current_patience'] < 0:
                        break


            if batch_count % args.display_freq == 0:
                writer.add_image('Pred_heatmap', torch.clip(torch.sum(heatmap, dim=1, keepdim=True), 0, 1), global_step=iterate)
                writer.add_image('Generated_heatmap', torch.clip(torch.sum(generated_heatmap, dim=1, keepdim=True), 0, 1), global_step=iterate)
                writer.add_image('GT_Heatmap', torch.clip(torch.sum(p2d, dim=1, keepdim=True), 0, 1), iterate)
                writer.add_image('GT_Image', img, iterate)
                #PLOT GT 3D pose, PRED 3D pose
                with torch.no_grad():
                    gt_pose = p3d.cpu().numpy()
                    pred_pose = pose.detach().cpu().numpy()
                    batch_dim = gt_pose.shape[0]
                    fig = plt.figure(figsize=(20*(batch_dim//8), 10))
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

            # store 5 most recent model_hm and model_pose checkpoints
            if batch_count % args.model_save_freq == 0:
                if iterate != 0:
                    torch.save(model_hm.state_dict(),
                               os.path.join(weight_save_dir_hm, '{}epo_{}step.ckpt'.format(epo, iterate)))
                    torch.save(model_pose.state_dict(),
                               os.path.join(weight_save_dir_pose, '{}epo_{}step.ckpt'.format(epo, iterate)))
                    if len(os.listdir(os.path.join(weight_save_dir_hm))) > 5:
                        model_dict = {}
                        for model_path in os.listdir(os.path.join(weight_save_dir_hm)):
                            print(model_path)
                            print(model_path.split('epo_'))
                            iter = model_path.split('epo_')[1].split('step')[0]
                            model_dict[model_path] = int(iter)
                        total_files = len(model_dict)
                        for k, v in sorted(model_dict.items(), key=lambda item: item[1]):
                            os.remove(os.path.join(weight_save_dir_hm, k))
                            total_files -= 1
                            if total_files == 5:
                                break

                    if len(os.listdir(os.path.join(weight_save_dir_pose))) > 5:
                        model_dict = {}
                        for model_path in os.listdir(os.path.join(weight_save_dir_pose)):
                            iter = model_path.split('epo_')[1].split('step')[0]
                            model_dict[model_path] = int(iter)
                        total_files = len(model_dict)
                        for k, v in sorted(model_dict.items(), key=lambda item: item[1]):
                            os.remove(os.path.join(weight_save_dir_pose, k))
                            total_files -= 1
                            if total_files == 5:
                                break


            if iterate // (args.batch_size * (decay_step // args.batch_size)) > decay_max and batch_count != 0:
                decay_max = iterate // (args.batch_size * (decay_step // args.batch_size))
                learning_rate_hm *= lr_decay
                learning_rate_pose *= lr_decay
                opt_hm = torch.optim.AdamW(model_hm.parameters(), lr=learning_rate_hm, weight_decay=0.01)
                opt_pose = torch.optim.AdamW(model_pose.parameters(), lr=learning_rate_pose, weight_decay=0.01)
            iterate += args.batch_size

        
