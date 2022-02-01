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
from xREgoPose.net.xRNet import *
from dataset.mocap import Mocap
from utils import config, evaluate
from base import SetType
import dataset.transform as trsf
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--load",
                        help="Directory of pre-trained model,  \n"
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
    parser.add_argument('--lr', help='learning_rate for pose. default = 0.001', default=0.001, type=float)
    parser.add_argument('--lr_decay', help='Learning rate decrease by lr_decay time per decay_step, default = 0.1',
                        default=0.1, type=float)
    parser.add_argument('--decay_step', help='Learning rate decrease by lr_decay time per decay_step,  default = 7000',
                        default=1E100, type=int)
    parser.add_argument('--display_freq', help='Frequency to display result image on Tensorboard, in batch units',
                        default=64, type=int)
    parser.add_argument('--load_resnet', help='Directory of ResNet 101 weights', default=None)
    parser.add_argument('--hm_train_steps', help='Number of steps to pre-train heatmap predictor', default=100000, type=int)


    args = parser.parse_args()
    device = torch.device(args.cuda)
    batch_size = args.batch_size
    epoch = args.epoch
    hm_train_steps = args.hm_train_steps

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

    
    load = args.load
    start_iter = 0
    model = xREgoPose().to(device=args.cuda)


    # Xavier Initialization
    def weight_init(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model.heatmap.resnet101.apply(weight_init)
    if args.load_resnet:
        model.heatmap.resnet101.load_state_dict(torch.load(args.load_resnet))
    model.heatmap.update_resnet101()
    model.heatmap.heatmap_deconv.apply(weight_init)

    model.encoder.apply(weight_init)
    model.pose_decoder.apply(weight_init)
    model.heatmap_decoder.apply(weight_init)

    now = datetime.datetime.now()
    start_epo = 0

    if load is not None:
        state_dict_hm = torch.load(load, map_location=args.cuda)


        start_iter = int(load.split('epo_')[1].strip('step.ckpt'))
        start_epo = int(load.split('/')[-1].split('epo')[0])
        now = datetime.datetime.strptime(load.split('/')[-2], '%m%d%H%M')

        print("Loading Model from {}".format(load))
        print("Start_iter : {}".format(start_iter))
        print("now : {}".format(now.strftime('%m%d%H%M')))
        model.load_state_dict(state_dict_hm)
        print('Loading_Complete')


    # Optimizer Setup
    learning_rate = args.lr
    lr_decay = args.lr_decay
    decay_step = args.decay_step
    learning_rate = learning_rate * (lr_decay ** (start_iter // decay_step))

    opt = torch.optim.SGD(model.heatmap.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

    # Logger Setup
    os.makedirs(os.path.join('log', now.strftime('%m%d%H%M')), exist_ok=True)
    weight_save_dir = os.path.join(args.logdir, os.path.join('models', 'state_dict', now.strftime('%m%d%H%M')))
    val_weight_save_dir = os.path.join(args.logdir, os.path.join('validation', 'state_dict', now.strftime('%m%d%H%M')))
    plot_3d_dir = os.path.join(args.logdir, os.path.join('3d_plot', now.strftime('%m%d%H%M')))
    os.makedirs(os.path.join(weight_save_dir), exist_ok=True)
    os.makedirs(os.path.join(val_weight_save_dir), exist_ok=True)
    os.makedirs(os.path.join(plot_3d_dir), exist_ok=True)
    writer = SummaryWriter(os.path.join(args.logdir, os.path.join('log', now.strftime('%m%d%H%M'))))
    iterate = start_iter

    validation_metrics = {
            "best_mpjpe": None,
            "best_step": None,
            "current_patience": args.es_patience,
            }
    decay_max = iterate // (args.batch_size * (decay_step // args.batch_size))
    update_optim_flag = True
    for epo in range(start_epo, epoch):
        print("\nEpoch : {}".format(epo))
        for batch_count, batch in enumerate(tqdm(dataloader_train)):
            if hm_train_steps <= 0 and update_optim_flag:
                opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
                update_optim_flag = False
            img, p2d, p3d, action = batch
            img = img.cuda()
            p2d = p2d.cuda()
            p3d = p3d.cuda()
            real_batch_size = img.size(0)
            model.train()
            opt.zero_grad()
            heatmap, pose, generated_heatmap = model(img)
            if hm_train_steps > 0:
                heatmap = torch.sigmoid(heatmap)
                loss = mse(heatmap, p2d)
                writer.add_scalar('Total HM loss', loss.item(), global_step=iterate)
            else:
                heatmap = torch.sigmoid(heatmap)
                generated_heatmap = torch.sigmoid(generated_heatmap)
                hm_loss = mse(heatmap, p2d)
                loss_3d_pose, loss_2d_ghm = auto_encoder_loss(pose, p3d, generated_heatmap, heatmap)
                ae_loss = loss_2d_ghm + loss_3d_pose
                loss = hm_loss + ae_loss
                writer.add_scalar('Total HM loss', hm_loss.item(), global_step=iterate)
                writer.add_scalar('Total 3D loss', loss_3d_pose.item(), global_step=iterate)
                writer.add_scalar('Total GHM loss', loss_2d_ghm.item(), global_step=iterate)
                with torch.no_grad():
                    MPJPE = torch.mean(torch.sqrt(torch.sum(torch.pow(p3d-pose, 2), dim=2)))
                    writer.add_scalar('Mean Per-Joint Position Error', MPJPE, global_step=iterate)

            loss.backward()
            opt.step()
            
            with torch.no_grad(): 
                l2_reg = torch.tensor(0., device=device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)


            writer.add_scalar('Regularization', l2_reg, global_step=iterate)
            writer.add_scalar('LR', learning_rate, global_step=iterate)
            
                        
            # TODO iterate is updated in increments of batch_size so it will skip
            # `iterate % ... == 0` checks
            if batch_count % args.val_freq == 0 and iterate != 0:
                # evaluate the validation set
                model.eval()
                # Initialize evaluation pipline
                eval_body = evaluate.EvalBody()
                eval_upper = evaluate.EvalUpperBody()
                eval_lower = evaluate.EvalLowerBody()
                with torch.no_grad():
                    val_hm_loss = torch.tensor(0., device=device)
                    val_loss_3d_pose_total = torch.tensor(0., device=device)
                    for img_val, p2d_val, p3d_val, action_val in tqdm(dataloader_val):
                        img_val = img_val.cuda()
                        p3d_val = p3d_val.cuda()
                        p2d_val = p2d_val.cuda()
                        heatmap_val, pose_val, GHM_val = model(img_val)
                        heatmap_val = torch.sigmoid(heatmap_val)
                        GHM_val = torch.sigmoid(GHM_val)
   
                        val_hm_loss += mse(heatmap_val, p2d_val)
                        val_loss_3d_pose, val_loss_2d_ghm = auto_encoder_loss(pose_val, p3d_val, GHM_val, heatmap_val)
                        val_loss_3d_pose_total += val_loss_3d_pose
                        # evaluate mpjpe for upper, lower and full body
                        # converting to numpy might cost time
                        
                        
                        y_output = pose_val.data.cpu().numpy()
                        y_target = p3d_val.data.cpu().numpy()
                        eval_body.eval(y_output, y_target, action_val)
                        eval_upper.eval(y_output, y_target, action_val)
                        eval_lower.eval(y_output, y_target, action_val)


                    writer.add_image('Val_Pred_heatmap', torch.clip(torch.sum(heatmap_val, dim=1, keepdim=True), 0, 1), global_step=iterate)
                    writer.add_image('Val_GT_Heatmap', torch.clip(torch.sum(p2d_val, dim=1, keepdim=True), 0, 1), iterate)
                    writer.add_image('Val_GT_Image', img_val, iterate)
                    val_mpjpe = eval_body.get_results()
                    val_mpjpe_upper = eval_upper.get_results()
                    val_mpjpe_lower = eval_lower.get_results()

                    writer.add_scalar("Validation MPJPE Fully Body", val_mpjpe['All']['mpjpe'], global_step=iterate)
                    writer.add_scalar("Validation MPJPE Upper Body", val_mpjpe_upper['All']['mpjpe'], global_step=iterate)
                    writer.add_scalar("Validation MPJPE Lower Body", val_mpjpe_lower['All']['mpjpe'], global_step=iterate)
                    writer.add_scalar("Validation HM loss", val_hm_loss, global_step=iterate)
                    writer.add_scalar("Validation 3D loss", val_loss_3d_pose_total, global_step=iterate)

                    if validation_metrics['best_mpjpe'] is None or validation_metrics['best_mpjpe'] > val_mpjpe['All']['mpjpe']:
                        validation_metrics['best_step'] = iterate
                        validation_metrics['best_mpjpe'] = val_mpjpe['All']['mpjpe']
                        # reset patience
                        validation_metrics['current_patience'] = args.es_patience

                        # list previously stored checkpoint
                        model_paths = os.listdir(val_weight_save_dir)

                        # remove the previous ckpt
                        if len(model_paths) > 0:
                            for model_path in model_paths:
                                os.remove(os.path.join(val_weight_save_dir, model_path))

                        # save model checkpoints
                        torch.save(model.state_dict(),
                                   os.path.join(val_weight_save_dir, '{}epo_{}step.ckpt'.format(epo, iterate)))

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
                    torch.save(model.state_dict(),
                               os.path.join(weight_save_dir, '{}epo_{}step.ckpt'.format(epo, iterate)))

                    if len(os.listdir(os.path.join(weight_save_dir))) > 5:
                        model_dict = {}
                        for model_path in os.listdir(os.path.join(weight_save_dir)):
                            iter = model_path.split('epo_')[1].split('step')[0]
                            model_dict[model_path] = int(iter)
                        total_files = len(model_dict)
                        for k, v in sorted(model_dict.items(), key=lambda item: item[1]):
                            os.remove(os.path.join(weight_save_dir, k))
                            total_files -= 1
                            if total_files == 5:
                                break




            if iterate // (args.batch_size * (decay_step // args.batch_size)) > decay_max and batch_count != 0:
                decay_max = iterate // (args.batch_size * (decay_step // args.batch_size))
                learning_rate *= lr_decay
                if hm_train_steps > 0:
                    opt = torch.optim.SGD(model.heatmap.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
                else:
                    opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

            iterate += real_batch_size
            hm_train_steps -= real_batch_size


        
