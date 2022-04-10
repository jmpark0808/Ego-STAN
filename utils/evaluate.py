# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Class for model evaluation

"""
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import bone
from mpl_toolkits.mplot3d.axes3d import get_test_data

import numpy as np
from base import BaseEval
import scipy.io
import os

__all__ = ["EvalBody", "EvalUpperBody", "EvalLowerBody"]

mean3D = scipy.io.loadmat(os.path.join(os.path.expanduser('~'), 'projects/def-pfieguth/mo2cap/code/util/mean3D.mat'))['mean3D']/1000. # 3x15 shape
# mean3D = scipy.io.loadmat('/home/eddie/scripts/code/util/mean3D.mat')['mean3D']/1000.
kinematic_parents = [ 0, 0, 1, 2, 0, 4, 5, 1, 7, 8, 9, 4, 11, 12, 13]
bones_mean = mean3D - mean3D[:,kinematic_parents]
bone_length = np.sqrt(np.sum(np.power(bones_mean, 2), axis=0)) # 15 shape

def get_p3ds_t(p3d_preds, p3d_gts):
    """
    Retrieve the 3D Poses, Predicted, Ground Truth and Procrustes Aligned
    Ground Truth.

    :param p3d_preds: Predicted 3D joints [batch_size x N x 3]
    :param p3d_gts: Ground Truth 3D joints [batch_size x N x 3]
    :return p3d_preds_t, p3d_gt_rot_t: [batch_size x N x 3], [batch_size x N x 3]
        Rescaled Predicted 3D joints, 
        Rescaled Procrustes Aligned Ground Truth 3D joints 
    """

    gt_rots_t = np.zeros((p3d_gts.shape[0], p3d_gts.shape[1], p3d_gts.shape[2]))
    preds_t = np.zeros((p3d_gts.shape[0], p3d_gts.shape[1], p3d_gts.shape[2]))

    for i, (p3d_pred, p3d_gt) in enumerate(zip(p3d_preds, p3d_gts)):
        if p3d_pred.shape[0] != 3:
            pred = np.transpose(p3d_pred, [1, 0])

        if p3d_gt.shape[0] != 3:
            gt = np.transpose(p3d_gt, [1, 0])
        assert pred.shape == gt.shape

        gt = skeleton_rescale(gt, bone_length[1:], kinematic_parents)
        pred = skeleton_rescale(pred, bone_length[1:], kinematic_parents)
        _, gt_rot, _ = procrustes(np.transpose(pred), np.transpose(gt), True, False)

        gt_rots_t[i] = gt_rot
        preds_t[i] = np.transpose(pred)

    return preds_t, gt_rots_t

def plot_skels(p3ds, savepath=None):

    """
    Returns matplotlib figure based on inputted 3D-Pose co-ordinates.
    Currently only supports 16 or 15 joints and even batch sizes.

    :param p3d: 3D Pose in batch_size x N x 3
    :return fig: matplotlib figure
    """

    fig = plt.figure(figsize=(6*len(p3ds)//2, 2*4))
    fig.tight_layout()
   
    # Check if there are any p3ds, return the fig as is if none
    if p3ds is None or len(p3ds) == 0:
        return fig

    if len(p3ds)%2 == 1:
        print("WARNING: Function cannot deal with odd batch-size")
        return fig

    for i, p3d in enumerate(p3ds):

        ax = fig.add_subplot(2, len(p3ds)//2, i+1, projection='3d')

        if p3d.shape[0] == 15:
            p3d_a = np.zeros((16, 3))
            p3d_a[1:, :] = p3d
        else:
            p3d_a = p3d

        #ax.set_xlim(-1, 1)
        #ax.set_ylim(-1, 1)
        #ax.set_zlim(-1, 1)
        plt.axis("off")

        ax.view_init(elev=27.0, azim=41.0)

        bone_links = [
            #    [0, 1],
                [1, 2],
                [1, 5],
                [2, 3],
                [3, 4],
                [2, 8],
                [8, 9],
                [9, 10],
                [10, 11],
                [8, 12],
                [5, 12],
                [5, 6],
                [6, 7],
                [12, 13],
                [13, 14],
                [14, 15],
            ]

        pose = p3d_a
        xs = pose[:, 0]
        ys = pose[:, 1]
        zs = -pose[:, 2]

        # draw bones
        for bone in bone_links:
            index1, index2 = bone[0], bone[1]
            ax.plot3D(
                [xs[index1], xs[index2]],
                [ys[index1], ys[index2]],
                [zs[index1], zs[index2]],
                linewidth=1, color = 'r'
            )
        # draw joints
        ax.scatter(xs, ys, zs, color = 'r')
        
        ax.set_title("X: {0}, Y: {1}, Z: {2}".format(
                                    [round(lim, 2) for lim in ax.get_xlim()], 
                                    [round(lim, 2) for lim in ax.get_ylim()], 
                                    [round(lim, 2) for lim in ax.get_zlim()]))
    
    if savepath is not None:
        fig.savefig(savepath)

    return fig

def plot_skels_compare(p3ds_1, p3ds_2, label_1, label_2, savepath=None):

    """
    Returns matplotlib figure based on inputted 3D-Pose co-ordinates.
    Currently only supports 16 or 15 joints and even batch sizes.

    :param p3d: 3D Pose in batch_size x N x 3
    :return fig: matplotlib figure
    """

    fig = plt.figure(figsize=(6*len(p3ds_1)//2, 2*4))
    fig.tight_layout()
   
    # Check if there are any p3ds, return the fig as is if none
    if p3ds_1 is None or len(p3ds_1) == 0:
        return fig

    if len(p3ds_1)%2 == 1:
        print("WARNING: Function cannot deal with odd batch-size")
        return fig

    for i, (p3d, p3dx) in enumerate(zip(p3ds_1, p3ds_2)):

        ax = fig.add_subplot(2, len(p3ds_1)//2, i+1, projection='3d')

        if p3d.shape[0] == 15:
            p3d_a = np.zeros((16, 3))
            p3d_a[1:, :] = p3d
            p3d_b = np.zeros((16,3))
            p3d_b[1:, :] = p3dx
        else:
            p3d_a = p3d
            p3d_b = p3dx

        #ax.set_xlim(-1, 1)
        #ax.set_ylim(-1, 1)
        #ax.set_zlim(-1, 1)
        plt.axis("off")

        ax.view_init(elev=27.0, azim=41.0)

        bone_links = [
            #    [0, 1],
                [1, 2],
                [1, 5],
                [2, 3],
                [3, 4],
                [2, 8],
                [8, 9],
                [9, 10],
                [10, 11],
                [8, 12],
                [5, 12],
                [5, 6],
                [6, 7],
                [12, 13],
                [13, 14],
                [14, 15],
            ]

        pose_a = p3d_a
        xs_a = pose_a[:, 0]
        ys_a = pose_a[:, 1]
        zs_a = -pose_a[:, 2]

        pose_b = p3d_b
        xs_b = pose_b[:, 0]
        ys_b = pose_b[:, 1]
        zs_b = -pose_b[:, 2]

        # draw bones A
        for bone in bone_links:
            index1, index2 = bone[0], bone[1]
            ax.plot3D(
                [xs_a[index1], xs_a[index2]],
                [ys_a[index1], ys_a[index2]],
                [zs_a[index1], zs_a[index2]],
                linewidth=1, color = 'r'
            )
        # draw joints A
        ax.scatter(xs_a, ys_a, zs_a, color = 'r', label=label_1)

        # draw bones B
        for bone in bone_links:
            index1, index2 = bone[0], bone[1]
            ax.plot3D(
                [xs_b[index1], xs_b[index2]],
                [ys_b[index1], ys_b[index2]],
                [zs_b[index1], zs_b[index2]],
                linewidth=1, color = 'b'
            )
        # draw joints B
        ax.scatter(xs_b, ys_b, zs_b, color = 'b', label=label_2)

        
        ax.set_title("X: {0}, Y: {1}, Z: {2}".format(
                                            [round(lim, 2) for lim in ax.get_xlim()], 
                                            [round(lim, 2) for lim in ax.get_ylim()], 
                                            [round(lim, 2) for lim in ax.get_zlim()]))

        ax.legend()
    
    if savepath is not None:
        fig.savefig(savepath)

    return fig


def skeleton_rescale(joints, bone_length, kinematic_parents):
    bones = joints[:, 1:] - joints[:,kinematic_parents[1:]] # 3 x 14 
    bones_rescale = bones * bone_length/np.sqrt(np.sum(np.power(bones, 2), axis=0)) # 3 x 14
    #bones_rescale = bsxfun(@times, bones, bone_length./sqrt(sum(bones.^2,1))) 

    joints_rescaled = np.zeros_like(joints)
    joints_rescaled[:, 0] = joints[:, 0]
    for i in range(1, 15):
        joints_rescaled[:, i] = joints_rescaled[:, kinematic_parents[i]] + bones_rescale[:, i-1]
    # for i=2:(size(joints_rescaled,2))
    #     joints_rescaled(:,i) = joints_rescaled(:,kinematic_parents(i))+bones_rescale(:,i-1);
    return joints_rescaled

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)
    
    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}
   
    return d, Z, tform

def create_results_csv(mpjpe_dict: dict, csv_path: str, mode: str = 'baseline'):
    """
    Save a csv of mpjpe evalutions stored in a dict.
    Refer to the `test_results` dict in DirectRegression.test_epoch_end
    for the expected structure for `mpjpe_dict`.
    """

    m_to_mm = 1000

    # get csv column names
    action_list = list(mpjpe_dict["Full Body"].keys())
    action_list.sort()
    columns = ["Evalution Error [mm]"]
    columns.extend(action_list)
    print(f"[print] columns: {columns}")
    if mode == 'baseline' or mode == 'sequential':
        joints = ['Head','Neck', 'LeftArm', 'LeftForeArm',
     'LeftHand', 'RightArm', 'RightForeArm', 'RightHand',
     'LeftUpLeg', 'LeftLeg','LeftFoot','LeftToeBase',
     'RightUpLeg','RightLeg','RightFoot','RightToeBase']
    elif mode == 'mo2cap2':
        joints = ['Neck', 'LeftArm', 'LeftForeArm',
     'LeftHand', 'RightArm', 'RightForeArm', 'RightHand',
     'LeftUpLeg', 'LeftLeg','LeftFoot','LeftToeBase',
     'RightUpLeg','RightLeg','RightFoot','RightToeBase']
    else:
        raise('Not a valid mode')

    with open(csv_path, mode="w") as f:
        mpjpe_writer = csv.writer(f)
        mpjpe_writer.writerow(columns)
        for body_split, action_dict in mpjpe_dict.items():
            if body_split != 'Per Joint':
                # the first column is the body split (e.g. "Full Body")
                row = [body_split]
                row_std = [body_split + " Error STD"]
                # store mpjpe in order of sorted 'action_list'
                for action in action_list:
                    row.append(action_dict[action]["mpjpe"] * m_to_mm)
                    row_std.append(action_dict[action]["std_mpjpe"] * m_to_mm)

                mpjpe_writer.writerow(row)
                mpjpe_writer.writerow(row_std)

        mpjpe_writer.writerow(joints)
        mpjpe_writer.writerow((mpjpe_dict['Per Joint']*m_to_mm).tolist())

        


    


def compute_error(pred, gt, return_mean=True, mode='baseline'):
    """Compute error

    Arguments:
        pred {np.ndarray} -- format (N x 3)
        gt {np.ndarray} -- format (N x 3)

    Returns:
        float -- error
    """
    if mode == 'baseline' or mode == 'sequential':
        if pred.shape[1] != 3:
            pred = np.transpose(pred, [1, 0])

        if gt.shape[1] != 3:
            gt = np.transpose(gt, [1, 0])

        assert pred.shape == gt.shape
        error = np.sqrt(np.sum((pred - gt) ** 2, axis=1))
        if return_mean:
            return np.mean(error)
        else:
            return error
    elif mode == 'mo2cap2':
        if pred.shape[0] != 3:
            pred = np.transpose(pred, [1, 0])

        if gt.shape[0] != 3:
            gt = np.transpose(gt, [1, 0])
        assert pred.shape == gt.shape
        gt_rescale = skeleton_rescale(gt, bone_length[1:], kinematic_parents)
        pred_rescale = skeleton_rescale(pred, bone_length[1:], kinematic_parents)
        _, gt_rot, _ = procrustes(np.transpose(pred_rescale), np.transpose(gt_rescale), True, False)
        error = pred - np.transpose(gt_rot)
        joint_error = np.sqrt(np.sum(np.power(error, 2), axis=0)) 
        if return_mean:
            return np.mean(joint_error)
        else:
            return joint_error

    else:
        raise('Not a valid mode')

class EvalBody(BaseEval):
    """Eval entire body"""
    def __init__(self, mode='baseline'):
        super().__init__()
        self.mode = mode

    def eval(self, pred, gt, actions=None):
        """Evaluate

        Arguments:
            pred {np.ndarray} -- predictions, format (N x 3)
            gt {np.ndarray} -- ground truth, format (N x 3)

        Keyword Arguments:
            action {str} -- action name (default: {None})
        """

        for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
            err = compute_error(pose_in, pose_target, mode=self.mode)

            if actions:
                act_name = self._map_action_name(actions[pid])

                # add element to dictionary if not there yet
                if not self._is_action_stored(act_name):
                    self._init_action(act_name)
                self.error[act_name].append(err)

            # add to all
            act_name = "All"
            self.error[act_name].append(err)

    def desc(self):
        return "Average3DError"


class EvalUpperBody(BaseEval):
    """Eval upper body"""
    def __init__(self, mode='baseline'):
        super().__init__()
        self.errors = []
        if mode == 'baseline' or mode == 'sequential':
            self._SEL = [0, 1, 2, 3, 4, 5, 6, 7]
        elif mode == 'mo2cap2':
            self._SEL = [0, 1, 2, 3, 4, 5, 6]
        else:
            raise('Not a valid mode')
        self.mode = mode

    def eval(self, pred, gt, actions=None):
        """Evaluate

        Arguments:
            pred {np.ndarray} -- predictions, format (N x 3)
            gt {np.ndarray} -- ground truth, format (N x 3)

        Keyword Arguments:
            action {str} -- action name (default: {None})
        """

        for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
            if self.mode == 'baseline' or self.mode =='sequential':
                err = compute_error(pose_in[self._SEL], pose_target[self._SEL], mode=self.mode)
            elif self.mode == 'mo2cap2':
                err = compute_error(pose_in, pose_target, return_mean=False, mode=self.mode)
                err = np.mean(err[self._SEL])
                

            if actions:
                act_name = self._map_action_name(actions[pid])

                # add element to dictionary if not there yet
                if not self._is_action_stored(act_name):
                    self._init_action(act_name)
                self.error[act_name].append(err)

            # add to all
            act_name = "All"
            self.error[act_name].append(err)

    def desc(self):
        return "UpperBody_Average3DError"


class EvalLowerBody(BaseEval):
    """Eval lower body"""

    def __init__(self, mode='baseline'):
        super().__init__()
        self.errors = []
        if mode == 'baseline' or mode == 'sequential':
            self._SEL = [8, 9, 10, 11, 12, 13, 14, 15]
        elif mode == 'mo2cap2':
            self._SEL = [7, 8, 9, 10, 11, 12, 13, 14]
        else:
            raise('Not a valid mode')
        self.mode = mode

    def eval(self, pred, gt, actions=None):
        """Evaluate

        Arguments:
            pred {np.ndarray} -- predictions, format (N x 3)
            gt {np.ndarray} -- ground truth, format (N x 3)

        Keyword Arguments:
            action {str} -- action name (default: {None})
        """

        for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
            if self.mode == 'baseline' or self.mode =='sequential':
                err = compute_error(pose_in[self._SEL], pose_target[self._SEL], mode=self.mode)
            elif self.mode == 'mo2cap2':
                err = compute_error(pose_in, pose_target, return_mean=False, mode=self.mode)
                err = np.mean(err[self._SEL])

            if actions:
                act_name = self._map_action_name(actions[pid])

                # add element to dictionary if not there yet
                if not self._is_action_stored(act_name):
                    self._init_action(act_name)
                self.error[act_name].append(err)

            # add to all
            act_name = "All"
            self.error[act_name].append(err)

    def desc(self):
        return "LowerBody_Average3DError"


class EvalPerJoint(object):
    """Eval MPJPE per joint body"""
    def __init__(self, mode='baseline'):
        super().__init__()
        self.errors = []
        self.mode = mode

    def eval(self, pred, gt):
        """Evaluate

        Arguments:
            pred {np.ndarray} -- predictions, format (N x 3)
            gt {np.ndarray} -- ground truth, format (N x 3)

        """

        for (pose_in, pose_target) in zip(pred, gt):
            err = compute_error(pose_in, pose_target, return_mean=False, mode=self.mode)
            # err = Error per joint
            self.errors.append(err)

    def get_results(self):
        stacked = np.array(self.errors)
        stacked = np.mean(stacked, axis=0)
        return stacked

    
