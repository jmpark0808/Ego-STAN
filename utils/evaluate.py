# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Class for model evaluation

"""
import csv

import numpy as np
from base import BaseEval

__all__ = ["EvalBody", "EvalUpperBody", "EvalLowerBody"]


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

        


    


def compute_error(pred, gt, return_mean=True):
    """Compute error

    Arguments:
        pred {np.ndarray} -- format (N x 3)
        gt {np.ndarray} -- format (N x 3)

    Returns:
        float -- error
    """

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


class EvalBody(BaseEval):
    """Eval entire body"""

    def eval(self, pred, gt, actions=None):
        """Evaluate

        Arguments:
            pred {np.ndarray} -- predictions, format (N x 3)
            gt {np.ndarray} -- ground truth, format (N x 3)

        Keyword Arguments:
            action {str} -- action name (default: {None})
        """

        for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
            err = compute_error(pose_in, pose_target)

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

    def eval(self, pred, gt, actions=None):
        """Evaluate

        Arguments:
            pred {np.ndarray} -- predictions, format (N x 3)
            gt {np.ndarray} -- ground truth, format (N x 3)

        Keyword Arguments:
            action {str} -- action name (default: {None})
        """

        for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
            err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

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

    def eval(self, pred, gt, actions=None):
        """Evaluate

        Arguments:
            pred {np.ndarray} -- predictions, format (N x 3)
            gt {np.ndarray} -- ground truth, format (N x 3)

        Keyword Arguments:
            action {str} -- action name (default: {None})
        """

        for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
            err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

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
    def __init__(self):
        super().__init__()
        self.errors = []

    def eval(self, pred, gt):
        """Evaluate

        Arguments:
            pred {np.ndarray} -- predictions, format (N x 3)
            gt {np.ndarray} -- ground truth, format (N x 3)

        """

        for (pose_in, pose_target) in zip(pred, gt):
            err = compute_error(pose_in, pose_target, return_mean=False)
            # err = Error per joint
            self.errors.append(err)

    def get_results(self):
        stacked = np.array(self.errors)
        stacked = np.mean(stacked, axis=0)
        return stacked

    