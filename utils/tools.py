# -*- coding: utf-8 -*-
import numpy as np


def iou(bboxes1, bboxes2):
    bboxes1_area = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    bboxes2_area = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    left_up = np.maximum(bboxes1[..., :2], bboxes2[..., :2])
    right_down = np.minimum(bboxes1[..., 2:], bboxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area
    ious = inter_area / union_area
    return ious


def compute_regression(bboxes1, bboxes2):
    '''
    :param bboxes1: ground-truth boxes
    :param bboxes2: anchor boxes
    :return:
    '''
    target_reg = np.zeros(shape=[4, ])
    w1 = bboxes1[2] - bboxes1[0]
    h1 = bboxes1[3] - bboxes1[1]
    w2 = bboxes2[2] - bboxes2[0]
    h2 = bboxes2[3] - bboxes2[1]

    target_reg[0] = (bboxes1[0] - bboxes2[0]) / w2
    target_reg[1] = (bboxes1[1] - bboxes2[1]) / h2
    target_reg[2] = np.log(w1 / w2)
    target_reg[3] = np.log(h1 / h2)

    return target_reg


def get_valid_num_bboxes(bboxes):
    count = 0
    for index in range(bboxes.shape[0]):
        if np.any(bboxes[index] != 0.0):
            count += 1
    return count
