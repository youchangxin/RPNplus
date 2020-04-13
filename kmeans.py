# -*- coding: utf-8 -*-
import glob
import numpy as np


def load_gt_boxes(path):
    bboxes = open(path).readlines()[1:]
    roi = np.zeros([len(bboxes), 4], dtype=np.int64)
    for iter_, bb in zip(range(len(bboxes)), bboxes):
        bb = bb.replace('\n', '').split(' ')
        bbtype = bb[0]
        bba = np.array([int(bb[i]) for i in range(1, 5)])
        # occ = float(bb[5])
        # bbv = np.array([float(bb[i]) for i in range(6, 10)])
        # ignore = int(bb[10])

        # ignore = ignore or (bbtype != 'person')
        # ignore = ignore or (bba[3] < 40)
        bba[2] += bba[0]
        bba[3] += bba[1]

        roi[iter_, :4] = bba
    return roi


def get_wh_from_boxes(boxes):
    """
    box shape: [-1, 4], return the width and height of boxes
    """
    return boxes[..., 2:4] - boxes[..., 0:2]



def iou(box, clusters):
    '''
    :param box:      np.array of shape (2,) containing w and h
    :param clusters: np.array of shape (N cluster, 2)
    '''
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def kmeans(boxes, k, dist=np.median,seed=1):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances     = np.empty((rows, k)) ## N row x N cluster
    last_clusters = np.zeros((rows,))

    np.random.seed(seed)

    # initialize the cluster centers to be k items
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for icluster in range(k): # I made change to lars76's code here to make the code faster
            distances[:,icluster] = 1 - iou(clusters[icluster], boxes)

        nearest_clusters = np.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


text_paths = glob.glob("./data/synthetic_dataset/imageAno/*.txt")
all_boxes = [load_gt_boxes(path) for path in text_paths]
all_boxes = np.vstack(all_boxes)
all_boxes_wh = get_wh_from_boxes(all_boxes)
anchors = kmeans(all_boxes_wh, k=9)
print(anchors)
