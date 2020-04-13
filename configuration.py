# -*- coding: utf-8 -*-
from easydict import EasyDict as edict
import numpy as np

__C                         = edict()
# Consumers can config by : from config import cfg
cfg                         = __C

__C.RPN                     = edict()
__C.RPN.INPUT_SIZE          = [720, 960]
__C.RPN.STRIDE              = 16
__C.RPN.POS_THRESH          = 0.5
__C.RPN.NEG_THRESH          = 0.1
__C.RPN.MAX_PER_IMAGE       = 20
__C.RPN.SAVE_MODEL_DIR      = "./saved_model/"
__C.RPN.ANCHORS             = [[ 74., 149.],
                               [ 34., 149.],
                               [ 86.,  74.],
                               [109., 132.],
                               [172., 183.],
                               [103., 229.],
                               [149.,  91.],
                               [ 51., 132.],
                               [ 57., 200.]]


# Tain section
__C.TRAIN                   = edict()
__C.TRAIN.EPOCH             = 20
__C.TRAIN.BATCH_SIZE        = 2
__C.TRAIN.LR                = 1e-4
__C.TRAIN.LAMBDA_SCALE      = 1.
__C.TRAIN.SAVE_FREQ         = 5
__C.TRAIN.TFRECORD_DIR      = "./data/train.tfrecord"
__C.TRAIN.SAVE_MODEL_DIR    = "./saved_model/"




# Test section
__C.TEST                        = edict()
__C.TEST.TFRECORD_DIR          = "data/test.tfrecord"
__C.TEST.BATCH_SIZE             = 2
