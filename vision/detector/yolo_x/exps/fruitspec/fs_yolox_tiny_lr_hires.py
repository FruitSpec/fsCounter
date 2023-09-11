#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from vision.detector.yolo_x.yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (1024, 1024)
        self.random_size = (10, 20)
        self.test_size = (1024, 1024)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False
        self.seed = 42
        self.max_detections = 300

        # Define yourself dataset path
        self.data_dir = "/home/fruitspec-lab/FruitSpec/Data/JAI_FSI_V6x_COCO_with_zoom"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"

        self.output_dir = '/home/mic-730ai/Documents/Expriments/weights/Run_2_7_oct/'
        self.num_classes = 1

        # -------------- training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 150
        self.warmup_lr = 0.0005
        self.basic_lr_per_img = 0.01 / 32.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 25
        self.min_lr_ratio = 0.3
        self.ema = True
        self.weight_decay = 5e-4
        self.momentum = 0.9

        # --------------- transform config ----------------- #
        # prob of applying mosaic aug
        self.mosaic_prob = 0.3
        # prob of applying mixup aug
        self.mixup_prob = 0.
        # prob of applying hsv aug
        self.hsv_prob = 0.3
        # prob of applying flip aug
        self.flip_prob = 0.3
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 10.0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.1
        self.mosaic_scale = (0.8, 1.2)
        # apply mixup aug or not
        self.enable_mixup = False
        self.mixup_scale = (0.5, 1.5)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 2.0

        self.data_num_workers = 4
        self.eval_interval = 10
        self.train_interval = 10

        # -----------------  testing config ------------------ #
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.01
        # nms threshold
        self.nmsthre = 0.65