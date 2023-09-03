import os
import sys
import torch
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import numpy as np

import time

cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'vision', 'detector', 'yolo_x'))

from vision.detector.yolo_x.yolox.exp import get_exp
from vision.detector.preprocess import Preprocess
from vision.detector.yolo_x.yolox.utils.boxes import postprocess
from vision.misc.help_func import scale_dets, scale
from vision.tracker.fsTracker.fs_tracker import FsTracker



class counter_detection():

    def __init__(self, cfg, args):

        # params
        self.confidence_threshold = cfg.detector.confidence
        self.nms_threshold = cfg.detector.nms
        self.num_of_classes = cfg.detector.num_of_classes
        self.fp16 = cfg.detector.fp16
        self.input_size = cfg.input_size
        self.max_detections = cfg.detector.max_detections
        self.device = cfg.device
        self.detector_type = cfg.detector.detector_type.lower()
        self.nms_thresh = cfg.detector.nms
        self.input_size = cfg.input_size

        # init
        self.tracker = self.init_tracker(cfg, args)

        if self.detector_type == 'yolox':
            self.preprocess = Preprocess(cfg.device, self.input_size)
            self.detector, self.decoder_ = self.init_detector(cfg)
        elif self.detector_type == 'yolov8':
            self.detector = YOLO(cfg.ckpt_file) # self.detector = model
        else:
            raise NotImplementedError(f"The '{self.detector_type}' detector algorithm is not implemented.")



    @staticmethod
    def init_detector(cfg):
        exp = get_exp(cfg.exp_file)
        model = exp.get_model()

        model.cuda(cfg.device)

        if cfg.detector.fp16:   # can only run on gpu
            model.half()

        model.eval()

        decoder_ = None

        if not cfg.detector.trt:

            print("loading checkpoint from {}".format(cfg.ckpt_file))
            ckpt = torch.load(cfg.ckpt_file, map_location=cfg.device)
            model.load_state_dict(ckpt["model"])
            print("loaded checkpoint done.")

        if cfg.detector.trt:
            model.head.decode_in_inference = False
            decoder_ = model.head.decode_outputs

            from torch2trt import TRTModule
            model_trt = TRTModule()

            # replace model weights with "model_trt.pth" in the same dir:
            weights_path = os.path.abspath(cfg.ckpt_file)
            if os.path.basename(weights_path) != "model_trt.pth":
                weights_path = os.path.join(os.path.dirname(weights_path), "model_trt.pth")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"File {weights_path} not found.")

            # load trt-pytorch model
            model_trt.load_state_dict(torch.load(weights_path))
            print("loaded TensorRT model.")
            x = torch.ones(cfg.batch_size, 3, exp.test_size[0], exp.test_size[1]).cuda()
            tensor_type = torch.cuda.HalfTensor if cfg.detector.fp16 else torch.cuda.FloatTensor
            x = x.type(tensor_type)
            model(x)
            model = model_trt

        return model, decoder_

    @staticmethod
    def init_tracker(cfg, args):

        return FsTracker(frame_size=args.frame_size,
                         minimal_max_distance=cfg.tracker.minimal_max_distance,
                         score_weights=cfg.tracker.score_weights,
                         match_type=cfg.tracker.match_type,
                         det_area=cfg.tracker.det_area,
                         max_losses=cfg.tracker.max_losses,
                         translation_size=cfg.tracker.translation_size,
                         major=cfg.tracker.major,
                         minor=cfg.tracker.minor,
                         compile_data=cfg.tracker.compile_data_path,
                         debug_folder=None)


    def detect(self, frames):

        if self.detector_type == 'yolox':
            output = self.detect_yolox(frames)

        elif self.detector_type == 'yolov8':
            output = self.detect_yolov8(frames)

        return output

    def detect_yolov8(self, frames):

        results = self.detector.predict(
            source = frames,
            conf=self.confidence_threshold,
            half=self.fp16,
            iou=self.nms_thresh,
            imgsz=self.input_size[0],
            show=False,
            save=False,
            hide_labels=True,
            max_det=self.max_detections,
            project="projects/debug",
            name="debuging",
            line_width=2)  # todo add device

        output = self._convert_yolov8_results_to_detections_format(results)
        return output


    def _convert_yolov8_results_to_detections_format(self, results):
        output = []
        for img_res in results:
            xyxy_coordinates = img_res.boxes.xyxy
            # todo - replace class conf (=1) with real value.
            stacked_tensors = torch.stack((img_res.boxes.conf,
                                           torch.ones(img_res.boxes.cls.shape[0]).to(img_res.boxes.conf.device),
                                           img_res.boxes.cls)).t()
            conc = torch.cat((xyxy_coordinates, stacked_tensors), dim=1)
            conc = conc.to('cpu').tolist()  # load to cpu, convert tensor to list
            conc = [[int(x) if i < 4 else x for i, x in enumerate(sublist)] for sublist in
                    conc]  # convert the 4 bbox coordinates to ints
            output.append(conc)  # Output ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        return output



    def detect_yolox(self, frames): # todo - detect yolox
        input_ = self.preprocess_batch(frames)

        if self.fp16:
            input_ = input_.half()

        with torch.no_grad():
            output = self.detector(input_)
            if self.decoder_ is not None:
                output = self.decoder_(output, dtype=output.type())

        # Filter results below confidence threshold and nms threshold
        output = postprocess(output, self.num_of_classes, self.confidence_threshold)

        # Scale bboxes to orig image coordinates
        output = self.scale_output(output, frames[0].shape)

        # Output ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        return output

    def track(self, outputs, translations, frame_id=None):

        batch_results = []
        batch_windows = []
        for i, frame_output in enumerate(outputs):
            if frame_output is not None:
                tx, ty = translations[i]
                if frame_id is not None:
                    id_ = frame_id + i
                else:
                    id_ = None
                online_targets, track_windows = self.tracker.update(frame_output, tx, ty, id_)
                tracking_results = []
                for target in online_targets:
                    target.append(id_)
                    tracking_results.append(target)

            batch_results.append(tracking_results)
            batch_windows.append(track_windows)

        return batch_results, batch_windows

    def preprocess_batch(self, batch, workers=4):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(self.preprocess, batch))
        preprc_batch = torch.stack(results)
        #preprc_batch = preprc_batch.

        return preprc_batch

    def get_imgs_info(self, frame_id):

        return (self.orig_height, self.orig_width, frame_id)

    @staticmethod
    def targets_to_results(online_targets, frame_id, min_box_area):

        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            #vertical = tlwh[2] / tlwh[3] > 1.6
            vertical = False
            if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)

        return frame_id, online_tlwhs, online_ids, online_scores

    def scale_output(self, output, frame_size):

        scale_ = scale(self.input_size, frame_size)
        output = scale_dets(output, scale_)

        return output





