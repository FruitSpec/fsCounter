import os
import sys
import torch

cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'vision', 'detector', 'yolo_x'))

from vision.detector.yolo_x.yolox.exp import get_exp
from vision.detector.preprocess import Preprocess
from vision.detector.yolo_x.yolox.utils.boxes import postprocess
#from vision.tracker.byteTrack.tracker.byte_tracker import BYTETracker
from vision.tracker.fsTracker.fs_tracker import FsTracker


class counter_detection():

    def __init__(self, cfg):

        self.preprocess = Preprocess(cfg.device, cfg.input_size)

        self.detector = self.init_detector(cfg)
        self.confidence_threshold = cfg.detector.confidence
        self.nms_threshold = cfg.detector.nms
        self.num_of_classes = cfg.detector.num_of_classes
        self.fp16 = cfg.detector.fp16

        self.tracker = self.init_tracker(cfg)

        self.device = cfg.device

    @staticmethod
    def init_detector(cfg):
        exp = get_exp(cfg.exp_file)
        model = exp.get_model()

        print("loading checkpoint from {}".format(cfg.ckpt_file))
        ckpt = torch.load(cfg.ckpt_file, map_location=cfg.device)
        model.load_state_dict(ckpt["model"])
        print("loaded checkpoint done.")
        model.cuda(cfg.device)
        model.eval()

        if cfg.detector.fp16:
            model.half()

        return model

    def init_tracker(self, cfg):

        self.frame_rate = cfg.tracker.frame_rate
        self.orig_width = cfg.tracker.orig_width
        self.orig_height = cfg.tracker.orig_height
        self.min_box_area = cfg.tracker.min_box_area
        self.input_size = cfg.input_size

        return FsTracker()

    def detect(self, frame):
        preprc_frame = self.preprocess(frame)
        input_ = preprc_frame.to(self.device)

        if self.fp16:
            input_ = input_.half()

        with torch.no_grad():
            output = self.detector(input_)

        # Filter results below confidence threshold and nms threshold
        output = postprocess(output, self.num_of_classes, self.confidence_threshold)

        # Output ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        return output

    def track(self, outputs, frame_id, frame):

        if outputs is not None and outputs[0] is not None:
            online_targets = self.tracker.update(outputs, frame)
            tracking_results = []
            for target in online_targets:
                target.append(frame_id)
                tracking_results.append(target)

            return tracking_results


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




