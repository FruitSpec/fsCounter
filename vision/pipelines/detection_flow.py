import os
import sys
import torch
from concurrent.futures import ThreadPoolExecutor
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

        self.preprocess = Preprocess(cfg.device, cfg.input_size)

        self.detector = self.init_detector(cfg)
        self.confidence_threshold = cfg.detector.confidence
        self.nms_threshold = cfg.detector.nms
        self.num_of_classes = cfg.detector.num_of_classes
        self.fp16 = cfg.detector.fp16
        self.input_size = cfg.input_size

        self.tracker = self.init_tracker(cfg, args)

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
                         debug_folder=None)


    def detect(self, frames):
        input_ = self.preprocess_batch(frames)

        if self.fp16:
            input_ = input_.half()

        with torch.no_grad():
            output = self.detector(input_)

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





