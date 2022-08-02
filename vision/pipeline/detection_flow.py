import os
import sys
import torch

cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'vision', 'Detector', 'YOLOX'))

from vision.detector.YOLOX.yolox.exp import get_exp
from vision.detector.preprocess import Preprocess
from vision.detector.YOLOX.yolox.utils.boxes import postprocess
#from vision.tracker.byteTrack.tracker.byte_tracker import BYTETracker


class counter_detection():

    def __init__(self, cfg):

        self.preprocess = Preprocess(cfg.input_size)

        self.detector = self.init_detector(cfg)
        self.confidence_threshold = cfg.detector.confidence
        self.nms_threshold = cfg.detector.nms
        self.num_of_classes = cfg.num_of_classes

        #self.tracker = self.init_tracker(cfg)

        self.device = cfg.device


    def init_detector(self, cfg):
        exp = get_exp(cfg.exp_file)
        model = exp.get_model()

        print("loading checkpoint from {}".format(cfg.ckpt_file))
        ckpt = torch.load(cfg.ckpt_file, map_location=cfg.device)
        model.load_state_dict(ckpt["model"])
        print("loaded checkpoint done.")

        model.cuda(cfg.device)
        model.eval()

        return model

    def init_tracker(self, cfg):
        self.tracker = BYTETracker(cfg)
        self.orig_width = cfg.tracker.orig_width
        self.orig_height = cfg.tracker.orig_height
        self.img_size = cfg.input_size

    def detect(self, frame):
        preprc_frame = self.preprocess(frame)
        input_ = preprc_frame.to(self.device)
        output = self.detector(input_)

        output = postprocess(output, self.num_of_classes)

        return output

    def track(self, outputs, frame_id):
        info_imgs = self.get_imgs_info(frame_id)
        online_targets = self.tracker.update(outputs, info_imgs, self.input_size)

    def get_imgs_info(self, frame_id):

        return (self.orig_height, self.orig_width, frame_id)




