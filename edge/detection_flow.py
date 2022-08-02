import os
import sys
import torch

cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'vision', 'Detector', 'YOLOX'))
from vision.Detector.YOLOX.yolox.exp import get_exp
from vision.tracker.byteTrack.tracker.byte_tracker import BYTETracker

class counter_detection():

    def __init__(self, cfg):

        self.detector = self.init_detector(cfg)
        self.tracker = self.init_tracker(cfg)


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

    def detect(self, batch):

        output = self.detector(batch)

        return output

    def track(self, outputs, frame_id):
        info_imgs = self.get_imgs_info(frame_id)
        online_targets = self.tracker.update(outputs, info_imgs, self.input_size)

    def get_imgs_info(self, frame_id):

        return (self.orig_height, self.orig_width, frame_id)


