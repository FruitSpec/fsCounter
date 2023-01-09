import os
import sys
from torch.utils.data import DataLoader

import cv2
from omegaconf import OmegaConf

cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.pipelines.run_args import make_parser
from vision.data.results_collector import ResultsCollector
from vision.data import COCO_utils

def run(cfg, args):
    detector = counter_detection(cfg)
    results_collector = ResultsCollector()

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)