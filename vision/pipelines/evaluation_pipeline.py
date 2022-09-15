import os
import sys
from torch.utils.data import DataLoader

import cv2
from omegaconf import OmegaConf

from vision.misc.help_func import get_repo_dir

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.pipelines.run_args import make_parser
from vision.detector.yolo_x.yolox.evaluators.coco_evaluator import COCOEvaluator
from vision.detector.yolo_x.yolox.data import COCODataset
from vision.data import COCO_utils

def run(cfg, args):

    detector = counter_detection(cfg)
    dataloader = get_evaluation_dataloader(args, img_size=cfg.input_size, preprocess=detector.preprocess)
    evaluator = COCOEvaluator(dataloader,
                              img_size=cfg.input_size,
                              confthre=cfg.detector.confidence,
                              nmsthre=cfg.detector.nms,
                              num_classes=cfg.detector.num_of_classes)

    eval_results = evaluator.evaluate(detector.detector,
                                      half=cfg.detector.fp16, output_eval=True)

    return eval_results


def get_evaluation_dataloader(args , img_size, preprocess=None):

    evaldataset = COCODataset(
        data_dir=args.data_dir,
        json_file=args.coco_gt,
        name=args.ds_name,
        img_size=img_size,
        preproc=preprocess,
        batch_size=args.eval_batch
    )

    return evaldataset


if __name__ == "__main__":

    repo_dir = get_repo_dir()
    config_file = "/vision/pipelines/config/pipeline_config.yaml"
    #config_file = "/config/pipeline_config.yaml"
    cfg = OmegaConf.load(repo_dir + config_file)


    args = make_parser()

    args.data_dir = "/home/fruitspec-lab/FruitSpec/Data/YOLO-Fruits_COCO/COCO"
    args.coco_gt = "/home/fruitspec-lab/FruitSpec/Data/YOLO-Fruits_COCO/COCO/annotations/instances_val.json"
    args.ds_name = "val2017"
    args.eval_batch = 8

    run(cfg, args)