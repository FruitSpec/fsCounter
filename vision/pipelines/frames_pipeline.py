import os
import sys

import torch
import collections
from omegaconf import OmegaConf
import cv2
import json
from tqdm import tqdm

from vision.misc.help_func import get_repo_dir, scale_dets

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.data.results_collector import ResultsCollector, scale
from vision.pipelines.run_args import make_parser
#from vision.detector.yolo_x.yolox.evaluators.coco_evaluator import COCOEvaluator
from vision.detector.yolo_x.yolox.data import COCODataset
#from vision.detector.yolo_x.yolox.data.data_augment import ValTransform
from vision.detector.yolo_x.yolox.utils import xyxy2xywh
from vision.data import COCO_utils


def run(cfg, args):

    detector = counter_detection(cfg)
    results_collector = ResultsCollector()

    img_list = os.listdir(args.data_dir)

    # assuming string frame_<id> or FSI_<id>
    print('Running')
    files_dict = {}
    for img in img_list:
        if 'png' in img or 'jpg' in img:
            if 'frame' in img or 'FSI' in img:
                full_name = img.split('.')[0]
                id_ = int(full_name.split('_')[1])
                files_dict[id_] = img

    # sort by ids
    files_dict = collections.OrderedDict(sorted(files_dict.items()))

    for id_, img_name in tqdm(files_dict.items()):

        results_collector.collect_file_name(img_name)
        results_collector.collect_id(id_)

        frame = cv2.imread(os.path.join(args.data_dir, img_name))

        det_outputs = detector.detect(frame)

        scale_ = scale(detector.input_size, frame.shape)
        det_outputs = scale_dets(det_outputs, scale_)
        # track:
        trk_outputs = detector.track(det_outputs, id_, frame)

        # collect results:
        results_collector.collect_detections(det_outputs, id_, scale_)
        results_collector.collect_tracks(trk_outputs)


    results_collector.draw_and_save_dir(args.data_dir, args.output_path, True)
    return results_collector.detections, results_collector.tracks


if __name__ == "__main__":
    repo_dir = get_repo_dir()
    config_file = "/vision/pipelines/config/pipeline_config.yaml"
    # config_file = "/config/pipeline_config.yaml"
    cfg = OmegaConf.load(repo_dir + config_file)

    args = make_parser()
    args.eval_batch = 1
    #args.data_dir = "/home/fruitspec-lab/FruitSpec/Sandbox/Sliced_data/RA_3_A_2/RA_3_A_2"
    folder_path = "/media/fruitspec-lab/262/JAI-test/300322"
    folder_list = os.listdir(folder_path)
    parent_folder = "/home/fruitspec-lab/FruitSpec/Sandbox/Sliced_data/count"

    res = []
    for folder in folder_list:

        args.data_dir = os.path.join(folder_path, folder)

        args.output_path = os.path.join(parent_folder, folder)
        if not os.path.isdir(args.output_path):
            os.mkdir(args.output_path)

        dets, tracks = run(cfg, args)
        tot_tracks = []
        for t in tracks:
            tid = t[-2]
            if tid not in tot_tracks:
                tot_tracks.append(tid)
        res.append({'row': folder, 'tot_tracks': len(tot_tracks)})

    with open(os.path.join(parent_folder, 'res.json'), 'w') as f:
        json.dump(res, f)


