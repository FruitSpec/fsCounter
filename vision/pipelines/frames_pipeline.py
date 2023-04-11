import os
import sys

import torch
import collections
from omegaconf import OmegaConf
import cv2
import numpy as np
import json
from tqdm import tqdm

from vision.misc.help_func import get_repo_dir, scale_dets

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.data.results_collector import ResultsCollector, scale
from vision.pipelines.run_args import make_parser
from vision.tools.translation import translation as T
from vision.misc.help_func import validate_output_path


def run(cfg, args, detector=None):
    if isinstance(detector, type(None)):
        detector = counter_detection(cfg, args)
    results_collector = ResultsCollector()
    translation = T(cfg.translation.translation_size, cfg.translation.dets_only, cfg.translation.mode)

    img_list = os.listdir(args.data_dir)

    # assuming string frame_<id> or FSI_<id>
    print('Running')
    files_dict = {}
    for img in img_list:
        if 'png' in img or 'jpg' in img:
            #if 'frame' in img or 'FSI' in img:
            if 'FSI' in img:
                full_name = img.split('.')[0]
                id_ = int(full_name.split('_')[-1])
                files_dict[id_] = img

    # sort by ids
    files_dict = collections.OrderedDict(sorted(files_dict.items()))

    for id_, img_name in files_dict.items():

        results_collector.collect_file_name(img_name)
        results_collector.collect_id(id_)

        frame = cv2.imread(os.path.join(args.data_dir, img_name))

        det_outputs = detector.detect(frame)

        tx, ty = translation.get_translation(frame, det_outputs)

        # track:
        trk_outputs, track_windows = detector.track(det_outputs, tx, ty, id_)

        if args.save_windows:
            results_collector.save_tracker_windows(id_, args, trk_outputs, track_windows)

        # collect results:
        results_collector.collect_detections(det_outputs, id_)
        results_collector.collect_tracks(trk_outputs)

    if args.draw_on_img:
        results_collector.draw_and_save_dir(args.data_dir, os.path.join(args.output_folder, "dets"), True)

    results_collector.dump_to_csv(os.path.join(args.output_folder, "det.csv"))
    results_collector.dump_to_csv(os.path.join(args.output_folder, "tracker.csv"), False)

    return results_collector.detections, results_collector.tracks


if __name__ == "__main__":
    repo_dir = get_repo_dir()
    config_file = "/vision/pipelines/config/pipeline_config.yaml"
    # config_file = "/config/pipeline_config.yaml"
    cfg = OmegaConf.load(repo_dir + config_file)

    args = make_parser()
    args.eval_batch = 1
    args.draw_on_img = True
    args.frame_size = [2048, 1536]
    #args.data_dir = "/home/fruitspec-lab/FruitSpec/Sandbox/Sliced_data/RA_3_A_2/RA_3_A_2"
    folder_path = "/media/fruitspec-lab/easystore/track_detect_analysis"
    folder_list = [folder for folder in os.listdir(folder_path) if "." not in folder]
    parent_folder = "/media/fruitspec-lab/easystore/track_detect_analysis"
    res = []
    for folder in folder_list:
        if folder == "clean":
            continue
        args.data_dir = os.path.join(folder_path, folder)

        args.output_folder = os.path.join(parent_folder, folder, "det")
        if not os.path.isdir(args.output_folder):
            os.mkdir(args.output_folder)

        dets, tracks = run(cfg, args)
        tot_tracks = []
        for t in tracks:
            tid = t[-2]
            if tid not in tot_tracks:
                tot_tracks.append(tid)
        res.append({'row': folder, 'tot_tracks': len(tot_tracks)})

    with open(os.path.join(parent_folder, 'res.json'), 'w') as f:
        json.dump(res, f)

    print(f"finished {folder}")


