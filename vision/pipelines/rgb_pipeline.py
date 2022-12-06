import copy

import os
import sys
import pyzed.sl as sl
import cv2
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

from vision.misc.help_func import get_repo_dir, scale_dets, validate_output_path, scale
from vision.depth.zed.svo_operations import get_frame, get_depth, get_point_cloud

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.data.results_collector import ResultsCollector
from vision.depth.zed.clip_depth_viewer import init_cam
from vision.pipelines.misc.filters import filter_by_distance, filter_by_duplicates, filter_by_size, filter_by_height, filter_by_intersection
from vision.tools.camera import is_sturated
from vision.depth.zed.svo_operations import get_measures, get_det_depth


def run(cfg, args):
    detector = counter_detection(cfg, args)
    results_collector = ResultsCollector(rotate=args.rotate)

    cam, runtime = init_cam(args.movie_path, 0.1, 2.5)

    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Read until video is completed
    print(f'Inferencing on {args.movie_path}\n')
    number_of_frames = sl.Camera.get_svo_number_of_frames(cam)
    frame_mat = sl.Mat()
    depth_mat = sl.Mat()
    point_cloud_mat = sl.Mat()
    f_id = 0
    pbar = tqdm(total=number_of_frames)
    while True:
        pbar.update(1)
        res = cam.grab(runtime)

        if res == sl.ERROR_CODE.SUCCESS and f_id < 2230:

            frame = get_frame(frame_mat, cam)
            depth = get_depth(depth_mat, cam)
            point_cloud = get_point_cloud(point_cloud_mat, cam)

            if is_sturated(frame):
                f_id += 1
                continue

            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
                point_cloud = cv2.rotate(point_cloud, cv2.ROTATE_90_CLOCKWISE)

            # detect:
            det_outputs = detector.detect(frame)
            scale_ = scale(detector.input_size, frame.shape)
            det_outputs = scale_dets(det_outputs, scale_)


            # filter
            if f_id == 140:
                a = 1
                #det_outputs = get_det_depth(point_cloud, det_outputs)
            filtered_outputs = filter_by_distance(det_outputs, depth, cfg.filters.distance.threshold)
            #filtered_outputs = filter_by_height(filtered_outputs, depth, cfg.filters.height.bias, cfg.filters.height.y_crop)
            #filtered_outputs = filter_by_height(det_outputs, depth, cfg.filters.height.bias, cfg.filters.height.y_crop)
            filtered_outputs = filter_by_size(filtered_outputs, cfg.filters.size.size_threshold)
            #filtered_outputs = filter_by_intersection(filtered_outputs)
      #      filtered_outputs = filter_by_duplicates(filtered_outputs, cfg.filters.duplicates.iou_threshold)

            # track:
            trk_outputs, trk_windows = detector.track(filtered_outputs, f_id, frame)

            # collect results:
            results_collector.collect_detections(det_outputs, f_id)
            results_collector.collect_tracks(trk_outputs)
            # TODO Matan to go through implementation
            results_collector.collect_size_measure(point_cloud, copy.deepcopy(trk_outputs))

            if args.debug.is_debug:
                results_collector.debug(f_id, args, trk_outputs, det_outputs, frame, depth, trk_windows)

            f_id += 1

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cam.close()

    results_collector.dump_to_csv(os.path.join(args.output_folder, 'detections.csv'))
    results_collector.dump_to_csv(os.path.join(args.output_folder, 'tracks.csv'), detections=False)
    # TODO Matan to go through implementation
    results_collector.dump_to_csv(os.path.join(args.output_folder, 'measures.csv'), detections=False, size=True)

    # results_collector.write_results_on_movie(args.movie_path, args.output_folder, write_tracks=True, write_frames=True)


def get_id_and_categories(cfg):
    category = []
    category_ids = []
    for category, id_ in cfg.classes.items():
        category.append(category)
        category_ids.append(id_)

    return category, category_ids


if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/home/yotam/FruitSpec/Code/fsCounter/vision/pipelines/config/runtime_config.yaml"
    # config_file = "/config/pipeline_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(runtime_config)

    validate_output_path(args.output_folder)
    run(cfg, args)
