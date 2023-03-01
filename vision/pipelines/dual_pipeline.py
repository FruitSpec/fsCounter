import os
import sys
import pyzed.sl as sl
import cv2
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

from vision.misc.help_func import get_repo_dir, scale_dets
#from vision.depth.zed.svo_operations import get_frame, get_depth, get_point_cloud
from vision.tools.video_wrapper import video_wrapper

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.data.results_collector import ResultsCollector, scale
from vision.tools.translation import translation as T
from vision.depth.zed.clip_depth_viewer import init_cam
from vision.tracker.fsTracker.score_func import get_intersection
from vision.pipelines.misc.filters import filter_by_distance, filter_by_duplicates, filter_by_size

def run(cfg, args):
    detector = counter_detection(cfg, args)
    results_collector = ResultsCollector(rotate=args.rotate)
    translation = T(cfg.translation.translation_size, cfg.translation.dets_only, cfg.translation.mode)

    zed_cam = video_wrapper(args.zed.movie_path, args.zed.rotate, args.zed.depth_minimum, args.zed.depth_maximum)
    jai_cam = video_wrapper(args.jai.movie_path, args.jai.rotate)

    # Read until video is completed
    print(f'Inferencing on {args.jai.movie_path}\n')
    f_id = 0
    pbar = tqdm(total=jai_cam.get_number_of_frames())
    while True:
        pbar.update(1)

        zed_frame, depth, point_cloud = zed_cam.get_zed()
        ret, jai_frame = jai_cam.get_frame()
        if not ret and not zed_cam.res:  # couldn't get frames
            # Break the loop
            break

        # detect:
        try:
            det_outputs = detector.detect(jai_frame)
        except:
            continue
        # find translation
        tx, ty = translation.get_translation(jai_frame, [])

        # track:
        trk_outputs, trk_windows = detector.track(det_outputs, tx, ty, f_id)

        #collect results:
        results_collector.collect_detections(det_outputs, f_id)
        results_collector.collect_tracks(trk_outputs)

        f_id += 1

    zed_cam.close()
    jai_cam.close()

    results_collector.dump_to_csv(os.path.join(args.output_folder, 'detections.csv'))
    results_collector.dump_to_csv(os.path.join(args.output_folder, 'tracks.csv'), type="tracks")


def validate_output_path(output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)


if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/home/yotam/FruitSpec/Code/fsCounter/vision/pipelines/config/dual_runtime_config.yaml"
    # config_file = "/config/pipeline_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(runtime_config)


    run(cfg, args)
