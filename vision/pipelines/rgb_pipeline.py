import os
import sys
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import collections

from vision.misc.help_func import get_repo_dir, validate_output_path, copy_configs
from vision.depth.zed.svo_operations import get_frame, get_depth, get_point_cloud, get_dimensions, sl_get_dimensions, measure_depth

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.pipelines.ops.fruit_cluster import FruitCluster
from vision.pipelines.misc.filters import filter_by_distance, filter_by_size, filter_by_height, sort_out
from vision.tracker.fsTracker.score_func import compute_dist_on_vec
from vision.data.results_collector import ResultsCollector
from vision.tools.translation import translation as T
from vision.tools.camera import is_sturated
from vision.tools.color import get_hue, get_tomato_color
from vision.tools.video_wrapper import video_wrapper


def run(cfg, args):
    print(f'Inferencing on {args.movie_path}\n')
    detector = counter_detection(cfg, args)
    results_collector = ResultsCollector(rotate=args.rotate)
    translation = T(cfg.translation.translation_size, cfg.translation.dets_only, cfg.translation.mode)
    fs = FruitCluster(cfg.clusters.max_single_fruit_dist,
                      cfg.clusters.range_diff_threshold,
                      cfg.clusters.max_losses)

    cam = video_wrapper(args.movie_path, args.rotate, args.depth_minimum, args.depth_maximum)

    # Read until video is completed
    number_of_frames = cam.get_number_of_frames()

    f_id = 0
    pbar = tqdm(total=number_of_frames)
    while True:
        pbar.update(1)
        frame, depth, point_cloud = cam.get_zed()
        if not cam.res:  # couldn't get frames
            #     Break the loop
            break

        if is_sturated(frame):
            f_id += 1
            continue

        # detect:
        det_outputs = detector.detect(frame)

        # filter by size:
        filtered_outputs = filter_by_size(det_outputs, cfg.filters.size.size_threshold)

        outputs_depth = measure_depth(filtered_outputs, point_cloud)

        # find translation
        tx, ty = translation.get_translation(frame, filtered_outputs)

        # track:
        trk_outputs, trk_windows = detector.track(filtered_outputs, tx, ty, f_id, outputs_depth)

        # filter by distance:
        filtered_outputs, ranges = filter_by_distance(trk_outputs, point_cloud, cfg.filters.distance.threshold)

        # clutser:
        filtered_outputs = fs.cluster(filtered_outputs, ranges)

        # sort out
        #indices_out = list(set(range(len(trk_outputs))) - (set(indices_in_distance)))
        #trk_outputs, trk_windows = sort_out(trk_outputs, trk_windows, indices_out)


        # measure:
        colors, hists_hue = get_colors(filtered_outputs, frame)
        dimensions = get_dimensions(point_cloud, frame, filtered_outputs, cfg)
        # dimensions = sl_get_dimensions(trk_outputs, cam)

        # collect results:
        results_collector.collect_detections(det_outputs, f_id)

        frame_results = results_collector.collect_results(filtered_outputs, dimensions, colors)

        if args.debug.is_debug:
            depth = None
            results_collector.debug(f_id, args, frame_results, det_outputs, frame, hists_hue, depth, trk_windows)

        f_id += 1

    # When everything done, release the video capture object
    cam.close()
    filter_suffix = f'{"_hue" if cfg.filters.hue else ""}{"_depth" if cfg.filters.depth else ""}'
    out_name = f'measures_{cfg.dim_method}_{str(cfg.margin).split(".")[-1]}{filter_suffix}.csv'
    results_collector.dump_to_csv(os.path.join(args.output_folder, out_name), type='measures')
    #detector.release()


def get_id_and_categories(cfg):
    category = []
    category_ids = []
    for category, id_ in cfg.classes.items():
        category.append(category)
        category_ids.append(id_)

    return category, category_ids


def get_colors(trk_results, frame):
    colors = []
    hists = []
    for res in trk_results:
        rgb_crop = frame[max(int(res[1]), 0):int(res[3]), max(int(res[0]), 0):int(res[2]), :]
        h, b = get_hue(rgb_crop)
        mean = np.sum(b[:-1] * h) / np.sum(h)
        std = np.sqrt(np.sum(((mean - b[:-1]) ** 2) * h) / np.sum(h))
        colors.append([get_tomato_color(rgb_crop), std * 2])  # multiply by 2 to correct hue to 360 angles
        hists.append(h)

    return colors, hists


if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = os.path.join(repo_dir, "vision/pipelines/config/pipeline_config.yaml")
    runtime_config = os.path.join(repo_dir, "vision/pipelines/config/runtime_config.yaml")
    cfg = OmegaConf.load(pipeline_config)
    args = OmegaConf.load(runtime_config)

    validate_output_path(args.output_folder)
    copy_configs(pipeline_config, runtime_config, args.output_folder)

    run(cfg, args)

