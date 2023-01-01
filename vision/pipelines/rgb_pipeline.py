import copy

import os
import sys
import pyzed.sl as sl
import cv2
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import collections

from vision.misc.help_func import get_repo_dir, scale_dets, validate_output_path, scale
from vision.depth.zed.svo_operations import get_frame, get_depth, get_point_cloud

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.pipelines.misc.filters import filter_by_distance, filter_by_size, filter_by_height, sort_out
from vision.tracker.fsTracker.score_func import compute_dist_on_vec
from vision.data.results_collector import ResultsCollector
from vision.depth.zed.svo_operations import get_dimentions
from vision.tools.translation import translation as T
from vision.tools.camera import is_sturated
from vision.tools.color import get_hue
from vision.tools.video_wrapper import video_wrapper


def run(cfg, args):
    detector = counter_detection(cfg, args)
    results_collector = ResultsCollector(rotate=args.rotate)
    translation = T(cfg.translation.translation_size, cfg.translation.dets_only, cfg.translation.mode)

    cam = video_wrapper(args.movie_path, args.rotate, args.depth_minimum, args.depth_maximum)

    # Read until video is completed
    print(f'Inferencing on {args.movie_path}\n')
    number_of_frames = cam.get_number_of_frames()

    f_id = 0
    pbar = tqdm(total=number_of_frames)
    while True:
        pbar.update(1)
        frame, depth, point_cloud = cam.get_zed()
        if not cam.res:  # couldn't get frames
            # Break the loop
            break

        if is_sturated(frame):
            f_id += 1
            continue

        # detect:
        det_outputs = detector.detect(frame)

        # filter by size:
        filtered_outputs = filter_by_size(det_outputs, cfg.filters.size.size_threshold)

        # find translation
        tx, ty = translation.get_translation(frame, filtered_outputs)

        # track:
        trk_outputs, trk_windows = detector.track(filtered_outputs, tx, ty, f_id)

        # filter by distance:
        indices_in_distance = filter_by_distance(filtered_outputs, point_cloud, cfg.filters.distance.threshold)

        # sort out
        indices_out = list(set(range(len(trk_outputs))) - (set(indices_in_distance)))
        trk_outputs, trk_windows = sort_out(trk_outputs, trk_windows, indices_out)

        # measure:
        colors, hists_hue = get_colors(trk_outputs, frame)
        clusters = get_clusters(trk_outputs, cfg.clusters.min_single_fruit_distance)
        dimensions = get_dimentions(point_cloud, trk_outputs)

        # collect results:
        results_collector.collect_detections(det_outputs, f_id)
        frame_results = results_collector.collect_results(trk_outputs, clusters, dimensions, colors)

        if args.debug.is_debug:
            results_collector.debug(f_id, args, frame_results, det_outputs, frame, hists_hue, depth, trk_windows)

        f_id += 1

    # When everything done, release the video capture object
    cam.close()

    # results_collector.dump_to_csv(os.path.join(args.output_folder, 'detections.csv'))
    results_collector.dump_to_csv(os.path.join(args.output_folder, 'measures.csv'), type='measures')

    # results_collector.write_results_on_movie(args.movie_path, args.output_folder, write_tracks=True, write_frames=True)


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
        # TODO
        h, b = get_hue(frame[max(res[1],0):res[3], max(res[0],0):res[2], :])
        mean = np.sum(b[:-1] * h) / np.sum(h)
        std = np.sqrt(np.sum(((mean - b[:-1]) ** 2) * h) / np.sum(h))
        colors.append([mean * 2, std * 2])  # multiply by 2 to correct hue to 360 angles
        hists.append(h)

    return colors, hists


def get_clusters(trk_results, max_single_fruit_dist=200):
    if len(trk_results) == 0:
        return []
    trk_results = np.array(trk_results)
    dist = compute_dist_on_vec(trk_results, trk_results)

    t_ids = np.array([trk[-2] for trk in trk_results])
    neighbors = []
    for t_dist in dist:
        t_id_neighbors = t_ids[t_dist < max_single_fruit_dist]
        neighbors.append(t_id_neighbors)

    cluster_id = 0
    clusters = {}
    for t_id, t_id_neighbors in enumerate(neighbors):
        clusters_list = list(clusters.keys())
        id_found = False
        for c in clusters_list:
            for n in t_id_neighbors:
                if n in clusters[c]:
                    id_found = True
                    for t_n in t_id_neighbors:
                        if t_n in clusters[c]:
                            continue
                        clusters[c].append(t_n)
                    break
            if id_found:
                break

        if not id_found:
            clusters[cluster_id] = list(t_id_neighbors)
            cluster_id += 1

    id_to_cluster = {}
    for k, ids in clusters.items():
        for id_ in ids:
            id_to_cluster[id_] = k

    id_to_cluster = collections.OrderedDict(sorted(id_to_cluster.items()))
    clusters = list(id_to_cluster.values())

    return clusters


if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/home/yotam/FruitSpec/Code/fsCounter/vision/pipelines/config/runtime_config.yaml"
    # config_file = "/config/pipeline_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(runtime_config)

    validate_output_path(args.output_folder)
    run(cfg, args)
