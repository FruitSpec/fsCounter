import os
import sys
import cv2
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pickle

from vision.misc.help_func import get_repo_dir, load_json, validate_output_path

# repo_dir = get_repo_dir()
# sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.data.results_collector import ResultsCollector
from vision.tools.translation import translation as T
from vision.depth.slicer.slicer_flow import post_process
from vision.tools.sensors_alignment import SensorAligner
from vision.tools.camera import batch_is_saturated
from vision.pipelines.ops.simulator import get_n_frames, write_metadata, init_cams, get_frame_drop
from vision.pipelines.ops.frame_loader import FramesLoader
from vision.data.fs_logger import Logger
from vision.feature_extractor.boxing_tools import xyz_center_of_box, cut_zed_in_jai
from concurrent.futures import ThreadPoolExecutor
from vision.feature_extractor.percent_seen import get_percent_seen
from vision.feature_extractor.tree_size_tools import stable_euclid_dist, get_pix_size
from vision.pipelines.ops.bboxes import depth_center_of_box, cut_zed_in_jai
from vision.pipelines.adt_pipeline import Pipeline, update_metadata
from vision.tools.color import get_tomato_color
from vision.pipelines.ops.simulator import update_arg_with_metadata
from vision.pipelines.ops.kp_matching.infer import lightglue_infer


def run(cfg, args, metadata=None, n_frames=None, zed_shift=0, f_id=0):
    adt = Pipeline(cfg, args)
    results_collector = ResultsCollector(rotate=args.rotate, mode=cfg.result_collector.mode)
    print(f'Inferencing on {args.jai.movie_path}\n')
    max_cut_frame = metadata['max_cut_frame'] if metadata is not None else np.inf
    if isinstance(max_cut_frame, str):
        if max_cut_frame == "inf":
            max_cut_frame = np.inf
        else:
            max_cut_frame = int(max_cut_frame)
    if n_frames is None:
        n_frames = min(adt.frames_loader.jai_cam.get_number_of_frames(), max_cut_frame)
    n_frames = (n_frames//cfg.batch_size)*cfg.batch_size
    pbar = tqdm(total=n_frames)
    if adt.frames_loader.mode == "async":
        adt.sensor_aligner.zed_shift = zed_shift
    lg = lightglue_infer(cfg, len_size=cfg.len_size)
    while f_id < n_frames:
        pbar.update(adt.batch_size)
        zed_batch, pc_batch, jai_batch, rgb_batch = adt.get_frames(f_id)
        if not len(zed_batch) or not len(jai_batch):
            break
        # rgb_batch = [img[:,:, ::-1] for img in rgb_batch] # turn to BGR
        alignment_results = adt.align_cameras(zed_batch, rgb_batch)
        # alignment_results = [lg.align_sensors(zed_batch[0], rgb_batch[0])]
        # detect:
        det_outputs = adt.detect(jai_batch)
        # find translation
        translation_results = adt.get_translation(jai_batch, det_outputs)
        # track:
        trk_outputs, trk_windows = adt.track(det_outputs, translation_results, f_id)

        # location and dimensions
        trk_outputs = adt.get_xyzs_dims(pc_batch, jai_batch, alignment_results, trk_outputs)

        #color
        tomato_colors = list(map(get_trks_colors, rgb_batch, trk_outputs, zed_batch, alignment_results))
        tomato_colors_viz = list(map(tomato_color_class_to_rgb, tomato_colors))
        trk_outputs = list(map(add_colors_to_track, trk_outputs, tomato_colors))
        # collect results:
        results_collector.collect_adt(trk_outputs, alignment_results, None, f_id, translation_results)
        results_collector.debug_batch(f_id, args, trk_outputs, det_outputs, [img[:,:,::-1] for img in rgb_batch],
                                      None, trk_windows, tomato_colors_viz, zed_frames=zed_batch,
                                      alignment_results=alignment_results)

        f_id += adt.batch_size
        adt.logger.iterations += 1

    pbar.close()
    adt.frames_loader.close_cameras()
    adt.dump_log_stats(args)
    update_metadata(metadata, args)
    results_collector.dump_to_csv(os.path.join(args.output_folder, 'tracks.csv'), type="tracks")
    results_collector.dump_to_csv(os.path.join(args.output_folder, 'alignment.csv'), type="alignment")
    if adt.frames_loader.mode == "async":
        results_collector.dump_jai_zed_json(args.output_folder)
    return results_collector


def tomato_size_2_weight(widths, heights):
    valid_indexes = np.all([np.isfinite(widths), np.isfinite(heights)], axis=0)
    v_widths, v_heights = widths[valid_indexes], heights[valid_indexes]
    weights = 115.17 -2.10475268*v_widths -4.29288158*v_heights + 0.10813299*v_widths*v_heights
    return weights


def width_height_2_avg_mm(witdths, heights):
    return np.nanmean([np.round(witdths*100, 2), np.round(heights*100, 2)], axis=0)


def track2color(tracks):
    return tracks.groupby("track_id")["color"].mean().round()


def tracks2harvest(tracks, min_samp=2):
    if isinstance(tracks, str):
        tracks = pd.read_csv(tracks)
    uniq, counts = np.unique(tracks["track_id"], return_counts=True)
    valid_tracks = uniq[counts > min_samp]
    tracks_cleaned = tracks[np.isin(tracks["track_id"], valid_tracks)]
    count = len(np.unique(tracks_cleaned["track_id"]))
    bins = np.histogram(track2color(tracks_cleaned), list(range(1, 7)))[0]
    # filter only whole fruits
    tracks_cleaned = tracks_cleaned[tracks_cleaned['class_pred'] == 0]
    avg_mm = width_height_2_avg_mm(tracks_cleaned["width"], tracks_cleaned["height"])
    avg_mm = avg_mm[np.isfinite(avg_mm)]
    size_avg_mm, size_std = np.nanmean(avg_mm), np.nanstd(avg_mm)
    weights = tomato_size_2_weight(tracks_cleaned["width"], tracks_cleaned["height"])
    total_weights, avg_weight, weight_std = np.sum(weights), np.mean(weights), np.std(weights)
    return {"count": count,
            **{f"bin{bin_i+1}": bins[bin_i] for bin_i in range(5)},
            "total_weight_kg": total_weights/1000,
            "weight_avg_gr": avg_weight,
            "weight_std": weight_std,
            "size_avg_mm": size_avg_mm,
            "size_std": size_std}



def get_trks_colors(rgb_img, trk_outputs, zed_img=None, align_res=None):
    colors = []
    corr = align_res[0]
    use_zed = not isinstance(zed_img, type(None))
    if use_zed:
        zed_aligned = zed_img[int(corr[1]):int(corr[3]), int(corr[0]):int(corr[2]), :]
        zed_aligned = cv2.resize(zed_aligned, rgb_img.shape[:2][::-1])
    for trk in trk_outputs:
        trk_bbox = trk[:4]
        x1, y1, x2, y2 = trk_bbox
        x1, y1 = max(x1, 0), max(y1, 0)
        h, w = rgb_img.shape[:2]
        x2, y2 = min(x2, w - 1), min(y2, h - 1)
        if use_zed:
            colors.append(get_tomato_color(zed_aligned[y1:y2, x1:x2]))
        else:
            colors.append(get_tomato_color(rgb_img[y1:y2, x1:x2]))
    return colors


def add_colors_to_track(trk_output, tomato_color):
    for i in range(len(tomato_color)):
        trk_output[i].append(tomato_color[i])
    return trk_output


def tomato_color_class_to_rgb(colors):
    colors_out = []
    colors_dict = {0:(255,255,255), # white
        1:(0,0,255), # red
    2:(0,125,255), # orange
    3:(128,0,128), # purpule
    4:(0,255,255), # yellow
    5:(0,255,0)} # green
    for color in colors:
        colors_out.append(colors_dict[color])
    return colors_out


if __name__ == "__main__":
    tracks2harvest("/media/fruitspec-lab/TEMP SSD/Tomato/PackoutDataNondealeaf/pre/1/tracks.csv", min_samp=2)
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)

    validate_output_path(args.output_folder)
    #copy_configs(pipeline_config, runtime_config, args.output_folder)

    tomato_folder = "/media/fruitspec-lab/TEMP SSD/Tomato/PackoutDataNondealeaf"
    folders = []
    for phenotype in os.listdir(tomato_folder):
        type_path = os.path.join(tomato_folder, phenotype)
        if not os.path.isdir(type_path):
            continue
        for scan_number in os.listdir(type_path):
            scan_path = os.path.join(type_path, scan_number)
            folders.append(scan_path)
    cfg.frame_loader.mode = "async"
    cfg.ckpt_file = cfg.ckpt_file_tomato
    cfg.exp_file = cfg.exp_file_tomato
    cfg.batch_size = 1
    len_size = 83
    cfg.detector.confidence = 0.3
    cfg.detector.nms = 0.3
    cfg.detector.max_detections = 300
    cfg.detector.num_of_classes = 2
    cfg.sensor_aligner.apply_zed_shift = True
    overwrite = False
    f_id = 0
    for folder in folders:
        try:
            args.zed.movie_path = os.path.join(folder, "ZED_1.svo")
            args.depth.movie_path = os.path.join(folder, "DEPTH.mkv")
            args.jai.movie_path = os.path.join(folder, "Result_FSI_1.mkv")
            args.rgb_jai.movie_path = os.path.join(folder, "Result_RGB_1.mkv")
            args.sync_data_log_path = os.path.join(folder, "jai_zed.json")
            args, metadata = update_arg_with_metadata(args)
            # args.sync_data_log_path = os.path.join(folder, "jaized_timestamps.csv")
            scan = os.path.basename(folder)
            row = os.path.basename(os.path.dirname(folder))
            block = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(folder))))
            args.output_folder = folder
            cfg.len_size = metadata.get("len_size", len_size)
            if not metadata.get("align_detect_track", True) and not overwrite:
                print("skipping: ", folder)
                continue
            rc = run(cfg, args, zed_shift=metadata.get('zed_shift', 0), f_id=f_id)
        except:
            print("failed: ", folder)

    # rc.dump_feature_extractor(args.output_folder)
