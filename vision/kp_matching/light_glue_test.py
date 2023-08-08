import os
import cv2
import numpy as np
import pandas as pd
import time

from omegaconf import OmegaConf
from tqdm import tqdm

from vision.kp_match.infer import lightglue_infer
from vision.pipelines.ops.frame_loader import FramesLoader
from vision.misc.help_func import validate_output_path, get_repo_dir

def run(cfg, args, n_frames=200, type="superpoint"):

    cfg.batch_size = 1
    matcher = lightglue_infer(cfg, type)
    #res = []
    frame_loader = FramesLoader(cfg, args)
    #crop = [cfg.sensor_aligner.x_s, cfg.sensor_aligner.y_s, cfg.sensor_aligner.x_e, cfg.sensor_aligner.y_e]

    keypoints_path = os.path.join(args.output_folder, 'kp_match')
    validate_output_path(keypoints_path)

    f_id = 0
    n_frames = len(frame_loader.sync_zed_ids) if n_frames is None else min(n_frames, len(frame_loader.sync_zed_ids))
    pbar = tqdm(total=n_frames)
    while f_id < n_frames:
        pbar.update(cfg.batch_size)
        zed_batch, depth_batch, jai_batch, rgb_batch = frame_loader.get_frames(f_id)

        s = time.time()
        input0, r0, input1, r1 = matcher.preprocess_images(zed_batch[0], rgb_batch[0])
        e = time.time()
        print(f'preprocess time: {e - s}')
        s = e
        points0, points1, matches = matcher.match(input0, input1)
        print(f'matching time: {time.time() - s}')

        input0 = input0.cpu().numpy()
        input1 = input1.cpu().numpy()
        points0 = points0.cpu().numpy()
        points1 = points1.cpu().numpy()
        matches = matches.cpu().numpy()
        out_img = draw_matches(input0, input1, points0, points1)
        cv2.imwrite(os.path.join(keypoints_path, f"lightglue_f{f_id}.jpg"), out_img)

        f_id += 1

def draw_matches(input0, input1, points0, points1):
    input0 = np.moveaxis(input0, 0, 2) * 255
    input1 = np.moveaxis(input1, 0, 2) * 255
    h = max(input0.shape[0], input1.shape[0])
    w = input0.shape[1] + input1.shape[1] + 1
    canvas = np.zeros((h, w, 3))

    canvas[:input0.shape[0], :input0.shape[1], :] = input0
    canvas[:input1.shape[0], input0.shape[1]: input0.shape[1] + input1.shape[1], :] = input1

    for p0, p1 in zip(points0, points1):
        canvas = cv2.circle(canvas, (int(p0[0]), int(p0[1])), 3, (255, 0, 0))
        canvas = cv2.circle(canvas, (input0.shape[1] + int(p1[0]), int(p1[1])), 3, (255, 0, 0))
        canvas = cv2.line(canvas, (int(p0[0]), int(p0[1])), (input0.shape[1] + int(p1[0]), int(p1[1])), (0, 255, 0))

    return canvas






if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)

    #folder = "/media/matans/My Book/FruitSpec/NWFMXX/G10000XX/070623/row_12/1"
    folder = "/media/matans/My Book/FruitSpec/Customers_data/Fowler/daily/FREDIANI/210723/row_6/1"
    args.zed.movie_path = os.path.join(folder, "ZED.mkv")
    args.depth.movie_path = os.path.join(folder, "DEPTH.mkv")
    args.jai.movie_path = os.path.join(folder, "Result_FSI.mkv")
    args.rgb_jai.movie_path = os.path.join(folder, "Result_RGB.mkv")
    args.sync_data_log_path = os.path.join(folder, "jaized_timestamps.csv")
    args.output_folder = os.path.join("/media/matans/My Book/FruitSpec/Fowler_FREDIANI_210723_row_6_sp")
    validate_output_path(args.output_folder)

    run(cfg, args)

