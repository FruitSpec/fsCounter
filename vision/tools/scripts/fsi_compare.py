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
from vision.data.results_collector import ResultsCollector

from vision.tools.translation import translation as T
from vision.tools.video_wrapper import video_wrapper
from vision.tools.camera import fsi_from_channels



def run(cfg, args):
    width, height = 1536, 2048
    output_video = cv2.VideoWriter(os.path.join(args.output_folder, "Result_newFSI_1.mkv"), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   15, (width, height))

    fsi_cam = video_wrapper(os.path.join(args.movie_path, f"Result_FSI_1.mkv"), args.rotate)
    rgb_cam = video_wrapper(
        os.path.join(args.movie_path, f"Result_RGB_1.mkv"), args.rotate)
    c_800_cam = video_wrapper(
        os.path.join(args.movie_path, f"Result_800_1.mkv"), args.rotate)
    c_975_cam = video_wrapper(
        os.path.join(args.movie_path, f"Result_975_1.mkv"), args.rotate)

    # Read until video is completed
    print(f'Inferencing on {args.movie_path}\n')
    number_of_frames = fsi_cam.get_number_of_frames()

    f_id = 0
    pbar = tqdm(total=number_of_frames)
    while True:
        pbar.update(1)
        res, fsi_frame = fsi_cam.get_frame()
        _, rgb_frame = rgb_cam.get_frame()
        _, c_800_frame = c_800_cam.get_frame()
        _, c_975_frame = c_975_cam.get_frame()
        if not res:  # couldn't get frames
            # Break the loop
            break
        frame, rgb_frame = fsi_from_channels(rgb_frame, c_800_frame, c_975_frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output_video.write(frame)

    output_video.release()
    fsi_cam.close()
    rgb_cam.close()
    c_800_cam.close()
    c_975_cam.close()









if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/runtime_config.yaml"
    # config_file = "/config/pipeline_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)

    validate_output_path(args.output_folder)
    run(cfg, args)
