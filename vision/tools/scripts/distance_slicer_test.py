import os
import sys
import cv2
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from vision.misc.help_func import get_repo_dir, load_json, validate_output_path

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.ops.frame_loader import FramesLoader
from vision.pipelines.ops.simulator import get_n_frames, init_cams
from vision.tools.video_wrapper import video_wrapper

def slices_to_frames(slices):

    slices = slices.query('tree_id != -1')
    frames = slices.frame_id.unique()

    data = {}
    for frame in frames:
        frame_slices = slices.query(f'frame_id == {frame}')
        starts = frame_slices.start.unique()
        ends = frame_slices.end.unique()
        frame_slices = []
        for start in starts:
            frame_slices.append(start)
        for end in ends:
            frame_slices.append(end)
        data[frame] = frame_slices

    return data



if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)

    row_path = "/media/matans/My Book/FruitSpec/distance_slicing/Fowler/FREIDIANI/170723/row_14/1"
    slices_path = os.path.join(row_path, "all_slices.csv")
    mkv_path = os.path.join(row_path, "Result_FSI.mkv")
    output_path = "/media/matans/My Book/FruitSpec/distance_slicing/Fowler/FREIDIANI/170723/row_14"

    slices = pd.read_csv(slices_path)
    data = slices_to_frames(slices)
    width = 1536
    height = 2048
    output_video_name = os.path.join(output_path, 'result_video.avi')
    output_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   10, (width, height))

    cam = video_wrapper(mkv_path, rotate=1)
    f_id = 0
    for frame_id, frame_slice in tqdm(data.items()):
        ret, frame = cam.get_frame()
        if ret != True:
            break  # co

        for slice in frame_slice:
            frame = cv2.line(frame, (slice, 500), (slice, 1500), (255, 0, 0), 4)
        output_video.write(frame)
        f_id += 1
    output_video.release()
    cam.close()



