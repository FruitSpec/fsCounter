import pandas as pd
import cv2
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from vision.misc.help_func import get_repo_dir
from vision.tools.video_wrapper import video_wrapper

def downsample_by_log(jai_fp, jai_rgb_fp, output_movie_fp):

    cam_jai = video_wrapper(jai_fp, rotate=1)
    cam_jai_rgb = video_wrapper(jai_rgb_fp, rotate=1)

    f_id = 0

    output_video = cv2.VideoWriter(output_movie_fp, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   15, (800, 700), isColor=True)
    _, jai_frame = cam_jai.get_frame(99)
    _, jai_rgb_frame = cam_jai_rgb.get_frame(99)
    for i in tqdm(range(100, 400)):

        _, jai_frame = cam_jai.get_frame()
        _, jai_rgb_frame = cam_jai_rgb.get_frame()

        jai_rgb_frame = cv2.cvtColor(jai_rgb_frame, cv2.COLOR_BGR2RGB)
        jai_rgb_frame = cv2.resize(jai_rgb_frame, (480, 640))
        jai_rgb_frame = jai_rgb_frame[:, :jai_rgb_frame.shape[1] // 2, :]

        jai_frame = cv2.resize(jai_frame, (480, 640))
        jai_frame = jai_frame[:, jai_frame.shape[1] // 2:, :]

        canvas = np.zeros((700, 800, 3), dtype=np.uint8)
        canvas[10:10 + jai_rgb_frame.shape[0], 400 - jai_rgb_frame.shape[1]:400, :] = jai_rgb_frame
        canvas[10:10 + jai_frame.shape[0], 400:400 + jai_frame.shape[1], :] = jai_frame

        f_id += 1
        output_video.write(canvas)

    output_video.release()



if __name__ == "__main__":
    repo_dir = get_repo_dir()
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    args = OmegaConf.load(repo_dir + runtime_config)

    jai_fp = "/media/fruitspec-lab/cam175/APPLECHILE04/290323/APPCALIB/R1A/Result_FSI_1.mkv"
    jai_rgb_fp = "/media/fruitspec-lab/cam175/APPLECHILE04/290323/APPCALIB/R1A/Result_RGB_1.mkv"

    output_movie_fp = "/home/fruitspec-lab/FruitSpec/Data/counter/FSI_clip.mkv"

    downsample_by_log(jai_fp, jai_rgb_fp, output_movie_fp)
