import os
import sys
import cv2
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


from vision.tools.video_wrapper import video_wrapper

def random_split(movie_path, output_path, frames_to_split=100, rotation=1):

    cam = video_wrapper(movie_path, rotate=rotation)

    plot, date, row_id = get_movie_meta(movie_path)

    frames_ids_vec = np.arange(0, cam.get_number_of_frames()).astype(np.uint16)

    counter = 0
    progress_bar = tqdm(total=frames_to_split, desc="Processing")
    while counter < frames_to_split and len(frames_ids_vec) > 0:

        progress_bar.update(1)
        index_ = np.random.randint(0, len(frames_ids_vec))
        res, frame = cam.get_frame(frames_ids_vec[index_])
        if not res:
            break

        f_id = frames_ids_vec[index_]
        output_file_path = os.path.join(output_path, f"{plot}_{date}_{row_id}_f{f_id}.jpg")
        cv2.imwrite(output_file_path, frame)

        frames_ids_vec = np.delete(frames_ids_vec, index_)
        counter += 1

    print('Done')

def get_movie_meta(movie_path):

    splited = movie_path.split('/')
    row_id = splited[-3]
    date = splited[-4]
    plot = splited[-5]

    return plot, date, row_id

if __name__ == "__main__":
    movie_path = "/media/matans/My Book/FruitSpec/Customers_data/Fowler/daily/PAULBLOC/220723/row_13/1/Result_FSI.mkv"
    output_path = "/media/matans/My Book/FruitSpec/Detector_testset"
    random_split(movie_path, output_path)