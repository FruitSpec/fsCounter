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
from vision.pipelines.adt_pipeline import run

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))




if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)

    zed_name = "ZED.mkv"
    depth_name = "DEPTH.mkv"
    fsi_name = "Result_FSI.mkv"
    rgb_name = "Result_RGB.mkv"
    time_stamp = "jaized_timestamps.csv"

    output_path = "/home/matans/Documents/fruitspec/sandbox/counter/June_13/plots_res"
    validate_output_path(output_path)
    plots_dir = "/home/matans/Documents/fruitspec/sandbox/counter/June_13/plots"
    #plots_dir = "/media/matans/My Book/FruitSpec/jun6"
    plots = os.listdir(plots_dir)
    #rows = ["/home/matans/Documents/fruitspec/sandbox/NWFM/val"]
    for plot in plots:
        plot_folder = os.path.join(plots_dir, plot)
        if os.path.isdir(plot_folder):
            cur_output = os.path.join(output_path, plot)
            validate_output_path(cur_output)
            dates = os.listdir(plot_folder)
            for date in dates:
                date_folder = os.path.join(plot_folder, date)
                if os.path.isdir(date_folder):
                    cur_output = os.path.join(output_path, plot, date)
                    validate_output_path(cur_output)
                    rows = os.listdir(date_folder)
                    for row in rows:
                        row_folder = os.path.join(date_folder, row, '1')
                        if os.path.isdir(row_folder):
                            cur_output = os.path.join(output_path, plot, date, row)
                            #cur_output = os.path.join(output_path, plot, row)
                            validate_output_path(cur_output)
                            if os.path.exists(os.path.join(cur_output, 'tracks.csv')):
                                # Done running on the row in previous run, skip
                                continue


                            args.output_folder = cur_output
                            args.sync_data_log_path = os.path.join(row_folder, time_stamp)
                            if not os.path.exists(args.sync_data_log_path):
                                continue
                            args.zed.movie_path = os.path.join(row_folder, zed_name)
                            if not os.path.exists(args.zed.movie_path):
                                continue
                            args.depth.movie_path = os.path.join(row_folder, depth_name)
                            if not os.path.exists(args.depth.movie_path):
                                continue
                            args.jai.movie_path = os.path.join(row_folder, fsi_name)
                            if not os.path.exists(args.jai.movie_path):
                                continue
                            args.rgb_jai.movie_path = os.path.join(row_folder, rgb_name)
                            if not os.path.exists(args.rgb_jai.movie_path):
                                continue

                            validate_output_path(args.output_folder)

                            #try:
                            rc = run(cfg, args)
                            rc.dump_feature_extractor(args.output_folder)
                            #except:
                            #    print(f'failed {row_folder}')

