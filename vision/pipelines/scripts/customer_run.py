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

def run_plots_anlysis(plots_dir, output_path):
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

    analyzed_path = os.path.join(output_path, 'analysis.csv')
    if os.path.exists(analyzed_path):
        analyzed = pd.read_csv(analyzed_path)
    else:
        analyzed = pd.DataFrame(columns=["plot_code", "scan_date", "row", "status"])



    plots = os.listdir(plots_dir)
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

                            if len(analyzed.query(f"plot_code=='{plot}' and scan_date=={date} and row=='{row}'")) > 0: # already analyzed
                                continue

                            cur_output = os.path.join(output_path, plot, date, row)
                            validate_output_path(cur_output)

                            valid = True
                            #if os.path.exists(os.path.join(cur_output, 'tracks.csv')):
                            #    # Done running on the row in previous run, skip
                            #    valid = False

                            args.output_folder = cur_output
                            args.sync_data_log_path = os.path.join(row_folder, time_stamp)
                            if not os.path.exists(args.sync_data_log_path):
                                valid = False
                            args.zed.movie_path = os.path.join(row_folder, zed_name)
                            if not os.path.exists(args.zed.movie_path):
                                valid = False
                            args.depth.movie_path = os.path.join(row_folder, depth_name)
                            if not os.path.exists(args.depth.movie_path):
                                valid = False
                            args.jai.movie_path = os.path.join(row_folder, fsi_name)
                            if not os.path.exists(args.jai.movie_path):
                                valid = False
                            args.rgb_jai.movie_path = os.path.join(row_folder, rgb_name)
                            if not os.path.exists(args.rgb_jai.movie_path):
                                valid = False

                            validate_output_path(args.output_folder)

                            if valid:
#                                try:
                                rc = run(cfg, args, n_frames=500)
                                rc.dump_results(args.output_folder)

                                analyzed_data = {
                                    "plot_code": [plot],
                                    "scan_date": [str(date)],
                                    "row": [str(row)],
                                    "status": ["succeed"],
                                }
                               #except:
                                #    print(f'failed {row_folder}')
                                #   analyzed_data = {
                                #        "plot_code": [plot],
                                #        "scan_date": [str(date)],
                                #        "row": [str(row)],
                                #        "status": ["failed"],
                                #    }
                            else:
                                analyzed_data = {
                                    "plot_code": [plot],
                                    "scan_date": [str(date)],
                                    "row": [str(row)],
                                    "status": ["failed_missing_data"],
                                }
                            write_to_file(analyzed_data, analyzed_path)

def write_to_file(analyzed_data, analyzed_path):
    analyzed_df = pd.DataFrame(data=analyzed_data, index=[0])
    is_first = not os.path.exists(analyzed_path)
    analyzed_df.to_csv(analyzed_path, header=is_first, index=False, mode="a+")


if __name__ == "__main__":


    output_path = "/home/matans/Documents/fruitspec/sandbox/counter/June_13/plots_test"
    validate_output_path(output_path)
    plots_dir = "/home/matans/Documents/fruitspec/sandbox/counter/June_13/plots"
    run_plots_anlysis(plots_dir, output_path)