import os
import pandas as pd
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from skimage import exposure
from vision.tools.camera import jai_to_channels
from vision.depth.zed.svo_utils import svo_to_frames
from vision.tools.scripts.mkv_to_frames import folder_to_frames
from frames_pipeline import run as frames_pipeline_run
from vision.tools.sensors_alignment import align_folder
from vision.misc.help_func import get_repo_dir, scale_dets
from omegaconf import OmegaConf
from vision.pipelines.run_args import make_parser

# flow:
# break video to frames
# order frames in tree folders
# track folder
# align folder


def all_slices_aggregation(main_folder):
    df = pd.DataFrame({})
    for tree in os.listdir(main_folder):
        tree_path = os.path.join(main_folder, tree)
        slicespath = os.path.join(tree_path, "slices.csv")
        slice_df = pd.read_csv(slicespath)
        slice_df["start"] = slice_df["start"] * 4
        slice_df["end"] = slice_df["end"] * 4
        slice_df["start"] = slice_df["start"].replace(-4, 0)
        slice_df["end"] = slice_df["end"].replace(-4, 1536)
        df = pd.concat([df, slice_df])
    df.to_csv(os.path.join(main_folder, "all_slices.csv"))


def agg_to_trees(frames_path, slices):
    tree_ids = slices["tree_id"].unique()
    for id in tqdm(tree_ids):
        tree_folder = os.path.join(frames_path, f"T{id}")
        if not os.path.exists(tree_folder):
            os.mkdir(tree_folder)
        subslice = slices[slices["tree_id"] == id]
        frame_ids = subslice["frame_id"]
        for frame in frame_ids:
            frame_imgs = [f"channel_FSI_frame_{frame}.jpg", f"channel_RGB_frame_{frame}.jpg", f"channel_800_frame_{frame}.jpg",
            f"channel_975_frame_{frame}.jpg", f"frame_{frame}.jpg", f"depth_frame_{frame}.jpg", f"xyz_frame_{frame}.npy"]
            [shutil.copyfile(os.path.join(frames_path, img), os.path.join(tree_folder, img)) for img in frame_imgs]
        subslice.to_csv(os.path.join(tree_folder, "slices.csv"))


def get_tracker_args(config_file="/vision/pipelines/config/pipeline_config.yaml"):
    repo_dir = get_repo_dir()
    # config_file = "/config/pipeline_config.yaml"
    cfg = OmegaConf.load(repo_dir + config_file)

    args = make_parser()
    args.eval_batch = 1
    args.draw_on_img = False
    return args, cfg


def track_row(folder_path):
    args, cfg = get_tracker_args()
    res = []
    for folder in tqdm([folder for folder in os.listdir(folder_path) if "." not in folder]):
        args.data_dir = os.path.join(folder_path, folder)
        args.output_folder = args.data_dir
        if not os.path.isdir(args.output_folder):
            os.mkdir(args.output_folder)
        dets, tracks = frames_pipeline_run(cfg, args)


def preprocess_videos_to_trees(folder_path):
    # print("breaking videos to frames")
    # folder_to_frames(folder_path)
    print("aggtregating tree frames to folders")
    slices = pd.read_json(os.path.join(folder_path, "all_slices.json"))
    frames_path = os.path.join(folder_path, "frames")
    agg_to_trees(frames_path, slices)
    print("detecting and tracking for each tree")
    track_row(folder_path)
    print("aligning folders")
    align_folder(folder_path)


def preprocess_rows_to_trees(plot_path):
    for row in os.listdir(plot_path):
        row_path = os.path.join(plot_path, row_path)
        if os.path.isdir(row_path):
            preprocess_videos_to_trees(row_path)


if __name__ == "__main__":
    movies_path = "/media/fruitspec-lab/Extreme Pro/JAIZED_CaraCara_151122/R_1"
    preprocess_videos_to_trees(movies_path)
