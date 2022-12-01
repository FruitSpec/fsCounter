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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# flow:
# break video to frames
# order frames in tree folders
# track folder
# align folder


def update_index(k, index):
    """
    :param k: current key value
    :param index: current index
    :return: next index
    """
    cont = False
    if k == 83 or k ==100:
        index = max(index - 1, 0)
        cont = True
    if k == 81 or k ==97:
        index = max(index + 1, 0)
        cont = True
    if k == 82 or k == 119:
        index = max(index + 100, 0)
        cont = True
    if k == 84 or k == 115:
        index = max(index - 100, 0)
        cont = True
    if k == 32:
        global x_0, y_0, x_1, y_1
        x_0, y_0, x_1, y_1 = None, None, None, None
    return index, cont


def make_one_image(depth_img, img, size):
    """
    combines depth image and rgb image
    :param depth_img: depth_img
    :param img: rgb image
    :param size: size rgb/depth image
    :return: a combined image of rgb and depth side by side
    """
    img_display = np.zeros((int(size[1]), size[0] * 2, 3), dtype='uint8')
    im_depth_display = cv2.resize(depth_img, size)
    im_left_display = cv2.resize(img, size)
    img_display[:, :size[0]] = im_left_display
    img_display[:, size[0]:] = im_depth_display
    return img_display


def dual_frame_viewer(master_frames_folder="/media/fruitspec-lab/Extreme Pro/JAIZED_CaraCara_151122/R_1/frames"):
    all_frames = os.listdir(master_frames_folder)
    fsi_imgs = [img for img in all_frames if "FSI" in img]
    zed_imgs = [img for img in all_frames if img.startswith("frame")]
    fsi_imgs.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
    zed_imgs.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
    max_frame = max(len(fsi_imgs), len(zed_imgs))
    index = 0
    cv2.namedWindow('img')
    while index < max_frame:
        print(index)
        fsi = cv2.imread(os.path.join(master_frames_folder, fsi_imgs[index]))
        # if index >= 1677:
        #     zed_ind = index - 18
        # else:
        #     zed_ind = index
        zed_ind = index
        zed = cv2.imread(os.path.join(master_frames_folder, zed_imgs[zed_ind]))
        zed = cv2.resize(zed, (400, 600))
        fsi = cv2.resize(fsi, (400, 600))
        one_img = make_one_image(zed, fsi, (800, 600))
        cv2.imshow('img', one_img)
        k = cv2.waitKey()
        index, cont = update_index(k, index)


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


def copy_frames(frame, tree_folder, frames_path, zed_shift=0):
    frame_imgs = [f"channel_FSI_frame_{frame}.jpg", f"channel_RGB_frame_{frame}.jpg",
                  f"frame_{int(frame) + zed_shift}.jpg", f"xyz_frame_{int(frame) + zed_shift}.npy"]
    if not np.all([os.path.exists(os.path.join(frames_path, img)) for img in frame_imgs]):
        pass
    [shutil.copyfile(os.path.join(frames_path, img), os.path.join(tree_folder, img)) for img in frame_imgs
     if os.path.exists(os.path.join(frames_path, img))]


def agg_to_trees(frames_path, slices, zed_shift=0,
                 m_threds=16, m_procs=0):
    trees_path = os.path.join(os.path.dirname(frames_path), "trees")
    if not os.path.exists(trees_path):
        os.mkdir(trees_path)
    tree_ids = slices["tree_id"].unique()
    for id in tqdm(tree_ids):
        tree_folder = os.path.join(trees_path, f"T{id}")
        if not os.path.exists(tree_folder):
            os.mkdir(tree_folder)
        subslice = slices[slices["tree_id"] == id]
        frame_ids = subslice["frame_id"]
        n_frames = len(frame_ids)
        if m_threds > 1:
            with ThreadPoolExecutor(max_workers=m_threds) as executor:
                results = list(executor.map(copy_frames, frame_ids, [tree_folder]*n_frames, [frames_path]*n_frames, [zed_shift]*n_frames))
        elif m_procs > 1:
            with ProcessPoolExecutor(max_workers=m_procs) as executor:
                results = list(executor.map(copy_frames, frame_ids, [tree_folder]*n_frames, [frames_path]*n_frames, [zed_shift]*n_frames))
        else:
            for frame in frame_ids:
                copy_frames(frame, tree_folder, frames_path, zed_shift)
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


def preprocess_videos_to_trees(folder_path, zed_shift=0):
    print("breaking videos to frames")
    folder_to_frames(folder_path)
    print("aggtregating tree frames to folders")
    slices = pd.read_csv(os.path.join(folder_path, "all_slices.csv"))
    frames_path = os.path.join(folder_path, "frames")
    agg_to_trees(frames_path, slices, zed_shift)
    print("detecting and tracking for each tree")
    trees_path = os.path.join(folder_path, "trees")
    track_row(trees_path)
    print("aligning folders")
    pathed_trees = [os.path.join(trees_path, tree) for tree in os.listdir(trees_path)]
    n_trees = len(pathed_trees)
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(align_folder, pathed_trees, [""] * n_trees, [False] * n_trees, [False] * n_trees,
                                    [zed_shift] * n_trees))


def preprocess_rows_to_trees(plot_path, zed_shift=0):
    for row in os.listdir(plot_path):
        print(row)
        row_path = os.path.join(plot_path, row)
        if os.path.isdir(row_path):
            preprocess_videos_to_trees(row_path, zed_shift)


if __name__ == "__main__":
    # TODO break trees to sides
    #dual_frame_viewer("/media/fruitspec-lab/easystore/JAIZED_CaraCara_301122/r6/frames")
    plot_path = "/media/fruitspec-lab/easystore/JAIZED_CaraCara_301122"
    preprocess_videos_to_trees("/media/fruitspec-lab/easystore/JAIZED_CaraCara_301122/r6", zed_shift=0)
    for row in os.listdir(plot_path):
        print(row)
        row_path = os.path.join(plot_path, row)
        if os.path.isdir(row_path) and "frames" not in os.listdir(row_path):
            folder_to_frames(row_path)
    movies_path = "/media/fruitspec-lab/easystore/JAIZED_CaraCara_151122/R_1_testing"
    preprocess_videos_to_trees(movies_path)
    for i in range(1, 10):
        movies_path = f"/media/fruitspec-lab/easystore/JAIZED_CaraCara_151122/R_{i}"
        print(movies_path)
        folder_to_frames(movies_path)
    preprocess_videos_to_trees(movies_path)
