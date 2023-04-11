import os
import numpy as np
import json

from vision.misc.help_func import load_json, write_json
from vision.tools.video_wrapper import video_wrapper

def init_cams(args):
    """
    initiates all cameras based on arguments file
    :param args: arguments file
    :return: zed_cam, rgb_jai_cam, jai_cam
    """
    zed_cam = video_wrapper(args.zed.movie_path, args.zed.rotate, args.zed.depth_minimum, args.zed.depth_maximum)
    rgb_jai_cam = video_wrapper(args.rgb_jai.movie_path, args.rgb_jai.rotate)
    jai_cam = video_wrapper(args.jai.movie_path, args.jai.rotate)
    return zed_cam, rgb_jai_cam, jai_cam

def load_logs(args):
    """
    Load logs from the provided file paths and update args with metadata.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Tuple containing:
            - slice_data_zed (dict): Dictionary containing ZED slice data.
            - slice_data_jai (dict): Dictionary containing JAI slice data.
    """
    slice_data_zed, slice_data_jai = load_json(args.slice_data_path), load_json(args.jai_slice_data_path)

    return slice_data_zed, slice_data_jai


def get_frame_drop(args):
    try:
        frame_drop_jai = get_jai_drops(args.frame_drop_path)
    except Exception as e:
        print(e)
        frame_drop_jai = np.array([])

    return frame_drop_jai


def get_max_cut_frame(args, slice_data_jai, slice_data_zed):
    frames_with_slices = [key for key, item in slice_data_jai.items() if not item["end"] is None] + \
                         [key for key, item in slice_data_zed.items() if not item["end"] is None]
    max_cut_frame = np.max(frames_with_slices) if len(frames_with_slices) else np.inf

    return max_cut_frame


def update_arg_with_metadata(args):
    """
    updates the arguments base on metadata json, this is for passsing specific argumetns for each row
    :param args: current args
    :return: updated args
    """
    meta_data_path = get_metadata_path(args)
    metadata = {}
    if os.path.exists(meta_data_path):
        with open(meta_data_path, 'r') as f:
            metadata = json.load(f)
            metadata_keys = metadata.keys()
            if "zed_shift" in metadata_keys:
                args.zed_shift = metadata["zed_shift"]
            if "block_name" in metadata_keys:
                args.block_name = metadata["block_name"]
            if "max_z" in metadata_keys:
                args.max_z = metadata["max_z"]
            if "zed_rotate" in metadata_keys:
                args.zed.rotate = metadata["zed_rotate"]
    return args, metadata

def get_metadata_path(args):
    """
    returns metadata path based on moviepath in args
    :param args: argumetns hash
    :return: meta_data_path
    """
    row_folder = os.path.dirname(args.jai.movie_path)
    meta_data_path = os.path.join(row_folder, "metadata.json")
    return meta_data_path

def get_jai_drops(frame_drop_path):
    """
    reads jai log file and extracts the number of the dropped frames
    :param frame_drop_path: path to log file
    :return: numbers of dropped frames
    """
    jai_drops = np.array([])
    if not os.path.exists(frame_drop_path):
        return jai_drops
    with open(frame_drop_path, "r") as logfile:
        lines = logfile.readlines()
        for line in lines:
            if "FRAME DROP" in line:
                jai_drops = np.append(jai_drops, line.strip().split(" ")[-1])
    jai_drops_uniq = np.unique(jai_drops).astype(int)
    jai_drops_uniq.sort()
    jai_drops_uniq -= range(len(jai_drops_uniq))
    return jai_drops_uniq


def get_n_frames(max_cut_frame, jai_number_of_frames, metadata=None):
    """
    Returns the number of frames to run on.

    Args:
        max_cut_frame (float): The maximum frame to run on. If `max_cut_frame` is not finite,
         `n_frames` will be `max_cut_frame + 1`.
        jai_number_of_frames: The number of frames in Jai clip
        metadata (dict): A dictionary containing metadata related to the camera setup and configuration.

    Returns:
        int: The number of frames to run on. If `metadata` contains a key "cut_frames",
         `n_frames` will be the number of frames adjusted by the cut ratio.
          If `max_cut_frame` is not finite, `n_frames` will be `max_cut_frame + 1`.

    Note:
        `n_frames` is the number of frames the program will run on.
                    If tree slicing was done will take n_frames as last frames siced
    """
    n_frames = jai_number_of_frames
    if np.isfinite(max_cut_frame):
        n_frames = max_cut_frame + 1
    else:
        if metadata is not None:
            if "cut_frames" in metadata.keys():
                cut_frames = metadata["cut_frames"]
                n_frames = int(n_frames*cut_frames)+1
    return n_frames


def write_metadata(args, metadata):
    """
    writes metadata json
    :param args: argumetns file
    :param metadata: metadata dict
    :return:
    """
    meta_data_path = get_metadata_path(args)
    write_json(meta_data_path, metadata)


def reset_adt_metadata(folder_path="/media/fruitspec-lab/cam175/customers/DEWAGD"):
    for root, dirs, files in os.walk(folder_path):
        if "metadata.json" in files:
            metadata_path = os.path.join(root, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata["align_detect_track"] = True
            write_json(metadata_path, metadata)