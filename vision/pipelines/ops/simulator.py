import os
import numpy as np
import json
import pandas as pd
from vision.misc.help_func import load_json, write_json
from vision.tools.video_wrapper import video_wrapper
from vision.data.results_collector import ResultsCollector
from vision.depth.slicer.slicer_flow import post_process

def init_cams(args, mode):
    """
    initiates all cameras based on arguments file
    :param args: arguments file
    :return: zed_cam, rgb_jai_cam, jai_cam
    """
    if mode in ['async', 'sync_svo']:
        zed_cam = video_wrapper(args.zed.movie_path, args.zed.rotate, args.zed.depth_minimum, args.zed.depth_maximum)
        depth_cam = None
    elif mode in ['sync_mkv']:
        zed_cam = video_wrapper(args.zed.movie_path, rotate=args.zed.rotate, channels=args.zed.channels)
        depth_cam = video_wrapper(args.depth.movie_path, rotate=args.depth.rotate, channels=args.depth.channels)
    rgb_jai_cam = video_wrapper(args.rgb_jai.movie_path, rotate=args.rgb_jai.rotate, channels=args.rgb_jai.channels)
    jai_cam = video_wrapper(args.jai.movie_path, rotate=args.jai.rotate, channels=args.jai.channels)
    return zed_cam, rgb_jai_cam, jai_cam, depth_cam

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


def get_max_cut_frame(args, slice_data_jai, slice_data_zed, all_slices_path):
    max_slice = []
    if os.path.exists(all_slices_path):
        df_slice = pd.read_csv(all_slices_path)
        max_slice = [df_slice["frame_id"].max()]
    frames_with_slices = [key for key, item in slice_data_jai.items() if not item["end"] is None] + \
                         [key for key, item in slice_data_zed.items() if not item["end"] is None] + max_slice
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
    if not len(jai_drops_uniq):
        return jai_drops_uniq
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
    # if np.isfinite(max_cut_frame):
    #     n_frames = max_cut_frame + 1
    # else:
    #     if metadata is not None:
    #         if "cut_frames" in metadata.keys():
    #             cut_frames = metadata["cut_frames"]
    #             n_frames = int(n_frames*cut_frames)+1
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


def zed_slicing_to_jai(slice_data_path, output_folder, rotate=False):
    slice_data = load_json(slice_data_path)
    slice_data = ResultsCollector().converted_slice_data(slice_data)
    slice_df = post_process(slice_data=slice_data)
    slice_df.to_csv(os.path.join(output_folder, 'all_slices.csv'))


def get_assignments(metadata):
    align_detect_track = True
    if "align_detect_track" in metadata.keys():
        align_detect_track = metadata["align_detect_track"]
    tree_features = True
    if "tree_features" in metadata.keys():
        tree_features = metadata["tree_features"]

    return align_detect_track, tree_features


def reset_adt_metadata(folder_path="/media/fruitspec-lab/cam175/customers/DEWAGD"):
    for root, dirs, files in os.walk(folder_path):
        if "metadata.json" in files:
            metadata_path = os.path.join(root, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata["align_detect_track"] = True
            write_json(metadata_path, metadata)

def reset_tree_features(folder_path="/media/fruitspec-lab/cam175/customers/DEWAGD"):
    for root, dirs, files in os.walk(folder_path):
        if "metadata.json" in files:
            metadata_path = os.path.join(root, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata["tree_features"] = True
            write_json(metadata_path, metadata)


def reset_metadata(folder_path="/media/fruitspec-lab/cam175/customers/DEWAGD", complete=False):
    if complete:
        for root, dirs, files in os.walk(folder_path):
            if "metadata.json" in files:
                metadata_path = os.path.join(root, "metadata.json")
                metadata = {}
                write_json(metadata_path, metadata)
    else:
        reset_tree_features(folder_path)
        reset_adt_metadata(folder_path)


def modify_calibration_data(file_path, frame_start, frame_end, shift, fixed=False, reset_meta=True):
    # Load the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Modify the values
    for frame in range(frame_start, frame_end+1):
        if not fixed:
            if str(frame) in data:
                data[str(frame)] += shift
        else:
            data[str(frame)] = frame + shift

    # Save the modified data back to the JSON file
    with open(file_path, 'w') as f:
        json.dump(data, f)

    if reset_meta:
        reset_adt_metadata(os.path.dirname(file_path))
        reset_metadata(os.path.dirname(file_path))

    print(f"{file_path} data modified successfully.")


def find_folders_with_file(root_dir, file_name):
    # Initialize an empty list to hold the directories
    dirs_with_file = []
    # Walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if file_name in filenames:
            # If the file is found, add the directory path to the list
            dirs_with_file.append(dirpath)
    return dirs_with_file
    # # Test the function
    # root_dir = "/media/fruitspec-lab/cam175/FOWLER"  # Replace with your directory path
    # file_name = "row_features.csv"
    # for file in find_folders_with_file(root_dir, file_name):
    #     print(file)


def delete_files_with_name(starting_directory, file_name):
    for root, _, files in os.walk(starting_directory):
        for file in files:
            if file == file_name:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")


if __name__ == "__main__":
    folder_path = "/media/fruitspec-lab/cam175/customers_new/MOTCHADS"
    # reset_metadata(folder_path)
    # delete_files_with_name(folder_path, "row_features.csv")
    # data_path = ""
    # modify_calibration_data(data_path, 0, 0, 0)
    print("Done")
