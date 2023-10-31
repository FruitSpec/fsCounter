import os
import cv2
import numpy as np
import pandas as pd
from vision.misc.help_func import  safe_read_csv, post_process_slice_df
from vision.pipelines.tomato_rgb_adt import tracks2harvest
from vision.tools.manual_slicer import slice_to_trees_df


def read_slice_df_from_json(row_path):
    """Read the slice dataframe from a JSON file.

    Returns:
        The slice dataframe.

    """
    jai_slice_data_path = get_slice_data_path(row_path)
    if os.path.exists(jai_slice_data_path):
        slices_df = slice_to_trees_df(jai_slice_data_path, row_path)
        if not len(slices_df):
            slices_df = pd.DataFrame({}, columns=["tree_id", "frame_id", "start", "end"])
    else:
        slices_df = pd.DataFrame({}, columns=["tree_id", "frame_id", "start", "end"])
    return slices_df

def get_slice_data_path(row_path, scan_type = "multi_scans"):
    """
    formats the path to slicing data
    Returns:
        jai_slice_data_path (str): the path to slice data json.
    """
    side = 1 if row_path.endswith("A") else 2
    row = os.path.basename(row_path)
    if scan_type == "multi_scans":
        jai_slice_data_path = os.path.join(row_path, f"Result_FSI_slice_data_{row}.json")
        if not os.path.exists(jai_slice_data_path):
            jai_slice_data_path = os.path.join(row_path, f"Result_FSI_slice_data.json")
            if not os.path.exists(jai_slice_data_path):
                jai_slice_data_path = os.path.join(row_path, f"Result_RGB_slice_data.json")
    else:
        jai_slice_data_path = os.path.join(row_path, f"Result_FSI_{side}_slice_data_{row}.json")
        if not os.path.exists(jai_slice_data_path):
            jai_slice_data_path = os.path.join(row_path, f"Result_FSI_{side}_slice_data.json")
    return jai_slice_data_path

def get_width_height_cam(self):
    """Get the width and height of the JAI camera.

    Returns:
        A tuple containing the width and height of the camera.

    """
    width = int(self.cap_fsi.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(self.cap_fsi.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if self.rotate:
        width, height = height, width
    return width, height

def filter_outside_tree_boxes(tree_slices, tree_tracks):
    """
    Filters the tree tracks that are outside of the given tree slices.

    Args:
        tree_slices (pd.DataFrame): The tree slices.
        tree_tracks (pd.DataFrame): The tree tracks.

    Returns:
        pd.DataFrame: The filtered tree tracks.
    """
    dfs = []
    for frame in tree_slices["frame_id"]:
        slices = tree_slices[tree_slices["frame_id"] == frame][["start", "end"]].values[0]
        tree_tracks_frame = tree_tracks[tree_tracks["frame"] == frame]
        val_1 = tree_tracks_frame["x1"].values > slices[0]
        val_2 = tree_tracks_frame["x2"].values < slices[1]
        dfs.append(tree_tracks_frame[val_1 & val_2])
    return pd.concat(dfs)

def tracker_df_2_dict(tracker_results_frame):
    """Convert the tracker results dataframe to a dictionary.

    Args:
        tracker_results_frame(pd,DataFrame): The tracker results dataframe for a specific frame.

    Returns:
        A dictionary containing the track IDs as keys and the corresponding bounding box points as values.

    """
    point_1 = tuple(zip(tracker_results_frame["x1"], tracker_results_frame["y1"]))
    point_2 = tuple(zip(tracker_results_frame["x2"], tracker_results_frame["y2"]))
    points = tuple(zip(point_1, point_2))
    return dict(zip(tracker_results_frame["track_id"], points))

def get_tree_slice_track(tree_id, slices_df,tracks_df, depth_filter=False, min_samp_filter=False, min_samples = 2):
    """
    Gets the tree slice and track data for the given tree ID.

    Args:
        tree_id (int): The tree ID.
        depth_filter (bool, optional): Whether to apply depth filtering. Defaults to False.
        min_samp_filter (bool, optional): Whether to apply minimum sample filtering. Defaults to False.

    Returns:
        pd.DataFrame: The tree slice and track data.
    """
    tree_slices = slices_df[slices_df["tree_id"] == tree_id]
    tree_tracks = tracks_df[np.isin(tracks_df["frame"], tree_slices["frame_id"])]
    tree_tracks = filter_outside_tree_boxes(tree_slices, tree_tracks)
    if min_samp_filter:
        unique_tracks, counts = np.unique(tree_tracks["track_id"], return_counts=True)
        tree_tracks = tree_tracks[np.isin(tree_tracks["track_id"], unique_tracks[counts >= min_samples])]
    # if depth_filter:
    #     tree_tracks = tree_tracks[tree_tracks["depth"] < self.max_depth]
    unique_tracks, counts = np.unique(tree_tracks["track_id"], return_counts=True)
    new_ids = dict(zip(unique_tracks, range(len(unique_tracks))))
    mapped_ids = tree_tracks["track_id"].map(new_ids).values
    tree_tracks.loc[:, "track_id"] = mapped_ids
    tracker_results = {}
    for frame in tracks_df["frame"].unique():
        tracker_results_frame = tree_tracks[tree_tracks["frame"] == frame]
        tracker_results[frame] = tracker_df_2_dict(tracker_results_frame)
    return tree_tracks, tracker_results, tree_slices

def rows_report_main(TOMATO_FOLDER, MIN_SAMPLE, jai_width):

    global mode
    mode = "analysis"

    folders = []
    for phenotype in os.listdir(TOMATO_FOLDER):
        type_path = os.path.join(TOMATO_FOLDER, phenotype)
        if not os.path.isdir(type_path):
            continue
        for scan_number in os.listdir(type_path):
            if scan_number.isdigit():
                scan_path = os.path.join(type_path, scan_number)
                folders.append(scan_path)

    concatenated_df = pd.DataFrame()
    for row_path in folders:

        tracks_path = os.path.join(row_path, f'tracks.csv')
        slices_path = os.path.join(row_path, f'slices.csv')
        tracks_df, slices_df = safe_read_csv(tracks_path), safe_read_csv(slices_path)

        if "frame_id" in tracks_df.columns:
            tracks_df.rename({"frame_id": "frame"}, axis=1, inplace=True)
        tracks_df["frame"] = tracks_df["frame"].astype(int)

        if not len(slices_df):
            slices_df = read_slice_df_from_json(row_path)
        slices_df = post_process_slice_df(slices_df)
        slices_df["start"] = slices_df["start"].replace(-1, 0)
        slices_df["end"] = slices_df["end"].replace(-1, int(jai_width - 1))

        uniq_trees = slices_df["tree_id"].unique()
        for i, tree_id in enumerate(uniq_trees):
            print(
                f'phenotype_{os.path.basename(os.path.dirname(row_path))}_row_{os.path.basename(row_path)}_slice_{tree_id}')
            tree_tracks, tracker_results, tree_slices = get_tree_slice_track(tree_id, slices_df, tracks_df,
                                                                             depth_filter=False,
                                                                             min_samp_filter=False)
            tree_summary = tracks2harvest(tree_tracks, min_samp=MIN_SAMPLE)

            concatenated_df = pd.concat([concatenated_df, pd.DataFrame([tree_summary], index=[
                f'phenotype_{os.path.basename(os.path.dirname(row_path))}_row_{os.path.basename(row_path)}_slice_{tree_id}'])])

    output_report_path = os.path.join(TOMATO_FOLDER, 'rows_report.csv')
    concatenated_df.to_csv(output_report_path)
    print(f'Saved: {output_report_path}')


if __name__ == "__main__":

    TOMATO_FOLDER = "/media/fruitspec-lab/TEMP SSD/Tomato/PackoutDataNondealeaf"
    MIN_SAMPLE = 2
    jai_width = 1536

    rows_report_main(TOMATO_FOLDER, MIN_SAMPLE, jai_width)
    print ('Done')
