import os
import cv2
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from vision.misc.help_func import safe_read_csv, post_process_slice_df
from vision.tools.manual_slicer import slice_to_trees_df

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


def read_tracks_and_slices(tracks_path, slice_json_path):
    tracks_df = safe_read_csv(tracks_path)

    h = 2048 if 'FSI' in slice_json_path.split('/')[-1] else 1920  # 2048
    w = 1536 if 'FSI' in slice_json_path.split('/')[-1] else 1080  # 1536
    slices_df = slice_to_trees_df(slice_json_path, h=h, w=w)

    if "frame_id" in tracks_df.columns:
        tracks_df.rename({"frame_id": "frame"}, axis=1, inplace=True)
    tracks_df["frame"] = tracks_df["frame"].astype(int)

    return tracks_df, slices_df


def count_trees_fruits(tracks_df, slices_df, block=None, row=None, frame_width=1536, cv_filter=[1,2,3]):
    row_results = []

    slices_df = post_process_slice_df(slices_df)
    slices_df["start"] = slices_df["start"].replace(-1, 0)
    slices_df["end"] = slices_df["end"].replace(-1, int(frame_width - 1))

    uniq_trees = slices_df["tree_id"].unique()
    trees_tracks = {}
    for i, tree_id in enumerate(uniq_trees):

        tree_tracks, tracker_results, tree_slices = get_tree_slice_track(tree_id, slices_df, tracks_df,
                                                                         depth_filter=False,
                                                                         min_samp_filter=False)

        tree_data = {"tree_id": tree_id}
        if block is not None:
            tree_data['block'] = block
        if row is not None:
            tree_data['row'] = row
        for cv_threshold in cv_filter:
            _, count = count_fruits(tree_tracks, cv_threshold)
            tree_data[f"{cv_threshold}"] = count

        row_results.append(tree_data)
        trees_tracks[tree_id] = tree_tracks

    return row_results, trees_tracks


def count_fruits(tracks, min_samp):
    uniq, counts = np.unique(tracks["track_id"], return_counts=True)
    valid_tracks = uniq[counts >= min_samp]
    tracks_cleaned = tracks[np.isin(tracks["track_id"], valid_tracks)]
    count = len(np.unique(tracks_cleaned["track_id"]))

    return tracks_cleaned, count

def get_block_count(block_path):
    block_counts = []
    row_tracks = {}

    block = block_path.split('/')[-1]
    block_dates = os.listdir(block_path)

    for date in block_dates:
        date_path = os.path.join(block_path, date)
        if not os.path.isdir(date_path):
            continue
        row_list = os.listdir(date_path)

        for row in row_list:
            row_path = os.path.join(date_path, row)
            if not os.path.isdir(row_path):
                continue
            row_path = os.path.join(row_path, '1')
            if not os.path.exists(row_path):
                continue

            tracks_path = os.path.join(row_path, 'tracks.csv')
            slice_json_path = os.path.join(row_path, 'Result_FSI_slice_data.json')
            tracks_df, slices_df = read_tracks_and_slices(tracks_path, slice_json_path)
            trees_counts, trees_tracks = count_trees_fruits(tracks_df, slices_df, block, row)

            block_counts += trees_counts
            row_tracks[row] = trees_tracks

    return block_counts, row_tracks


def coarse_filter_depth(tracks_df, depth_threshold=2):
    tracks_depth = tracks_df.groupby('track_id').depth.mean()

    depth = tracks_depth.values.tolist()
    valid_depth = np.array(depth) < depth_threshold

    tracks_ids = np.array(tracks_depth.keys())
    valid_tracks = tracks_ids[valid_depth]

    tracks_updated = []
    df_columns = list(tracks_df.columns)
    for id_, track in tracks_df.iterrows():
        if track['track_id'] in valid_tracks:
            tracks_updated.append(track.values.tolist())

    tracks_updated = pd.DataFrame(data=tracks_updated, columns=df_columns)

    return tracks_updated

def fine_filter_depth(tracks_df):
    data = np.array(tracks_df.depth.values).reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=0).fit(data)
    arg_ = np.argmin(gm.means_)
    depth_class = gm.predict(data)
    tf = depth_class == arg_
    #tracks_updated = tracks_df[np.logical_not(depth_class)]
    tracks_updated = tracks_df[tf]

    return tracks_updated


if __name__ == "__main__":

    fp = "/media/matans/My Book/FruitSpec/Apples_SA/block 13/block 13"
    get_block_count(fp)