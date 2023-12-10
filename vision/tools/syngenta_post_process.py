import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

from vision.misc.help_func import get_repo_dir

sys.path.append(get_repo_dir())

from vision.tools.jupyter_notebooks.notebook_analysis_help_funcs import *
from vision.tools.post_process_analysis import read_tracks_and_slices, get_block_count, count_trees_fruits, coarse_filter_depth, fine_filter_depth
from vision.visualization.draw_bb_from_csv import draw_tree_bb_from_tracks


def get_clusters_average_depth(tracks_df):
    t_depth = tracks_df.groupby('track_id').depth.mean()
    c_ids = list(tracks_df['cluster_id'].unique())
    cluster_depth = {}
    for c in c_ids:
        c_df = tracks_df.query(f'cluster_id == {c}')
        c_t_ids = list(c_df['track_id'].unique())
        c_depths = []
        for t in c_t_ids:
            c_depths.append(t_depth[t])
        cluster_depth[c] = np.mean(c_depths)

    return cluster_depth


def curate_clusters(tracks_df):
    tracks_to_clusters = dict()
    for id_, track in tracks_df.iterrows():
        found_track_ids = list(tracks_to_clusters.keys())
        track_id = track['track_id']
        cluster_id = track['cluster_id']
        if track_id in found_track_ids:
            if cluster_id not in tracks_to_clusters[track_id]:
                tracks_to_clusters[track_id].append(cluster_id)
        else:
            tracks_to_clusters[track_id] = [cluster_id]

    curated = []
    for t_id, clusters in tracks_to_clusters.items():
        if len(clusters) == 2:
            first_cluster = clusters[0]
            second_cluster = clusters[1]

            if first_cluster in curated or second_cluster in curated:
                continue

            first_df = tracks_df.query(f'cluster_id == {first_cluster}')
            first_frames = first_df.frame.unique()

            second_df = tracks_df.query(f'cluster_id == {second_cluster}')
            second_frames = second_df.frame.unique()

            frame_delta = second_frames[0] - first_frames[-1]
            if frame_delta > 0:  # no overlap
                tracks_df.loc[tracks_df['cluster_id'] == second_cluster, 'cluster_id'] = first_cluster
                curated.append(first_cluster)
                curated.append(second_cluster)

    return tracks_df, curated

def argmax_tracks_colors(tracks_df):
    track_ids = list(tracks_df['track_id'].unique())

    tracks_to_color = {}
    for track in track_ids:
        t_df = tracks_df.query(f'track_id == {track}')
        colors, count = np.unique(t_df['color'], return_counts=True)
        final_color_id = np.argmax(count)
        final_color = colors[final_color_id]
        tracks_df.loc[tracks_df['track_id'] == track, 'color'] = final_color
        if track not in list(tracks_to_color.keys()):
            tracks_to_color[track] = final_color

    return tracks_df, tracks_to_color


def is_picked(tracks_df, tracks_to_color, color_break_threshold=0.75):
    tracks_df['is_picked'] = 0
    picked_clusters = []
    breaking_colors = [1, 2, 3, 4]  # all other than green
    c_ids = list(tracks_df['cluster_id'].unique())
    for c in c_ids:
        cluster_picked = False
        cluster_colors = []
        c_df = tracks_df.query(f'cluster_id == {c}')

        track_ids = list(c_df['track_id'].unique())
        for track in track_ids:
            cluster_colors.append(int(tracks_to_color[track]))

        if len(cluster_colors) < 3:
            for color in cluster_colors:
                if color in breaking_colors:
                    cluster_picked = True
        else:
            color_breaking_ratio = np.sum(np.array(cluster_colors) < 5) / len(cluster_colors)
            if color_breaking_ratio >= color_break_threshold:
                cluster_picked = True

        if cluster_picked:
            tracks_df.loc[tracks_df['cluster_id'] == c, 'is_picked'] = 1
            picked_clusters.append(c)

    return tracks_df, picked_clusters


if __name__ == '__main__':
    row = "/media/matans/My Book/FruitSpec/Syngenta/Calibration_data/291123/row_1/1"
    tracks = '/home/matans/Documents/fruitspec/sandbox/syngenta/lean_flow_test_data_291123_5/row_1/zed'
    tracks_path = os.path.join(tracks, 'tracks.csv')
    slice_json_path = os.path.join(row, 'ZED_slice_data.json')

    tracks_df, slices_df = read_tracks_and_slices(tracks_path, slice_json_path)

    row_results, trees_tracks = count_trees_fruits(tracks_df, slices_df, block='291123', row='row_1', frame_width=1080)
    section_df = trees_tracks[1]
    def get_picked_in_section(section_df, min_samp=3):

        uniq, counts = np.unique(section_df["track_id"], return_counts=True)

        """ filter tracks below threshold"""
        valid_tracks = uniq[counts >= min_samp]
        tracks_cleaned = section_df[np.isin(section_df["track_id"], valid_tracks)]

        """ filter depth"""
        tracks_updated = coarse_filter_depth(tracks_cleaned, 1.5)
        tracks_updated = fine_filter_depth(tracks_updated)

        """ curate cluster"""
        tracks_updated, curated_clusters = curate_clusters(tracks_updated)

        """ get picked clusters"""
        tracks_updated, tracks_to_color = argmax_tracks_colors(tracks_updated)
        tracks_updated, picked_clusters = is_picked(tracks_updated, tracks_to_color)

        return tracks_updated, picked_clusters, tracks_to_color

