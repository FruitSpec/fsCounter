import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

from vision.misc.help_func import get_repo_dir, validate_output_path

sys.path.append(get_repo_dir())

from vision.tools.jupyter_notebooks.notebook_analysis_help_funcs import *
from vision.tools.post_process_analysis import read_tracks_and_slices, get_block_count, count_trees_fruits, coarse_filter_depth, fine_filter_depth
from vision.visualization.draw_bb_from_csv import draw_tree_bb_from_tracks


def analyze_section(section_df, debug=None):

    tracks_updated, picked_clusters, tracks_to_color = get_picked_in_section(section_df)
    tracks_width, tracks_height = get_tracks_width_and_height(tracks_updated)

    color_bins = get_color_bins(tracks_to_color)

    section_report = {'clusters': len(picked_clusters),
                      'total_fruit': len(tracks_to_color),
                      'width mean': np.mean(tracks_width),
                      'width std': np.std(tracks_width),
                      'height mean': np.mean(tracks_height),
                      'height std': np.std(tracks_height),
                      'bin_1': color_bins[1],
                      'bin_2': color_bins[2],
                      'bin_3': color_bins[3],
                      'bin_4': color_bins[4],
                      'bin_5': color_bins[5],
                      }

    if debug is not None:
        draw_tree_bb_from_tracks(tracks_updated, debug['path'], debug['tree_id'], is_zed=True,
                                 data_index=debug['picked_col'])
        draw_tree_bb_from_tracks(tracks_updated, debug['path'], debug['tree_id'], is_zed=True,
                                 data_index=debug['cluster_col'], output_folder=debug['cluster_output'])

    return section_report




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

    tracks_to_color = filter_non_picked_tracks(tracks_updated, tracks_to_color)

    return tracks_updated, picked_clusters, tracks_to_color


def filter_non_picked_tracks(tracks_df, tracks_to_color):
    picked_df = tracks_df.query('is_picked == 1')
    picked_tracks = np.unique(picked_df['track_id'])

    final_tracks_to_color = dict()
    for key, value in tracks_to_color.items():
        if key in picked_tracks:
            final_tracks_to_color[key] = value

    return final_tracks_to_color



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

    tracks_df['center_x'] = (tracks_df['x1'] + tracks_df['x2']) / 2
    tracks_df['center_y'] = (tracks_df['y1'] + tracks_df['y2']) / 2
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

            else: # there is overlap. see when cluster center changes
                f_y_centers = first_df.groupby('frame').center_y.mean()

                s_y_centers = second_df.groupby('frame').center_y.mean()

                f_y = np.array(f_y_centers)
                f_y_std = np.std(f_y)

                s_y = np.array(s_y_centers)
                s_y_std = np.std(s_y)


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


def get_tracks_width_and_height(tracks_df, upper_filter=0.09, lower_filter=0.04):
    """ default filters assume tomato can't be bigger than 9 cm any axis"""
    """ default filters assume tomato can't be smaller than 4 cm any axis"""
    picked_df = tracks_df.query('is_picked == 1')
    tracks_width = picked_df.groupby('track_id').width.mean()
    tracks_height = picked_df.groupby('track_id').height.mean()

    tracks_width = tracks_width[tracks_width < upper_filter].copy()
    tracks_width = tracks_width[tracks_width > lower_filter].copy()

    tracks_height = tracks_height[tracks_height < upper_filter].copy()
    tracks_height = tracks_height[tracks_height > lower_filter].copy()

    return tracks_width, tracks_height

def get_color_bins(tracks_to_color):

    color_bins = {}
    colors = []
    for color in tracks_to_color.values():
        colors.append(color)

    unique_colors, color_counts = np.unique(colors, return_counts=True)

    for bin in range(1, 6): # we have colors in range 1 to 5 include
        if bin in unique_colors:
            bin_index = np.argwhere(unique_colors == bin)
            color_bins[bin] = color_counts[bin_index[0, 0]]
        else:
            color_bins[bin] = 0

    return color_bins



if __name__ == '__main__':
    #row = "/media/matans/My Book/FruitSpec/Syngenta/Calibration_data/291123/row_1/1"
    folder_path = "/media/matans/My Book/FruitSpec/Syngenta/Calibration_data/291123"
    #tracks = '/home/matans/Documents/fruitspec/sandbox/syngenta/lean_flow_test_data_291123_5/row_1/zed'
    #tracks_path = os.path.join(tracks, 'tracks.csv')
    to_debug = True


    rows = os.listdir(folder_path)
    res = []
    for row in rows:
        row_path = os.path.join(folder_path, row)
        if not os.path.isdir(row_path):
            continue
        repetitions = os.listdir(row_path)
        for rep in repetitions:
            rep_path = os.path.join(row_path, rep)
            tracks_path = os.path.join(rep_path, 'tracks.csv')
            slice_json_path = os.path.join(rep_path, 'ZED_slice_data.json')

            if not os.path.exists(slice_json_path):
                continue

            tracks_df, slices_df = read_tracks_and_slices(tracks_path, slice_json_path)

            row_results, trees_tracks = count_trees_fruits(tracks_df, slices_df, frame_width=1080)
            section_df = trees_tracks[1]
            if to_debug:
                validate_output_path(os.path.join(rep_path, 'tree_cluster'))
                debug = {'path': rep_path,
                         'tree_id': 1,
                         'picked_col': -1,
                         'cluster_col': -3,
                         'cluster_output': os.path.join(rep_path, 'tree_cluster')
                         }
            section_results = analyze_section(section_df, debug)

            section_results['row'] = row
            section_results['rep'] = rep

            res.append(section_results)


    results = pd.DataFrame(res, columns=list(section_results.keys()))
    results.to_csv(os.path.join(folder_path, 'results.csv'))
