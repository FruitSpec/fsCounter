import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.cluster import KMeans

from vision.misc.help_func import get_repo_dir, validate_output_path

sys.path.append(get_repo_dir())

from vision.tools.jupyter_notebooks.notebook_analysis_help_funcs import *
from vision.tools.post_process_analysis import read_tracks_and_slices, get_block_count, count_trees_fruits, coarse_filter_depth, fine_filter_depth
from vision.visualization.draw_bb_from_csv import draw_tree_bb_from_tracks

from syngenta_post_process import apply_model, convert_dist, argmax_tracks_colors, get_tracks_width_and_height, get_color_bins

def get_weight_std(conv_width, conv_height, model_coef):
    var_width = np.var(conv_width)
    var_height = np.var(conv_height)
    weight_std = np.sqrt((var_width * np.power(model_coef[0], 2)) + (var_height * np.power(model_coef[1], 2)))

    return weight_std

def analyze_plot_results(plot_df, plot_name, min_samp=5):
    uniq, counts = np.unique(plot_df["track_id"], return_counts=True)

    """ filter tracks below threshold"""
    valid_tracks = uniq[counts >= min_samp]
    tracks_updated = plot_df[np.isin(plot_df["track_id"], valid_tracks)]

    """ get colors"""
    tracks_updated, tracks_to_color = argmax_tracks_colors(tracks_updated)

    """ convert width and height according to model"""
    conv_width, conv_height = convert_width_and_height(tracks_updated)

    mean_width = np.nanmean(conv_width)

    mean_height = np.nanmean(conv_height)


    weight_mean = apply_model(mean_width, mean_height, model_coef, model_intercept)
    weight_std = get_weight_std(conv_width, conv_height, model_coef)

    """ get color bins"""
    color_bins = get_color_bins(tracks_to_color)

    gt_data = gt_df.query(f'Plot_name == "{plot_name}"')

    plot_results = create_results_dict(gt_data,
                                       tracks_to_color,
                                       color_bins,
                                       weight_mean,
                                       weight_std,
                                       conv_width,
                                       row,
                                       rep)

    return plot_results

def convert_width_and_height(tracks_updated):
    tracks_updated['is_picked'] = 1
    tracks_width, tracks_height = get_tracks_width_and_height(tracks_updated)

    width = np.array(tracks_width) * 1000  # convert to mm
    height = np.array(tracks_height) * 1000  # convert to mm

    conv_width = convert_dist(width,
                              counter_width_mean,
                              counter_width_std,
                              gt_width_mean,
                              gt_width_std)

    conv_height = convert_dist(height,
                               counter_height_mean,
                               counter_height_std,
                               gt_height_mean,
                               gt_height_std)

    return conv_width, conv_height


def create_results_dict(gt_data, tracks_to_color, color_bins, weight, weight_std, conv_width, row, rep):

    plot_results = dict()
    plot_results['Farm'] = gt_data['Farm'].values[0]
    plot_results['Plot_name'] = gt_data['Plot_name'].values[0]
    plot_results['FS_variety'] = gt_data['FS_variety'].values[0]
    plot_results['count'] = len(tracks_to_color)
    plot_results['bin_1'] = color_bins[1]
    plot_results['bin_2'] = color_bins[2]
    plot_results['bin_3'] = color_bins[3]
    plot_results['bin_4'] = color_bins[4]
    plot_results['bin_5'] = color_bins[5]
    plot_results['total_weight_kg'] = (len(tracks_to_color) * np.mean(weight)) / 1000  # in kg
    plot_results['weight_avg_gr'] = weight
    plot_results['weight_std'] = weight_std
    plot_results['size_avg_mm'] = np.mean(conv_width)
    plot_results['size_std'] = np.std(conv_width)
    plot_results['row'] = row
    plot_results['rep'] = rep

    return plot_results


if __name__ == '__main__':

    folder_path = "/home/matans/Documents/fruitspec/sandbox/syngenta/110124"
    gt_data_path = "/home/matans/Documents/fruitspec/sandbox/syngenta/almeria_plot_meta.csv"
    if gt_data_path is not None:
        gt_df = pd.read_csv(gt_data_path)
    else:
        gt_df = None

    to_debug = True
    min_samp = 7
    dist_center_threshold = 200
    color_break_threshold = 0.9

    model_coef = [4.25467987, 2.40293413]
    model_intercept = -271.53407231486557

    gt_width_mean = 66.64207831325301
    gt_width_std = 6.524271084973641
    gt_height_mean = 54.35472891566266
    gt_height_std = 6.012594102523396

    counter_width_mean = 67.55363882906708
    counter_width_std = 12.210693006488093
    counter_height_mean = 67.61603085290156
    counter_height_std = 10.08302167961458

    rows = os.listdir(folder_path)
    #rows = ['row_2', 'row_6', 'row_7', 'row_8']
    res = []
    width = []
    height = []
    for row in tqdm(rows):
        row_path = os.path.join(folder_path, row)
        if not os.path.isdir(row_path):
            continue
        repetitions = os.listdir(row_path)
        row_number = int(row.split('_')[-1])
        direction = 'right' if row_number % 2 != 0 else 'left'
        for rep in repetitions:
            rep_path = os.path.join(row_path, rep)
            tracks_path = os.path.join(rep_path, 'tracks.csv')
            row_meta_path = os.path.join(rep_path, 'row_meta.csv')

            if not os.path.exists(row_meta_path):
                print(f'row meta is missing for row {row}')
                continue
            try:
                tracks_df = pd.read_csv(tracks_path)
                meta_df = pd.read_csv(row_meta_path)

                plots = meta_df['plot'].unique()

                plot_data = []
                for p in plots:
                    sub_meta_df = meta_df.query(f'plot == "{p}"')
                    plot_tracks = list(sub_meta_df['track_id'].unique())

                    for id_, df_row in tracks_df.iterrows():
                        if int(df_row['track_id']) in plot_tracks:
                            plot_data.append(df_row.to_list())


                    plot_df = pd.DataFrame(data=plot_data, columns=list(tracks_df.columns))

                    for id_, meta_row in sub_meta_df.iterrows():
                        plot_df.loc[plot_df['track_id'] == meta_row['track_id'], 'cluster_id'] = meta_row['cluster_id']

                    plot_results = analyze_plot_results(plot_df, plot_name=p, min_samp=min_samp)

                    res.append(plot_results)
            except:
                print(f'failed to run {rep_path}')

    results = pd.DataFrame(res, columns=list(plot_results.keys()))
    results.to_csv(os.path.join(folder_path, 'results_m.csv'))


