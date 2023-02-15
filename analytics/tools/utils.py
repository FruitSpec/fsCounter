import pandas as pd
import math
import os
import numpy as np
from vision.tools.manual_slicer import slice_to_trees


def open_measures(path):
    df = pd.read_csv(os.path.join(path, 'measures.csv'))
    return df


def get_trees(path):
    trees_slices = slice_to_trees(os.path.join(path, [i for i in os.listdir(path) if 'slice_data' in i][0]), None, None)
    iter_trees = trees_slices.groupby('tree_id')
    return iter_trees


def bound_red_fruit(df):
    # filter out clusters that have only green fruits
    df_red = df[(df['color'] < 60) & (df['color'] > 0)]
    if df_red.empty:
        return pd.DataFrame(columns=df.columns)
    min_red = df_red['y1'].min()
    df = df[df['y1'] > min_red]
    return df


def get_size_set(df):
    df = df[(df['height'] < 90) & (df['width'] < 90)]
    df_group = df.groupby('track_id')
    measures = df_group.apply(lambda x: x.width.max() if x.width.mean() > x.height.mean() else x.height.max())
    return measures


def get_color_set(df):
    def color(x):
        if x > 0 and x < 25:
            return 1
        elif x > 25 and x < 45:
            return 2
        elif x > 45 and x < 65:
            return 3
        elif x > 90 and x < 165:
            return 4

    df = df[(df['color'] < 65) | ((df['color'] > 90) & (df['color'] < 165))]
    df_group = df.groupby('track_id')
    colors = df_group.apply(lambda x: color(x.color.mean())).dropna().astype(int)
    return colors


def get_count_value(df):
    count = len(df['track_id'].unique())
    return count


def trackers_into_values(df_res, df_tree=None):
    """
    :param df_res: df of all detections in a file
    :param df_tree: df of relevent frame per tree and its start_x end_x , deafult is None in case that no subset of df_res is needed
    :return: counter, measures, colors_class
    """

    def extract_tree_det():
        df_tree['end'].replace([-1], math.inf, inplace=True)
        plot_det = []
        margin = 10
        frames = frames[frames.index == df_tree['frame_id'].unique()]
        for frame_id, df_frame in frames:
            # filtter out first red fruit and above
            df_frame = bound_red_fruit(df_frame)
            if df_frame.empty:
                continue
            frames_bounds = df_tree[df_tree['frame_id'] == frame_id].iloc[0]
            df = df_frame[(df_frame['x2'] + margin > frames_bounds['start']) & (df_frame['x1'] < frames_bounds['end'] - margin)]
            plot_det.append(df)

    frames = df_res.groupby('frame')
    if df_tree is None:
        df_res = pd.concat(frames, axis=0)
    else:
        plot_det = []
        extract_tree_det()
        df_res = pd.concat(plot_det, axis=0)

    counter = get_count_value(df_res)
    measures = get_size_set(df_res)
    colors_class = get_color_set(df_res)

    return counter, measures, colors_class


def predict_weight_values(miu, sigma):
    weight_miu = 11.475 * np.exp(0.0359 * miu)
    weight_sigma = 11.475 * np.exp(0.0359 * sigma)
    return weight_miu, weight_sigma
