import pandas as pd
import math
import os
import numpy as np
from vision.tools.manual_slicer import slice_to_trees


def open_measures(path, measures_name='measures.csv'):
    df = pd.read_csv(os.path.join(path, measures_name))
    return df


def get_trees(path):
    trees_slices = slice_to_trees(os.path.join(path, [i for i in os.listdir(path) if 'slice_data' in i][0]), None, None)
    iter_trees = trees_slices.groupby('tree_id')
    return iter_trees


def bound_red_fruit(df):
    # filter out clusters that have only green fruits
    df_red = df[(df['color'] < 5)]
    if df_red.empty:
        return pd.DataFrame(columns=df.columns)
    min_red = df_red['y1'].min()
    df = df[df['y1'] > min_red]
    return df


def get_size_set(df):
    df = df[(df['height'] < 90) & (df['width'] < 90)]
    df["pix_w"] = df["x2"] - df["x1"]
    df["pix_h"] = df["y2"] - df["y1"]
    ratio = df["pix_w"]/df["pix_h"]
    df = df[(ratio > 0.6) & (1/ratio > 0.6)] # filter fruits with uneven BBOX # remove below 0.6
    df.loc[(ratio > 0.6) & (ratio < 0.9), "pix_w"] = np.nan #width axis occluded # keep major axis if in 0.6, 0.9
    df.loc[(ratio > 1.11), "pix_h"] = np.nan  # width axis occluded # keep major axis if in 0.6, 0.9
    df_group = df.groupby('track_id')
    measures = df_group.apply(lambda x: max(np.nanmean(x.width), np.nanmean(x.height)))
    return measures


def get_valid_by_color(color):
    """
    computes the mode and returns for each observation if it color is valid to keep or nor
    :param color: pandas sieres contaianing color values
    :return: boolean series indicating valid observations
    """
    counts = color.value_counts()
    frequent_color = counts.idxmax()
    return color.isin([frequent_color-1, frequent_color, frequent_color+1])


def filter_by_color(df):
    """
    filters dataframe of a track id by color
    :param df: dataframe to filter
    :return: filtered dataframe or an empty dataframe if track id is invalid by color
    """
    valids = get_valid_by_color(df["color"])
    df = df[valids]
    if len(df) < 3 and np.mean(valids) < 1:
        return pd.DataFrame({})
    return df


def get_color_set(df):
    df_group = df.groupby('track_id')
    # TODO more robust desicion rule
    # filter out noise based on color
    colors = df_group.apply(lambda x: x.color.mean()).dropna().astype(int)
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
        margin = 0
        for frame_id, df_frame in frames:
            # filtter out first red fruit and above
            df = bound_red_fruit(df_frame)
            df = filter_by_color
            if df_frame.empty:
                continue
            if df_tree is not None:
                frames_bounds = df_tree[df_tree['frame_id'] == frame_id].iloc[0]
                df = df[(df['x2'] + margin > frames_bounds['start']) & (df['x1'] < frames_bounds['end'] - margin)]
            plot_det.append(df)

    plot_det = []
    if df_tree is not None:
        tree_frames = df_tree['frame_id'].unique()
        frames = df_res[df_res['frame'].isin(tree_frames)].groupby('frame')
        df_tree['end'].replace([-1], math.inf, inplace=True)
    else:
        frames = df_res.groupby('frame')

    extract_tree_det()
    try:
        df_res = pd.concat(plot_det, axis=0)
    except:
        print("err")
    # TODO coloer track id filter

    counter = get_count_value(df_res)
    measures = get_size_set(df_res)
    colors_class = get_color_set(df_res)

    return counter, measures, colors_class


def predict_weight_values(miu, sigma):
    # using exponential regression
    weight_miu = 11.475 * np.exp(0.0359 * miu)
    weight_sigma = 11.475 * np.exp(0.0359 * sigma)
    return weight_miu, weight_sigma


def append_results(df, data):
    _df = pd.DataFrame({"side": [data[0]],
                        "plot_id": [data[1]],
                        "count": [data[2]],
                        "avg_size": [data[3]],
                        "std_size": [data[4]],
                        "avg_weight": [data[5]],
                        "std_weight": [data[6]],
                        "bin1": [data[7]],
                        "bin2": [data[8]],
                        "bin3": [data[9]],
                        "bin4": [data[10]],
                        "bin5": [data[11]]})
    df = pd.concat([df, _df], axis=0)
    return df
