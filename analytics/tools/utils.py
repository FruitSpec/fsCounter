import pandas as pd
import math
import os
import numpy as np
from vision.tools.manual_slicer import slice_to_trees
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def open_measures(path, measures_name='measures.csv'):
    df = pd.read_csv(os.path.join(path, 'measures_pix_size_median_2_hue_depth.csv'))
    # df = pd.read_csv(os.path.join(path, 'measures.csv'))
    return df


def get_trees(path):
    trees_slices, border_df = slice_to_trees(os.path.join(path, [i for i in os.listdir(path) if 'slice_data' in i][0]),
                                             None, None, h=1920, w=1080)
    iter_trees = trees_slices.groupby('tree_id')
    return iter_trees, border_df


def bound_red_fruit(df):
    # filter out clusters that have only green fruits
    df_red = df[(df['color'] < 5)]
    if df_red.empty:
        return pd.DataFrame(columns=df.columns)
    min_red = df_red['y1'].min()
    df = df[df['y1'] > min_red]
    return df


def get_size_helper(sub_df):
    n_finite_widths = np.sum(np.isfinite(sub_df['width']))
    n_finite_heights = np.sum(np.isfinite(sub_df['height']))

    # return the surface of a fruit
    if n_finite_widths > 0 and n_finite_heights > 0:
        # return max(np.nanmedian(sub_df['width']), np.nanmedian(sub_df['height']))
        return np.nanmedian(sub_df['width']) * np.nanmedian(sub_df['height'])
    # if n_finite_widths > 0:
    #     return np.nanmedian(sub_df['width'])
    # if n_finite_heights > 0:
    #     return np.nanmedian(sub_df['height'])
    return np.nan


def get_size_set(df, filter_by_ratio=True):
    df = df[(df['height'] < 90) & (df['width'] < 90)]
    if filter_by_ratio:
        pix_w = df["x2"] - df["x1"]
        pix_h = df["y2"] - df["y1"]
        ratio = pix_w / pix_h
        # filter fruits with uneven BBOX # remove below 0.6
        df.loc[(ratio < 0.6) | (1 / ratio < 0.6), ["width",
                                                   "height"]] = np.nan
        # width axis occluded # keep major axis if in 0.6, 0.9
        df.loc[(ratio > 0.6) & (ratio < 0.8), "width"] = np.nan
        # width axis occluded # keep major axis if in 0.6, 0.9
        df.loc[(ratio > 1.25), "height"] = np.nan
    measures = pd.Series([get_size_helper(df_track) for ind, df_track in df.groupby("track_id")])
    return measures


def filter_by_color(df):
    counts = df["color"].value_counts()
    frequent_color = counts.idxmax()
    valids = df["color"].isin([frequent_color - 1, frequent_color, frequent_color + 1])
    return df[valids]

def filter_df_by_color(df):
    return pd.concat([filter_by_color(df_track) for ind, df_track in df.groupby("track_id")], axis=0)


def get_color_set(df):
    df_group = df.groupby('track_id')
    colors = df_group.apply(lambda x: np.nanmean(x.color)).dropna().astype(int)
    return colors


def get_count_value(df):
    ids = df['track_id'].unique()
    count = len(ids)
    return count, ids


def get_intersection_point(df_res, max_dist=3, debug=False):
    """
    Calculate the intersection point of a bimodal distribution of distances in a DataFrame.

    The function fits a Gaussian mixture model to the distance data in the input DataFrame, and calculates the
    intersection point of the two Gaussian distributions in the model by finding the minimum point of the kernel density
    estimate of the data between the two means.

    Parameters
    ----------
    df_res : pandas.DataFrame
        A DataFrame containing the distance data in a column named "distance".

    max_dist: float
        every fruit farther than max_dist will be dropped

    Returns
    -------
    intersection_point : float
        The intersection point of the two Gaussian distributions, which represents the threshold between the two modes
        of the bimodal distribution.

    """
    df_res["distance"].replace(0, np.nan, inplace=True)
    gmm = GaussianMixture(n_components=2)
    clean_dist = df_res["distance"][df_res["distance"] < max_dist].dropna().to_numpy().reshape(-1, 1)
    gmm.fit(clean_dist)
    left_distribution_index = np.argmin(gmm.means_)
    mean1, mean2 = gmm.means_[[left_distribution_index, 1 - left_distribution_index]]
    std1, std2 = np.sqrt(gmm.covariances_.flatten())[[left_distribution_index, 1 - left_distribution_index]]
    kernel = gaussian_kde(clean_dist.reshape(-1))
    vals_between_dists = np.arange(mean1, mean2, 0.05)
    density_between_dists = kernel(vals_between_dists)
    intersection_point = vals_between_dists[np.argmin(density_between_dists)]
    #  intersection_point = mean1 + 2*std1
    if debug:
        counts = plt.hist(clean_dist, bins=50, color="blue")
        max_y = np.max(counts[0])
        plt.vlines(intersection_point, 0, max_y, color="black")
        plt.vlines(mean1 + 2 * std1, 0, max_y, color="purple")
        plt.vlines(mean1 + 3 * std1, 0, max_y, color="purple", linestyle="--")
        plt.show()

    return intersection_point


def filter_df_by_min_samp(df_res, min_tracks):
    dfs_list = []
    for ind, df_track in df_res.groupby("track_id"):
        if len(df_track) > min_tracks:
            dfs_list.append(df_track)
    return pd.concat(dfs_list, axis=0)


def filter_trackers(df_res, dist_threshold):
    # df_res = filter_df_by_min_samp(df_res, min_tracks=3)  # 230123,010323
    if dist_threshold == 0:
        _dist = get_intersection_point(df_res)
    else:
        _dist = dist_threshold
    df_res = df_res[df_res["distance"] < _dist]
    df_res = filter_df_by_min_samp(df_res, min_tracks=2)  # 150223,150323,200323 - foliage
    # df_res = filter_df_by_color(df_res)
    # df_res = df_res[df_res["x1"] > 200]
    return df_res


def extract_tree_dets(df_res, df_tree=None, df_border=None):
    plot_det = []
    margin = 0
    # phenotyping analysis
    tree_frames = df_tree['frame_id'].unique()
    frames = df_res[df_res['frame'].isin(tree_frames)].groupby('frame')
    df_tree['end'].replace([-1], math.inf, inplace=True)
    for frame_id, df_frame in frames:
        if df_frame.empty:
            continue
        if df_tree is not None:
            frames_bounds = df_tree[df_tree['frame_id'] == frame_id].iloc[0]
            df = df_frame[(df_frame['x2'] + margin > frames_bounds['start']) & (
                    df_frame['x1'] < frames_bounds['end'] - margin)]
            plot_det.append(df)
            if df_border is not None:
                border_boxes = df_border[df_border['frame_id'] == frame_id]
                for index, box in border_boxes.iterrows():
                    df_b = df_frame[
                        ((df_frame['x2'] + df_frame['x1']) / 2 > box['x1']) &
                        ((df_frame['x2'] + df_frame['x1']) / 2 < box['x2']) &
                        ((df_frame['y2'] + df_frame['y1']) / 2 < box['y2']) &
                        ((df_frame['y2'] + df_frame['y1']) / 2 > box['y1'])]
                    plot_det.append(df_b)
        else:
            plot_det.append(df_frame)
    if not len(plot_det):
        raise Exception
    return pd.concat(plot_det, axis=0)

def clusters_into_values(df_res):
    """
    :param df_res:
    :return: count ,empty, empty
    """
    fruits = 0
    for cluster_id, df_cluster in df_res.groupby('cluster'):
        frames_info = {'count': [], 'redness': []}
        for frame_id, df_frame in df_cluster.groupby('frame'):
            if len(df_frame) < 2:
                continue
            frames_info['count'].append(len(df_frame))
            red_count = df_frame[df_frame['color'] <= 3]['color'].count()
            frames_info['redness'].append(red_count / len(df_frame))

        if not frames_info['count']:
            continue
        count_value = np.max(frames_info['count'])
        normed_count = frames_info['count'] / np.sum(frames_info['count'])
        weighted_color = normed_count * frames_info['redness']
        redness_value = np.sum(weighted_color)

        # picking decision
        if redness_value > 0:
            fruits += count_value

    return fruits, pd.Series(), pd.Series()

def fruits_into_values(df_res):
    """
    :param df_res: df of all detections in a file
    :return: counter, measures, colors_class
    """

    counter, extract_ids = get_count_value(df_res)
    measures = get_size_set(df_res)
    colors_class = get_color_set(df_res)
    return counter, measures, colors_class, extract_ids


def predict_weight_values(miu, sigma, observation=[]):
    a = 0.0562
    b = -70.965
    # using exponential regression
    if not len(observation):
        weight_miu = a * miu + b
        weight_sigma = a * sigma

        return weight_miu, weight_sigma

    return a * observation + b


def append_results(df, data):
    total_weight_kg = None
    if data[4] != None and data[1] != None:
        total_weight_kg = round((data[4] * data[1]) / 1000, 2)
    _df = pd.DataFrame({"plot_id": [data[0]],
                        "count": [data[1]],
                        "bin1": [data[6]],
                        "bin2": [data[7]],
                        "bin3": [data[8]],
                        "bin4": [data[9]],
                        "bin5": [data[10]],
                        "total_weight_kg": [total_weight_kg],
                        "weight_avg_gr": [data[4]],
                        "weight_std": [data[5]],
                        "size_avg_mm": [data[2]],
                        "size_std": [data[3]]})
    df = pd.concat([df, _df], axis=0)
    return df


def run_on_blocks(blocks_folder):
    res = []
    blocks = os.listdir(blocks_folder)

    for block in blocks:
        if not os.path.isdir(os.path.join(blocks_folder, block)):
            continue
        row_path = os.path.join(blocks_folder, block)
        if not np.any(["slice_data" in file for file in os.listdir(row_path)]):
            continue
        df_res = open_measures(row_path, "measures.csv")
        df_res = filter_trackers(df_res, dist_threshold=0)
        trees, borders = get_trees(row_path)
        for tree_id, df_tree in trees:
            counter, size, color = trackers_into_values(df_res, df_tree)
            res.append({"tree_id": tree_id, "count": counter, "block": block})

    res = pd.DataFrame(data=res, columns=['tree_id', 'count', 'block'])
    return res


if __name__ == "__main__":
    run_on_blocks('')
