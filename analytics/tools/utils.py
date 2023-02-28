import pandas as pd
import math
import os
import numpy as np
from vision.tools.manual_slicer import slice_to_trees
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture


def open_measures(path, measures_name='measures.csv'):
    df = pd.read_csv(os.path.join(path, measures_name))
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
    if n_finite_widths > 0 and n_finite_heights > 0:
        return np.nanmax([np.nanmedian(sub_df['width']), np.nanmedian(sub_df['height'])])
    if n_finite_widths > 0:
        return np.nanmedian(sub_df['width'])
    if n_finite_heights > 0:
        np.nanmedian(sub_df['height'])
    return np.nan


def get_size_set(df, filter_by_ratio=True):
    df = df[(df['height'] < 90) & (df['width'] < 90)]
    if filter_by_ratio:
        pix_w = df["x2"] - df["x1"]
        pix_h = df["y2"] - df["y1"]
        ratio = pix_w / pix_h
        df.loc[(ratio < 0.6) | (1 / ratio < 0.6), ["width",
                                                   "height"]] = np.nan  # filter fruits with uneven BBOX # remove below 0.6
        df.loc[(ratio > 0.6) & (ratio < 0.8), "width"] = np.nan  # width axis occluded # keep major axis if in 0.6, 0.9
        df.loc[(ratio > 1.25), "height"] = np.nan  # width axis occluded # keep major axis if in 0.6, 0.9
    measures = pd.DataFrame([get_size_helper(df_track) for ind, df_track in df.groupby("track_id")])
    return measures


def get_valid_by_color(color):
    """
    computes the mode and returns for each observation if it color is valid to keep or nor
    :param color: pandas sieres contaianing color values
    :return: boolean series indicating valid observations
    """
    counts = color.value_counts()
    frequent_color = counts.idxmax()
    return color.isin([frequent_color - 1, frequent_color, frequent_color + 1])


def filter_by_color(df):
    """
    Filters a DataFrame of objects by color.

    Uses the get_valid_by_color function to determine which objects have valid
    colors. Filters the DataFrame using the resulting boolean array.
    If there are fewer than three objects in the DataFrame and the mean of the
    valid boolean array is less than one, an empty DataFrame is returned.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of objects with a color column.

    Returns
    -------
    pandas.DataFrame
        A filtered DataFrame of objects with a color column.

    """
    valids = get_valid_by_color(df["color"])
    return df[valids]


def filter_df_by_color(df):
    """
    Filter a DataFrame of tomatos based on the color of the objects in the crops.

    The function groups the DataFrame by "track_id" and applies the filter_by_color function to each group, which
    filters the crops in the group based on their color. The filtered crops are concatenated into a single DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing columns "track_id" and "color"

    Returns
    -------
    filtered_df : pandas.DataFrame
        A DataFrame containing the same columns as the input DataFrame, but with only the observations that passed the
        color filter.

    """
    return pd.concat([filter_by_color(df_track) for ind, df_track in df.groupby("track_id")], axis=0)


def get_color_set(df):
    df_group = df.groupby('track_id')
    colors = df_group.apply(lambda x: np.nanmean(x.color)).dropna().astype(int)
    return colors


def get_count_value(df):
    ids = df['track_id'].unique()
    count = len(ids)
    return count, ids


def get_intersection_point(df_res):
    """
    Calculate the intersection point of a bimodal distribution of distances in a DataFrame.

    The function fits a Gaussian mixture model to the distance data in the input DataFrame, and calculates the
    intersection point of the two Gaussian distributions in the model by finding the minimum point of the kernel density
    estimate of the data between the two means.

    Parameters
    ----------
    df_res : pandas.DataFrame
        A DataFrame containing the distance data in a column named "distance".

    Returns
    -------
    intersection_point : float
        The intersection point of the two Gaussian distributions, which represents the threshold between the two modes
        of the bimodal distribution.

    """
    df_res["distance"].replace(0, np.nan, inplace=True)
    gmm = GaussianMixture(n_components=2)
    clean_dist = df_res["distance"].dropna().to_numpy().reshape(-1, 1)
    gmm.fit(clean_dist)
    mean1, mean2 = gmm.means_
    kernel = gaussian_kde(clean_dist.reshape(-1))
    vals_between_dists = np.arange(min(mean1, mean2), max(mean1, mean2), 0.05)
    density_between_dists = kernel(vals_between_dists)
    intersection_point = vals_between_dists[np.argmin(density_between_dists)]
    return intersection_point


def filter_df_by_min_samp(df_res, min_samples=3):
    """
    Filter a DataFrame of image crops based on the number of samples in each track.

    The function groups the input DataFrame by "track_id" and filters the groups to only include those with a number
    of samples greater than or equal to the specified "min_samples" value. The filtered groups are then concatenated
    into a single DataFrame.

    Parameters
    ----------
    df_res : pandas.DataFrame
        A DataFrame containing columns "track_id" and any other columns with data on the image crops.
    min_samples : int, optional
        The minimum number of samples required for a track to be included in the filtered DataFrame. Defaults to 3.

    Returns
    -------
    filtered_df : pandas.DataFrame
        A DataFrame containing the same columns as the input DataFrame, but with only the observations that are part
        of tracks with a number of samples greater than or equal to the specified "min_samples" value.
    """
    dfs_list = []
    for ind, df_track in df_res.groupby("track_id"):
        if len(df_track) > min_samples:
            dfs_list.append(df_track)
    return pd.concat(dfs_list, axis=0)


def filter_trackers(df_res, apply_filter_by_color=True, apply_filter_by_dist=True, min_samples=3, min_x1=50,
                    dist_threshold=''):
    """
    Filter a DataFrame of image crops based on various criteria.

    The function applies up to three filters to the input DataFrame, depending on the values of the "apply_filter_by_color",
    "apply_filter_by_dist", and "min_samples" parameters. The filters are applied in the order: filter_by_color, filter_by_dist,
    filter_by_min_samples, filter_by_location. If "apply_filter_by_color" is True, the filter_by_color function is applied to the input DataFrame.
    If "apply_filter_by_dist" is True, the filter_by_dist function is applied to the input DataFrame. If "min_samples" is
    greater than 0, the filter_by_min_samp function is applied to the input DataFrame. If "min_x1" > 0,
     the input DataFrame will be cleaned from values with "x1" larger then min_x1.

    Parameters
    ----------
    df_res : pandas.DataFrame
        A DataFrame containing columns "distance", "track_id", "color", and any other columns with data on the image crops.
    apply_filter_by_color : bool, optional
        Whether or not to apply the filter_by_color function to the input DataFrame. Defaults to True.
    apply_filter_by_dist : bool, optional
        Whether or not to apply the filter_by_dist function to the input DataFrame. Defaults to True.
    min_samples : int, optional
        The minimum number of samples required for a track to be included in the filtered DataFrame. Defaults to 3.
    min_x1 : int, optional
        Every fruit with x1 < "min_x1" will be dropped
    dist_threshold : int
        filter values upper the value

    Returns
    -------
    filtered_df : pandas.DataFrame
        A DataFrame containing the same columns as the input DataFrame, but with only the observations that passed the
        specified filters.

    """
    if apply_filter_by_color:
        df_res = df_res[df_res["distance"] < dist_threshold]
    if apply_filter_by_dist:
        df_res = filter_df_by_color(df_res)
    if min_samples > 0:
        df_res = filter_df_by_min_samp(df_res, min_samples)
    if min_x1 > 0:
        df_res = df_res[df_res["x1"] > min_x1]
    return df_res


def trackers_into_values(df_res, dist_threshold, df_tree=None, df_border=None):
    """
    :param df_res: df of all detections in a file
    :param df_tree: df of relevent frame per tree and its start_x end_x , deafult is None in case that no subset of df_res is needed
    :return: counter, measures, colors_class, extract_ids
    """

    df_res = filter_trackers(df_res, apply_filter_by_color=True, apply_filter_by_dist=True, min_samples=3, min_x1=50,
                             dist_threshold=dist_threshold)

    def extract_tree_det():
        margin = 0
        for frame_id, df_frame in frames:
            # filtter out first red fruit and above
            df_frame = bound_red_fruit(df_frame)
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

    plot_det = []
    if df_tree is not None:
        tree_frames = df_tree['frame_id'].unique()
        frames = df_res[df_res['frame'].isin(tree_frames)].groupby('frame')
        df_tree['end'].replace([-1], math.inf, inplace=True)
    else:
        frames = df_res.groupby('frame')
    extract_tree_det()
    df_res = pd.concat(plot_det, axis=0)

    counter, extract_ids = get_count_value(df_res)
    measures = get_size_set(df_res)
    colors_class = get_color_set(df_res)

    return counter, measures, colors_class, extract_ids


def predict_weight_values(miu, sigma):
    # using exponential regression
    weight_miu = 6.305 * np.exp(0.045 * miu)
    weight_sigma = 6.305 * np.exp(0.045 * sigma)

    # using linear regression
    # weight_miu = 4.04 * miu - 142.06
    # weight_sigma = 4.04 * sigma - 142.06

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
