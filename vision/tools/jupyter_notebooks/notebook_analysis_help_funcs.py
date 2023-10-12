from math import radians, cos, sin, asin, sqrt
from vision.misc.help_func import go_up_n_levels
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from MHS.scoring import cross_validate_with_mean
from sklearn.linear_model import LinearRegression, PoissonRegressor

sns.set_style("whitegrid")


def distance(lat1, lat2, lon1, lon2):
    # convert from degrees to radians
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2

    c = 2 * asin(sqrt(a))

    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371

    # calculate the result
    return (c * r) * 1000  # results in M


def get_row_length(df_jz, return_time=False):
    try:
        if isinstance(df_jz, str):
            df_jz = pd.read_csv(df_jz)
        df_merged = df_jz
        df_merged = df_merged.fillna(method='bfill')

        if df_merged is None:  # None where is no overlap between GPS and JZ
            return 0
        first_index = min(df_merged.index) + 1
        last_index = max(df_merged.index)
        lat1 = float(df_merged.latitude[first_index])
        lat2 = float(df_merged.latitude[last_index])
        lon1 = float(df_merged.longitude[first_index])
        lon2 = float(df_merged.longitude[last_index])
        if return_time:
            return distance(lat1, lat2, lon1, lon2), pd.to_datetime(df_merged["JAI_timestamp"]).iloc[0].hour
        return distance(lat1, lat2, lon1, lon2)
    except:
        if return_time:
            return 0, pd.to_datetime(df_merged["JAI_timestamp"]).iloc[0].hour
        return 0


def get_valid_row_paths(master_folder):
    paths_list = []
    for root, dirs, files in os.walk(master_folder):
        if np.all([file in files for file in ["jaized_timestamps.csv"]]):
            row_scan_path = os.path.abspath(root)
            paths_list.append(os.path.join(row_scan_path, "jaized_timestamps.csv"))
    return paths_list


def get_full_name_from_path(path_to_row_jz):
    customer_name = os.path.basename(go_up_n_levels(path_to_row_jz, 5))
    block_name = os.path.basename(go_up_n_levels(path_to_row_jz, 4))
    row_name = "R" + os.path.basename(go_up_n_levels(path_to_row_jz, 2)).split("_")[-1]
    scan_name = "S" + os.path.basename(go_up_n_levels(path_to_row_jz, 1)).split("_")[
        -1] + f"({os.path.basename(go_up_n_levels(path_to_row_jz, 3))})"
    full_name = f"{customer_name}_{block_name}_{row_name}_{scan_name}"
    return full_name


def run_on_folder(master_folder, njobs=1, return_time=False):
    paths_list = get_valid_row_paths(master_folder)
    n = len(paths_list)
    if njobs > 1:
        with ProcessPoolExecutor(max_workers=njobs) as executor:
            res = list(executor.map(get_row_length, paths_list, [return_time] * len(paths_list)))
    else:
        res = list(map(get_row_length, paths_list, [return_time] * len(paths_list)))
    res_names = list(map(get_full_name_from_path, paths_list))
    return dict(zip(res_names, res))


def get_valid_row_paths_n_tracks(master_folder):
    paths_list = []
    for root, dirs, files in os.walk(master_folder):
        if np.all([file in files for file in ["tracks.csv"]]):
            row_scan_path = os.path.abspath(root)
            paths_list.append(os.path.join(row_scan_path, "tracks.csv"))
    return paths_list


def get_n_tracks(tracks_path, max_depth=5, full_cv=False):
    if full_cv:
        return get_n_tracks_full_cv(tracks_path, max_depth=max_depth)
    df_tracks = pd.read_csv(tracks_path)
    df_tracks = df_tracks[df_tracks["depth"] < max_depth].reset_index(drop=True)
    uniq, counts = np.unique(df_tracks["track_id"], return_counts=True)
    return len(uniq), len(uniq[counts > 1]), len(uniq[counts > 2])


def run_on_folder_tracks(master_folder, njobs=1, max_depth=5, full_cv=False):
    paths_list = get_valid_row_paths_n_tracks(master_folder)
    n = len(paths_list)
    if njobs > 1:
        with ProcessPoolExecutor(max_workers=njobs) as executor:
            res = list(
                executor.map(get_n_tracks, paths_list, [max_depth] * len(paths_list), [full_cv] * len(paths_list)))
    else:
        res = list(map(get_n_tracks, paths_list, [max_depth] * len(paths_list), [full_cv] * len(paths_list)))
    res_names = list(map(get_full_name_from_path, paths_list))
    return dict(zip(res_names, res))


def get_n_tracks_full_cv(tracks_path, max_depth=5):
    try:
        df_tracks = pd.read_csv(tracks_path)
        df_tracks = df_tracks[df_tracks["depth"] < max_depth].reset_index(drop=True)
        uniq, counts = np.unique(df_tracks["track_id"], return_counts=True)
        df_alignment = pd.read_csv(tracks_path.replace("tracks", "alignment"))
        n_dropped = df_alignment["frame"].max() + 1 - len(df_alignment["frame"].unique())
        n_dets = len(df_tracks["track_id"])
    #     track_id_depth = df_tracks.groupby("track_id")["depth"].median()
    #     depth_1 = track_id_depth[np.isin(track_id_depth.index, uniq)].median()
    #     depth_2 = track_id_depth[np.isin(track_id_depth.index, uniq[counts>1])].median()
    #     depth_3 = track_id_depth[np.isin(track_id_depth.index, uniq[counts>2])].median()
    # , depth_1, depth_2, depth_3
    except:
        return 0, 0, 0, 0, 0, 0, 0
    return len(uniq), len(uniq[counts > 1]), len(uniq[counts > 2]), len(uniq[counts > 3]), len(
        uniq[counts > 4]), n_dets, n_dropped


def plot_F_cv(df, min_samp="", hue=None, title="", col="", figsize=(10, 6), add_xy_line=True,
              y="F", order=1):
    if col == "":
        col = f"cv{min_samp}"
    max_val = np.min(np.max(df[[col, y]].values, axis=0))
    plt.figure(figsize=figsize)
    ax = sns.lmplot(data=df, x=col, y=y, hue=hue)
    sns.regplot(data=df, x=col, y=y, scatter_kws={'s': 2}, order=order, ci=0, ax=ax.axes[0, 0],
                x_ci=0, color="black", line_kws={"ls": "--"}, scatter=False)
    if add_xy_line:
        plt.plot([0, max_val], [0, max_val], color='grey')
    plt.xlim(0, np.max(df[col] * 1.1))  # Adjust x-axis limits
    plt.ylim(0, np.max(df[y] * 1.1))  # Adjust y-axis limits
    plt.title(title)
    plt.show()


def get_model_res(df, cv=1, include_fruits=True, include_interaction=True, group_col="block_name",
                 fit_intercept=False):
    groups = df[group_col]
    if include_fruits:
        X = df[[f"cv{cv}", "lemon", "mandarin"]]
        if include_interaction:
            X["cv_lemon"] = X["lemon"] * X[f"cv{cv}"]
            X["cv_mandarin"] = X["mandarin"] * X[f"cv{cv}"]
    else:
        X = df[[f"cv{cv}"]]
    y = df["F"]
    model = LinearRegression(fit_intercept=fit_intercept)
    gr_res, ge_std, tree_res, tree_std, preds = cross_validate_with_mean(model, X, y, groups=groups, ret_all_res=True)
    return gr_res, ge_std, tree_res, tree_std, preds


class MaxLinearRegressor(LinearRegression):
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None):
        super().__init__(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs)

    def predict(self, X):
        predicted_values = super().predict(X)
        return np.array([max(p, 0) for p in predicted_values])


def run_LROCV_by_block(df_f, cv_col = "cv1", fit_intercept=False):
    df = df_f.reset_index(drop=True).copy()
    for block in df["block_name"].unique():
        logic_vec = df["block_name"] == block
        if not isinstance(cv_col, list):
            X = df[logic_vec][[cv_col]].reset_index(drop=True)
        else:
            X = df[logic_vec][cv_col].reset_index(drop=True)
        y = df[logic_vec]["F"].reset_index(drop=True)
        model = LinearRegression(fit_intercept=fit_intercept)
        print(cross_validate_with_mean(model, X, y, groups=df[logic_vec]["row"].reset_index(drop=True)))
        model.fit(X, y)
        print(model.coef_)


def run_LBOCV_by_block(df_f, cv_col="cv1", fit_intercept=False):
    df = df_f.reset_index(drop=True).copy()
    if not isinstance(cv_col, list):
        X = df[[cv_col]].reset_index(drop=True)
    else:
        X = df[cv_col].reset_index(drop=True)
    y = df["F"].reset_index(drop=True)
    model = LinearRegression(fit_intercept=fit_intercept)
    print(cross_validate_with_mean(model, X, y, groups=df["block_name"].reset_index(drop=True)))
    model.fit(X, y)
    print(model.coef_)