from math import radians, cos, sin, asin, sqrt
#from vision.misc.help_func import go_up_n_levels
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, PoissonRegressor
from vision.misc.help_func import  safe_read_csv, post_process_slice_df
from vision.tools.manual_slicer import slice_to_trees_df

sns.set_style("whitegrid")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.linear_model import LinearRegression, LassoCV


def mape_scoring(y_true, y_pred):
    """
    mean average precentege error scoring function
    :param y_true: the true values of y
    :param y_pred: the predicted values of y
    :return: mape score
    """
    return np.mean(np.abs(y_true - y_pred)/y_true)


def naive_prediction(data, frames=False):
    """
    predicing for each tree the mean F of all trees.
    :param data: training dataframe
    :param frames: if using frame fitting mmethod then aggregate the data
    :return: prediction for each data point
    """
    df = data.copy(True)
    if frames:
        df = df[df["frame"] == 1]
    fruit_avg = {col: df["F"][df[col] == 1].mean()
                 for col in ["orange","lemon","mandarin","apple"] if col in df.columns}
    print(fruit_avg)
    predictions = data[fruit_avg.keys()] * fruit_avg
    return predictions.max(axis=1)


def naive_score(data, frames=False):
    """
    calculates mape scoring with naive predictions
    :param data: training dataframe
    :param frames: if using frame fitting mmethod then aggregate the data
    :return: mape scoring
    """
    predictions = naive_prediction(data, frames)
    return mape_scoring(data["F"], predictions)


def cross_validate_with_mean(model=None, X=None, y=None, cv=10, groups=None, ret_preds=False, use_log1p=False,
                             random_state=43, use_pandas=False, ret_all_res=False, sum_as_mean=False):
    """
    cross validate the model with mean value per row (oe fake row, meaning a random group of instances)
    :param model: model to predict with
    :param X: features matrix
    :param y: target
    :param cv: number of cross validation to run, works only if groups = None
    :param groups: groups for KFold group CV
    :param ret_preds: flag for returning the predictions
    :param use_log1p: flag for appling log to the target
    :return: mean score, std of scores
    """
    results = []
    tree_res = []
    if (not isinstance(X, type(np.array([])))) and (not use_pandas):
        X = X.to_numpy()
    all_preds = np.zeros(len(y))
    if (not isinstance(y, type(np.array([])))) and (not use_pandas):
        y = y.to_numpy()
    if not isinstance(groups, type(None)):
        cv = groups.nunique()
        kf = GroupKFold(n_splits=cv)
        iterable_kf = kf.split(X, y, groups)
    else:
        kf = KFold(n_splits=cv, random_state=random_state, shuffle=True)
        iterable_kf = kf.split(X)

    cv_counter = 1
    for train_index, test_index in iterable_kf:
        if not use_pandas:
            x_train, x_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        else:
            x_train, x_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y[train_index], y[test_index]
        if isinstance(model, type(None)):
            y_pred = np.array([np.mean(y_train)]*len(y_test))
        elif use_log1p:
            model.fit(x_train, np.log1p(y_train))
            y_pred = np.expm1(model.predict(x_test))
        else:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
        y_true_sum = y_test[~np.isnan(y_pred)].sum() if not sum_as_mean else y_test[~np.isnan(y_pred)].mean()
        if not sum_as_mean:
            results.append(abs(np.nansum(y_pred) - y_true_sum)/(y_true_sum))
        else:
            results.append(abs(np.nanmean(y_pred) - y_true_sum) / (y_true_sum))
        tree_res.append(np.nanmean(abs(y_pred-y_test)/(y_test)))
        test_group = ""
        if not isinstance(groups,type(None)):
            test_group = f"({groups[test_index[0]]})"
        y_pred_sum = np.nansum(y_pred) if not sum_as_mean else np.nanmean(y_pred)
        acc = np.abs(y_true_sum-y_pred_sum)/y_true_sum
        all_preds[test_index] = y_pred
        print(F"CV {cv_counter},  true: {y_true_sum},  pred: {y_pred_sum},  % error({acc*100 :.2f} %)  val group: {test_group}" )
        cv_counter +=1
    print(f'Mean {np.mean(tree_res)}, STD {np.std(tree_res)}')
    if ret_all_res:
        return np.mean(results), np.std(results), np.mean(tree_res), np.std(tree_res), all_preds
    if ret_preds:
        return np.mean(results), np.std(results), all_preds
    return np.mean(results), np.std(results)



def print_navie_scores(df, X_tr_lr, y, rows, ret_res = False):
    print(naive_score(df, frames=False))
    cols = ["cv"] + [col for col in ["orange", "lemon", "mandarin", "apple"] if col in X_tr_lr.columns]
    print(cross_validate_with_mean(LinearRegression(), X_tr_lr[cols], y))
    res_mean, res_std, preds = cross_validate_with_mean(LinearRegression(), X_tr_lr[cols], y, groups=rows,
                                                        ret_preds=True)
    print(res_mean, res_std)
    if ret_res:
        return preds, res_mean, res_std
    return preds


def print_cv_scores(model, X, y, groups):
    # print("regular cross validation score:")
    # print(cross_validate_with_mean(model, X, y))
    print("groups cross validation score:")
    print(cross_validate_with_mean(model, X, y, groups=groups))
    print("groups with log training cross validation score:")
    print(cross_validate_with_mean(model, X, y, groups=groups, use_log1p=True))



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


def plot_F_cv(df, output_dir = None, min_samp="", hue=None, title="", col="", figsize=(10, 6), add_xy_line=True,
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
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'f_to_cv1_scatter.png'))
        print(f"Saved: {os.path.join(output_dir, 'f_to_cv1_scatter.png')}")
    else:
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


def run_LROCV(df_f, cv_col="cv1", type_col="block_name", cross_val='row', fit_intercept=False, return_res=False):

    df = df_f.reset_index(drop=True).copy()
    for block in df[type_col].unique():
        print(cv_col, block)
        logic_vec = df[type_col] == block
        if not isinstance(cv_col, list):
            X = df[logic_vec][[cv_col]].reset_index(drop=True)
        else:
            X = df[logic_vec][cv_col].reset_index(drop=True)
        y = df[logic_vec]["F"].reset_index(drop=True)
        model = LinearRegression(fit_intercept=fit_intercept)

        cross_validate_with_mean(model, X, y, groups=df[logic_vec][cross_val].reset_index(drop=True))
        #print(f'Cross Validation with mean {cross_validate_with_mean(model, X, y, groups=df[logic_vec][cross_val].reset_index(drop=True))}')
        model.fit(X, y)
        print(f'Coefficient: {model.coef_}')
        if return_res:
            res_mean, res_std, tree_mean, tree_std, all_preds = cross_validate_with_mean(model, X, y,
                                                                                         groups=df[logic_vec][
                                                                                             cross_val].reset_index(
                                                                                             drop=True),
                                                                                         ret_all_res=return_res)

            return model.coef_, res_mean, res_std, tree_mean, tree_std, all_preds
        else:
            return model.coef_


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


