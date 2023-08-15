import json
import os

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import pandas as pd

pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', 100)
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import (MinMaxScaler, StandardScaler, PowerTransformer,
                                   OneHotEncoder, RobustScaler, QuantileTransformer)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer, mean_squared_log_error
import time
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import optuna
from vision.feature_extractor.feature_extractor import *
from MHS.scoring import *
from MHS.model_hybrid_selector import model_hybrid_selector, LassoSelector
import pickle
from omegaconf import OmegaConf
from MHS.sqlread.MySQLRead import get_table
from MHS.F_model_models_dict import F_model_models, model_folder
from MHS.models import models
from vision.misc.help_func import validate_output_path

import warnings
warnings.filterwarnings("ignore")

global scalers
scalers = {"Standard": StandardScaler(),
           "MinMax": MinMaxScaler(),
           "Power": PowerTransformer(),
           "Quantile": QuantileTransformer(),
           "Robust": RobustScaler()}

def read_f_df(cfg):
    """
    Reads and concatenates all dataframes from the specified folder.

    Args:
        cfg (dict): Dictionary containing configuration parameter f_fdfs_path.

    Returns:
        pd.DataFrame: Concatenation of all dataframes.
    """
    f_df_clean = pd.DataFrame({})
    for df_path in os.listdir(cfg.f_fdfs_path):
        f_df_clean = pd.concat([f_df_clean, pd.read_csv(os.path.join(cfg.f_fdfs_path, df_path))])
    return f_df_clean


def get_rel_cols(cfg):
    """
    Returns the relevant columns to use for training, the sides to drop, the trees to drop,
    and the columns to drop from training.

    Args:
        cfg (dict): A dictionary containing configuration parameters.

    Returns:
        tuple: A tuple containing four lists: final_cols (list of str), drop_sides (list of str),
               drop_trees (list of str), and drop_final (list of str).
    """
    phisical_features_names = init_physical_parmas([]).keys()
    location_features_names = cfg.location_features_names
    tree_features_names = init_fruit_params([]).keys()
    v_i_features = transform_to_vi_features({**get_additional_vegetation_indexes(0, 0, 0, fill=[0])}).keys()
    all_cols = list(phisical_features_names) + list(location_features_names) + list(tree_features_names) + list(
        v_i_features)
    drop_cols_vi = list(cfg.drop_cols_vi)
    drop_cols = list(cfg.drop_cols)
    final_cols = list(set(all_cols) - set(drop_cols) - set(drop_cols_vi) - {f"{col}_skew" for col in drop_cols_vi})
    drop_final = list(set(drop_cols_vi + [f"{col}_skew" for col in drop_cols_vi] + drop_cols))
    return final_cols, drop_final


def read_data(cfg):
    """
    Read and concatenate csv file(s) from specified path(s) and return a cleaned DataFrame.

    Args:
        cfg (dict): Dictionary containing configuration parameters.
            csv_paths (list): List of file path(s) of csv file(s) to be read.

    Returns:
        clean_set (pd.DataFrame): DataFrame containing the data from the csv file(s).
    """
    df = pd.DataFrame()
    for row_csv_path in cfg.csv_paths:
        df = pd.concat([df, pd.read_csv(row_csv_path)], ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    return df.drop_duplicates()


def get_ranges(cfg):
    """
    gets range for each features from config, it will be used to drop whatever values that are out of range

    Parameters:
        cfg (Namespace): Configuration file containing cleaning settings

    Returns:
        hidden_range (np.array): A numpy array for filtering % hidden
        cv_range (np.array): A numpy array for filtering cv values
        F_range (np.array): A numpy array for filtering F values
        fruits_exclude (List): A list for dropping fruit types
    """
    hidden_range = np.array(eval(cfg.cleaning.hidden_range))
    cv_range = np.array(eval(cfg.cleaning.cv_range))
    F_range = np.array(eval(cfg.cleaning.F_range))
    fruits_exclude = list(cfg.cleaning.fruits_exclude)
    return hidden_range, cv_range, F_range, fruits_exclude


def process_fruit_type(df, fruits_exclude, cfg):
    """
    One hot encodes the fruit types and drops fruit types that are in 'fruits_exclude'

    Parameters:
        df (pd.DataFrame): Features DataFrame
        fruits_exclude (list): what fruit names to drop from the dataframe
        cfg (Namespace): Configuration file containing output_folder
    Returns:
        clean_df (pd.DataFrame): A DataFrame with one hot encoded fruits
    """
    df.loc[:, "fruit_type"] = df["fruit_type"].apply(lambda x: x.lower())
    df = df[df["fruit_type"].map(lambda x: x not in fruits_exclude)]
    for fruit in cfg.supported_fruits:
        df[fruit] = (df["fruit_type"] == fruit)*1
    return df


def get_full_name(df):
    df["block_name"] = df["block_name"].replace({"2020injx": "2020injX", "injertos": "Injertos"})
    return df["customer"].str.cat(df["block_name"].str.cat(df["name"], sep="_"), sep="_")


def train_clean(df, df_cols, hidden_range, F_range, cfg):
    if "% hidden" in df_cols:
        df = df[(df["% hidden"] > hidden_range[0]) & (df["% hidden"] < hidden_range[1])]
    if "F" in df_cols:
        df = df[(df["F"] > F_range[0]) & (df["F"] < F_range[1])]
    if "customer" in df_cols:
        df = df[df["customer"].apply(lambda x: x not in list(cfg.drop_customer))]
    return df

def clean_the_df(df, drop_final, cfg, mode="train", drop_nas=True):
    """
    Cleans the features DataFrame for training.

    Parameters:
        df (pd.DataFrame): Features DataFrame
        drop_final (list): Columns to drop
        cfg (Namespace): Configuration file containing cleaning settings, drop_sides, drop_trees
        add_variety (Boolen): flag for adding a column for variety

    Returns:
        clean_df (pd.DataFrame): A cleaned features DataFrame
    """
    hidden_range, cv_range, F_range, fruits_exclude = get_ranges(cfg)
    df["block_name"] = df["block_name"].replace({"2020injx": "2020injX", "injertos": "Injertos"})

    df_cols = df.columns
    df["cv"] = df[f"cv{cfg.cv_vol}"]

    if mode == "train":
        df = train_clean(df, df_cols, hidden_range, F_range, cfg)

    df = df[(df["cv"] > cv_range[0]) & (df["cv"] < cv_range[1])]

    if "fruit_type" not in df_cols:
        plot_fruit_var = get_plot_fruit_variety_df(cfg)
        df = pd.merge(df, plot_fruit_var.drop_duplicates(), how="left", on="block_name")
    df = process_fruit_type(df, fruits_exclude, cfg)

    if "side" in df_cols:
        side_full_name = df["customer"].str.cat(df["block_name"].str.cat(df["side"], sep="_"), sep="_")
        df = df[side_full_name.apply(lambda x: x not in list(cfg.drop_sides))]

    full_name = get_full_name(df)
    df = df[full_name.apply(lambda x: x not in list(cfg.drop_trees))]
    drop_final = [col for col in drop_final if col in df.columns]
    df.drop(["fruit_type"]+[col for col in df.columns if col.startswith("Unnamed")] + drop_final, axis=1, inplace=True)
    quantile_cols = [f"q_{i}_precent_fruits" for i in range(1, 4)]
    if quantile_cols[0] in df.columns:
        df[quantile_cols] = df[quantile_cols].fillna(1 / 3)
    if drop_nas:
        return df[np.all(np.isfinite(df.select_dtypes(exclude=object)), axis=1)] #remove observations with na/inf
    return df


def add_fs(features_df, f_df):
    """
    Adds F values to the features DataFrame.

    Parameters:
        features_df (pd.DataFrame): DataFrame of features
        f_df (pd.DataFrame): DataFrame of F values

    Returns:
        features_df (pd.DataFrame): Features DataFrame with F values merged by {block_name}_{tree_name}
    """
    f_df["fullname"] = f_df["block_name"].str.cat(f_df["new_tree_name"], sep="_")
    features_df["fullname"] = features_df["block_name"].str.cat(features_df["name"], sep="_")
    features_df["F"] = features_df["fullname"].map(dict(zip(f_df["fullname"], f_df["F"])))
    features_df = features_df[np.isfinite(features_df["F"])]
    f_df.drop("fullname", axis=1, inplace=True)
    features_df.drop("fullname", axis=1, inplace=True)
    if "Unnamed: 0" in features_df.columns:
        features_df.drop("Unnamed: 0", axis=1, inplace=True)
    return features_df


def get_groups(df, cfg):
    """
    Get groups for cross validation

    Args:
    - df: (pd.DataFrame) Features dataframe with F values.
    - cfg: Configuration object with preprocessing parameters.

    Returns:
    - groups: (pd.Series) groups for KFold grouping.
    """
    if cfg.group_method == "rows":
        rows = df["name"].apply(lambda x: x.split("_")[0])
        groups = df["block_name"].str.cat(rows, sep="_")  # new
    elif cfg.group_method == "blocks":
        groups = df["block_name"]
    elif cfg.group_method == "customers":
        groups = df["customer"]
    elif cfg.group_method == "summergold":
        fruits = ['lemon', 'mandarin', 'orange']
        groups = df[["customer"] + fruits].apply(lambda x:
                                f"{x['customer']}_{fruits[np.argmax([x.lemon, x.mandarin, x.orange])]}", axis=1)
    else:
        groups = None
    return groups


def get_X_y(df, cfg, apply_preprocess=True):
    """
    Get the data for training.

    Args:
    - df: (pd.DataFrame) Features dataframe with F values.
    - cfg: Configuration object with preprocessing parameters.

    Returns:
    - X: (pd.DataFrame) Train matrix.
    - y: (pd.Series) Target variable.
    - rows: (pd.Series) Rows for KFold grouping.
    """
    groups = get_groups(df, cfg)
    X_tr_lr = df.copy()
    y = X_tr_lr['F']
    if apply_preprocess:
        X_tr_lr = preprocess(X_tr_lr, cfg)
        X_tr_lr.fillna(0, inplace=True)
    if not isinstance(groups, type(None)):
        groups.reset_index(inplace=True, drop=True)
    X_tr_lr.reset_index(inplace=True, drop=True)
    y.reset_index(inplace=True, drop=True)
    return X_tr_lr, y, groups


def add_features(df):
    """
    Adds new features to the data frame

    Args:
    - df: (pd.DataFrame) Dataframe to preprocess.

    Returns:
    - df: (pd.DataFrame) Processed dataframe with the new features
    """
    df_cols = df.columns
    df["fruit_foliage_ratio"] = df["fruit_foliage_ratio"].replace(np.inf, 100)
    df["clusters_area_mean_arr"] = df["clusters_area_mean_arr"].clip(0, 700)
    df["clusters_area_med_arr"] = df["clusters_area_med_arr"].clip(0, 600)
    df['cv^2'] = df["cv"] ** 2
    df['cv/center_height'] = df["cv"] / df["center_height"]
    if "center_width" in df_cols:
        df['cv/center_width'] = df["cv"] / df["center_width"]
    df['cv/surface_area'] = df["cv"] / df["surface_area"]
    df['cv/total_foliage'] = df["cv"] / df["total_foliage"]
    df['cv/fruit_foliage_ratio'] = df["cv"] / df["fruit_foliage_ratio"]
    # df['cv/foliage_fullness'] = df["cv"] / df["foliage_fullness"]
    # df['cv_fruit_dist_center'] = df["cv"] * df["fruit_dist_center"] # new 09.3
    # df['cv_w_h_ratio'] = df["cv"] * df["w_h_ratio"] # new 09.3
    # df['cv_q_1_precent_fruits'] = df["cv"] * df["q_1_precent_fruits"] # new 09.3
    # df['cv_q_2_precent_fruits'] = df["cv"] * df["q_2_precent_fruits"] # new 09.3
    # df['cv_q_3_precent_fruits'] = df["cv"] * df["q_3_precent_fruits"] # new 09.3
    # df['mst_sums_arr^2'] = df["mst_sums_arr"] ** 2
    # df['1/mst_sums_arr^2'] = 1/df["mst_sums_arr"] ** 2 # new 09.3
    # df['1/mst_sums_arr'] = 1 / df["mst_sums_arr"] # new 09.3
    for col in ["orange", "lemon", "mandarin", "apple"]:
        if col in df_cols:
            df[f"{col}_cv"] =df[col]*df["cv"]
            df[f"{col}_surface_area"] = df[col] * df["surface_area"] # new 09.3
            if "center_width" in df_cols:
                df[f"{col}_center_width"] = df[col] * df["center_width"] # new 09.3
    return df


def preprocess(df, cfg):
    """
    Preprocess the input dataframe for training and inference.

    Args:
    - df: (pd.DataFrame) Dataframe to preprocess.
    - cfg: (Config) Configuration object with preprocessing parameters.

    Returns:
    - df: (pd.DataFrame) Processed dataframe.
    """
    df.replace("[0]", 0, inplace=True)
    # drop nontrain cols and clean cols
    df_cols = df.columns
    for col in cfg.drop_preprocess:
        if col in df_cols:
            df.drop(col, axis=1, inplace=True)
    if "fruit_type" in df_cols:
        df = process_fruit_type(df, cfg.cleaning.fruits_exclude, cfg)
        df.drop("fruit_type", axis=1, inplace=True)
    for col in cfg.change_cols:
        if col in df_cols:
            df[col] = df[col].astype(float)
    df = add_features(df)
    return df


def mhs_fit_save(X_tr_lr, y, fitting_method="iterative_fitting", num_stuck=3, cols=[], use_alwyas=["cv"], n_iter=15,
                 statisticly=False, groups=None):
    """
    Fits a hybrid model selector to the input data and saves the model.

    Parameters:
    -----------
    X_tr_lr : pandas.DataFrame
        Input features of training set.

    y : pandas.Series
        Target variable of training set.

    fitting_method : str, optional
        Fitting method used by model_hybrid_selector.
        Options: "iterative_fitting", "fitplusplus". Default is "iterative_fitting".

    num_stuck : int, optional
        Maximum number of times to allow hybrid model selector to not add any models in a row during fitting. Default is 3.

    cols : list, optional
        List of column names to use in fitting. Default is empty list.

    use_alwyas : list, optional
        List of strings to add to always use list. Default is ["cv"].

    n_iter : int, optional
        Number of iterations to run in fitplusplus method. Default is 15.

    statisticly : bool, optional
        Whether to use statistical selection in hybrid model selector. Default is False.

    Returns:
    --------
    tuple
        Mean and standard deviation of negative mean absolute percentage error (MAPE) from cross-validation.
    """
    mhs = model_hybrid_selector(statisticly=statisticly)
    if fitting_method == "iterative_fitting":
        mhs.iterative_fitting(X_tr_lr, y, num_stuck=num_stuck, cols=cols, use_alwyas=use_alwyas)
    if fitting_method == "fitplusplus":
        mhs.fitplusplus(X_tr_lr, y, num_stuck=num_stuck, n_iter=n_iter, use_alwyas=use_alwyas)
    fitting_method = f"{fitting_method}_statisticly" if statisticly else fitting_method
    score = cross_validate(LinearRegression(), X_tr_lr.loc[:, mhs.cols], y=y, cv=6, n_jobs=6,
                           scoring="neg_mean_absolute_percentage_error")
    print(f"{fitting_method} results:")
    print(np.mean(score['test_score']), np.std(score['test_score']))
    print("chosen cols")
    print(mhs.cols)
    print_cv_scores(LinearRegression(), X_tr_lr[mhs.cols], y, groups=groups)
    mhs.fit(X_tr_lr, y, prod=True)
    with open(os.path.join(cfg.output_folder, f'model_mhs_{fitting_method}.pkl'), 'wb') as f:
        pickle.dump(mhs, f)
    return np.mean(score['test_score']), np.std(score['test_score'])


def xgb_fit_save(X_tr_lr, y, cv=6, n_jobs=6, use_best_study=False, groups=None):
    """
    This function fits an XGBoost Regressor on the input data, performs cross-validation and saves the trained model to disk. It also prints out the mean and standard deviation of the cross-validation score, as well as the cross-validation score for different subsets of the data.

    Args:
    X_tr_lr (array-like): Training data, with features and samples.
    y (array-like): Training target values.
    cv (int, optional): Number of cross-validation folds. Default is 8.
    n_jobs (int, optional): Number of CPU cores to use during cross-validation. Default is -1 (use all available cores).
    use_best_study (bool, optional): Whether to use the best hyperparameters found in a previous study. Default is False.

    Returns:
    Tuple of the mean and standard deviation of the cross-validation score.
    """
    if not use_best_study or not os.path.exists(os.path.join(cfg.output_folder, "study_xgb.pkl")):
        xgb_params = {'eta': 0.08045293046930338,
                      'min_child_weight': 1.2169041832751055,
                      'max_depth': 9,
                      'gamma': 0.8726360469177697,
                      'colsample_bytree': 1.0,
                      'colsample_bylevel': 0.9000000000000001,
                      'reg_alpha': 2.9518939876884756,
                      'reg_lambda': 4.759526161107639,
                      'subsample': 0.4,
                      'n_estimators': 40}
    else:
        study = joblib.load(os.path.join(cfg.output_folder, "study_xgb.pkl"))
        xgb_params = study.best_params
    model_xgb = XGBRegressor(**xgb_params)
    score = cross_validate(model_xgb, X_tr_lr, y=y, cv=cv, n_jobs=n_jobs, scoring="neg_mean_absolute_percentage_error")
    print("XGB results:")
    print((np.mean(score['test_score']), np.std(score['test_score'])))
    print_cv_scores(model_xgb, X_tr_lr, y, groups=groups)
    model_xgb.fit(X_tr_lr, y)
    with open(os.path.join(cfg.output_folder, f'model_xgb.pkl'), 'wb') as f:
        pickle.dump(model_xgb, f)
    return np.mean(score['test_score']), np.std(score['test_score'])


def rf_fit_save(X_tr_lr, y, cv=6, n_jobs=6, use_best_study=False, groups=None):
    """
    Fits and saves a random forest model using the given input data, and returns the mean and standard deviation of the
     negative mean absolute percentage error obtained through cross-validation.

    Args:
    X_tr_lr (pandas.DataFrame): Input feature matrix for training the model.
    y (pandas.Series): Target variable vector for training the model.
    cv (int, optional): Number of folds to use for cross-validation. Defaults to 8.
    n_jobs (int, optional): Number of CPU cores to use for parallel computation during cross-validation. Defaults to -1.
    use_best_study (bool, optional): Whether to use the best hyperparameters found through a previous hyperparameter
     optimization study or use the default hyperparameters. Defaults to False.

    Returns:
    tuple: A tuple containing the mean and standard deviation of the negative mean absolute percentage error obtained
     through cross-validation.
    """
    if not use_best_study or not os.path.exists(os.path.join(cfg.output_folder, "study_rf.pkl")):
        rf_params = {'n_estimators': 10,
                     'min_samples_split': 3,
                     'min_samples_leaf': 2,
                     'max_features': 0.6000000000000001,
                     'ccp_alpha': 0.01712760331733212,
                     'max_samples': 0.9}
    else:
        study = joblib.load(os.path.join(cfg.output_folder, "study_rf.pkl"))
        rf_params = study.best_params
    model_rf = RandomForestRegressor(**rf_params)
    score = cross_validate(model_rf, X_tr_lr, y=y, cv=cv, n_jobs=n_jobs, scoring="neg_mean_absolute_percentage_error")
    print("random forest results:")
    print((np.mean(score['test_score']), np.std(score['test_score'])))
    print_cv_scores(model_rf, X_tr_lr, y, groups=groups)
    model_rf.fit(X_tr_lr, y)
    with open(os.path.join(cfg.output_folder, f'model_rf.pkl'), 'wb') as f:
        pickle.dump(model_rf, f)
    return (np.mean(score['test_score']), np.std(score['test_score']))


def get_models_results(X_tr_lr, y, use_best_study=False, drop_from_trees=[], groups=None):
    """
    Fits and saves the results of several models on the given data.

    Args:
        X_tr_lr (pandas.DataFrame): The training data.
        y (pandas.Series): The target variable.
        use_best_study (bool): Whether to use the best hyperparameters found in previous studies.
        drop_from_trees (list): List of columns to be dropped when using tree-based models.

    Returns:
        pandas.DataFrame: A summary of the results of each model, including the model name, mean score, and standard deviation.
    """
    X_tr_trees = X_tr_lr.copy()
    for col in drop_from_trees:
        if col in X_tr_trees.columns:
            X_tr_trees.drop(col, axis=1, inplace=True)
    results_summary = []
    score_mean, score_std = xgb_fit_save(X_tr_trees, y, cv=6, n_jobs=6, use_best_study=use_best_study, groups=groups)
    results_summary.append({"model": f"XGB",
                            "score": score_mean, "std": score_std})
    score_mean, score_std = rf_fit_save(X_tr_trees, y, cv=6, n_jobs=6, use_best_study=use_best_study, groups=groups)
    results_summary.append({"model": f"RF",
                            "score": score_mean, "std": score_std})
    # for statisticly in [True, False]:
    #     for fitting_method in ["iterative_fitting", "fitplusplus"]:
    #         score_mean, score_std = mhs_fit_save(X_tr_lr, y, fitting_method=fitting_method, num_stuck=3, cols=[],
    #                                              use_alwyas=["cv"], n_iter=15, statisticly=statisticly)
    #         results_summary.append({"model": f"{fitting_method}{'_statisticly' if statisticly else ''}",
    #                                 "score": score_mean, "std": score_std})
    score_mean, score_std = lasso_fit_save(X_tr_lr, y, cv=6, n_jobs=6)
    results_summary.append({"model": f"lasso",
                            "score": score_mean, "std": score_std})
    res_df = pd.DataFrame.from_records(results_summary)
    res_df.to_csv("results.csv")
    return res_df


def lasso_fit_save(X_tr_lr, y, cv=6, n_jobs=6, scaler="Robust", groups=None):
    """
    Fits a LassoCV model to the input data and saves the fitted pipeline to a pickle file.

    Parameters:
    -----------
    X_tr_lr : pandas.DataFrame
        The training dataset with features.
    y : pandas.Series
        The target variable for regression.
    cv : int, optional (default=8)
        The number of cross-validation folds to use.
    n_jobs : int, optional (default=-1)
        The number of CPU cores to use for cross-validation.
    scaler : str, optional (default="Robust")
        The name of the scaler to use in the pipeline. One of "MinMax", "Power", "Quantile", "Standard", or "Robust".

    Returns:
    --------
    tuple of floats
        The mean and standard deviation of the negative mean absolute percentage error obtained via cross-validation.
    """
    lasso = LassoCV(n_alphas=2500, cv=cv, n_jobs=n_jobs, max_iter=10000)
    pipe = Pipeline([("scaler", scalers[scaler]),
                     ("lasso_selector", LassoSelector(lasso)),
                     ("final_estimator", LinearRegression())])
    las_pipe_score = cross_validate(pipe, X_tr_lr,
                                    y=y, cv=cv, n_jobs=n_jobs, scoring="neg_mean_absolute_percentage_error")
    print("lasso results")
    print((np.mean(las_pipe_score['test_score']), np.std(las_pipe_score['test_score'])))
    print("regular cross validation score:")
    print(cross_validate_with_mean(pipe, X_tr_lr, y))
    print("groups cross validation score:")
    print(cross_validate_with_mean(pipe, X_tr_lr, y, groups=groups))
    print("groups with log training cross validation score:")
    print(cross_validate_with_mean(pipe, X_tr_lr, y, groups=groups, use_log1p=True))
    pipe.fit(X_tr_lr, y)
    with open(os.path.join(cfg.output_folder, f'model_lassopipe.pkl'), 'wb') as f:
        pickle.dump(pipe, f)
    return np.mean(las_pipe_score['test_score']), np.std(las_pipe_score['test_score'])


def get_all_features(customer_path, suffix="features.csv", save_path=""):
    """
    Concatenates all CSV files in the given directory (and its subdirectories) that end with the given suffix,
    drops columns starting with "Unnamed", and adds a "customer_name" column with the name of the top-level directory if
    such a column does not exsist

    Args:
        customer_path (str): The path to the top-level directory containing the CSV files to be concatenated.
        suffix (str): The file name suffix of the CSV files to be concatenated (default: "features.csv").
        save_path (str): The path to save the concatenated DataFrame as a CSV file (default: "").

    Returns:
        pd.DataFrame: The concatenated DataFrame.
    """
    dfs_list = []
    customer_name = os.path.basename(customer_path)
    for root, dirs, files in os.walk(customer_path):
        for file in files:
            if file.endswith(suffix):
                dfs_list.append(pd.read_csv(os.path.join(root, file)))
    out_df = pd.concat(dfs_list)
    out_df.drop([col for col in out_df.columns if col.startswith("Unnamed")], axis=1, inplace=True)
    if "customer" not in out_df.columns:
        out_df["customer"] = customer_name
    if save_path != "":
        out_df.to_csv(save_path)
    return out_df


def get_plot_fruit_variety_df(cfg):
    """
    Retrieves the plot code, fruit type, and variety ID for each plot from the portal database and returns
     them as a DataFrame.

    Args:
        cfg (Config): A Config object containing the necessary database connection information and table names.

    Returns:
        pd.DataFrame: A DataFrame with columns 'plot_code', 'fruit_type', and 'variety_id'.
    """
    table_cfg = cfg.table_config
    plots = get_table("SELECT plot_code,plot_fruit_type,plot_fruit_variety FROM plot", table_cfg)
    fruit_types = get_table("SELECT * FROM fruit_type", table_cfg).convert_dtypes()
    fruit_var = get_table("SELECT fruit_id,variety_id,variety_name FROM fruit_variety", table_cfg).convert_dtypes()
    fruit_types.rename({"value": "fruit_type"}, axis=1, inplace=True)
    plots.rename({"plot_fruit_type": "fruit_id", "plot_fruit_variety": "variety_id"}, axis=1, inplace=True)
    plots = pd.merge(plots, fruit_types, on="fruit_id", how="left")
    plots = pd.merge(plots, fruit_var, on="variety_id", how="left")
    plot_fruit_var = plots[['plot_code', 'fruit_type', 'variety_name']]
    plot_fruit_var = plot_fruit_var.append({'plot_code': 'SUMGLD', 'fruit_type': 'Orange',
                                            'variety_name': 'Summer gold Navel'}, ignore_index=True)
    plot_fruit_var = plot_fruit_var.append({'plot_code': 'FREDIANI175', 'fruit_type': 'Mandarin',
                                            'variety_name': 'Clem'}, ignore_index=True)
    plot_fruit_var = plot_fruit_var.rename({"plot_code": "block_name"}, axis=1)
    if not cfg.add_variety:
        plot_fruit_var.drop("variety_name", axis=1, inplace=True)
    return plot_fruit_var


def load_data_from_model_args(model_args):
    """
    load X and y as pandas dataframe from model_args
    Args:
        model_args: a dictionary contatning X_data, y_data
    Returns:
        X_train (pandas.DataFrame): X data
        y_train (pandas.DataFrame): y data
    """
    if isinstance(model_args["X_data"], str):
        X_train = pd.read_csv(model_args["X_data"])
    else:
        X_train = model_args["X_data"]
    if isinstance(model_args["y_data"], str):
        y_train = pd.read_csv(model_args["y_data"])
    else:
        y_train = model_args["y_data"]
    return X_train, y_train


def model_fit_save(model, X, y, save_name, cv=6, n_jobs=6, scoring="neg_mean_absolute_percentage_error", groups=None,
                   perform_reg_cross_val=False, perform_log=False, use_pandas=False):
    """
    Fits a model to the input data and saves the fitted model to a pickle file.

    Parameters:
    -----------
    model:
        a model to fit and crossvalidate (an object with fit and predict)
    X : pandas.DataFrame
        The training dataset with features.
    y : pandas.Series
        The target variable for regression.
    save_name: str
        where to save the model after fitting on the full data
    cv : int, optional (default=8)
        The number of cross-validation folds to use.
    n_jobs : int, optional (default=-1)
        The number of CPU cores to use for cross-validation.
    scoring: str/function
        A scoring for the cross validation, should be compatiable with sklearn

    Returns:
    --------
    tuple of floats
        The mean and standard deviation of the scoring error obtained via cross-validation.
        The mean and standard deviation of the scoring error obtained via group cross-validation.
        The mean and standard deviation of the scoring error obtained via group cross-validation and log sacle for y.
    """
    if perform_reg_cross_val:
        las_pipe_score = cross_validate(model, X, y=y, cv=cv, n_jobs=n_jobs, scoring=scoring)
        print("regular cross validation score:")
        print((np.mean(las_pipe_score['test_score']), np.std(las_pipe_score['test_score'])))
    else:
        las_pipe_score = {'test_score': 100}
    print("groups cross validation score:")
    groups_mean_score, groups_std_score, preds = cross_validate_with_mean(model, X, y, groups=groups, ret_preds=True,
                                                                          use_pandas=use_pandas)
    print(groups_mean_score, groups_std_score)
    if perform_log:
        print("groups with log training cross validation score:")
        groups_mean_score_log, groups_std_score_log = cross_validate_with_mean(model, X, y, groups=groups,
                                                                               use_log1p=True, use_pandas=use_pandas)
        print(groups_mean_score_log, groups_std_score_log)
    else:
        groups_mean_score_log, groups_std_score_log = 100, 0
    model.fit(X, y)
    with open(save_name, 'wb') as f:
        pickle.dump(model, f)
    return (np.mean(las_pipe_score['test_score']), np.std(las_pipe_score['test_score']), groups_mean_score,
        groups_std_score, groups_mean_score_log, groups_std_score_log, preds)


def load_model_cols(model_args):
    model_cols = model_args["columns"]
    if isinstance(model_cols, str):
        model_cols = json.load(model_cols)["best_features"]
    return model_cols


def get_dict_models_results(X, y, groups=None, naive_preds=None, results_summary=[]):
    """
    Fits and saves the results of several models.
    F_model_models format should be: 'model_name': {'model_params'(None,dict,path to optuna study),
                                          'X_data'(DataFrame or path), 'y_data'(DataFrame or path), 'output_path'(path)}
                                          dictioinary can also contain sclaer
    Args:

    Returns:
        pandas.DataFrame: A summary of the results of each model, including the model name, mean score, and standard deviation.
    """
    trees = pd.read_csv(os.path.join(model_folder, "trees.csv"))
    if not isinstance(naive_preds, type(None)):
        trees["cv pred"] = naive_preds
    for model_name, model_args in F_model_models.items():
        model = models[model_name.split("_")[0]]
        model_cols = load_model_cols(model_args)
        # X_train, y_train = load_data_from_model_args(model_args)
        X_train, y_train = X[model_cols], y
        model_params = model_args["model_params"]
        if isinstance(model_params, str):
            study = joblib.load(model_params)
            model_params = study.best_params
        model.set_params(**model_params)
        if "scaler" in model_args.keys():
            scaler = scalers[model_args["scaler"]]
            model = Pipeline([("scaler", scaler),
                              ("final_estimator", model)])
        save_name = model_args["output_path"]
        validate_output_path(os.path.dirname(save_name))
        print(("#"*20 + " " + model_name + " " + "#"*20)*2)
        mean_score, std_score, group_score, groups_std, groups_score_log, groups_std_log, preds = model_fit_save(
                                                                                                        model, X_train,
                                                                                    y_train.values.flatten(), save_name,
                                                                                    groups=groups)
        results_summary.append({"model": model_name, "score": np.abs(mean_score), "std": std_score,
                                "group_score": group_score, "groups_std": groups_std,
                                "group_log_score": groups_score_log, "groups_log_std": groups_std_log})
        trees[model_name] = preds
        trees.to_csv(os.path.join(model_folder, "trees.csv"))
    trees["cv"] = X_train["cv"]
    trees.to_csv(os.path.join(model_folder, "trees.csv"))
    res_df = pd.DataFrame.from_records(results_summary)
    res_df.to_csv(os.path.join(model_folder, "results_F_models.csv"))
    return res_df


def infer_on_features(features_df, cfg,
                model_pkl="/home/fruitspec-lab/FruitSpec/Code/roi/fsCounter/MHS/models_04_07_SUMGLD/xgb_notebook.pkl",
                      tree_model=True, save_path="outputs", name="pred_res.csv",
                      features = None):
    """
    Perform inference on the given features using a trained model.

    Args:
        features_df (pandas.DataFrame): The input features DataFrame.
        cfg (Config): The configuration object.
        model_pkl (str, optional): Path to the trained model pickle file. Default is "models_04_07_SUMGLD/xgb_notebook.pkl".
        tree_model (bool, optional): Whether the model is a tree-based model. Default is True.
        save_path (str, optional): Path to save the prediction results. Default is "outputs".
        name (str, optional): Name of the output file. Default is "pred_res.csv".

    Returns:
        numpy.ndarray: The predicted values.

    """
    final_cols, drop_final = get_rel_cols(cfg)
    features_df_clean = clean_the_df(features_df, drop_final, cfg, mode="infer")
    out_frame = features_df_clean[["block_name", "name", "customer"]].copy()
    X = preprocess(features_df_clean, cfg)
    if isinstance(features, type(None)):
        if tree_model:
            X = X[[col for col in cfg.tree_cols if col in X.columns]]
    else:
        X = X[features]
    model = joblib.load(model_pkl)
    out_frame["F_pred"] = model.predict(X)
    out_frame.to_csv(os.path.join(save_path, name))
    return out_frame


def get_mape(csv_path, group_col="block_name"):
    """
    Calculate the mean absolute percentage error (MAPE) for the given prediction results.

    Args:
        csv_path (str): Path to the prediction results CSV file.
        group_col (str, optional): Column name to group the data for calculating MAPE. Default is "block_name".

    Returns:
        float: The average MAPE.
        pandas.Series: The MAPE for each group.

    """
    if isinstance(csv_path, str):
        df_res = pd.read_csv(csv_path)
    else:
        df_res = csv_path
    df_res = df_res[~df_res["F_pred"].isna()]
    print(f"{sum(df_res['F_pred'].isna())} na values")
    mape = np.abs(df_res["F_pred"] - df_res["F"])/df_res["F"]
    avg_mape = np.mean(mape)
    mape_std = np.std(mape)
    group_sum = df_res.groupby(group_col)[["F", "F_pred"]].sum()
    group_mape = np.abs(group_sum["F_pred"] - group_sum["F"])/group_sum["F"]
    return {"avg_mape": avg_mape, "mape_std": mape_std, "group_mape": group_mape,
            "group_mape_avg": group_mape.mean(), "group_mape_std": group_mape.std()}


def train_model(cfg, include_only_2_sided_tree=False):
    """
    Train a model using the given configuration.

    Args:
        cfg (Config): The configuration object.

    Returns:
        pandas.DataFrame: The results of the trained models.

    """
    validate_output_path(cfg.output_folder)
    final_cols, drop_final = get_rel_cols(cfg)
    f_df = read_f_df(cfg)
    features_df = read_data(cfg)
    full_name_org = get_full_name(features_df)
    features_df_clean = clean_the_df(features_df, drop_final, cfg)
    features_df_w_f = add_fs(features_df_clean, f_df)
    if include_only_2_sided_tree:
        features_df_w_f["base_tree_name"] = features_df_w_f["block_name"] + "_" + features_df_w_f["name"].apply(
            lambda x: x.split("_")[0] + "_" + x.split("_")[-1])
        counts = features_df_w_f["base_tree_name"].value_counts() > 1
        features_df_w_f = features_df_w_f[features_df_w_f["base_tree_name"].map(dict(zip(counts.index, counts.values)))]
        features_df_w_f.drop("base_tree_name", axis=1, inplace=True)
        features_df_w_f.reset_index(inplace=True, drop=True)
    full_name_clean = get_full_name(features_df_w_f)
    dropped_obs = features_df[~full_name_org.isin(full_name_clean)]
    validate_output_path(model_folder)
    dropped_obs.to_csv(os.path.join(model_folder, "dropped_obs.csv"), index=False)
    X_tr_lr, y, groups = get_X_y(features_df_w_f, cfg)
    X_train_trees = X_tr_lr[[col for col in cfg.tree_cols if col in X_tr_lr.columns]]
    X_tr_lr.to_csv(os.path.join(model_folder, "X_train.csv"), index=False)
    y.to_csv(os.path.join(model_folder, "y_train.csv"), index=False)
    pd.concat([full_name_clean, features_df_w_f["F"]],
              axis=1).rename({"customer": "full_name"}, axis=1).to_csv(os.path.join(model_folder, "trees.csv"), index=False)
    X_train_trees.to_csv(os.path.join(model_folder, "X_train_trees.csv"), index=False)
    preds, res_mean, res_std = print_navie_scores(features_df_w_f, X_tr_lr, y, groups, ret_res=True)
    results_summary = [{"model": "cv_model", "score": 100, "std": 0,
                            "group_score": res_mean, "groups_std": res_std,
                            "group_log_score": 100, "groups_log_std": 0}]
    res_df = get_dict_models_results(X_tr_lr, y, groups, naive_preds=preds, results_summary=results_summary)
    return res_df

def folwer_analysis(features_df):
    FREDIANI175 = features_df[features_df["block_name"] == "FREDIANI"].copy()
    FREDIANI175["block_name"] = "FREDIANI175"
    FREDIANI175["cv"] = FREDIANI175["cv"] * 1.75
    FREDIANI175["cv1"] = FREDIANI175["cv1"] * 1.75
    FREDIANI175["cv2"] = FREDIANI175["cv2"] * 1.75
    FREDIANI175["cv3"] = FREDIANI175["cv3"] * 1.75
    features_df_w_feredini_175 = pd.concat([features_df, FREDIANI175]).reset_index(drop=True)

    out_frame = infer_on_features(features_df_w_feredini_175, cfg, model_pkl="models_04_07_SUMGLD/lasso_pipe.pkl", tree_model=False)
    output = out_frame
    out_frame = infer_on_features(features_df_w_feredini_175, cfg, model_pkl="models_with_Fowler/lasso_pipe.pkl", tree_model=False)
    output["Lasso_retrain_pred"] = out_frame["F_pred"]
    out_frame = infer_on_features(features_df_w_feredini_175, cfg, model_pkl="models_04_07_SUMGLD/xgb_notebook.pkl", tree_model=True)
    output["XGB_pred"] = out_frame["F_pred"]
    out_frame = infer_on_features(features_df_w_feredini_175, cfg, model_pkl="models_with_Fowler/xgb_notebook.pkl", tree_model=True)
    output["XGB_retrain_pred"] = out_frame["F_pred"]

    lens_path = "/home/fruitspec-lab/Downloads/fruits(1).csv"
    lens_df = pd.read_csv(lens_path)
    lens_df["row"] = lens_df["block_name"] + "_" + lens_df["row"].str.split("_").str[0]
    FREDIANI175_len = lens_df[lens_df["row"].str.startswith("FREDIANI")]
    FREDIANI175_len["row"] = FREDIANI175_len["row"].apply(lambda x: x.replace("FREDIANI", "FREDIANI175"))
    lens_df = pd.concat([lens_df, FREDIANI175_len]).reset_index(drop=True)

    output["Row"] = output["block_name"] + "_" + output["name"].str.split("_").str[0]
    output.rename({"F_pred": "Lasso"}, axis=1, inplace=True)
    output.to_csv("/media/fruitspec-lab/cam175/FOWLER/FOWLER_preds_feredini175_pc_fix.csv")
    output.groupby("block_name").mean().to_csv("/media/fruitspec-lab/cam175/FOWLER/FOWLER_block_summary_mean_pc_fix.csv")
    output.groupby("block_name").std().to_csv("/media/fruitspec-lab/cam175/FOWLER/FOWLER_block_summary_std_pc_fix.csv")

    grouped_df = output.groupby("Row").agg({
        "name": "count",  # Count the occurrences of each group
        "Lasso": [np.mean, np.std],
        "Lasso_retrain_pred": [np.mean, np.std],
        "XGB_pred": [np.mean, np.std],
        "XGB_retrain_pred": [np.mean, np.std]
    })

    # Rename the columns for better readability
    grouped_df.columns = ["count", "Lasso mean", "Lasso std", "Lasso_retrain_pred mean", "Lasso_retrain_pred std",
                          "XGB_pred mean", "XGB_pred std", "XGB_retrain_pred mean", "XGB_retrain_pred std"]

    rows = [name.split("_")[1] for name in grouped_df.index]
    blocks = [name.split("_")[0] for name in grouped_df.index]
    grouped_df["row"] = rows
    grouped_df["block_name"] = blocks
    grouped_df["row_len"] = grouped_df.index.map(dict(zip(lens_df["row"], lens_df["row_length"])))
    grouped_df = grouped_df[grouped_df["row_len"] > 1]
    col = "Lasso"
    for col in ["Lasso", "Lasso_retrain_pred", "XGB_pred", "XGB_retrain_pred"]:
        grouped_df[f"{col} per meter"] = grouped_df[f"{col} mean"] * grouped_df["count"] / grouped_df["row_len"]
    grouped_df.to_csv("/media/fruitspec-lab/cam175/FOWLER/FOWLER_summary_row_pc_fix.csv")
    grouped_df.groupby("block_name").mean().to_csv(
        "/media/fruitspec-lab/cam175/FOWLER/FOWLER_summary_block_per_meter_pc_fix.csv")
    grouped_df.groupby("block_name").std().to_csv(
        "/media/fruitspec-lab/cam175/FOWLER/FOWLER_summary_block_per_meter_std_pc_fix.csv")

    grouped_df = output.groupby("block_name").agg({
        "name": "count",  # Count the occurrences of each group
        "Lasso": [np.mean, np.std],
        "Lasso_retrain_pred": [np.mean, np.std],
        "XGB_pred": [np.mean, np.std],
        "XGB_retrain_pred": [np.mean, np.std]
    })

    # Rename the columns for better readability
    grouped_df.columns = ["count", "Lasso mean", "Lasso std", "Lasso_retrain_pred mean", "Lasso_retrain_pred std",
                          "XGB_pred mean", "XGB_pred std", "XGB_retrain_pred mean", "XGB_retrain_pred std"]
    grouped_df.to_csv("/media/fruitspec-lab/cam175/FOWLER/FOWLER_summary_block_pc_fix.csv")


if __name__ == "__main__":
    use_best_study = True
    cfg = OmegaConf.load("model_config.yaml")
    final_cols, drop_final = get_rel_cols(cfg)
    f_df = read_f_df(cfg)
    features_df = read_data(cfg)

    # order = pd.read_csv("/media/fruitspec-lab/cam175/customers_new/FOWLER_features_manual_slice_new_det.csv").columns
    # features_df = features_df[order]
    #
    # folwer_analysis(features_df)

    # out_frame = infer_on_features(features_df, cfg, model_pkl="models_with_Fowler/xgb_notebook.pkl", tree_model=True)
    # features_df["F_pred"] = out_frame["F_pred"]
    # features_df = add_fs(features_df, f_df)
    train_model(cfg)
    # features_df.to_csv("/media/fruitspec-lab/cam175/customers_new/motcha_features_manual_slice_new_det_w_F.csv")
    # print(get_mape(features_df, group_col="block_name"))



