from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
import optuna
import warnings
from MHS.F_model_training import get_rel_cols, read_f_df, read_data, clean_the_df, add_fs, get_X_y
from MHS.scoring import print_navie_scores
from omegaconf import OmegaConf
import os


def objective_xgb_w_p_import(trial, n_jobs=-1, scoring="neg_mean_absolute_percentage_error",
                             n_splits=8, shuffle=True, random_state=42, param=None, stratify=None):
    """
    obkective function for optuna, this objective fits XGB
    then uses the permutation importance to choose most relevnt features and fits the model again
    :param trial: optuna trail object
    :param n_jobs: number of parrlel runs -1= max
    :param scoring: scoring function for crossvalidation
    :param n_splits: number of splits for cross validation
    :param shuffle: wheter to shuffle the data or not
    :param random_state: random state for repreducibility
    :param param: paramaters for XGB
    :param stratify: for crossvalidation stratification
    :return: model score
    """
    if not isinstance(stratify, type(None)):
        kfolds = StratifiedKFold(n_splits, shuffle=shuffle, random_state=random_state)
    else:
        kfolds = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    if isinstance(param, type(None)):
        param = {
            'objective': trial.suggest_categorical('objective', ["reg:squarederror",
                                                                "reg:pseudohubererror","reg:squaredlogerror"]),
            'eta': trial.suggest_float("eta", 0.0001, 0.5, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-2, 5+1e-2, step=1e-2),
            'max_depth': trial.suggest_int('max_depth', 1, 7),
            'gamma': trial.suggest_float("gamma", 1e-3, 2+1e-3, step=1e-3),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1, step=0.05),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1, step=0.05),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 20, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20, log=True),
            'subsample': trial.suggest_float('subsample', 0.4, 0.95, step=0.05),
            'n_estimators': trial.suggest_int('n_estimators', 5, 2500, step=5)
        }
    reg = XGBRegressor(**param)

    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X.values, y, test_size=0.2)
    reg.fit(X_train_lr, y_train_lr)

    result = permutation_importance(reg, X_test_lr, y_test_lr.values, n_repeats=8, n_jobs=n_jobs)
    sorted_idx = result.importances_mean.argsort()

    permutation_importance_res = pd.DataFrame({"names": X.columns, "imp": result.importances_mean,
                                               "std": result.importances_std})
    imp_params = permutation_importance_res["names"][permutation_importance_res["imp"] > 0].values
    if not len(imp_params):
        imp_params = permutation_importance_res["names"][:5].values
    reg = XGBRegressor(**param)

    scores = cross_val_score(reg, X[imp_params].values, y.values, cv=kfolds, scoring=scoring)
    return scores.mean()


def objective_rf(trial, n_jobs=-1, scoring="neg_mean_absolute_percentage_error",
                 n_splits=8, shuffle=True, random_state=42, param=None, stratify=None):
    """
    obkective function for optuna, this objective fits Random Forest
    :param trial: optuna trail object
    :param n_jobs: number of parrlel runs -1= max
    :param scoring: scoring function for crossvalidation
    :param n_splits: number of splits for cross validation
    :param shuffle: wheter to shuffle the data or not
    :param random_state: random state for repreducibility
    :param param: paramaters for XGB
    :param stratify: for crossvalidation stratification
    :return: model score
    """
    if not isinstance(stratify, type(None)):
        kfolds = StratifiedKFold(n_splits, shuffle=shuffle, random_state=random_state)
    else:
        kfolds = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    if isinstance(param, type(None)):
        param = {
            'n_estimators': trial.suggest_int("n_estimators", 5, 5000, step=5),
#             'max_depth': trial.suggest_int('max_depth', 1,150),
#             'criterion': trial.suggest_categorical('criterion', ["squared_error","absolute_error"]),
            'min_samples_split': trial.suggest_int("min_samples_split", 2, 50), ###
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 50),
            'max_features': trial.suggest_float('max_features', 0.3, 1, step=0.05),
            "ccp_alpha" : trial.suggest_float("ccp_alpha", 1e-16, 0.1, log=True),
            'max_samples': trial.suggest_float('max_samples', 0.4, 0.95, step=0.05),
        }
    reg = RandomForestRegressor(**param)
    scores = cross_val_score(reg, X.reset_index(), y, cv=kfolds, n_jobs=n_jobs, scoring=scoring)
    return scores.mean()



def objective_xgb(trial, n_jobs=-1, scoring="neg_mean_absolute_percentage_error",
                 n_splits=8, shuffle=True, random_state=42, param=None, stratify=None):
    """
    obkective function for optuna, this objective fits XGB
    :param trial: optuna trail object
    :param n_jobs: number of parrlel runs -1= max
    :param scoring: scoring function for crossvalidation
    :param n_splits: number of splits for cross validation
    :param shuffle: wheter to shuffle the data or not
    :param random_state: random state for repreducibility
    :param param: paramaters for XGB
    :param stratify: for crossvalidation stratification
    :return: model score
    """
    if not isinstance(stratify, type(None)):
        kfolds = StratifiedKFold(n_splits, shuffle=shuffle, random_state=random_state)
    else:
        kfolds = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    if isinstance(param, type(None)):
        param = {
            'objective': trial.suggest_categorical('objective', ["reg:squarederror",
                                                                "reg:pseudohubererror","reg:squaredlogerror"]),
            'eta': trial.suggest_float("eta", 0.0001, 0.5, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-2, 5+1e-2, step=1e-2),
            'max_depth': trial.suggest_int('max_depth', 1, 7),
            'gamma': trial.suggest_float("gamma", 1e-3, 2+1e-3, step=1e-3),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1, step=0.05),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1, step=0.05),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 20, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20, log=True),
            'subsample': trial.suggest_float('subsample', 0.4, 0.95, step=0.05),
            'n_estimators': trial.suggest_int('n_estimators', 5, 2500, step=5)
        }
    reg = XGBRegressor(**param)
    scores = cross_val_score(reg, X.values, y, cv=kfolds, scoring=scoring, n_jobs=n_jobs)
    return scores.mean()


def optimize_estimator(objective, direction='maximize', n_trails=200, save_int=0, save_dst="", read_from=""):
    """
    hyperparamater searching for a given objectivfe
    :param objective:  objective to optimize
    :param direction: optimization direction (which is better)
    :param n_trails: number of trails to conduct
    :param save_int: save interval, will save the study every save_int iterations
    :param save_dst: where to save
    :param read_from: path for a pkl to read an already started trail
    :return: best score out of all models tested, best params.
    """
    if read_from != "":
        study = joblib.load(read_from)
    else:
        study = optuna.create_study(direction=direction)
    if save_int == 0:
        study.optimize(objective, n_trials=n_trails, n_jobs=-1)
        if save_dst!="":
            joblib.dump(study, save_dst)
    else:
        trails_per_save = np.ceil(n_trails/save_int)
        for i in range(save_int):
            study.optimize(objective, n_trials=trails_per_save)
            if save_dst!="":
                joblib.dump(study, save_dst)
    best_params = study.best_params
    best_score = study.best_value
    print(f"Best score: {best_score}\n")
    print(f"Optimized parameters: {best_params}\n")
    return best_params, best_score


if __name__ == "__main__":
    output_folder = "/home/fruitspec-lab/FruitSpec/Code/roi/fsCounter/MHS/studies_30_07"
    cfg = OmegaConf.load("model_config.yaml")
    final_cols, drop_final = get_rel_cols(cfg)
    f_df = read_f_df(cfg)
    features_df = read_data(cfg)
    features_df = clean_the_df(features_df, drop_final, cfg)
    features_df = add_fs(features_df, f_df)
    global y
    X_tr_lr, y, rows = get_X_y(features_df, cfg)
    print_navie_scores(features_df, X_tr_lr, y, rows)

    drop_from_trees = ["cv^2", "clusters_area_med_arr", "avg_height", "height", "avg_perimeter", "perimeter",
                      "width", "avg_width", "mst_sums_arr^2", "cv*fruit_foliage_ratio", "avg_volume"]
    X_tr_trees = X_tr_lr.drop(drop_from_trees, axis=1, errors="ignore")
    global X
    X = X_tr_trees
    # can continue stopped session via the read_from argument
    optimize_estimator(objective_rf, n_trails=10000, save_int=50,
                       save_dst=os.path.join(output_folder, "study_rf_training.pkl"))
    optimize_estimator(objective_xgb, n_trails=10000, save_int=50,
                       save_dst=os.path.join(output_folder, "study_xgb_training.pkl"))
    optimize_estimator(objective_xgb_w_p_import, n_trails=10000, save_int=50,
                       save_dst=os.path.join(output_folder, "study_xgb_perm_training.pkl"))
