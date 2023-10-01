import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_validate, StratifiedKFold
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import numpy as np
from typing import List, Tuple, Callable
from omegaconf import OmegaConf
from MHS.F_model_training import (clean_the_df, get_full_name, read_f_df, read_data, add_fs, get_X_y,
                                  print_navie_scores, scalers, model_fit_save, get_plot_fruit_variety_df, add_features)
from vision.misc.help_func import contains_special_char
import joblib
from MHS.F_model_models_dict import F_model_models
import os
from vision.misc.help_func import validate_output_path
from MHS.models import models
from sklearn.pipeline import Pipeline

class F_model(BaseEstimator, TransformerMixin):
    def __init__(self, F_model_cfg: object, model_cfg: object, model_name: str = ""):
        model = self.get_model_from_conf(F_model_cfg, model_name)
        self.model = model
        if isinstance(model_cfg, str):
            model_cfg = OmegaConf.load(model_cfg)
        self.model_cfg = model_cfg
        self.F_model_cfg = F_model_cfg
        self.model_name = model_name
        self.columns = F_model_cfg["columns"]
        self.preprocess_type = F_model_cfg["preprocess_type"]

    @staticmethod
    def get_model_from_conf(F_model_cfg, model_name):
        model_pkl_path = F_model_cfg["model_pkl_path"]
        if os.path.exists(model_pkl_path):
            with open(F_model_cfg["model_pkl_path"], 'rb') as file:
                model = joblib.load(file)
        else:
            model = models[model_name.split("_")[0]]
            model_params = F_model_cfg["model_params"]
            if isinstance(model_params, str):
                study = joblib.load(model_params)
                model_params = study.best_params
            model.set_params(**model_params)
        if "scaler" in F_model_cfg.keys():
            scaler = scalers[F_model_cfg["scaler"]]
            model = Pipeline([("scaler", scaler),
                              ("final_estimator", model)])
        return model

    @staticmethod
    def validate_tree_name(X):
        x_cols = X.columns
        if "customer" not in x_cols:
            X["customer"] = "Fake_customer"
        if "block_name" not in x_cols:
            X["block_name"] = "Fake_block"
        if "name" not in x_cols:
            X["name"] = [f"R0_T{i}" for i in range(X.shape[0])]
        return X

    @staticmethod
    def add_fruit_interaction(X):
        for col in ["orange", "lemon", "mandarin", "apple"]:
            if col in X.columns:
                X[f"{col}_cv"] = X[col] * X["cv"]
                X[f"{col}_surface_area"] = X[col] * X["surface_area"]
        return X

    def feature_engineering(self, X):
        eng_cols = [col for col in self.columns if contains_special_char(col)]
        if "fruit_foliage_ratio" in self.columns:
            X["fruit_foliage_ratio"] = X["fruit_foliage_ratio"].replace(np.inf, 100)
        if "clusters_area_mean_arr" in self.columns:
            X["clusters_area_mean_arr"] = X["clusters_area_mean_arr"].clip(0, 700)
        if "clusters_area_med_arr" in self.columns:
            X["clusters_area_med_arr"] = X["clusters_area_med_arr"].clip(0, 600)
        for col in eng_cols:
            if "^" in col:
                col1, val = col.split("^")
                X[col] = X[col1]**int(val)
            if "*" in col:
                col1, col2 = col.split("*")
                X[col] = X[col1]*X[col2]
            if "/" in col:
                col1, col2 = col.split("/")
                X[col] = X[col1]/X[col2]
        X = self.add_fruit_interaction(X)
        return X

    def preprocess(self, X, y=None, return_out_frame=False):
        X = self.validate_tree_name(X)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y, columns=["F"])
        df_org = pd.concat([X, y], axis=1) if not isinstance(y, type(None)) else X
        df_org.reset_index(drop=True, inplace=True)
        full_name_org = get_full_name(df_org)
        # df = clean_the_df(df_org, [], self.model_cfg, "infer", drop_nas=False)
        # df = self.feature_engineering(df_org)
        df = df_org[np.all(np.isfinite(df_org[self.columns].select_dtypes(exclude=object)), axis=1)]
        df.reset_index(drop=True, inplace=True)
        if "F" in df.columns:
            df = df[~df["F"].isna()]
        full_name_clean = get_full_name(df)
        dropped_obs = df_org.loc[~full_name_org.isin(full_name_clean), :] # For logging
        X = df[self.columns]
        y = df["F"] if not isinstance(y, type(None)) else y
        out_frame = df[["block_name", "name", "customer"]].copy()
        if return_out_frame:
            return X, y, out_frame, full_name_org.isin(full_name_clean)
        return X, y

    def fit(self, X, y):
        X, y = self.preprocess(X, y)
        self.model.fit(X, y)

    def predict(self, X, y=None, return_out_frame=False, ret_final_obs=False):
        X, y, out_frame, final_obs = self.preprocess(X, y, True)
        y_pred_final = np.repeat(np.nan, len(final_obs))
        y_pred = self.model.predict(X.to_numpy())
        y_pred_final[final_obs] = y_pred
        if return_out_frame:
            out_frame["F_pred"] = y_pred
            if ret_final_obs:
                return out_frame, final_obs
            return out_frame
        if ret_final_obs:
            return y_pred_final, final_obs
        return y_pred_final


class FSStackingRegressor(BaseEstimator, TransformerMixin):
    def __init__(self, models_dict, meta_model, model_cfg):
        self.meta_model = meta_model
        self.model_cfg = model_cfg
        self.models = self.init_F_models(models_dict)

    def init_F_models(self, models_dict):
        models = {}
        for model_name, model_params in models_dict.items():
            model = F_model(model_params, self.model_cfg, model_name)
            models[model_name] = model
        return models

    def fit(self, X, y, refit_models=True):
        meta_features = []
        kept_obs = []
        for model_name, model in self.models.items():
            if refit_models:
                model.fit(X, y)
            y_preds, final_obs = model.predict(X, y, ret_final_obs=True)
            kept_obs.append(final_obs)
            meta_features.append(y_preds)
        meta_features = np.array(meta_features).T
        kept_obs = np.array(kept_obs)
        self.meta_model.fit(meta_features, y[np.mean(kept_obs, axis=0) == 1])
        return self

    def predict(self, X):
        meta_features = []
        for model_name, model in self.models.items():
            print("Getting predictions of: ", model_name)
            meta_features.append(model.predict(X))
        print("Predicted with sub models.")
        meta_features = np.column_stack(meta_features)
        predictions = self.meta_model.predict(meta_features)
        return predictions


def run_f_model():
    X_trees_cols = ["cv/center_width", "cv", "mst_mean_arr", "cv/surface_area", "orange_surface_area",
                    "cv/fruit_foliage_ratio",
                    "orange_center_width", "orange_cv", "cv/total_foliage", "cv/center_height",
                    "w_h_ratio", "fruit_dist_center", "mst_sums_arr",
                    "center_width", "center_height", "center_perimeter", "clusters_area_med_arr"]

    cfg = OmegaConf.load("model_config.yaml")
    features_df = pd.read_csv("/media/fruitspec-lab/cam175/customers_new/MOTCHA_auto_slice.csv")

    model_dict = {"model_params": {},
                     "X_data": "X_train.csv",
                     "y_data": "y_train.csv",
                     "columns": X_trees_cols,
                     "output_path": "models_04_07_SUMGLD/xgb_notebook.pkl",
                     "preprocess_type": 0,
                     "model_pkl_path": "models_04_07_SUMGLD/xgb_notebook.pkl"}

    f_model = F_model(model_dict, cfg)
    out_frame = f_model.predict(features_df, return_out_frame=True)
    print(out_frame)


def get_dict_models_results(cfg, X, y, groups=None, use_pandas=False):
    results_summary = []
    for model_name, model_args in F_model_models.items():
        model = F_model(model_args, cfg, model_name)
        save_name = model_args["output_path"]
        print(("#"*20 + " " + model_name + " " + "#"*20)*2)
        mean_score, std_score, group_score, groups_std, groups_score_log, groups_std_log, preds = model_fit_save(
                                                                                                        model, X,
                                                                                    y.values.flatten(), save_name,
                                                                                    groups=groups,
            use_pandas=use_pandas)
        results_summary.append({"model": model_name, "score": np.abs(mean_score), "std": std_score,
                                "group_score": group_score, "groups_std": groups_std,
                                "group_log_score": groups_score_log, "groups_log_std": groups_std_log})
    res_df = pd.DataFrame.from_records(results_summary)
    res_df.to_csv("results_F_models.csv")
    return res_df


def train_F_models(cfg):
    validate_output_path(cfg.output_folder)
    f_df = read_f_df(cfg)
    features_df = read_data(cfg)
    full_name_org = get_full_name(features_df)
    features_df_w_f = add_fs(features_df, f_df)
    features_df_w_f = features_df_w_f[~features_df_w_f["F"].isna()]

    # clean for train
    features_df_w_f = clean_the_df(features_df_w_f, [], cfg)
    features_df_w_f = add_features(features_df_w_f)


    X, y, groups = get_X_y(features_df_w_f, cfg, apply_preprocess=False)
    if "F" in X.columns:
        X.drop("F", axis=1, inplace=True)
    X.to_csv("X_train_Fmodel.csv", index=False)
    y.to_csv("y_train_Fmodel.csv", index=False)
    # print_navie_scores(features_df_w_f, X, y, groups)
    res_df = get_dict_models_results(cfg, X, y, groups, use_pandas=True)


if __name__ == "__main__":
    cfg = OmegaConf.load("model_config.yaml")
    train_F_models(cfg)

    f_df = read_f_df(cfg)
    features_df = read_data(cfg)
    features_df_w_f = add_fs(features_df, f_df)
    y = features_df_w_f['F']
    features_df = features_df_w_f.drop("F", axis=1)
    meta_model = LinearRegression()

    stacked_model = FSStackingRegressor(F_model_models, meta_model, cfg)
    stacked_model.fit(features_df, y, refit_models=False)
    with open(os.path.join(cfg.output_folder, f'stacked_model.pkl'), 'wb') as f:
        pickle.dump(stacked_model, f)
    stacked_model.predict(features_df)