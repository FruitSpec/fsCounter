import os.path

import optuna
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import optuna
from sklearn.ensemble import GradientBoostingRegressor
from models import models
from optuna_dict_config import models_params
from sklearn.preprocessing import RobustScaler
from MHS.F_model_training import *

class HyperparameterTuner:
    def __init__(self, model, X, y, param_distributions={}, n_trials=2500, direction='maximize', save_int=50,
                 save_dst="", read_from="", verbose=True, scoring="neg_mean_absolute_percentage_error",
                 n_splits=8, scaling_models=["KNeighborsRegressor", "SVR"], suffix="",
                 feature_selection=False, optimization_technique='tpe', train_until_n_trails=True):
        if isinstance(model, dict):
            self.mode = "multi"
        else:
            self.mode = "single"
        self.model = model
        self.X = X.reset_index()
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.reshape(-1)
        self.y = y
        self.param_distributions = param_distributions
        self.n_trials = n_trials
        self.direction = direction
        self.save_int = save_int
        self.save_dst = save_dst
        if save_dst != "":
            if not os.path.exists(save_dst):
                os.makedirs(save_dst, exist_ok=True)
        self.read_from = read_from
        self.verbose = verbose
        self.study = None
        self.scoring = scoring
        self.n_splits = n_splits
        self.scaling_models = scaling_models
        self.suffix = suffix
        self.feature_selection = feature_selection
        self.optimization_technique = optimization_technique
        self.train_until_n_trails = train_until_n_trails

    def objective(self, trial, shuffle: bool=True, random_state: int=42):
        kfolds = KFold(n_splits=self.n_splits, shuffle=shuffle, random_state=random_state)
        params = {}
        for param_name, param_distribution in self.param_distributions.items():
            dist_type = param_distribution["type"]
            if dist_type == 'suggest_int':
                params[param_name] = trial.suggest_int(name=param_name, **param_distribution["args"])
            if dist_type == 'suggest_float':
                params[param_name] = trial.suggest_float(name=param_name, **param_distribution["args"])
            if dist_type == 'suggest_categorical':
                params[param_name] = trial.suggest_categorical(name=param_name, **param_distribution["args"])
        self.model.set_params(**params)

        X = self.X.copy()
        if self.feature_selection:
            features_to_use = []
            for i in range(self.X.shape[1]):
                if trial.suggest_int(f'use_feature_{self.X.columns[i]}', 0, 1):
                    features_to_use.append(i)
            X = X.iloc[:, features_to_use]

        try:
            scores = cross_val_score(self.model, X, self.y, cv=kfolds, scoring=self.scoring)
        except ValueError:
            return -np.inf if self.direction == "maximize" else np.inf
        if np.nan in scores:
            return -np.inf if self.direction == "maximize" else np.inf
        return np.nanmean(scores)

    def get_study(self, model_name):
        got_study = False
        if self.read_from != "":
            if self.suffix != "":
                read_from_name = os.path.join(self.read_from, f"{model_name}_{self.suffix}_study.pkl")
            else:
                read_from_name = os.path.join(self.read_from, f"{model_name}_study.pkl")
            if os.path.exists(read_from_name):
                self.study = joblib.load(read_from_name)
                got_study = True

        if not got_study:
            if self.optimization_technique == 'tpe':
                sampler = optuna.samplers.TPESampler()
            elif self.optimization_technique == 'cmaes':
                sampler = optuna.samplers.CmaEsSampler()
            elif self.optimization_technique == 'random':
                sampler = optuna.samplers.RandomSampler()
            else:
                raise ValueError("Unsupported optimization technique")

            self.study = optuna.create_study(sampler=sampler, direction=self.direction)

    def single_tune(self, n_jobs: int = -1, model_name: str = ""):
        self.get_study(model_name)
        if self.save_int == 0:
            self.study.optimize(self.objective, n_trials=self.n_trials, n_jobs=n_jobs, show_progress_bar=self.verbose)
            if self.save_dst != "":
                joblib.dump(self.study, self.save_dst)
        else:
            total_trails = self.n_trials - len(self.study.trials)*self.train_until_n_trails
            if total_trails > 0:
                total_runs = int(np.ceil(total_trails / self.save_int))
                for i in range(total_runs):
                    self.study.optimize(self.objective, n_trials=self.save_int, n_jobs=n_jobs, show_progress_bar=self.verbose)
                    if self.save_dst != "":
                        joblib.dump(self.study, self.save_dst)
        best_params = self.study.best_params
        best_score = self.study.best_value
        print(f"Best score: {best_score}\n")
        print(f"Optimized parameters: {best_params}\n")
        return best_params, best_score

    def multi_models_tune(self, models_params: dict, n_jobs: int = -1):
        for model_name, model_params_distribution in models_params.items():
            print(f"tuning {model_name}")
            if model_name in self.scaling_models:
                org_x = self.X.copy()
                self.X = RobustScaler().fit_transform(org_x)
            cur_save_dist = self.save_dst
            if cur_save_dist != "":
                if self.suffix != "":
                    self.save_dst = os.path.join(cur_save_dist, f"{model_name}_{self.suffix}_study.pkl")
                else:
                    self.save_dst = os.path.join(cur_save_dist, f"{model_name}_study.pkl")
            self.param_distributions = model_params_distribution
            self.model = models[model_name]
            self.single_tune(n_jobs, model_name)
            self.save_dst = cur_save_dist
            if model_name in self.scaling_models:
                self.X = org_x

    def tune(self, n_jobs: int = -1, model_name=""):
        if self.mode == "single":
            self.single_tune(self, n_jobs, model_name)
        else:
            self.multi_models_tune(self.model, n_jobs)

if __name__ == "__main__":
    cfg = OmegaConf.load("model_config.yaml")
    validate_output_path(cfg.output_folder)
    final_cols, drop_final = get_rel_cols(cfg)
    f_df = read_f_df(cfg)
    features_df = read_data(cfg)
    full_name_org = get_full_name(features_df)
    features_df_clean = clean_the_df(features_df, drop_final, cfg)
    features_df_w_f = add_fs(features_df_clean, f_df)
    full_name_clean = get_full_name(features_df_w_f)
    dropped_obs = features_df[~full_name_org.isin(full_name_clean)]
    dropped_obs.to_csv("dropped_obs.csv", index=False)
    X, y, groups = get_X_y(features_df_w_f, cfg)
    X_trees = X[[col for col in cfg.tree_cols if col in X.columns]]
    org_cols = features_df_clean.drop(["fullname", "F", "block_name", "name", "customer"], axis=1).columns
    X_org = features_df_w_f[org_cols]

    # trees
    tuner = HyperparameterTuner(models_params, X_trees, y, n_trials=500, save_int=24,
                                direction='maximize', save_dst="studies_31_07", suffix="trees", read_from="studies_31_07")
    tuner.tune()

    #original
    tuner = HyperparameterTuner(models_params, X_org, y, n_trials=500, save_int=24,
                                direction='maximize', save_dst="studies_31_07", suffix="org", read_from="studies_31_07")
    tuner.tune()


    # all
    tuner = HyperparameterTuner(models_params, X, y, n_trials=500, save_int=24,
                                direction='maximize', save_dst="studies_31_07", suffix="", read_from="studies_31_07")
    tuner.tune()

    # with feature selection
    # trees
    tuner = HyperparameterTuner(models_params, X_trees, y, n_trials=500, save_int=24,
                                direction='maximize', save_dst="studies_31_07", suffix="trees_fs",
                                read_from="studies_31_07", feature_selection=True)
    tuner.tune()

    # original
    tuner = HyperparameterTuner(models_params, X_org, y, n_trials=500, save_int=24,
                                direction='maximize', save_dst="studies_31_07", suffix="org_fs",
                                read_from="studies_31_07", feature_selection=True)
    tuner.tune()

    # all
    tuner = HyperparameterTuner(models_params, X, y, n_trials=500, save_int=24,
                                direction='maximize', save_dst="studies_31_07", suffix="fs",
                                read_from="studies_31_07", feature_selection=True)
    tuner.tune()


    # CAMES - genetic algorithm optimization
    # trees
    tuner = HyperparameterTuner(models_params, X_trees, y, n_trials=500, save_int=24,
                                direction='maximize', save_dst="studies_31_07", suffix="trees_cmaes",
                                optimization_technique='cmaes')
    tuner.tune()

    #org
    tuner = HyperparameterTuner(models_params, X_org, y, n_trials=500, save_int=24,
                                direction='maximize', save_dst="studies_31_07", suffix="org_cmaes",
                                optimization_technique='cmaes')
    tuner.tune()

    #all
    tuner = HyperparameterTuner(models_params, X, y, n_trials=500, save_int=24,
                                direction='maximize', save_dst="studies_31_07", suffix="cmaes",
                                optimization_technique='cmaes')
    tuner.tune()

    # with feature selection
    # trees
    tuner = HyperparameterTuner(models_params, X_trees, y, n_trials=500, save_int=24,
                                direction='maximize', save_dst="studies_31_07", suffix="trees_cmaes_fs",
                                optimization_technique='cmaes', feature_selection=True)
    tuner.tune()

    # original
    tuner = HyperparameterTuner(models_params, X_org, y, n_trials=500, save_int=24,
                                direction='maximize', save_dst="studies_31_07", suffix="org_cmaes_fs",
                                optimization_technique='cmaes', feature_selection=True)
    tuner.tune()

    # all
    tuner = HyperparameterTuner(models_params, X, y, n_trials=500, save_int=24,
                                direction='maximize', save_dst="studies_31_07", suffix="cmaes_fs",
                                optimization_technique='cmaes', feature_selection=True)
    tuner.tune()