import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate, StratifiedKFold
from tqdm import tqdm

class model_hybrid_selector(BaseEstimator):

    def __init__(self, base_estimator=LinearRegression(), params={},
                 scoring="neg_mean_absolute_percentage_error", statisticly=False):
        print(params)
        if len(params) > 0:
            self.model = base_estimator(**params)
        else:
            self.model = base_estimator
        self.cols = []
        self.scoring = scoring
        self.stratify = None
        self.statisticly = statisticly

    def get_score(self, X_tr, y_tr, cols, cv=16, n_jobs=-1, random_state=None):
        if not isinstance(self.stratify, type(None)):
            cv = StratifiedKFold(cv, shuffle=True, random_state=random_state)
        score = cross_validate(self.model, X_tr[cols], y=y_tr, cv=cv, n_jobs=n_jobs,
                               scoring=self.scoring)
        test_scores = score["test_score"]
        return np.mean(test_scores * (-1)), np.std(test_scores)

    def get_candidate(self, results_dict, results_stds, cur_score):
        sort_by = "z_score" if self.statisticly else "new_result"
        new_results = list(results_dict.values())
        z_score = (np.array(new_results) - cur_score) / np.array(results_stds)
        df = pd.DataFrame({"feature": list(results_dict.keys()),
                           "new_result": list(results_dict.values()),
                           "z_score": z_score}).sort_values(sort_by)
        return df.iloc[0] if self.statisticly else df.iloc[0]

    def print_results(self, canidate, cur_score, cols, added=True):
        change = 0
        status = "added" if added else "removed"
        new_score = canidate["new_result"]
        if np.isnan(new_score):
            return cur_score, change, cols
        if new_score < cur_score:
            feature = canidate["feature"]
            if added:
                cols.append(feature)
            elif feature in cols:
                cols.remove(feature)
            change = 1
            print(f"{status} {feature}, new_score: {new_score}")
        else:
            print(f"no feature could be {status}, new_score: {new_score}")
        return new_score, change, cols

    def step(self, X_tr, y_tr, use_alwyas, cols, cur_score, forward=True):
        results_dict, results_stds, change = {}, [], 0
        canidate_cols = set(X_tr.columns) - set(use_alwyas) - set(cols) if forward else set(cols) - set(use_alwyas)
        if len(canidate_cols) == 0:
            return cols, change, cur_score
        random_state = np.random.randint(0, 1000, 1)[0]
        assured_cols = cols + use_alwyas
        for col in tqdm(canidate_cols):
            train_cols = assured_cols + [col] if forward else set(assured_cols) - {col}
            results_dict[col], cur_std = self.get_score(X_tr, y_tr, train_cols, random_state=random_state)
            results_stds.append(cur_std)
        candidate = self.get_candidate(results_dict, results_stds, cur_score)
        new_score, change, cols = self.print_results(candidate, cur_score, cols, added=forward)
        return cols, change, min(new_score, cur_score)

    def hybrid_stepwise_selection(self, X_tr, y_tr, num_stuck=3, cols=[], use_alwyas=[]):
        no_change, cur_score = 0, np.inf
        print(cur_score)
        while no_change < num_stuck:
            cols, added_result, cur_score = self.step(X_tr, y_tr, use_alwyas, cols, cur_score)
            cols, removed_result, cur_score = self.step(X_tr, y_tr, use_alwyas, cols, cur_score, forward=False)
            if added_result or removed_result:
                no_change = 0
            else:
                no_change += 1
            print(f"score: {cur_score},cols: {cols}")
        return cols + use_alwyas, cur_score

    def fit(self, X_tr, y_tr, num_stuck=3, cols=[], use_alwyas=[], stratify=None):
        self.stratify = stratify
        self.cols, cur_score = self.hybrid_stepwise_selection(X_tr, y_tr, num_stuck, cols, use_alwyas)
        self.model.fit(X_tr[self.cols], y_tr)
        return cur_score

    def fitplusplus(self, X_tr, y_tr, num_stuck=3, stratify=None, n_iter=15):
        self.stratify = stratify
        all_cols_list = list(X_tr.columns)
        n_cols = len(all_cols_list)
        cols_arr = {}
        for i in range(1, n_iter + 1):
            if i == 1:
                cols = []
            elif i == 2:
                cols = all_cols_list.copy()
            else:
                size = np.random.randint(1, n_cols - 1)
                cols = np.random.choice(all_cols_list.copy(), size=size, replace=False).tolist()
            self.fit(X_tr, y_tr, num_stuck=num_stuck, cols=cols, stratify=stratify)
            cols_arr[i] = {"cols": self.cols, "score": self.get_score(X_tr, y_tr, self.cols, n_jobs=-1)}
            print(self.cols)
            print(i)
        cols_arr_sorted = {k: v for k, v in sorted(cols_arr.items(), key=lambda item: item[1]["score"])}
        best_iteration = cols_arr[list(cols_arr_sorted.keys())[0]]
        self.cols = best_iteration["cols"]
        return best_iteration["score"]

    def predict(self, X_pred):
        return self.model.predict(X_pred[self.cols])

    def fit_predict(self, X_tr, y_tr, num_stuck=3, cols=[], use_alwyas=[], stratify=None):
        self.fit(X_tr, y_tr, num_stuck, cols, use_alwyas, stratify)
        return self.predict(X_tr)

    def iterative_fitting(self, X_tr, y_tr, num_stuck=3, cols=[], use_alwyas=[], stratify=None, n_iter=15,
                          sampling=0.8):
        self.stratify = stratify
        cols_arr = {}
        for i in range(1, n_iter + 1):
            self.fit(X_tr, y_tr, num_stuck=num_stuck, cols=cols, stratify=stratify)
            cols = self.cols
            if i > 1:
                cols_arr_sorted = {k: v for k, v in sorted(cols_arr.items(), key=lambda item: item[1]["score"])}
                best_iteration = cols_arr[list(cols_arr_sorted.keys())[0]]
                cols = best_iteration["cols"]
            cols = np.random.choice(np.array(cols), replace=False, size=int(sampling * len(cols))).tolist()
            cols_arr[i] = {"cols": self.cols, "score": self.get_score(X_tr, y_tr, self.cols, n_jobs=-1)}
            print(self.cols)
            print(i)
        cols_arr_sorted = {k: v for k, v in sorted(cols_arr.items(), key=lambda item: item[1]["score"])}
        best_iteration = cols_arr[list(cols_arr_sorted.keys())[0]]
        self.cols = best_iteration["cols"]
        return best_iteration["score"]
