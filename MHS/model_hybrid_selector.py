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

class model_hybrid_selector(BaseEstimator):
    """
    A hybrid model selector class which implements a forward and backward feature selection algorithm.
    """
    def __init__(self, base_estimator=LinearRegression(), params={},
                 scoring="neg_mean_absolute_percentage_error", statisticly=False, cols=[]):
        """
        Initialize the hybrid selector with a base estimator, and a set of parameters.
        :param base_estimator: (estimator object) the base estimator to use
        :param params: (dict) the parameters to pass to the base estimator
        :param scoring: (str) the scoring metric to use for cross validation
        :param statisticly: (bool) whether to use a statistical method to choose the best feature
        :param cols: (list) the initial set of features to use
        """
        print(params)
        if len(params) > 0:
            self.model = base_estimator(**params)
        else:
            self.model = base_estimator
        self.cols = cols
        self.scoring = scoring
        self.stratify = None
        self.statisticly = statisticly

    def get_score(self, X_tr, y_tr, cols, cv=8, n_jobs=-1, random_state=None):
        """
        Get the score of a set of features using cross validation
        :param X_tr: (pd.DataFrame) the training data
        :param y_tr: (pd.Series) the target variable
        :param cols: (list) the set of features to use
        :param cv:(int) the number of cross validation splits
        :param n_jobs: (int) the number of parallel jobs to run
        :param random_state: (int) random state for reproducibility
        :return: (float) the mean score of the cross validation, (float) the standard deviation of the scores
        """
        if len(cols) == 0:
            return np.inf, np.inf
        if isinstance(cols, set):
            cols = list(cols)
        if not isinstance(self.stratify, type(None)):
            cv = StratifiedKFold(cv, shuffle=True, random_state=random_state)
        score = cross_validate(self.model, X_tr[cols], y=y_tr, cv=cv, n_jobs=n_jobs,
                               scoring=self.scoring)
        test_scores = score["test_score"]
        return np.mean(test_scores * (-1)), np.std(test_scores)

    def get_candidate(self, results_dict, results_stds, cur_score):
        """
        Get the best feature based on the results dictionary and the current score
        :param results_dict: (dict) a dictionary of feature:score
        :param results_stds: (list) a list of standard deviations of the scores
        :param cur_score: (float) the current score
        :return: (pd.DataFrame) the best feature based on the method specified
        """
        sort_by = "z_score" if self.statisticly else "new_result"
        new_results = list(results_dict.values())
        z_score = (np.array(new_results) - cur_score) / np.array(results_stds)
        df = pd.DataFrame({"feature": list(results_dict.keys()),
                           "new_result": list(results_dict.values()),
                           "z_score": z_score}).sort_values(sort_by)
        return df.iloc[0] if self.statisticly else df.iloc[0]

    def print_results(self, canidate, cur_score, cols, added=True):
        """
        Print the results of the step and update the current score and columns
        :param candidate: (pd.DataFrame) the best feature based on the method specified
        :param cur_score: (float) the current score
        :param cols: (list) the current columns being used
        :param added: (bool) whether the feature is being added or removed
        :return: (float) the new score, (int) the number of changes made, (list) the new columns
        """
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
        """
        Perform one step of the feature selection process
        :param X_tr: (pd.DataFrame) the training data
        :param y_tr: (pd.Series) the training labels
        :param use_always: (list) a list of features that should always be used
        :param cols: (list) the current columns being used
        :param cur_score: (float) the current score
        :param forward: (bool) whether to add or remove a feature
        :return: (list) the new columns, (int) the number of changes made, (float) the new score
        """
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
        """
        Perform hybrid stepwise selection on the input data
        :param X_tr: (pd.DataFrame) the training data
        :param y_tr: (pd.Series) the target variable
        :param num_stuck: (int) number of times the algorithm can get stuck before stopping
        :param cols: (list) the columns to start with
        :param use_alwyas: (list) the columns to always use
        :return: (list, float) the selected columns and final score
        """
        no_change, cur_score = 0, np.inf
        print(cur_score)
        while no_change < num_stuck:
            cols, added_result, cur_score = self.step(X_tr, y_tr, use_alwyas, cols, cur_score)
            cols, removed_result, cur_score = self.step(X_tr, y_tr, use_alwyas, cols, cur_score, forward=False)
            if added_result or removed_result:
                no_change = 0
            else:
                no_change += 1
            print(f"score: {cur_score},cols: {cols + use_alwyas}")
        return cols + use_alwyas, cur_score

    def fit(self,X_tr, y_tr, num_stuck=3, cols=[], use_alwyas=[], stratify=None, prod=False):
        """
        Fit the model to the input data
        :param X_tr: (pd.DataFrame) the training data
        :param y_tr: (pd.Series) the target variable
        :param num_stuck: (int) number of times the algorithm can get stuck before stopping
        :param cols: (list) the columns to start with
        :param use_alwyas: (list) the columns to always use
        :param stratify: (str) column to use for stratified sampling
        :param prod: (bool) if True, will not perform feature selection, and will use current column
        :return: (float) final score if feature selection is performed
        """
        self.stratify = stratify
        if not prod:
            self.cols, cur_score = self.hybrid_stepwise_selection(X_tr, y_tr, num_stuck, cols, use_alwyas)
        self.model.fit(X_tr[self.cols], y_tr)
        if not prod:
            return cur_score

    def fitplusplus(self, X_tr, y_tr, num_stuck=3, stratify=None, n_iter=15, use_alwyas =[]):
        """
        Perform multiple iterations of the hybrid stepwise selection process,
         selecting the best iteration based on the final score.
        :param X_tr: (pd.DataFrame) the training data
        :param y_tr: (pd.Series) the target variable
        :param num_stuck: (int) number of times the algorithm can get stuck before stopping
        :param stratify: (str) column to use for stratified sampling
        :param n_iter: (int) number of iterations to perform
        :param use_alwyas: (list) the columns to always use
        :return: (float) final score of the best iteration
        """
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
            self.fit(X_tr, y_tr, num_stuck=num_stuck, cols=cols, stratify=stratify, use_alwyas=use_alwyas)
            cols_arr[i] = {"cols": self.cols, "score": self.get_score(X_tr, y_tr, self.cols, n_jobs=-1)}
            print(f"final cols for iteration {i}: {self.cols}")
        cols_arr_sorted = {k: v for k, v in sorted(cols_arr.items(), key=lambda item: item[1]["score"])}
        best_iteration = cols_arr[list(cols_arr_sorted.keys())[0]]
        self.cols = best_iteration["cols"]
        return best_iteration["score"]

    def predict(self, X_pred):
        """
        Predict using the model
        :param X_pred: (pd.DataFrame) the data to predict on
        :return: (np.array) the predictions
        """
        return self.model.predict(X_pred[self.cols])

    def fit_predict(self, X_tr, y_tr, num_stuck=3, cols=[], use_alwyas=[], stratify=None):
        """
        Fit the model to the input data and make predictions
        :param X_tr: (pd.DataFrame) the training data
        :param y_tr: (pd.Series) the target variable
        :param num_stuck: (int) number of times the algorithm can get stuck before stopping
        :param cols: (list) the columns to start with
        :param use_alwyas: (list) the columns to always use
        :param stratify: (str) column to use for stratified sampling
        :return: (np.array) the predictions
        """
        self.fit(X_tr, y_tr, num_stuck, cols, use_alwyas, stratify)
        return self.predict(X_tr)


    def iterative_fitting(self, X_tr, y_tr, num_stuck=3, cols=[], use_alwyas=[], stratify=None, n_iter=15,
                          sampling=0.8):
        """
        Perform multiple iterations of the hybrid stepwise selection process,
         selecting the best iteration based on the final score.
        :param X_tr: (pd.DataFrame) the training data
        :param y_tr: (pd.Series) the target variable
        :param num_stuck: (int) number of times the algorithm can get stuck before stopping
        :param stratify: (str) column to use for stratified sampling
        :param n_iter: (int) number of iterations to perform
        :param use_alwyas: (list) the columns to always use
        :return: (float) final score of the best iteration
        """
        self.stratify = stratify
        cols_arr = {}
        for i in range(1, n_iter + 1):
            self.fit(X_tr, y_tr, num_stuck=num_stuck, cols=cols, stratify=stratify, use_alwyas=use_alwyas)
            cols = self.cols
            if i > 1:
                cols_arr_sorted = {k: v for k, v in sorted(cols_arr.items(), key=lambda item: item[1]["score"])}
                best_iteration = cols_arr[list(cols_arr_sorted.keys())[0]]
                cols = best_iteration["cols"]
            cols = np.random.choice(np.array(cols), replace=False, size=int(sampling * len(cols))).tolist()
            cols_arr[i] = {"cols": self.cols, "score": self.get_score(X_tr, y_tr, self.cols, n_jobs=-1)}
            print(f"final cols for iteration {i}: {self.cols}")
        cols_arr_sorted = {k: v for k, v in sorted(cols_arr.items(), key=lambda item: item[1]["score"])}
        best_iteration = cols_arr[list(cols_arr_sorted.keys())[0]]
        self.cols = best_iteration["cols"]
        return best_iteration["score"]




class LassoSelector(BaseEstimator, TransformerMixin):
    """
    A lasso features selection column transformer
    """
    def __init__(self, estimator):
        """
        init the estimator (a lasso class example LassoCV)
        :param estimator:
        """
        self.estimator = estimator

    def fit(self, X, y=None):
        """
        fit the transformer
        :param X: train matrix
        :param y: target vector
        :return: self
        """
        self.estimator.fit(X, y)
        self.coefs_ = self.estimator.coef_
        return self

    def transform(self, X):
        """
        :param X: matrix to transform
        :return: the matrix with columns that the Lasso choose
        """
        return X[:, self.coefs_ != 0]



