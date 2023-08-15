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
                             random_state=43, use_pandas=False, ret_all_res=False):
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
        y_true_sum = y_test[~np.isnan(y_pred)].sum()
        results.append(abs(np.nansum(y_pred) - y_true_sum)/(y_true_sum))
        tree_res.append(np.nanmean(abs(y_pred-y_test)/(y_test)))
        test_group = ""
        if not isinstance(groups,type(None)):
            test_group = f"({groups[test_index[0]]})"
        y_pred_sum = np.nansum(y_pred)
        acc = np.abs(y_true_sum-y_pred_sum)/y_true_sum
        all_preds[test_index] = y_pred
        print(F"true: {y_true_sum},    pred: {y_pred_sum}. ({acc*100 :.2f} %) {test_group}" )
    print(np.mean(tree_res), np.std(tree_res))
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
