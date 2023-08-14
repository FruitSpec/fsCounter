param_distributions_gbm = {
    'n_estimators': {'type': 'suggest_int', "args": {'low': 10, "high": 2000, "log": False, "step": 10}},
    'max_depth': {'type': 'suggest_int', "args": {'low': 1, "high": 20, "log": False}},
    'learning_rate': {'type': 'suggest_float', 'args': {'low': 1e-5, "high": 1e-1, "log": True}},
    'subsample': {'type': 'suggest_float', 'args': {'low': 0.5, "high": 1, "log": False}},
    'alpha': {'type': 'suggest_float', 'args': {'low': 0.1, "high": 0.9, "log": False}},
    'ccp_alpha': {'type': 'suggest_float', 'args': {'low': 1e-10, "high": 10, "log": True}},
    'min_samples_split': {'type': 'suggest_int', 'args': {'low': 2, "high": 20, "log": False}},
    'min_samples_leaf': {'type': 'suggest_int', 'args': {'low': 1, "high": 20, "log": False}},
    'max_features': {'type': 'suggest_float', 'args': {'low': 0.1, "high": 1.0, "log": False}},
    'loss': {'type': 'suggest_categorical', 'args': {'choices': ['squared_error', 'absolute_error', 'huber']}},
}

param_distributions_rf = {
    'n_estimators': {'type': 'suggest_int', "args": {'low': 10, "high": 2000, "log": False, "step": 10}},
    'max_depth': {'type': 'suggest_int', "args": {'low': 1, "high": 100, "log": False}},
    'min_samples_split': {'type': 'suggest_int', 'args': {'low': 2, "high": 20, "log": True}},
    'min_samples_leaf': {'type': 'suggest_int', 'args': {'low': 1, "high": 20, "log": True}},
    'max_features': {'type': 'suggest_float', 'args': {'low': 0.1, "high": 1.0, "log": False}},
    'max_samples': {'type': 'suggest_float', 'args': {'low': 0.5, "high": 1.0, "log": False}},
    'criterion': {'type': 'suggest_categorical',
                  'args': {'choices': ['squared_error', 'absolute_error', 'friedman_mse']}},
    'ccp_alpha': {'type': 'suggest_float', 'args': {'low': 1e-10, "high": 10, "log": True}},
}

param_distributions_hgb = {
    'loss': {'type': 'suggest_categorical', 'args': {'choices': ['squared_error', 'absolute_error']}},
    'max_iter': {'type': 'suggest_int', 'args': {'low': 10, 'high': 2000, 'log': False, 'step': 10}},
    'learning_rate': {'type': 'suggest_float', 'args': {'low': 1e-5, 'high': 1e-1, 'log': True}},
    'max_depth': {'type': 'suggest_int', 'args': {'low': 1, 'high': 50, 'log': False}},
    'min_samples_leaf': {'type': 'suggest_int', 'args': {'low': 1, 'high': 30, 'log': True}},
    'max_bins': {'type': 'suggest_int', 'args': {'low': 5, 'high': 255, 'log': True}},
    'max_leaf_nodes': {'type': 'suggest_int', 'args': {'low': 5, 'high': 150, 'log': False}},

    'l2_regularization': {'type': 'suggest_float', 'args': {'low': 1e-10, 'high': 1e-1, 'log': True}},
}

param_distributions_knr = {
    'n_neighbors': {'type': 'suggest_int', 'args': {'low': 1, 'high': 20, 'log': False}},
    'weights': {'type': 'suggest_categorical', 'args': {'choices': ['uniform', 'distance']}},
    'algorithm': {'type': 'suggest_categorical', 'args': {'choices': ['auto', 'brute']}},
    'metric': {'type': 'suggest_categorical', 'args': {'choices': ['manhattan', 'cosine', 'euclidean']}},
    'p': {'type': 'suggest_int', 'args': {'low': 1, 'high': 5, 'log': False}},
}

param_distributions_svr = {
    'kernel': {'type': 'suggest_categorical', 'args': {'choices': ['linear', 'poly', 'rbf', 'sigmoid']}},
    'C': {'type': 'suggest_float', 'args': {'low': 1e-3, 'high': 1e3, 'log': True}},
    'epsilon': {'type': 'suggest_float', 'args': {'low': 1e-3, 'high': 1, 'log': True}},
    'gamma': {'type': 'suggest_categorical', 'args': {'choices': ['scale', 'auto']}},
    'degree': {'type': 'suggest_int', 'args': {'low': 1, 'high': 5, 'log': False}},
}

param_distributions_dtr = {
    'max_depth': {'type': 'suggest_int', "args": {'low': 1, "high": 100, "log": False}},
    'min_samples_split': {'type': 'suggest_int', 'args': {'low': 2, "high": 20, "log": True}},
    'min_samples_leaf': {'type': 'suggest_int', 'args': {'low': 1, "high": 20, "log": True}},
    'criterion': {'type': 'suggest_categorical',
                  'args': {'choices': ['squared_error', 'absolute_error', 'friedman_mse']}},
    'ccp_alpha': {'type': 'suggest_float', 'args': {'low': 1e-10, "high": 10, "log": True}},
}

param_distributions_xgb = {
    'eval_metric': {'type': 'suggest_categorical', 'args': {'choices': ['rmse', 'rmsle', 'mape']}},
    'objective': {'type': 'suggest_categorical',
                  'args': {'choices': ['reg:squarederror', 'reg:squaredlogerror']}},
    'booster': {'type': 'suggest_categorical', 'args': {'choices': ['gbtree', 'gblinear', 'dart']}},
    'n_estimators': {'type': 'suggest_int', 'args': {'low': 50, 'high': 2500, 'log': False, 'step': 10}},
    'max_depth': {'type': 'suggest_int', 'args': {'low': 1, 'high': 30, 'log': False}},
    'learning_rate': {'type': 'suggest_float', 'args': {'low': 1e-5, 'high': 1e-1, 'log': True}},
    'subsample': {'type': 'suggest_float', 'args': {'low': 0.5, 'high': 1, 'log': False, 'step': 0.1}},
    'colsample_bytree': {'type': 'suggest_float', 'args': {'low': 0.5, 'high': 1, 'log': False, 'step': 0.1}},
    'colsample_bylevel': {'type': 'suggest_float', 'args': {'low': 0.5, 'high': 1, 'log': False, 'step': 0.1}},
    'reg_alpha': {'type': 'suggest_float', 'args': {'low': 1e-10, 'high': 10, 'log': False}},
    'reg_lambda': {'type': 'suggest_float', 'args': {'low': 1e-10, 'high': 10, 'log': False}},
    'gamma': {'type': 'suggest_float', 'args': {'low': 1e-10, 'high': 10, 'log': False}},
}

param_distributions_cat = {
    'iterations': {'type': 'suggest_int', 'args': {'low': 10, 'high': 2000, 'log': True}},
    'l2_leaf_reg': {'type': 'suggest_float', 'args': {'low': 1e-10, 'high': 10, 'log': True}},
    'depth': {'type': 'suggest_int', 'args': {'low': 1, 'high': 16}},
    'learning_rate': {'type': 'suggest_float', 'args': {'low': 1e-4, 'high': 0.5, 'log': True}},
    'subsample': {'type': 'suggest_float', 'args': {'low': 0.5, 'high': 1.0, 'step': 0.1}},
    'random_seed': {'type': 'suggest_int', 'args': {'low': 1, 'high': 100}}
}

param_distributions_lgbm = {
    'boosting_type': {'type': 'suggest_categorical', 'args': {'choices': ['gbdt', 'dart', 'rf']}},
    'n_estimators': {'type': 'suggest_int', 'args': {'low': 10, 'high': 2000, 'log': False, 'step': 10}},
    # 'max_depth': {'type': 'suggest_int', 'args': {'low': 5, 'high': 30, 'log': False}},
    'num_leaves': {'type': 'suggest_int', 'args': {'low': 8, 'high': 250, 'log': False}},
    'learning_rate': {'type': 'suggest_float', 'args': {'low': 1e-5, 'high': 5e-1, 'log': True}},
    'colsample_bytree': {'type': 'suggest_float', 'args': {'low': 0.5, 'high': 1, 'log': False, 'step': 0.05}},
    'bagging_fraction': {'type': 'suggest_float', 'args': {'low': 0.1, 'high': 0.9, 'log': False, 'step': 0.05}},
    'reg_alpha': {'type': 'suggest_float', 'args': {'low': 1e-10, 'high': 10, 'log': False}},
    'reg_lambda': {'type': 'suggest_float', 'args': {'low': 1e-10, 'high': 10, 'log': False}},
    'min_child_samples': {'type': 'suggest_int', 'args': {'low': 2, 'high': 20, 'log': False}},
}

models_params = {
    # "DecisionTreeRegressor": param_distributions_dtr,
    # "SVR": param_distributions_svr,
    # "KNeighborsRegressor": param_distributions_knr,
    "HistGradientBoostingRegressor": param_distributions_hgb,
    "RandomForestRegressor": param_distributions_rf,
    "LGBMRegressor": param_distributions_lgbm,
    "GradientBoostingRegressor": param_distributions_gbm,
    # "CatBoostRegressor": param_distributions_cat,
    "XGBRegressor": param_distributions_xgb,
}