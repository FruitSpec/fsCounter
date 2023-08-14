# if you want a model with sacling it should be in a pipe at models.py
import os

prefix_studies = r"/home/fruitspec-lab/FruitSpec/Code/roi/fsCounter/MHS/models_studies"
model_folder = r"/home/fruitspec-lab/FruitSpec/Code/roi/fsCounter/MHS/models_0908_new_translator"

gfs_train_cols = ['total_foliage', 'total_orange', 'volume', 'surface_area', 'avg_volume',
             'center_perimeter', 'center_height', 'w_h_ratio',
             'med_intens_arr', 'fruit_foliage_ratio', 'mst_sums_arr', 'mst_mean_arr',
             'mst_skew_arr', 'clusters_area_mean_arr', 'clusters_area_med_arr',
             'clusters_ch_area_mean_arr', 'clusters_ch_area_med_arr',
             'n_clust_arr_2', 'q_1_precent_fruits', 'q_2_precent_fruits',
             'q_3_precent_fruits', 'fruit_dist_center', 'ndvi', 'gli', 'sipi',
             'ndri', 'ndvi_skew', 'gli_skew', 'sipi_skew', 'ndri_skew', 'cv',
             'lemon', 'mandarin', 'orange', 'cv^2', 'cv/center_height',
             'cv/surface_area', 'cv/total_foliage',
             'cv/fruit_foliage_ratio', 'orange_cv', 'orange_surface_area',
             'lemon_cv', 'lemon_surface_area',
             'mandarin_cv', 'mandarin_surface_area', 'center_width', 'cv/center_width']

X_tr_cols = ['total_foliage', 'total_orange', 'volume', 'surface_area', 'avg_volume',
             'center_perimeter', 'center_height', 'w_h_ratio',
             'med_intens_arr', 'fruit_foliage_ratio', 'mst_sums_arr', 'mst_mean_arr',
             'mst_skew_arr', 'clusters_area_mean_arr', 'clusters_area_med_arr',
             'clusters_ch_area_mean_arr', 'clusters_ch_area_med_arr',
             'n_clust_arr_2', 'q_1_precent_fruits', 'q_2_precent_fruits',
             'q_3_precent_fruits', 'fruit_dist_center', 'ndvi', 'gli', 'sipi',
             'ndri', 'ndvi_skew', 'gli_skew', 'sipi_skew', 'ndri_skew', 'cv',
             'lemon', 'mandarin', 'orange', 'cv^2', 'cv/center_height',
             'cv/surface_area', 'cv/total_foliage',
             'cv/fruit_foliage_ratio', 'orange_cv', 'orange_surface_area',
             'lemon_cv', 'lemon_surface_area',
             'mandarin_cv', 'mandarin_surface_area']

X_trees_cols = ["cv", "mst_mean_arr", "cv/surface_area", "orange_surface_area", "cv/fruit_foliage_ratio",
        "orange_cv", "cv/total_foliage", "cv/center_height",
        "w_h_ratio", "fruit_dist_center", "mst_sums_arr",
        "center_height", "center_perimeter", "clusters_area_med_arr", "mandarin", "lemon",
            "foliage_fullness"]

dt_chosen_cols = ["total_orange", "surface_area", "avg_volume", "w_h_ratio", "fruit_foliage_ratio", "mst_mean_arr",
                  "mst_skew_arr", "clusters_area_mean_arr", "clusters_area_med_arr", "gli", "ndri_skew", "cv",
                  "mandarin", "cv^2", "cv/center_height", "cv/surface_area", "cv/total_foliage", "lemon_cv",
                  "mandarin_surface_area"]

gbm_chosen = ['volume', 'surface_area', 'avg_volume', 'center_perimeter', 'w_h_ratio',
       'mst_mean_arr', 'clusters_area_mean_arr', 'clusters_ch_area_mean_arr',
       'clusters_ch_area_med_arr', 'n_clust_arr_2', 'q_1_precent_fruits',
       'ndri', 'sipi_skew', 'cv', 'orange', 'cv/surface_area',
       'cv/total_foliage', 'orange_cv', 'orange_surface_area',
       'cv/center_width']

rf_chosen = ['total_foliage', 'surface_area', 'center_perimeter', 'center_height',
       'mst_sums_arr', 'mst_mean_arr', 'clusters_area_mean_arr',
       'clusters_area_med_arr', 'clusters_ch_area_mean_arr',
       'q_1_precent_fruits', 'q_2_precent_fruits', 'q_3_precent_fruits',
       'fruit_dist_center', 'ndvi', 'gli', 'sipi_skew', 'cv', 'mandarin',
       'cv/surface_area', 'lemon_surface_area', 'mandarin_cv']

lasso_chosen = ['total_foliage', 'total_orange', 'volume', 'avg_volume',
       'center_perimeter', 'w_h_ratio', 'fruit_foliage_ratio', 'mst_sums_arr',
       'mst_mean_arr', 'clusters_area_mean_arr', 'clusters_ch_area_mean_arr',
       'clusters_ch_area_med_arr', 'n_clust_arr_2', 'q_3_precent_fruits',
       'ndvi_skew', 'ndri_skew', 'cv', 'lemon', 'mandarin', 'orange',
       'cv/surface_area', 'cv/fruit_foliage_ratio', 'orange_cv',
       'orange_surface_area', 'lemon_cv', 'lemon_surface_area',
       'cv/center_width']

F_model_models = {
    # "SVR": {"model_params": "models_studies/SVR_study.pkl",
    #         "X_data": "X_train.csv",
    #         "y_data": "y_train.csv",
    #         "output_path": os.path.join(prefix, "models_studies/SVR_pipe.pkl"),
    #         "scaler": "Robust"},

    # "KNeighborsRegressor": {"model_params": "models_studies/KNeighborsRegressor_study.pkl",
    #                         "X_data": "X_train.csv",
    #                         "y_data": "y_train.csv",
    #                         "output_path": os.path.join(model_folder, "models_studies/KNeighborsRegressor_pipe.pkl"),
    #                         "scaler": "Robust"},
    #
    # "KNeighborsRegressor_trees": {"model_params": "models_studies/KNeighborsRegressor_study.pkl",
    #                         "X_data": "X_train_trees.csv",
    #                         "y_data": "y_train.csv",
    #                         "output_path": os.path.join(model_folder, "models_studies/KNeighborsRegressor_pipe_trees.pkl"),
    #                         "scaler": "Robust"},

    "DecisionTreeRegressor": {"model_params": os.path.join(prefix_studies, "DecisionTreeRegressor_study.pkl"),
                              "gfs_train_cols": gfs_train_cols,
                              "columns": X_trees_cols,
                              "output_path": os.path.join(model_folder, "DecisionTreeRegressor.pkl"),
                              "preprocess_type": 0,
                              "model_pkl_path": ""},

    "DecisionTreeRegressor_gfs_chosen": {"model_params": os.path.join(prefix_studies, "DecisionTreeRegressor_study.pkl"),
                              "gfs_train_cols": "",
                              "columns": dt_chosen_cols,
                              "output_path": os.path.join(model_folder, "DecisionTreeRegressor_gfs_chosen.pkl"),
                              "preprocess_type": 0,
                              "model_pkl_path": ""},

    "RandomForestRegressor": {"model_params": os.path.join(prefix_studies, "study_rf_notebook.pkl"),
                              "gfs_train_cols": "",
                              "columns": X_trees_cols,
                              "output_path": os.path.join(model_folder, "rf_notebook.pkl"),
                              "preprocess_type": 0,
                              "model_pkl_path": ""},

    "RandomForestRegressor_gfs_chosen": {"model_params": os.path.join(prefix_studies, "study_rf_notebook.pkl"),
                              "gfs_train_cols": "",
                              "columns": rf_chosen,
                              "output_path": os.path.join(model_folder, "rf_notebook_gfs_chosen.pkl"),
                              "preprocess_type": 0,
                              "model_pkl_path": ""},

    "XGBRegressor": {"model_params": os.path.join(prefix_studies, "study_xgb_notebook.pkl"),
                     "gfs_train_cols": gfs_train_cols,
                     "columns": X_trees_cols,
                     "output_path": os.path.join(model_folder, "xgb_notebook.pkl"),
                     "preprocess_type": 0,
                     "model_pkl_path": ""},

    "XGBRegressor_local": {"model_params": os.path.join(prefix_studies, "XGBRegressor_study.pkl"),
                           "gfs_train_cols": gfs_train_cols,
                           "columns": X_trees_cols,
                           "output_path": os.path.join(model_folder, "xgb_local.pkl"),
                           "preprocess_type": 0,
                           "model_pkl_path": ""},

    "LassoPipe": {"model_params": {},
                  "gfs_train_cols": gfs_train_cols,
                  "columns": X_tr_cols,
                  "output_path": os.path.join(model_folder, "lasso_pipe.pkl"),
                  "preprocess_type": 0,
                  "model_pkl_path": ""},

    "LassoPipe_gfs": {"model_params": {},
                  "gfs_train_cols": "",
                  "columns": lasso_chosen,
                  "output_path": os.path.join(model_folder, "lasso_pipe.pkl"),
                  "preprocess_type": 0,
                  "model_pkl_path": ""},


    "GradientBoostingRegressor": {"model_params": os.path.join(prefix_studies, "GradientBoostingRegressor_study.pkl"),
                                  "gfs_train_cols": gfs_train_cols,
                                  "columns": X_trees_cols,
                                  "output_path": os.path.join(model_folder, "GradientBoostingRegressor.pkl"),
                                  "preprocess_type": 0,
                                  "model_pkl_path": ""},

    "GradientBoostingRegressor_gfs_chosen": {"model_params": os.path.join(prefix_studies,
                                                                          "GradientBoostingRegressor_study.pkl"),
                                  "gfs_train_cols": "",
                                  "columns": gbm_chosen,
                                  "output_path": os.path.join(model_folder, "GradientBoostingRegressor_gfs_chosen.pkl"),
                                  "preprocess_type": 0,
                                  "model_pkl_path": ""},

    "GradientBoostingRegressor_3107_trees": {
        "model_params": "/home/fruitspec-lab/FruitSpec/Code/roi/fsCounter/MHS/studies_31_07/GradientBoostingRegressor_trees_study.pkl",
        "gfs_train_cols": gfs_train_cols,
        "columns": X_trees_cols,
        "output_path": os.path.join(model_folder, "GradientBoostingRegressor_3107_trees.pkl"),
        "preprocess_type": 0,
        "model_pkl_path": ""},

    "HistGradientBoostingRegressor": {
        "model_params": os.path.join(prefix_studies, "HistGradientBoostingRegressor_study.pkl"),
        "gfs_train_cols": gfs_train_cols,
        "columns": X_trees_cols,
        "output_path": os.path.join(model_folder, "HistGradientBoostingRegressor.pkl"),
        "preprocess_type": 0,
        "model_pkl_path": ""},

    "HistGradientBoostingRegressor_3107_trees": {
        "model_params": "/home/fruitspec-lab/FruitSpec/Code/roi/fsCounter/MHS/studies_31_07/HistGradientBoostingRegressor_trees_study.pkl",
        "gfs_train_cols": gfs_train_cols,
        "columns": X_trees_cols,
        "output_path": os.path.join(model_folder, "HistGradientBoostingRegressor_3107_trees.pkl"),
        "preprocess_type": 0,
        "model_pkl_path": ""},

    "LGBMRegressor": {"model_params": os.path.join(prefix_studies, "LGBMRegressor_study.pkl"),
                      "gfs_train_cols": gfs_train_cols,
                      "columns": X_trees_cols,
                      "output_path": os.path.join(model_folder, "LGBMRegressor.pkl"),
                      "preprocess_type": 0,
                      "model_pkl_path": ""},
}
