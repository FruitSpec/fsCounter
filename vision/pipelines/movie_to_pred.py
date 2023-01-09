from vision.pipelines.movies_to_trees_pipe import preprocess_videos_to_trees_aligmnet_fix
import os
import vision.feature_extractor.feature_extractor as feature_extractor
import pickle
import pandas as pd
from MHS.model_hybrid_selector import model_hybrid_selector
from MHS.pre_post_processing import preprocess


# TODO add a way to continue process if it failed, based on exisiting files- maybe creating a log where we are is better...


def process_plot(plot_path, model=None, block_name="", max_z=5, zed_shift=0, max_x=600, max_y=900, rows_skips={},
                 overwrite=False):
    all_rows = [row for row in os.listdir(plot_path) if os.path.isdir(os.path.join(plot_path, row))]
    skip_keys = list(rows_skips.keys())
    for row in all_rows:
        if row in skip_keys:
            row_skip = rows_skips[row]
        else:
            row_skip = []
        row_path = os.path.join(plot_path, row)
        if not overwrite:
            if os.path.exists(os.path.join(row_path, "trees", "row_features.csv")):
                continue
        preprocess_videos_to_trees_aligmnet_fix(row_path, zed_shift=zed_shift,
                                            zed_roi_params=dict(y_s=None, y_e=None, x_s=0, x_e=None),skip_steps=row_skip)
    path_to_plot_features = os.path.join(plot_path, "plot_features.csv")
    if (not overwrite) and os.path.exists(path_to_plot_features):
        features_df = pd.read_csv(path_to_plot_features)
    else:
        features_df = feature_extractor.create_plot_features(plot_path, zed_shift=zed_shift, max_x=max_x, max_y=max_y,
                                                             save_csv=True, block_name=block_name, max_z=max_z)
    if not isinstance(model, type(None)):
        if isinstance(model, str):
            model_str = model
            with open(model_str, 'rb') as f:
                model = pickle.load(f)
        features_df_porc = preprocess(features_df)
        preds = model.predict(features_df_porc)
        preds_out_df = features_df[["block_name", "name"]]
        preds_out_df["pred"] = preds
        preds_out_df.to_csv(os.path.join(plot_path, "predictions.csv"))
    return preds_out_df


def process_customer(customer_path, model=None, customer_name="",
                     max_z={"defult": 5}, zed_shift=0, max_x=600, max_y=900, skip_dict={}, overwrite=False):
    all_plots = [plot for plot in os.listdir(customer_path) if os.path.isdir(os.path.join(customer_path, plot))]
    preds_out_df = pd.DataFrame({})
    skip_keys = list(skip_dict.keys())
    max_z_keys = list(max_z.keys())
    for plot in all_plots:
        plot_path = os.path.join(customer_path, plot)
        if plot in skip_keys:
            rows_skips = skip_dict[plot]
        if plot in max_z_keys:
            max_z_plot = max_z[plot]
        else:
            max_z_plot = max_z["defult"]
        preds_out_df_tmp = process_plot(plot_path, model=model, block_name=plot, max_z=max_z_plot, zed_shift=zed_shift,
                                        max_x=max_x, max_y=max_y, rows_skips=rows_skips, overwrite=overwrite)
        preds_out_df = pd.concat([preds_out_df, preds_out_df_tmp], ignore_index=True)
    if not isinstance(model, type(None)):
        preds_out_df.to_csv(os.path.join(customer_path, "predictions.csv"))


if __name__ == "__main__":
    #TODO important data from outsource: cloudiness, 
    skip_dict = {"test_pred_pipe": {"R2": ["folder_to_frames", "align_folder", "agg_to_trees", "track_row"],
                                    "R3": ["folder_to_frames", "align_folder", "agg_to_trees", "track_row"]}}
    max_z_dict = {"test_pred_pipe": 5}
    process_plot("/media/fruitspec-lab/easystore/test_pred_pipe",
                 '/home/fruitspec-lab/PycharmProjects/foliage/counter/model.pkl',
                 max_z=5,rows_skips=skip_dict["test_pred_pipe"])

