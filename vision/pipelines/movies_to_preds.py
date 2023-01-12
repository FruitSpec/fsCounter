from vision.pipelines.movies_to_trees_pipe import preprocess_videos_to_trees_aligmnet_fix
import os
import vision.feature_extractor.feature_extractor as feature_extractor
import pickle
import pandas as pd
from MHS.model_hybrid_selector import model_hybrid_selector
from MHS.pre_post_processing import preprocess
from vision.feature_extractor import feature_extractor as feat_e
import pickle
import joblib


def get_prediction_df(features_df,scan_date, customer_name, path_to_plot, model = None):
    pred_frame = pd.DataFrame({})
    if not isinstance(model, type(None)):
        features_df_processed = preprocess(features_df)
        preds = model.predict(features_df_processed)
    else:
        preds = features_df["cv"] * 4
    pred_frame["preds"] = preds
    pred_frame["scan_date"] = scan_date
    pred_frame["customer_name"] = customer_name
    pred_frame[["name", "block_name"]] = features_df[["name", "block_name"]]
    pred_frame.to_csv(os.path.join(path_to_plot, "preds.csv"))
    return pred_frame

def plot_to_preds(path_to_plot, block_name, model=None, zed_shift={"default": 0}, max_z=5,
                  zed_roi_params={"default": dict(x_s=0, x_e=1080, y_s=310, y_e=1670)}, skip_steps=[],
                  scan_date=0, customer_name="customer_plot", skip_rows=[]):
    zed_shift_keys = list(zed_shift.keys())
    skip_steps_keys = list(skip_steps.keys())
    defualt_shift = zed_shift["default"]
    for row in os.listdir(path_to_plot):
        row_path = os.path.join(path_to_plot, row)
        zed_shift_row = zed_shift[row] if row in zed_shift_keys else defualt_shift
        skip_steps_row = skip_steps[row] if row in skip_steps_keys else []
        if os.path.isdir(row_path):
            preprocess_videos_to_trees_aligmnet_fix(row_path, zed_shift=zed_shift_row,
                                                    zed_roi_params=zed_roi_params, skip_steps=skip_steps_row)
    features_df = feat_e.create_plot_features(path_to_plot, block_name=block_name, max_z=max_z, skip_rows=skip_rows)
    pred_frame = get_prediction_df(features_df, scan_date, customer_name, path_to_plot, model=None)
    return pred_frame, features_df


def customer_to_preds(customer_path, customer_cnfg, model=None):
    all_preds = pd.DataFrame({})
    scan_dates = os.listdir(customer_path)
    customer_name = os.path.basename(customer_path)
    for scan in scan_dates:
        scan_path = os.path.join(customer_path, scan)
        if not os.path.isdir(scan_path):
            continue
        for customer_plot in os.listdir(scan_path):
            plot_path = os.path.join(scan_path, customer_plot)
            if os.path.isdir(plot_path):
                zed_shift = customer_cnfg["zed_shift_dict"][scan][customer_plot]
                zed_roi_params = customer_cnfg["roi_parms_dict"][scan]
                max_z = customer_cnfg["max_depth_dict"][scan][customer_plot]
                skip_steps = customer_cnfg["skip_rows_dict"][scan][customer_plot]
                skip_rows = customer_cnfg["skip_rows"][scan][customer_plot]
                plot_preds, plot_features = plot_to_preds(plot_path, customer_plot, model=model,
                                                          zed_shift=zed_shift, max_z=max_z,
                                                          zed_roi_params=zed_roi_params, skip_steps=skip_steps,
                                                          scan_date=scan, customer_name=customer_name,
                                                          skip_rows=skip_rows)
                all_preds = pd.concat([all_preds, plot_preds])
    all_preds.to_csv(os.path.join(customer_path, "preds.csv"))


if __name__ == '__main__':
    customer_path = "/media/fruitspec-lab/easystore/test_pred_pipe"
    skip_rows_dict = {"301022": {"test_plot": {"R2": ["folder_to_frames", "align_folder", "agg_to_trees", "track_row"],
                                               "R3": ["folder_to_frames", "align_folder", "agg_to_trees", "track_row"]}}}
    max_depth_dict = {"301022": {"test_plot": 5}}
    roi_parms_dict = {"301022": dict(x_s=0, x_e=1080, y_s=310, y_e=1670),
                      "default": dict(x_s=0, x_e=1080, y_s=310, y_e=1670)}
    zed_shift_dict = {"301022": {"test_plot": {"R2": 0,
                                               "R3": 0,
                                               "default": 0}}}
    skip_rows = {"301022": {"test_plot": []}}
    customer_cnfg = {"skip_rows_dict": skip_rows_dict,
           "max_depth_dict": max_depth_dict,
           "roi_parms_dict": roi_parms_dict,
           "zed_shift_dict": zed_shift_dict,
           "skip_rows": skip_rows}
    model = "/home/fruitspec-lab/FruitSpec/Code/fsCounter/model_mhs.pkl"
    customer_to_preds(customer_path, customer_cnfg, model=model)
    print("Done")