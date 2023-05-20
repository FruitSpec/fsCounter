import os
import shutil
import json
import pandas as pd

def scale(det_dims, frame_dims):
    r = min(det_dims[0] / frame_dims[0], det_dims[1] / frame_dims[1])
    return (1 / r)


def scale_det(detection, scale_):
    # Detection ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    # Detection ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    x1 = int(detection[0] * scale_)
    y1 = int(detection[1] * scale_)
    x2 = int(detection[2] * scale_)
    y2 = int(detection[3] * scale_)
    obj_conf = detection[4]
    class_conf = detection[5]
    class_pred = detection[6]

    # res ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred, image_id)
    return [x1, y1, x2, y2, obj_conf, class_conf, class_pred]


def scale_dets(det_outputs, scale_):

    dets = []
    for frame_dets in det_outputs:
        if frame_dets is None:
            dets.append([])
        else:
            scales = [scale_ for _ in frame_dets]
            scaled_dets = list(map(scale_det, frame_dets.cpu().numpy(), scales))
            dets.append(scaled_dets)

    return dets


def get_repo_dir():
    cwd = os.getcwd()
    if 'Windows' in os.environ['OS']:
        splited = cwd.split('\\')
    else:
        splited = cwd.split('/')
    ind = splited.index('fsCounter')
    repo_dir = '/'
    for s in splited[1:ind + 1]:
        repo_dir = os.path.join(repo_dir, s)

    return repo_dir


def validate_output_path(output_folder, flag=1):
    if flag == 0:
        return
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

def copy_configs(pipeline_config, runtime_config, output_path):
    shutil.copy(pipeline_config, os.path.join(output_path, "pipeline_config.yaml"))
    shutil.copy(runtime_config, os.path.join(output_path, "runtime_config.yaml"))


def read_json(filepath):
    """
    this function reads a json from filepath
    :param filepath: path to file
    :return: the read json
    """
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r') as f:
        loaded_data = json.load(f)
    return loaded_data


def load_json(filepath):
    """
    this function reads the json file and converts the keys to ints
    :param filepath: path to file
    :return: the converted json
    """
    if not os.path.exists(filepath):
        return {}
    loaded_data = read_json(filepath)
    data = {}
    for k, v in loaded_data.items():
        data[int(k)] = v

    return data

def write_json(file_path, data):

    with open(file_path, 'w') as f:
        json.dump(data, f)


def post_process_slice_df(slice_df):
    """
    Post processes the slices dataframe - if not all frames of the tree are on the json file they are not
        added to the data frame, this function fills in the missing trees with start and end value of -1

    Args:
        slice_df (pd.DataFrame): A dataframe contatining frame_id, tree_id, start, end

    Returns:
        (pd.DataFrame): A post process dataframe
    """
    row_to_add = []
    for tree_id in slice_df["tree_id"].unique():
        temp_df = slice_df[slice_df["tree_id"] == tree_id]
        min_frame, max_frame = temp_df["frame_id"].min(), temp_df["frame_id"].max()
        temp_df_frames = temp_df["frame_id"].values
        for frame_id in range(min_frame, max_frame +1):
            if frame_id not in temp_df_frames:
                row_to_add.append({"frame_id": frame_id, "tree_id": tree_id, "start": -1 ,"end": -1})
    return pd.concat([slice_df, pd.DataFrame.from_records(row_to_add)]).sort_values("frame_id")


def update_save_log(log_path, log, update_vals):
    if update_vals:
        for key, val in update_vals.items():
            log[key] = val
    with open(log_path, "w") as json_file:
        json.dump(log, json_file)
    return log


def safe_read_csv(file_path: str) -> pd.DataFrame:
    """
    read csv file, if path does not exists it will return an empty csv
    Args:
        file_path (str): path for dataframe

    Returns:
        pd.DataFrame
    """
    if not os.path.exists(file_path):
        return pd.DataFrame({})
    return pd.read_csv(file_path)


def pop_list_drom_dict(x_dict: dict, pop_list: list) -> dict:
    """
    pops each element of the list from the dictionary
    Args:
        x_dict (dict): dictionary to pop elements from
        pop_list (list): list of values to pop

    Returns:
        dict: new dict with the values popped
    """
    x_dict = x_dict.copy()
    for item in pop_list:
        if item in x_dict.keys():
            x_dict.pop(item)
    return x_dict