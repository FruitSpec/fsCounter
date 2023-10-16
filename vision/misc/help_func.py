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
        os.makedirs(output_folder)

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

def go_up_n_levels(path, n):
    current_path = os.path.abspath(path)
    for _ in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

def contains_special_char(s, special_chars=['^', '/', '*']):
    return any(char in s for char in special_chars)


def reset_adt_metadata(folder_path="/media/fruitspec-lab/cam175/customers/DEWAGD"):
    for root, dirs, files in os.walk(folder_path):
        if "metadata.json" in files:
            metadata_path = os.path.join(root, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata["align_detect_track"] = True
            write_json(metadata_path, metadata)

def reset_tree_features(folder_path="/media/fruitspec-lab/cam175/customers/DEWAGD"):
    for root, dirs, files in os.walk(folder_path):
        if "metadata.json" in files:
            metadata_path = os.path.join(root, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata["tree_features"] = True
            write_json(metadata_path, metadata)

def reset_direction(folder_path="/media/fruitspec-lab/cam175/customers/DEWAGD", direction=""):
    for root, dirs, files in os.walk(folder_path):
        if "metadata.json" in files:
            metadata_path = os.path.join(root, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata["direction"] = direction
            write_json(metadata_path, metadata)


def reset_metadata(folder_path="/media/fruitspec-lab/cam175/customers/DEWAGD", complete=False):
    if complete:
        for root, dirs, files in os.walk(folder_path):
            if "metadata.json" in files:
                metadata_path = os.path.join(root, "metadata.json")
                metadata = {}
                write_json(metadata_path, metadata)
    else:
        reset_tree_features(folder_path)
        reset_adt_metadata(folder_path)


def modify_calibration_data(file_path, frame_start, frame_end, shift, fixed=False, reset_meta=True):
    # Load the JSON file
    if "jai_zed" not in file_path:
        file_path = os.path.join(file_path, "jai_zed.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

    # Modify the values
    for frame in range(frame_start, frame_end+1):
        if not fixed:
            if str(frame) in data:
                data[str(frame)] += shift
        else:
            data[str(frame)] = frame + shift

    # Save the modified data back to the JSON file
    with open(file_path, 'w') as f:
        json.dump(data, f)

    if reset_meta:
        reset_adt_metadata(os.path.dirname(file_path))
        reset_metadata(os.path.dirname(file_path))

    print(f"{file_path} data modified successfully.")


def find_folders_with_file(root_dir, file_name):
    # Initialize an empty list to hold the directories
    dirs_with_file = []
    # Walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if file_name in filenames:
            # If the file is found, add the directory path to the list
            dirs_with_file.append(dirpath)
    return dirs_with_file
    # # Test the function
    # root_dir = "/media/fruitspec-lab/cam175/FOWLER"  # Replace with your directory path
    # file_name = "row_features.csv"
    # for file in find_folders_with_file(root_dir, file_name):
    #     print(file)


def delete_files_with_name(starting_directory, file_name):
    for root, _, files in os.walk(starting_directory):
        for file in files:
            if file == file_name:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")


if __name__ == "__main__":
    folder_path = "/media/fruitspec-lab/cam175/customers_new"
    reset_metadata(folder_path="/media/fruitspec-lab/TEMP SSD/Tomato/PackoutDataNondealeaf/pre")
    # delete_files_with_name(folder_path, "row_features.csv")
    data_path = ""
    # modify_calibration_data("/media/fruitspec-lab/cam175/customers_new/LDCBRA/LDC42200/190423/row_7/1", 3343, 3385, 2)

    # modify_calibration_data("/media/fruitspec-lab/cam175/customers_new/LDCBRA/LDC42200/190423/row_7/2", 100, 1837, -30)
    # modify_calibration_data("/media/fruitspec-lab/cam175/customers_new/LDCBRA/LDC42200/190423/row_7/2", 3950, 4010, -38)
    # modify_calibration_data("/media/fruitspec-lab/cam175/customers_new/LDCBRA/LDC42200/190423/row_7/2", 4300, 4500, -57)
    # modify_calibration_data("/media/fruitspec-lab/cam175/customers_new/LDCBRA/LDC42200/190423/row_7/2", 4700, 4910, -55)
    # modify_calibration_data("/media/fruitspec-lab/cam175/customers_new/LDCBRA/LDC42200/190423/row_15/1", 4100, 4250, 1)
    # modify_calibration_data("/media/fruitspec-lab/cam175/customers_new/LDCBRA/LDC42200/190423/row_15/2", 1422, 1460, -2)
    # modify_calibration_data("/media/fruitspec-lab/cam175/customers_new/LDCBRA/LDC42200/190423/row_28/1", 4989, 5050, 1)
    # modify_calibration_data("/media/fruitspec-lab/cam175/customers_new/LDCBRA/LDC42200/190423/row_28/2", 300, 500, -2)
    # modify_calibration_data("/media/fruitspec-lab/cam175/customers_new/LDCBRA/LDC42200/190423/row_46/1", 1276, 4000, -1)
    # modify_calibration_data("/media/fruitspec-lab/cam175/customers_new/LDCBRA/LDC42200/190423/row_46/2", 1500, 4200, 2)

    print("Done")