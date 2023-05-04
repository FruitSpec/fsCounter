import os
import shutil
import json

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