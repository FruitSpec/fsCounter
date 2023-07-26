import os
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import json
from vision.feature_extractor.feature_extractor_class import FeatureExtractor, run_on_tree
from vision.feature_extractor.adt_result_loader import ADTSBatchLoader
from vision.misc.help_func import get_repo_dir, write_json, post_process_slice_df
from vision.pipelines.ops.simulator import get_assignments
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from vision.tools.manual_slicer import slice_to_trees_df
from vision.pipelines.ops.frame_loader import FramesLoader
from tqdm import tqdm
import time

def get_row_name(row_scan_path):
    """
    Get the row name from the given row scan path.

    Args:
        row_scan_path (str): Path to the row scan.

    Returns:
        str: The row name in the format "R<row_number>_S<scan_number>".
    """
    scan_number = os.path.basename(row_scan_path)
    row_number = os.path.basename(os.path.dirname(row_scan_path)).split("_")[1]
    return f"R{row_number}_S{scan_number}"


def get_block_name(row_scan_path):
    """
    Get the block name from the given row scan path.

    Args:
        row_scan_path (str): Path to the row scan.

    Returns:
        str: The block name.
    """
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(row_scan_path))))


def get_customer_name(row_scan_path):
    """
    Get the customer name from the given row scan path.

    Args:
        row_scan_path (str): Path to the row scan.

    Returns:
        str: The customer name.
    """
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(row_scan_path)))))


def update_args(args_adts, row_scan_path):
    args_adts.zed.movie_path = os.path.join(row_scan_path, f"ZED.svo")
    args_adts.depth.movie_path = os.path.join(row_scan_path, f"DEPTH.mkv")
    args_adts.rgb_jai.movie_path = os.path.join(row_scan_path, f"Result_RGB.mkv")
    args_adts.jai.movie_path = os.path.join(row_scan_path, f"Result_FSI.mkv")
    args_adts.sync_data_log_path = os.path.join(row_scan_path, f"jaized_timestamps.csv")
    if not os.path.exists(args_adts.sync_data_log_path):
        args_adts.sync_data_log_path = os.path.join(row_scan_path, f"jai_zed.json")
    metadata, metadata_path = get_metadata(row_scan_path)
    metadata_keys = metadata.keys()
    if "max_z" in metadata_keys:
        args_adts.max_z = metadata["max_z"]
    if "zed_rotate" in metadata_keys:
        args_adts.zed.rotate = metadata["zed_rotate"]
    return args_adts


def init_fe_obj(row_scan_path, tree_id, adts_loader=None):
    """
    Initialize the FeatureExtractor object and ADTSBatchLoader object.

    Args:
        row_scan_path (str): Path to the row scan.
        tree_id (int): Tree ID.

    Returns:
        tuple: A tuple containing the FeatureExtractor object, ADTSBatchLoader object, and batch size.
    """
    row_name = get_row_name(row_scan_path)
    block_name = get_block_name(row_scan_path)
    repo_dir = get_repo_dir()
    fe_args = OmegaConf.load(repo_dir + "/vision/feature_extractor/feature_extractor_config.yaml")
    fe = FeatureExtractor(fe_args, tree_id, row_name, block_name)
    args_adts = OmegaConf.load(repo_dir + "/vision/pipelines/config/dual_runtime_config.yaml")
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    cfg.batch_size = fe_args.batch_size
    args_adts = update_args(args_adts, row_scan_path)
    if isinstance(adts_loader, type(None)):
        adts_loader = ADTSBatchLoader(cfg, args_adts, block_name, row_scan_path, tree_id=tree_id)
    else:
        adts_loader.tree_id = tree_id
        adts_loader.load_dfs()
    batch_size = fe_args.batch_size
    return fe, adts_loader, batch_size


def create_tree_features(fe, row_scan_path):
    """
     Create tree features dictionary from the FeatureExtractor object.

     Args:
         fe (FeatureExtractor): The FeatureExtractor object.
         row_scan_path (str): Path to the row scan.

     Returns:
         dict: Dictionary containing tree features.
     """
    tree_features = {k: v for features in fe.get_tree_results() for k, v in features.items()}
    tree_features["block_name"] = fe.block
    tree_features["name"] = fe.tree_name
    tree_features["customer"] = get_customer_name(row_scan_path)
    return tree_features


def run_on_row(row_scan_path, suffix="", print_fids=False):
    """
    Process a row scan.

    Args:
        row_scan_path (str): Path to the row scan.
        suffix (str, optional): Suffix for the output files. Defaults to "".
        print_fids (bool, optional): Whether to print frame IDs. Defaults to False.

    Returns:
        list: List of tree features dictionaries for the row scan.
    """
    slices_path = os.path.join(row_scan_path, "slices.csv")
    if os.path.exists(slices_path):
        slices = pd.read_csv(slices_path)
    else:
        slices = pd.read_csv(os.path.join(row_scan_path, "all_slices.csv"))

    trees = slices["tree_id"].unique()
    row_tree_features = []
    adts_loader = None
    sucess = True
    for tree_id in trees:
        if tree_id < 0:
            continue
        #try:
        tree_frames = slices["frame_id"][slices["tree_id"] == tree_id].apply(str).values
        s = time.time()
        fe, adts_loader, batch_size = init_fe_obj(row_scan_path, tree_id, adts_loader)
        e = time.time()
        print(f'init_time: {e-s}')
        tree_time = run_on_tree(tree_frames, fe, adts_loader, batch_size, print_fids)
        tree_time = pd.DataFrame(tree_time)
        tree_time.to_csv(os.path.join("/home/matans/Documents/fruitspec/sandbox/FE/time_test", 'tree_time.csv'))
        tree_features = create_tree_features(fe, row_scan_path)
        row_tree_features.append(tree_features)
        # except:
        #     print("bug: ", row_scan_path)
        #     sucess = False
    if sucess:
        row_post_process(row_scan_path, row_tree_features, suffix)
    return row_tree_features


def row_post_process(row_scan_path, row_tree_features, suffix=""):
    """
    Perform post-processing steps after processing a row scan:
        1. Saves the row_featuers to a csv
        2. updates metadata file

    Args:
        row_scan_path (str): Path to the row scan.
        row_tree_features (list): List of tree features dictionaries.
        suffix (str, optional): Suffix for the output files. Defaults to "".
    """
    if suffix != "":
        if not suffix.startswith("_"):
            suffix = "_" + suffix
    if len(row_tree_features):
        pd.DataFrame(row_tree_features).to_csv(os.path.join(row_scan_path, f"row_features{suffix}.csv"), index=False)
    metadata, metadata_path = get_metadata(row_scan_path)
    metadata["tree_features"] = False
    write_json(metadata_path, metadata)
    print("Done with: ", row_scan_path)


def get_metadata(row_scan_path):
    """
    Get the metadata and metadata path for a row scan.

    Args:
        row_scan_path (str): Path to the row scan.

    Returns:
        tuple: A tuple containing the metadata dictionary and metadata path.
    """
    metadata_path = os.path.join(row_scan_path, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    return metadata, metadata_path


def update_processed_data(process_data, row_scan_path):
    """
    Update the processed data list with the row scan features.

    Args:
        process_data (list): List of processed data.
        row_scan_path (str): Path to the row scan.
    """
    try:
        df = pd.read_csv(os.path.join(row_scan_path, "row_features.csv"))
    except:
        print("empty df: ", row_scan_path)
        return
    if "Unnamed: 0" in df:
        df.drop("Unnamed: 0", axis=1, inplace=True)
    for i in range(len(df)):
        process_data.append([df.iloc[i, :].to_dict()])


def validate_slice_data(row_scan_path, min_slice_len=5):
    """
    Validate the existence and format of slice data for a row scan.

    Args:
        row_scan_path (str): Path to the row scan.
        min_slice_len (int): a valid slice must have more than 'min_slice_len' trees

    Returns:
        bool: True if slice data is valid, False otherwise.
    """
    slices_path = os.path.join(row_scan_path, "slices.csv")
    all_slices_path = os.path.join(row_scan_path, "all_slices.csv")
    slice_data_jai = os.path.join(row_scan_path, "Result_FSI_slice_data.json")
    slice_data_rgb = os.path.join(row_scan_path, "Result_RGB_slice_data.json")
    if os.path.exists(slices_path):
        slice_df = pd.read_csv(slices_path)
        if slice_df["tree_id"].max() < min_slice_len:
            return False
        return True
    if os.path.exists(all_slices_path):
        slice_df = pd.read_csv(all_slices_path)
    elif os.path.exists(slice_data_jai):
            slice_df = slice_to_trees_df(slice_data_jai, row_scan_path)
    elif os.path.exists(slice_data_rgb):
        slice_df = slice_to_trees_df(slice_data_rgb, row_scan_path)
    else:
        return False
    slice_df = post_process_slice_df(slice_df)
    slice_df.to_csv(slices_path)
    if slice_df["tree_id"].max() < min_slice_len:
        return False
    return True


def validate_jai_zed_json(row_scan_path):
    """
    Validate the JAI-ZED JSON data for a row scan.

    Args:
        row_scan_path (str): Path to the row scan.

    Returns:
        bool: True if JAI-ZED JSON data is valid, False otherwise.
    """
    jai_zed_path = os.path.join(row_scan_path, "jai_zed.json")
    csv_path = os.path.join(row_scan_path, "jaized_timestamps.csv")
    log_path = os.path.join(row_scan_path, "jaized_timestamps.log")
    jai_zed_dict = {}
    if os.path.exists(jai_zed_path):
        return True
    if os.path.exists(csv_path):
        zed_ids, jai_ids = FramesLoader.get_cameras_sync_data(csv_path)
        jai_zed_dict = dict(zip(jai_ids, zed_ids))
    elif os.path.exists(log_path):
        zed_ids, jai_ids = FramesLoader.get_cameras_sync_data(log_path)
        jai_zed_dict = dict(zip(jai_ids, zed_ids))
    if not len(jai_zed_dict):
        return False
    write_json(jai_zed_path, jai_zed_dict)
    return True


def get_valid_row_paths(master_folder, over_write=False, run_only_done_adt=True, min_slice_len=5):
    """
    Get valid row scan paths for analysis from a master folder.

    Args:
        master_folder (str): Path to the master folder.
        over_write (bool, optional): Whether to overwrite existing features. Defaults to False.

    Returns:
        tuple: A tuple containing a list of valid row scan paths and a list of processed data.
    """
    paths_list = []
    process_data = []
    for root, dirs, files in os.walk(master_folder):
        if np.all([file in files for file in ["alignment.csv", "tracks.csv"]]):
            row_scan_path = os.path.abspath(root)
            metadata, _ = get_metadata(row_scan_path)
            align_detect_track, tree_features = get_assignments(metadata)
            try:
                slices_validation = validate_slice_data(row_scan_path, min_slice_len)
            except:
                print("bug: ", row_scan_path)
                continue
            if not slices_validation:
                print("no slices found for: ", row_scan_path)
                continue
            jai_zed_validation = validate_jai_zed_json(row_scan_path)
            if not jai_zed_validation:
                print("no jai_zed alignment found for: ", row_scan_path)
                continue
            #if align_detect_track and run_only_done_adt:
            #    continue
            if tree_features or over_write:
                paths_list.append(row_scan_path)
            else:
                update_processed_data(process_data, row_scan_path)
    return paths_list, process_data


def run_on_folder(master_folder, over_write=False, njobs=1, suffix="", print_fids=False, run_only_done_adt=False,
                  min_slice_len=5):
    """
    Process all row scans in a master folder.

    Args:
        master_folder (str): Path to the master folder.
        over_write (bool, optional): Whether to overwrite existing features. Defaults to False.
        njobs (int, optional): Number of parallel jobs. Defaults to 1.
        suffix (str, optional): Suffix for the output files. Defaults to "".
        print_fids (bool, optional): Whether to print frame IDs. Defaults to False.

    Returns:
        list: List of tree features dictionaries for all row scans.
    """
    paths_list, process_data = get_valid_row_paths(master_folder, over_write, run_only_done_adt, min_slice_len)
    n = len(paths_list)
    if njobs > 1:
        with ProcessPoolExecutor(max_workers=njobs) as executor:
            res = list(executor.map(run_on_row, paths_list, [suffix]*n, [print_fids]*n))
    else:
        res = list(map(run_on_row, paths_list, [suffix] * n, [print_fids] * n))
    return list(chain.from_iterable(res + process_data))

def get_tree_features(fe):
    return {k: v for features in fe.get_tree_results() for k, v in features.items()}

def run(cfg, fe_args, adt_args, slices, row):
    """
    Process a row scan.

    Args:
        row_scan_path (str): Path to the row scan.
        suffix (str, optional): Suffix for the output files. Defaults to "".
        print_fids (bool, optional): Whether to print frame IDs. Defaults to False.

    Returns:
        list: List of tree features dictionaries for the row scan.
    """

    block_name = row['plot_code']
    row_id = row['row']

    #fe = FeatureExtractor(fe_args, 0, row_id, block_name)
    adts_loader = ADTSBatchLoader(cfg, fe_args, block_name, adt_args.output_folder, tree_id=0)

    trees = slices["tree_id"].unique()
    row_tree_features = []
    for tree_id in trees:
        #fe.reset(fe_args, tree_id, row_id, block_name)
        s = time.time()
        fe = FeatureExtractor(fe_args, tree_id, row_id, block_name)
        fe, adts_loader, tree_frames = update_tree_data(fe, adts_loader, slices, tree_id)
        e = time.time()
        print('init tree FE and tree data')
        tree_time = run_on_tree(tree_frames, fe, adts_loader, cfg.batch_size)
        row_tree_features.append(get_tree_features(fe))



    if len(row_tree_features):
        row_tree_features = pd.DataFrame(row_tree_features)

    return row_tree_features

def update_tree_data(fe, adts_loader, slices, tree_id):
    fe.tree_id = tree_id
    adts_loader.tree_id = tree_id
    adts_loader.slicer_results = slices[slices["tree_id"] == tree_id]
    tree_frames = slices["frame_id"][slices["tree_id"] == tree_id].apply(str).values

    return fe, adts_loader, tree_frames

def update_run_args(fe_args, adt_args, row_folder):
    adt_args.zed.movie_path = os.path.join(row_folder, f"ZED.mkv")
    adt_args.depth.movie_path = os.path.join(row_folder, f"DEPTH.mkv")
    adt_args.jai.movie_path = os.path.join(row_folder, f"Result_FSI.mkv")
    adt_args.rgb_jai.movie_path = os.path.join(row_folder, f"Result_RGB.mkv")
    adt_args.output_folder = row_folder

    fe_args.zed.movie_path = os.path.join(row_folder, f"ZED.svo")
    fe_args.jai.movie_path = os.path.join(row_folder, f"Result_FSI.mkv")
    fe_args.rgb_jai.movie_path = os.path.join(row_folder, f"Result_RGB.mkv")
    fe_args.tracker_results = os.path.join(row_folder, f"tracks.csv")
    fe_args.alignment = os.path.join(row_folder,"alignment.csv")
    fe_args.jai_translations = os.path.join(row_folder, "jai_translations.csv")
    fe_args.sync_data_log_path = os.path.join(row_folder, "jaized_timestamps.csv")

    slices = pd.read_csv(os.path.join(row_folder, "slices.csv"))


    return fe_args, adt_args, slices



if __name__ == '__main__':
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    fe_config = "/vision/feature_extractor/feature_extractor_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    adt_args = OmegaConf.load(repo_dir + runtime_config)
    fe_args = OmegaConf.load(repo_dir + fe_config)
    folder_path = "/media/matans/My Book/FruitSpec/Customers_data/Fowler/daily/OLIVER12/180723/row_10/1"
    output_path = "/media/matans/My Book/FruitSpec/Customers_data/Fowler/daily/OLIVER12/180723/row_10/1/features.csv"
    row = {'plot_code': 'test_data',
           'row': 10}
    over_write = True
    njobs = 1
    suffix = ""
    print_fids = False
    run_only_done_adt = True
    min_slice_len = 5

    #fe_args, adt_args, slices = update_run_args(fe_args, adt_args, folder_path)
    #features_df = run(cfg, fe_args, adt_args, slices, row)
    #
    #features_df.to_csv(output_path)
    final_df_output = "/home/mic-730ai/fruitspec/test_data/MOTCHA_features_orig.csv"
    results = run_on_folder(folder_path, over_write, njobs, suffix, print_fids, run_only_done_adt)
    pd.DataFrame(results).to_csv(output_path)
    pd.DataFrame(results)