import os
import cv2
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import shutil
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from vision.misc.help_func import get_repo_dir, validate_output_path, post_process_slice_df
from vision.data.results_collector import ResultsCollector
from vision.pipelines.adt_pipeline import run as adt_run
from vision.pipelines.fe_pipeline import run as fe_run
from vision.pipelines.ops.simulator import update_arg_with_metadata, get_jai_drops, get_max_cut_frame, load_logs
from vision.pipelines.ops.simulator import init_cams
from vision.pipelines.ops.debug_methods import plot_alignmnt_graph, draw_on_tracked_imgaes
from vision.depth.slicer.slicer_flow import post_process
from vision.tools.manual_slicer import slice_to_trees_df

np.random.seed(123)
cv2.setRNGSeed(123)

def run_on_rows_inside_loop(args, cfg, block_name, key, row_runtime_params):
    """
    Runs the pipeline for a row.

    Args:
        args (Namespace): Namespace containing command line arguments.
        cfg (Namespace): Namespace containing configuration data.
        block_name (str): Name of the block being processed.
        key (str): Index of the row being processed.
        row_runtime_params (Namespace): Namespace containing runtime parameters for the row being processed.

    Returns:
        None
    """
    print("starting ", row_runtime_params["jai_movie_path"])
    run_args = update_args_with_row_runtime_params(args.copy(), row_runtime_params, block_name, key)
    validate_output_path(run_args.output_folder)

    run_args, metadata = update_arg_with_metadata(run_args)

    slice_data_zed, slice_data_jai = load_logs(run_args)
    all_slices_path = os.path.join(os.path.dirname(args.slice_data_path), "all_slices.csv")
    metadata['max_cut_frame'] = str(get_max_cut_frame(run_args, slice_data_jai, slice_data_zed))

    align_detect_track, tree_features = get_assignments(metadata)

    # perform align -> detect -> track
    if align_detect_track or run_args.overwtire.adt:
        rc = adt_run(cfg, run_args, metadata)
    else:
        rc = ResultsCollector(rotate=run_args.rotate)
        rc.set_self_params(run_args.output_folder, parmas=["alignment", "jai_zed", "detections",
                                                       "tracks", "percent_seen"])
    adt_slice_postprocess(run_args, rc, slice_data_zed, slice_data_jai, all_slices_path)
    if run_args.debug.align_graph:
        plot_alignmnt_graph(run_args, rc, get_jai_drops(run_args.frame_drop_path))

    # perform features extraction
    if tree_features or run_args.overwtire.trees:
        fe_run(run_args)
    print("finished ", row_runtime_params["jai_movie_path"])

def run_on_rows(input_dict, exclude=[], block_name="", njobs=1):
    """
    Runs the pipeline on each row of input data.

    Args:
        input_dict (dict): A dictionary with keys representing row indices and values as input data for the pipeline.
        exclude (list, optional): A list of row indices to be excluded from the pipeline run. Defaults to [].
        block_name (str, optional): The name of the processing block. Defaults to "".
        njobs (int, optional): The number of parallel jobs to run. Defaults to 1.

    Returns:
        float: max_z in the pipeline arguments.
    """
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)
    input_dict = {k: input_dict[k] for k in sorted(input_dict)} # sort it
    keys_list, row_rt_params_list = [], []
    for key, row_runtime_params in input_dict.items():
        if key in exclude:
            continue
        keys_list.append(key)
        row_rt_params_list.append(row_runtime_params)
    n = len(keys_list)
    args_list, cfg_list, block_name_list = [args] * n, [cfg] * n, [block_name] * n
    if njobs > 1:
        with ProcessPoolExecutor(max_workers=njobs) as executor:
            res = executor.map(run_on_rows_inside_loop, args_list, cfg_list, block_name_list, keys_list, row_rt_params_list)
    else:
        list(map(run_on_rows_inside_loop, args_list, cfg_list, block_name_list, keys_list, row_rt_params_list))
    return args.max_z

def adt_slice_postprocess(args, results_collector, slice_data_zed, slice_data_jai, all_slices_path):
    """
    Performs post-processing on slice data to generate trees and draw tracks on images if in debug mode.

    Args:
        args (dict): A dictionary of runtime parameters.
        results_collector (ResultsCollector): An object for collecting results.
        slice_data_zed (dict): A dictionary containing ZED slice data.
        slice_data_jai (dict): A dictionary containing JAI slice data.
        all_slices_path (str): The path to a CSV file containing slice data.

    Returns:
        None
    """
    if slice_data_zed or slice_data_jai or os.path.exists(all_slices_path):
        if slice_data_jai:
            """this function depends on how we sliced (before or after slicing bug)"""
            slice_df = slice_to_trees_df(args.jai_slice_data_path, args.output_folder, resize_factor=3, h=2048, w=1536)
        elif slice_data_zed:
            slice_data_zed = results_collector.converted_slice_data(
                slice_data_zed)  # convert from zed coordinates to jai
            slice_df = post_process(slice_data_zed, args.output_folder, save_csv=True)
        else:
            slice_df = pd.read_csv(all_slices_path)
        slice_df = post_process_slice_df(slice_df)
        slice_df.to_csv(os.path.join(args.output_folder, "slices.csv"))
        results_collector.dump_to_trees(args.output_folder, slice_df)
        filtered_trees = results_collector.save_trees_sliced_track(args.output_folder, slice_df)
        if args.debug.trees:
            _, _, jai_cam = init_cams(args)
            draw_on_tracked_imgaes(args, slice_df, filtered_trees, jai_cam, results_collector)
            jai_cam.close()


def update_args_with_row_runtime_params(args, row_runtime_params, block_name, key):
    """
    Updates arguments with runtime parameters for a given row.

    Args:
        args (Namespace): The existing argument namespace to update.
        row_runtime_params (dict): A dictionary of runtime parameters to add to the arguments namespace.
        block_name (str): The name of the block being processed.
        key (str): The name of the row being processed.

    Returns:
        Namespace: The updated argument namespace.
    """
    args.zed.movie_path = row_runtime_params["zed_movie_path"]
    args.jai.movie_path = row_runtime_params["jai_movie_path"]
    args.rgb_jai.movie_path = row_runtime_params["rgb_jai_movie_path"]
    args.output_folder = row_runtime_params["output_folder"]
    args.slice_data_path = row_runtime_params["slice_data_path"]
    args.jai_slice_data_path = row_runtime_params["jai_slice_data_path"]
    args.frame_drop_path = row_runtime_params["frame_drop_path"]
    args.block_name = block_name
    args.row_name = key

    return args


def get_assignments(metadata):
    """
    Extracts alignment and feature assignment (to do them or no) parameters from a metadata dictionary.

    Args:
        metadata (dict): A dictionary of metadata.

    Returns:
        Tuple: A tuple of the align_detect_track and tree_features parameters.
    """
    align_detect_track = True
    if "align_detect_track" in metadata.keys():
        align_detect_track = metadata["align_detect_track"]
    tree_features = True
    if "tree_features" in metadata.keys():
        tree_features = metadata["tree_features"]

    return align_detect_track, tree_features


def create_input(block_path, output_path, side=1, row_list=[], norgb=False):
    """
    Creates a dictionary of input parameters for a given block.

    Args:
        block_path (str): The path to the block directory.
        output_path (str): The path to the output directory.
        side (int, optional): The side of the block to process. Defaults to 1.
        row_list (list, optional): A list of row names to process. Defaults to [].
        norgb (bool, optional): Whether to ignore RGB data. Defaults to False.

    Returns:
        dict: A dictionary of input parameters.
    """
    if not row_list:
        row_list = os.listdir(block_path)
    input_dict = {}
    for row in row_list:
        if side == 1 and "B" in row:
            continue
        elif side == 2 and "A" in row:
            continue
        row_path = os.path.join(block_path, row)
        if os.path.isdir(row_path):
            row_output_path = os.path.join(output_path, row)
            validate_output_path(row_output_path)
            row_dict = {"zed_movie_path": os.path.join(row_path, f"ZED_{side}.svo"),
                         "jai_movie_path": os.path.join(row_path, f"Result_FSI_{side}.mkv"),
                         "rgb_jai_movie_path": os.path.join(row_path, f"Result_RGB_{side}.mkv"),
                         "output_folder": row_output_path,
                         "slice_data_path": os.path.join(row_path, f"ZED_{side}_slice_data_{row}.json"),
                         "jai_slice_data_path": os.path.join(row_path, f"Result_FSI_{side}_slice_data_{row}.json"),
                        "frame_drop_path": os.path.join(row_path, f"frame_drop_{side}.log")}
            if norgb:
                row_dict["rgb_jai_movie_path"] = row_dict["jai_movie_path"]
            input_dict[row] = row_dict

    return input_dict


def collect_cvs(block_folder):
    """
    Collects the CV results from each row in a given block folder and saves the results to a CSV file.

    Args:
        block_folder (str): The path to the block folder containing the rows to process.

    Returns:
        None
    """
    final = []
    block = block_folder.split('/')[-1]

    row_list = os.listdir(block_folder)
    for row in row_list:
        row_cv_path = os.path.join(block_folder, row, 'trees_cv.csv')
        if os.path.exists(row_cv_path):
            row_cv = pd.read_csv(row_cv_path)
            cv_index = np.where(row_cv.keys() == "cv")[0][0]
            tree_id_index = np.where(row_cv.keys() == "tree_id")[0][0]
            row_cv = row_cv.values.tolist()
            for cv in row_cv:
                final.append({"tree_id": cv[tree_id_index], "cv": cv[cv_index], "row_id": row})

    final = pd.DataFrame(data=final, columns=['tree_id', "cv", "row_id"])
    final.sort_values("row_id").to_csv(os.path.join(block_folder, f'block_{block}_cv.csv'))


def collect_unique_track(block_folder, depth=0):
    """
    Collects the unique track IDs and saves the results to a CSV file.

    Args:
        block_folder (str): The path to the block folder containing the rows to process.
        depth (int, optional): The maximum depth of tracks to consider. Defaults to 0 which means all tracks.

    Returns:
        None
    """
    block = block_folder.split('/')[-1]
    track_list, row_list, row_list_filtered, track_list_filtered = [], [], [], []
    cvs_min_samp = {f"n_unique_track_ids_{i}": [] for i in range(2, 6)}
    cvs_min_samp_filtered = {f"n_unique_track_ids_filtered_{depth}_{i}": [] for i in range(2, 6)}
    for row in os.listdir(block_folder):
        if row.endswith("csv") or row.endswith("png"):
            continue
        row_tracks_path = os.path.join(block_folder, row, 'tracks.csv')
        if os.path.exists(row_tracks_path):
            track_df = pd.read_csv(row_tracks_path)
            track_list.append(track_df["track_id"].nunique())
            row_list.append(row)
            uniq, counts = np.unique(track_df["track_id"], return_counts=True)
            for i in range(2, 6):
                cvs_min_samp[f"n_unique_track_ids_{i}"].append(len(uniq[counts >= i]))
            if "depth" in track_df.columns:
                track_df = track_df[track_df["depth"] < depth]
                uniq, counts = np.unique(track_df["track_id"], return_counts=True)
                track_list_filtered.append(track_df["track_id"].nunique())
                row_list_filtered.append(row)
                for i in range(2, 6):
                    cvs_min_samp_filtered[f"n_unique_track_ids_filtered_{depth}_{i}"].append(len(uniq[counts >= i]))
    final_csv_path = os.path.join(block_folder, f'{block}_n_track_ids.csv')
    final_csv_path_filtered = os.path.join(block_folder, f'{block}_n_track_ids_filtered_{depth}.csv')
    pd.DataFrame({"row": row_list, "n_unique_track_ids": track_list, **cvs_min_samp}).to_csv(final_csv_path)
    pd.DataFrame({"row": row_list_filtered,
                  f"n_unique_track_ids_filtered_{depth}": track_list_filtered, **cvs_min_samp_filtered})\
                    .to_csv(final_csv_path_filtered)


def collect_features(block_folder):
    """
    Collects the row features from each row in a given block folder and saves the results to a CSV file.

    Args:
        block_folder (str): The path to the block folder containing the rows to process.

    Returns:
        None
    """
    block = block_folder.split('/')[-1]
    dfs_list = []
    for row in os.listdir(block_folder):
        if row.endswith("csv") or row.endswith("png"):
            continue
        row_features_path = os.path.join(block_folder, row, 'row_features.csv')
        if os.path.exists(row_features_path):
            dfs_list.append(pd.read_csv(row_features_path))
    if dfs_list:
        features_df = pd.concat(dfs_list)
        features_df.to_csv(os.path.join(block_folder, f'{block}_features.csv'))


def get_rows_with_slicing(block_path):
    """
    Given a block path, returns a list of rows that contain data sliced to trees.

    Args:
        block_path (str): The path to the block folder.

    Returns:
        out_rows (list): A list of row names that contain sliced tree data.
    """
    out_rows = []
    for row in os.listdir(block_path):
        row_path = os.path.join(block_path, row)
        if os.path.isdir(row_path):
            if np.any([f"slice_data_{row}" in file or "all_slices.csv" in file for file in os.listdir(row_path)]):
                out_rows.append(row)
    return out_rows


def make_newly_dets_graph(frames, start, end, track_list_row, block, row, block_folder, interval):
    """
    Makes a graph for newly detection with average new detection per interval and sliced area.

    Args:
        frames (list): A list of frame numbers for x-axis.
        start (int): The start frame for calibration tree.
        end (int): The end frame for calibration tree.
        track_list_row (list): A list of new detection per frame.
        block (str): The name of the block for plot title.
        row (str): The name of the row for plot title.
        block_folder (str): The block folder for saving the graph.
        interval (int): Interval values taken also for naming.

    Returns:
        None
    """
    plt.figure(figsize=(15, 10))
    plt.plot(frames, track_list_row)
    # # idea for accumelated data, produce smoother graphs
    # frames = frames[::5][:-1]
    # track_list_row = np.diff(np.cumsum(track_list_row)[::5])
    if start * end:
        plt.vlines(start, 0, np.max(track_list_row) * 1.1, color="red")
        plt.vlines(end, 0, np.max(track_list_row) * 1.1, color="red")
    plt.hlines(np.mean(track_list_row), np.min(frames), np.max(frames), color="black")
    plt.title(f"{block}_{row}")
    custom_lines = [Line2D([0], [0], color="red", lw=4),
                    Line2D([0], [0], color="black", lw=4)]
    plt.legend(custom_lines, ['sliced tree', 'avg change'])
    plt.savefig(os.path.join(block_folder, f"{block}_{row}_{interval}.png"))
    plt.close()

    plt.figure(figsize=(15, 10))
    frames = frames[::5][:-1]
    track_list_row = np.diff(np.cumsum(track_list_row)[::5])
    plt.plot(frames, track_list_row)
    if start * end:
        plt.vlines(start, 0, np.max(track_list_row) * 1.1, color="red")
        plt.vlines(end, 0, np.max(track_list_row) * 1.1, color="red")
    plt.hlines(np.mean(track_list_row), np.min(frames), np.max(frames), color="black")
    plt.title(f"{block}_{row}")
    custom_lines = [Line2D([0], [0], color="red", lw=4),
                    Line2D([0], [0], color="black", lw=4)]
    plt.legend(custom_lines, ['sliced tree', 'avg change'])
    plt.savefig(os.path.join(block_folder, f"{block}_{row}_acc_5.png"))
    plt.close()


def newly_dets(block_folder, interval=1):
    """
    This function calculates the new track ids per interval,
    it is supposed to give us an idea of how much fruits we have per area unit.
    It will add calibration tree area according to the trees_sliced_track.csv in the folder,
    this might mess up if we have more than 1 tree.

    Args:
        block_folder (str): Path to block folder.
        interval (int): The number of frames between counts. Default is 1.

    Returns:
        None
    """
    block = block_folder.split('/')[-1]
    track_list = np.array([])
    row_list = np.array([])
    frame_number = np.array([])
    for row in os.listdir(block_folder):
        if row.endswith("csv") or row.endswith("png"):
            continue
        row_tracks_path = os.path.join(block_folder, row, 'tracks.csv')
        sliced_tracks_path = os.path.join(block_folder, row, "trees_sliced_track.csv")
        trees_sliced_track = {}
        start, end = 0, 0
        if os.path.exists(sliced_tracks_path):
            trees_sliced_track_df = pd.read_csv(sliced_tracks_path)
            # trees_sliced_track = dict(zip(trees_sliced_track["frame_id"].astype(int), trees_sliced_track["tree_id"]))
            start, end = trees_sliced_track_df["frame_id"].min(), trees_sliced_track_df["frame_id"].max()
        if os.path.exists(row_tracks_path):
            track_df = pd.read_csv(row_tracks_path)
            frames = track_df["frame"].unique()
            tracks_seen = np.array([])
            track_list_row = np.array([])
            # track_df["tree"] = track_df["frame"].map(trees_sliced_track).replace(np.nan, -1)
            for frame in frames[::interval]:
                frame_tracks = track_df[track_df["frame"] == frame]
                unique_tracks = frame_tracks["track_id"].unique().tolist()
                track_list_row = np.append(track_list_row, len([track for track in unique_tracks if track not in tracks_seen]))
                tracks_seen = np.append(tracks_seen, unique_tracks)
                row_list = np.append(row_list, row)
                frame_number = np.append(frame_number, frame)
            make_newly_dets_graph(frames, start, end, track_list_row, block, row, block_folder, interval)
            track_list = np.append(track_list, track_list_row)
    final_csv_path = os.path.join(block_folder, f'{block}_new_ids_per_{interval}_frames.csv')
    pd.DataFrame({"row": row_list, "n_new_ids": track_list, "frame": frame_number}).to_csv(final_csv_path)


def alignment_graph(block_folder):
    """
    Creates a graph of tx ~ frame number to check if the sensor alignment works.

    Args:
        block_folder (str): The path to the block folder.

    Returns:
        None
    """
    block = block_folder.split('/')[-1]
    for row in os.listdir(block_folder):
        if row.endswith("csv") or row.endswith("png"):
            continue
        jai_zed_cors_path = os.path.join(block_folder, row, 'jai_cors_in_zed.csv')
        if os.path.exists(jai_zed_cors_path):
            df_coors = pd.read_csv(jai_zed_cors_path)
        else:
            continue
        sliced_tracks_path = os.path.join(block_folder, row, "trees_sliced_track.csv")
        start, end = 0, 0
        if os.path.exists(sliced_tracks_path):
            trees_sliced_track_df = pd.read_csv(sliced_tracks_path)
            start, end = trees_sliced_track_df["frame_id"].min(), trees_sliced_track_df["frame_id"].max()
        plt.figure(figsize=(15, 10))
        df_coors["tx"].clip(-50, 150, inplace=True)
        tx_conv = np.convolve(df_coors["tx"], np.ones(10) / 10, mode='same')
        plt.plot(df_coors["frame"], df_coors["tx"])
        plt.plot(df_coors["frame"], tx_conv, color="orange")
        plt.title(f"Jai Zed alignment {block}_{row}")
        for frame in df_coors["frame"][:-1][df_coors["zed_shift"].values[1:] - df_coors["zed_shift"].values[:-1] == 1]:
            plt.vlines(frame, np.min(df_coors["tx"]) - 10, np.max(df_coors["tx"]) + 10, color="green",
                       linestyles="dotted")
        custom_lines = [Line2D([0], [0], color="green", lw=4)]
        legends = ['zed_shift']
        if start * end:
            plt.vlines(start, np.min(df_coors["tx"]) - 10, np.max(df_coors["tx"]) + 10, color="red")
            plt.vlines(end, np.min(df_coors["tx"]) - 10, np.max(df_coors["tx"]) + 10, color="red")
            custom_lines += [Line2D([0], [0], color="red", lw=4)]
            legends += ['sliced tree']
        plt.legend(custom_lines, legends)
        plt.savefig(os.path.join(block_folder, f"jai_zed_alignment {block}_{row}.png"))
        plt.close()


def agg_results_to_delivery(customer_path="/media/fruitspec-lab/cam175/DEWAGD", output_path="/media/fruitspec-lab/cam175/results"):
    """
     Copies only relevant data to send out.

     Args:
         customer_path (str, optional): The path to the master folder. Defaults to "/media/fruitspec-lab/cam175/DEWAGD".
         output_path (str, optional): The path to save files. Defaults to "/media/fruitspec-lab/cam175/results".

     Returns:
         None
     """
    for scan_date in os.listdir(customer_path):
        scan_path = os.path.join(customer_path, scan_date)
        for block in os.listdir(scan_path):
            block_path = os.path.join(scan_path, block)
            for file in os.listdir(block_path):
                if file.endswith("csv") or file.endswith("png"):
                    file_path = os.path.join(block_path, file)
                    new_file_path = os.path.join(output_path, scan_date, block, file)
                    new_scan_path = os.path.join(output_path, scan_date)
                    validate_output_path(new_scan_path)
                    new_block_path = os.path.join(new_scan_path, block)
                    validate_output_path(new_block_path)
                    shutil.copy(file_path, new_file_path)


def get_b_with_slicing(block_path):
    """Returns a list of rows from the B side that have slicing data.

    Args:
        block_path (str): Path to block.

    Returns:
        list: A list of rows from the B side that have slicing data.
    """
    rows_with_slice = []
    for row in os.listdir(block_path):
        if "A" in row or row.endswith("png") or row.endswith("csv"):
            continue
        row_path = os.path.join(block_path, row)
        slice_data_path = os.path.join(row_path, f"ZED_2_slice_data_{row}.json")
        jai_slice_data_path = os.path.join(row_path, f"Result_FSI_2_slice_data_{row}.json")
        if os.path.exists(slice_data_path) or os.path.exists(jai_slice_data_path):
            rows_with_slice.append(row)
    return rows_with_slice


def run_multi_customers(folder_path, use_sliced_rows_only=False, skip_blocks=[], sides=[1, 2], njobs=1, norgb=False,
                        rows_parralel=False):
    """Runs feature extraction pipeline on a folder of customers (a folder of customer -> scan -> blocks -> rows).

    Args:
        folder_path (str): Path to master folder.
        use_sliced_rows_only (bool, optional): Flag, if True will only run pipe on sliced rows. Defaults to False.
        skip_blocks (list, optional): Which blocks not to run. Defaults to [].
        sides (list, optional): Will run on all rows of sides[0] and only on rows with slicing for sides[1]. Defaults to [1, 2].
        njobs (int, optional): Number of jobs. Defaults to 1.
        norgb (bool, optional): Flag if there is no RGB video. Defaults to False.
        rows_parralel (bool, optional): Flag for running parallel on rows instead of blocks. Defaults to False.

    Returns:
        None
    """
    for customer in os.listdir(folder_path):
        customer_path = os.path.join(folder_path, customer)
        run_multi_block(customer_path, use_sliced_rows_only, skip_blocks, sides, njobs, norgb, rows_parralel)


def multi_block_multiprocess_wrapper(block, block_path, use_sliced_rows_only, sides, norgb=False, njobs=1):
    """Wrapper function for running feature extraction pipeline on a block.

    Args:
        block (str): Block name.
        block_path (str): Path to block.
        use_sliced_rows_only (bool): Flag, if True will only run pipe on sliced rows.
        sides (list): Will run on all rows of sides[0] and only on rows with slicing for sides[1].
        norgb (bool, optional): Flag if there is no RGB video. Defaults to False.
        njobs (int, optional): Number of jobs. Defaults to 1.

    Returns:
        None
    """
    print("#"*50 + "started: " + block + " " + "#"*50)
    output_path = block_path
    row_list = get_rows_with_slicing(block_path) if use_sliced_rows_only else []
    input_dict = create_input(block_path, output_path, side=sides[0], row_list=row_list, norgb=norgb)
    # take only B side if it has slicing data:
    b_side_rows = get_b_with_slicing(block_path)
    input_dictB = {}
    if use_sliced_rows_only:
        if b_side_rows and len(sides) > 1:
            input_dictB = create_input(block_path, output_path, side=sides[1], row_list=b_side_rows, norgb=norgb)
    else:
        if len(sides) > 1:
            input_dictB = create_input(block_path, output_path, side=sides[1], row_list=b_side_rows, norgb=norgb)
    max_z = run_on_rows({**input_dictB, **input_dict}, block_name=block, njobs=njobs)
    collect_cvs(block_path)
    collect_features(block_path)
    collect_unique_track(block_path, max_z)
    newly_dets(block_path, interval=1)
    alignment_graph(block_path)


def run_multi_block(customer_path, use_sliced_rows_only=False, skip_blocks=[], sides=[1, 2], njobs=1, norgb=False,
                    rows_parralel=False):
    """Runs feature extraction pipeline on a customer (a folder of scan -> blocks -> rows).

    Args:
        customer_path (str): Path to master folder.
        use_sliced_rows_only (bool, optional): Flag, if True will only run pipe on sliced rows. Defaults to False.
        skip_blocks (list, optional): Which blocks not to run. Defaults to [].
        sides (list, optional): Will run on all rows of sides[0] and only on rows with slicing for sides[1]. Defaults to [1, 2].
        njobs (int, optional): Number of jobs. Defaults to 1.
        norgb (bool, optional): Flag if there is no RGB video. Defaults to False.
        rows_parralel (bool, optional): Flag for running parallel on rows instead of blocks. Defaults to False.

    Returns:
        None
    """

    blocks, block_paths, use_sliced_rows_only_list, sides_list, norgbs = [], [], [], [], []
    for scan_date in os.listdir(customer_path):
        scan_path = os.path.join(customer_path, scan_date)
        if not os.path.isdir(scan_path):
            continue
        for block in os.listdir(scan_path):
            if block in skip_blocks or scan_date in skip_blocks:
                continue
            block_path = os.path.join(scan_path, block)
            if not os.path.isdir(block_path):
                continue
            block_paths.append(block_path)
            blocks.append(block)
            use_sliced_rows_only_list.append(use_sliced_rows_only)
            sides_list.append(sides)
            norgbs.append(norgb)
    if rows_parralel:
        list(map(multi_block_multiprocess_wrapper, blocks, block_paths, use_sliced_rows_only_list,
                 sides_list, norgbs, [njobs] * len(blocks)))
    else:
        if njobs > 1:
            with ProcessPoolExecutor(max_workers=njobs) as executor:
                executor.map(multi_block_multiprocess_wrapper, blocks, block_paths, use_sliced_rows_only_list,
                             sides_list, norgbs)
        else:
            list(map(multi_block_multiprocess_wrapper, blocks, block_paths, use_sliced_rows_only_list,
                     sides_list, norgbs))


    # for scan_date in os.listdir(customer_path):
    #     scan_path = os.path.join(customer_path, scan_date)
    #     for block in os.listdir(scan_path):
    #         if block in skip_blocks:
    #             continue
    #         block_path = os.path.join(scan_path, block)
    #         if not os.path.isdir(block_path):
    #             continue
    #         output_path = block_path
    #         row_list = get_rows_with_slicing(block_path) if use_sliced_rows_only else []
    #         input_dict = create_input(block_path, output_path, side=sides[0], row_list=row_list)
    #         # take only B side if it has slicing data:
    #         b_side_rows = get_b_with_slicing(block_path)
    #         input_dictB = {}
    #         if b_side_rows and len(sides) > 1:
    #             input_dictB = create_input(block_path, output_path, side=sides[1], row_list=b_side_rows)
    #         run_on_rows({**input_dictB, **input_dict}, block_name=block)
    #         collect_cvs(block_path)
    #         collect_features(block_path)
    #         collect_unique_track(block_path)
    #         newly_dets(block_path, interval=1)
    #         alignment_graph(block_path)
    #         # TODO add prediction

def fix_file_numbering(customer_folder):
    """
    Changes the file naming format for files uploaded in Result_channel_1 format for the B side. Replaces 1 with 2 in
    the file name.

    Args:
    - customer_folder (str): The path to the customer folder.

    Returns:
    - None
    """
    for scan in os.listdir(customer_folder):
        scan_path = os.path.join(customer_folder, scan)
        if not os.path.isdir(scan_path):
            continue
        for block in os.listdir(scan_path):
            block_path = os.path.join(scan_path, block)
            if not os.path.isdir(block_path):
                continue
            for row in os.listdir(block_path):
                row_path = os.path.join(block_path, row)
                if not os.path.isdir(row_path) or row.endswith("A"):
                    continue
                for file in os.listdir(row_path):
                    loc_1 = file.find("1")
                    if loc_1 == -1:
                        continue
                    if file.endswith("json"):
                        if loc_1 > 11:
                            continue
                    new_name = file[:loc_1] + "2" + file[loc_1+1:]
                    os.rename(os.path.join(row_path, file), os.path.join(row_path, new_name))


if __name__ == "__main__":
    customers_folder_path = "/media/fruitspec-lab/cam175/customers"
    customer_path = "/media/fruitspec-lab/cam175/APPLECHILE04_test"
    # customer_path = "/media/fruitspec-lab/cam175/customers/PROPAL"
    # fix_file_numbering("/media/fruitspec-lab/cam175/customers/PROPAL/280323")
    # apples_path = "/media/fruitspec-lab/cam175/customers/CHILEAPP"
    skip_blocks = []
    # run_multi_block(apples_path, use_sliced_rows_only=False, skip_blocks=skip_blocks, sides=[1, 2], njobs=1, norgb=True)
    run_multi_block(customer_path, use_sliced_rows_only=False, skip_blocks=skip_blocks, sides=[1, 2], njobs=1,
                    rows_parralel=True)
    # run_multi_block(customer_path, use_sliced_rows_only=True, skip_blocks=skip_blocks, sides=[2], njobs=3)
    # run_multi_customers(customers_folder_path, use_sliced_rows_only=True, skip_blocks=skip_blocks, sides=[1], njobs=4)
    # run_multi_customers(customers_folder_path, use_sliced_rows_only=True, skip_blocks=skip_blocks, sides=[2])

    # skip_blocks_2 = []
    # run_multi_block(customer_path, skip_blocks=skip_blocks_2, sides=[])

    #
    # block_path = "/media/fruitspec-lab/cam175/customers/DEWAGD/190123/DWDBLE33"
    # output_path = "/media/fruitspec-lab/cam175/customers/DEWAGD/190123/DWDBLE33"
    #
    # # validate_output_path(output_path)
    # # TODO add logic for non json calibraion (all_slices.csv)
    # input_dict_a = {}
    # input_dict_a = create_input(block_path, output_path, row_list=get_rows_with_slicing(block_path))
    # input_dict_b = {}
    # # input_dict_b = create_input(block_path, output_path, side=2, row_list=get_rows_with_slicing(block_path))
    #
    # exclude = []
    # # exclude = [file for file in os.listdir(block_path)
    # #            if not (file.endswith("csv") or file.endswith("png") or "11" in file)]
    # # alignment_graph(block_path)
    # run_on_rows({**input_dict_a, **input_dict_b}, exclude)
    #
    # # run_on_rows(input_dict, exclude_DWDBNC
    # # 47)

