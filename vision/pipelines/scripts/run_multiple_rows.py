import os
import cv2
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import shutil
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from vision.misc.help_func import get_repo_dir, validate_output_path
from vision.data.results_collector import ResultsCollector
from vision.pipelines.adt_pipeline import run as adt_run
#from vision.pipelines.fe_pipeline import run as fe_run
from vision.pipelines.ops.simulator import update_arg_with_metadata, get_jai_drops, get_max_cut_frame, load_logs
from vision.pipelines.ops.simulator import init_cams
from vision.pipelines.ops.debug_methods import plot_alignmnt_graph, draw_on_tracked_imgaes
from vision.depth.slicer.slicer_flow import post_process
from vision.tools.manual_slicer import slice_to_trees

np.random.seed(123)
cv2.setRNGSeed(123)

def run_on_rows(input_dict, exclude=[], block_name=""):
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    # config_file = "/config/pipeline_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)
    input_dict = {k: input_dict[k] for k in sorted(input_dict)} # sort it
    for key, row_runtime_params in input_dict.items():
        if key in exclude:
            continue
        print("starting ", row_runtime_params["jai_novie_path"])
        run_args = args.copy()
        run_args = update_args_with_row_runtime_params(run_args, row_runtime_params, block_name, key)
        validate_output_path(run_args.output_folder)

        run_args, metadata = update_arg_with_metadata(run_args)

        slice_data_zed, slice_data_jai = load_logs(run_args)
        all_slices_path = os.path.join(os.path.dirname(args.slice_data_path), "all_slices.csv")
        metadata['max_cut_frame'] = str(get_max_cut_frame(run_args, slice_data_jai, slice_data_zed))

        align_detect_track, tree_features = get_assignments(metadata)

        # perform align -> detect -> track
        if align_detect_track or args.overwrite.adt:
            rc = adt_run(cfg, run_args, metadata)
            #adt_slice_postprocess(run_args, rc, slice_data_zed, slice_data_jai, all_slices_path)
            if run_args.debug.align_graph:
                frame_drop_jai = get_jai_drops(run_args.frame_drop_path)
                plot_alignmnt_graph(run_args, rc, frame_drop_jai)

        # perform features extraction
        #if tree_features or run_args.overwrite.trees:
        #    fe_run(run_args)

def adt_slice_postprocess(args, results_collector, slice_data_zed, slice_data_jai, all_slices_path):

    if slice_data_zed or slice_data_jai or os.path.exists(all_slices_path):
        if slice_data_jai:
            """this function depends on how we sliced (before or after slicing bug)"""
            slice_df = slice_to_trees(args.jai_slice_data_path, "", args.output_folder, resize_factor=3, h=2048,
                                      w=1536)
            slice_df.to_csv(os.path.join(args.output_folder, "slices.csv"))
        elif slice_data_zed:
            slice_data_zed = results_collector.converted_slice_data(
                slice_data_zed)  # convert from zed coordinates to jai
            slice_df = post_process(slice_data_zed, args.output_folder, save_csv=True)
        else:
            slice_df = pd.read_csv(all_slices_path)
        results_collector.dump_to_trees(args.output_folder, slice_df)
        filtered_trees = results_collector.save_trees_sliced_track(args.output_folder, slice_df)
        if args.debug.trees:
            _, _, jai_cam = init_cams(args)
            draw_on_tracked_imgaes(args, slice_df, filtered_trees, jai_cam, results_collector)
            jai_cam.close()


def update_args_with_row_runtime_params(args, row_runtime_params, block_name, key):
    args.zed.movie_path = row_runtime_params["zed_novie_path"]
    args.jai.movie_path = row_runtime_params["jai_novie_path"]
    args.rgb_jai.movie_path = row_runtime_params["rgb_jai_novie_path"]
    args.output_folder = row_runtime_params["output_folder"]
    args.slice_data_path = row_runtime_params["slice_data_path"]
    args.jai_slice_data_path = row_runtime_params["jai_slice_data_path"]
    args.frame_drop_path = row_runtime_params["frame_drop_path"]
    args.block_name = block_name
    args.row_name = key

    return args
def get_assignments(metadata):
    align_detect_track = True
    if "align_detect_track" in metadata.keys():
        align_detect_track = metadata["align_detect_track"]
    tree_features = True
    if "tree_features" in metadata.keys():
        tree_features = metadata["tree_features"]

    return align_detect_track, tree_features


def create_input(block_path, output_path, side=1, row_list=[]):
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
            row_dict = {"zed_novie_path": os.path.join(row_path, f"ZED_{side}.svo"),
                         "jai_novie_path": os.path.join(row_path, f"Result_FSI_{side}.mkv"),
                         "rgb_jai_novie_path": os.path.join(row_path, f"Result_RGB_{side}.mkv"),
                         "output_folder": row_output_path,
                         "slice_data_path": os.path.join(row_path, f"ZED_{side}_slice_data_{row}.json"),
                         "jai_slice_data_path": os.path.join(row_path, f"Result_FSI_{side}_slice_data_{row}.json"),
                        "frame_drop_path": os.path.join(row_path, f"frame_drop_{side}.log")}
            input_dict[row] = row_dict

    return input_dict


def collect_cvs(block_folder):
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
    final.to_csv(os.path.join(block_folder, f'block_{block}_cv.csv'))


def collect_unique_track(block_folder):
    """
    collects the number of unique track id for eadch row per block
    :param block_folder:
    :return:
    """
    block = block_folder.split('/')[-1]
    track_list = []
    row_list = []
    for row in os.listdir(block_folder):
        if row.endswith("csv") or row.endswith("png"):
            continue
        row_tracks_path = os.path.join(block_folder, row, 'tracks.csv')
        if os.path.exists(row_tracks_path):
            track_list.append(pd.read_csv(row_tracks_path)["track_id"].nunique())
            row_list.append(row)
    final_csv_path = os.path.join(block_folder, f'{block}_n_track_ids.csv')
    pd.DataFrame({"row": row_list, "n_unique_track_ids": track_list}).to_csv(final_csv_path)


def collect_features(block_folder):
    """
    concatenates all features from rows to a dataframe of block features
    :param block_folder: path to block folder
    :return:
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
    returns the rows that have a slicing json for the block
    :param block_path: path to block
    :return: row with slicing jsons
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
    makes a graph for newly detection with average new detection per interval and sliced area
    :param frames: frame numbers for x axis
    :param start: start frame for calibaration tree
    :param end: end frame for calibaration tree
    :param track_list_row:  new detection per frame
    :param block: the name of the block for plot title
    :param row: the name of the row for plot title
    :param block_folder: block folder for saving the graph
    :param interval: initerval values taken also for naming
    :return:
    """
    plt.figure(figsize=(15, 10))
    plt.plot(frames, track_list_row)
    # # idea for accumelated data, produce smoother graphs
    # frames = frames[::5][:-1]
    # track_list_row = np.diff(np.cumsum(track_list_row)[::5])
    if start*end:
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
    if start*end:
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
    this function calculates the new track ids per interval,
    it is supposed to give us an idea of how much fruits we have per area unit
    it will add calibration tree area accoring to the trees_sliced_track.csv in the folder,
     this might mess up if we have more then 1 tree
    :param block_folder: path to block folder
    :param interval: number of frames between counts
    :return:
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
    creates a graph of tx ~ frame number, this is for understanding if the sensor alignment works ok
    :param block_folder: path to block folder
    :return:
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
    copies only relevant data to send out
    :param customer_path: path to master folder
    :param output_path: where to save files
    :return:
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
    """
    this function returen a list of rows from B side that has slicing data
    :param block_path: path to block
    :return: a list of rows from B side that has slicing data
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


def run_multi_customers(folder_path, use_sliced_rows_only=False, skip_blocks=[], sides=[1, 2]):
    """
    this function runs feature extraction pipeline on a folder of customers (a folder of customer -> scan -> blocks -> rows)
    :param customer_path: path to master folder
    :param use_sliced_rows_only: flag, if true will only run pipe on sliced rows
    :param skip_blocks: which blocks not to run
    :param sides: will run on all of rows side[0] and only on rows with slicing for side[1]
    :return:
    """
    for customer in os.listdir(folder_path):
        customer_path = os.path.join(folder_path, customer)
        run_multi_block(customer_path, use_sliced_rows_only, skip_blocks, sides)


def multi_block_multiprocess_wrapper(block ,block_path, use_sliced_rows_only, sides):
    output_path = block_path
    row_list = get_rows_with_slicing(block_path) if use_sliced_rows_only else []
    input_dict = create_input(block_path, output_path, side=sides[0], row_list=row_list)
    # take only B side if it has slicing data:
    b_side_rows = get_b_with_slicing(block_path)
    input_dictB = {}
    if b_side_rows and len(sides) > 1:
        input_dictB = create_input(block_path, output_path, side=sides[1], row_list=b_side_rows)
    run_on_rows({**input_dictB, **input_dict}, block_name=block)
    collect_cvs(block_path)
    collect_features(block_path)
    collect_unique_track(block_path)
    newly_dets(block_path, interval=1)
    alignment_graph(block_path)


def run_multi_block(customer_path, use_sliced_rows_only=False, skip_blocks=[], sides=[1, 2], njobs=1):
    """
    this function runs feature extraction pipeline on a customer (a folder of scan -> blocks -> rows)
    :param customer_path: path to master folder
    :param use_sliced_rows_only: flag, if true will only run pipe on sliced rows
    :param skip_blocks: which blocks not to run
    :param sides: will run on all of rows side[0] and only on rows with slicing for side[1]
    :return:
    """

    blocks, block_paths, use_sliced_rows_only_list, sides_list = [], [], [], []
    for scan_date in os.listdir(customer_path):
        scan_path = os.path.join(customer_path, scan_date)
        for block in os.listdir(scan_path):
            if block in skip_blocks:
                continue
            block_path = os.path.join(scan_path, block)
            if not os.path.isdir(block_path):
                continue
            block_paths.append(block_path)
            blocks.append(block)
            use_sliced_rows_only_list.append(use_sliced_rows_only)
            sides_list.append(sides)

    with ProcessPoolExecutor(max_workers=njobs) as executor:
        executor.map(multi_block_multiprocess_wrapper, blocks, block_paths, use_sliced_rows_only_list, sides_list)


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


if __name__ == "__main__":
    #customers_folder_path = "/media/fruitspec-lab/cam175/customers"
    #customer_path = "/media/fruitspec-lab/cam175/customers/SHANIR"
    #skip_blocks = []
    #run_multi_block(customer_path, use_sliced_rows_only=True, skip_blocks=skip_blocks, sides=[1])
    # run_multi_customers(customers_folder_path, use_sliced_rows_only=True, skip_blocks=skip_blocks, sides=[1])
    #run_multi_customers(customers_folder_path, use_sliced_rows_only=True, skip_blocks=skip_blocks, sides=[2])

    # skip_blocks_2 = []
    # run_multi_block(customer_path, skip_blocks=skip_blocks_2, sides=[])


    block_path = "/home/mic-730ai/fruitspec/test_data/validate_refactor/data"
    output_path = "/home/mic-730ai/fruitspec/test_data/validate_refactor"

    validate_output_path(output_path)
    # TODO add logic for non json calibraion (all_slices.csv)
    input_dict_a = {}
    input_dict_a = create_input(block_path, output_path, row_list=get_rows_with_slicing(block_path))
    input_dict_b = {}
    input_dict_b = create_input(block_path, output_path, side=2, row_list=get_rows_with_slicing(block_path))

    exclude = ['R2B', 'R3A', 'R3B', 'R4A', 'R4B', 'R5A', 'RBA']
    run_on_rows({**input_dict_a, **input_dict_b}, exclude)

