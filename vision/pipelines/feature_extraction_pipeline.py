import os
import sys
import time
import pandas as pd
import pyzed.sl as sl
import cv2
import json
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

from vision.misc.help_func import get_repo_dir, write_json, load_json, read_json
from vision.tools.video_wrapper import video_wrapper
from vision.tools.manual_slicer import slice_to_trees

from vision.feature_extractor.image_processing import get_percent_seen

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.data.results_collector import ResultsCollector
from vision.tools.translation import translation as T
from vision.depth.slicer.slicer_flow import post_process
from vision.tools.sensors_alignment import SensorAligner, FirstMoveDetector
from vision.tools.camera import is_sturated
from vision.feature_extractor.feature_extractor import create_row_features_fe_pipe, cut_zed_in_jai
import matplotlib.pyplot as plt
from vision.feature_extractor.boxing_tools import xyz_center_of_box


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


def init_cams(args):
    """
    initiates all cameras based on arguments file
    :param args: arguments file
    :return: zed_cam, rgb_jai_cam, jai_cam
    """
    zed_cam = video_wrapper(args.zed.movie_path, args.zed.rotate, args.zed.depth_minimum, args.zed.depth_maximum)
    rgb_jai_cam = video_wrapper(args.rgb_jai.movie_path, args.rgb_jai.rotate)
    jai_cam = video_wrapper(args.jai.movie_path, args.jai.rotate)
    return zed_cam, rgb_jai_cam, jai_cam

def get_jai_drops(frame_drop_path):
    """
    reads jai log file and extracts the number of the dropped frames
    :param frame_drop_path: path to log file
    :return: numbers of dropped frames
    """
    jai_drops = np.array([])
    if not os.path.exists(frame_drop_path):
        return jai_drops
    with open(frame_drop_path, "r") as logfile:
        lines = logfile.readlines()
        for line in lines:
            if "FRAME DROP" in line:
                jai_drops = np.append(jai_drops, line.strip().split(" ")[-1])
    jai_drops_uniq = np.unique(jai_drops).astype(np.int)
    jai_drops_uniq.sort()
    jai_drops_uniq -= range(len(jai_drops_uniq))
    return jai_drops_uniq


def init_run_objects(cfg, args):
    """
    Initializes the necessary objects for the main run function.

    Args:
        cfg (obj): Config object containing necessary parameters.
        args (argparse.Namespace): Namespace object containing arguments.

    Returns:
        tuple: A tuple of initialized objects consisting of:
            - detector (counter_detection): Counter detection object.
            - results_collector (ResultsCollector): Results collector object.
            - translation (T): Translation object.
            - sensor_aligner (SensorAligner): Sensor aligner object.
            - zed_cam (ZEDCamera): ZED camera object.
            - rgb_jai_cam (JAI_Camera): RGB camera object.
            - jai_cam (JAI_Camera): JAI camera object.
    """
    detector = counter_detection(cfg, args)
    results_collector = ResultsCollector(rotate=args.rotate)
    translation = T(cfg.translation.translation_size, cfg.translation.dets_only, cfg.translation.mode)
    sensor_aligner = SensorAligner(args=args.sensor_aligner, zed_shift=args.zed_shift)
    zed_cam, rgb_jai_cam, jai_cam = init_cams(args)
    return detector, results_collector, translation, sensor_aligner, zed_cam, rgb_jai_cam, jai_cam


def load_logs(args):
    """
    Load logs from the provided file paths and update args with metadata.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Tuple containing:
            - slice_data_zed (dict): Dictionary containing ZED slice data.
            - slice_data_jai (dict): Dictionary containing JAI slice data.
            - args (argparse.Namespace): Updated command-line arguments.
            - metadata (dict): Dictionary containing metadata.
            - frame_drop_jai (list): List containing frames dropped from JAI camera.
            - max_cut_frame (int): the last frame id that has a tree sliced in it

    """
    slice_data_zed, slice_data_jai = load_json(args.slice_data_path), load_json(args.jai_slice_data_path)
    args, metadata = update_arg_with_metadata(args)
    try:
        frame_drop_jai = get_jai_drops(args.frame_drop_path)
    except Exception as e:
        print(e)
        frame_drop_jai = np.array([])
    all_slices_path = os.path.join(os.path.dirname(args.slice_data_path), "all_slices.csv")
    if args.until_last_slice:
        frames_with_slices = [key for key, item in slice_data_jai.items() if not item["end"] is None] +\
                             [key for key, item in slice_data_zed.items() if not item["end"] is None]
        max_cut_frame = np.max(frames_with_slices) if len(frames_with_slices) else np.inf
    else:
        max_cut_frame = np.inf
    return slice_data_zed, slice_data_jai, args, metadata, frame_drop_jai, all_slices_path, max_cut_frame


def get_frames(zed_cam, jai_cam, rgb_jai_cam, f_id, sensor_aligner):
    """
    Returns frames from the ZED, JAI, and RGB JAI cameras for a given frame ID.

    Args:
        zed_cam (ZEDCam): A ZED camera object.
        jai_cam (JAICam): A JAI camera object.
        rgb_jai_cam (JAICam): A JAI camera object used for capturing RGB images.
        f_id (int): The frame ID for which frames are to be retrieved.
        sensor_aligner (SensorAligner): A SensorAligner object for synchronizing the cameras.

    Returns:
        tuple: A tuple containing the following:
            - `zed_frame`: The ZED camera frame for the given frame ID.
            - `point_cloud`: The point cloud data generated by the ZED camera for the given frame ID.
            - `fsi_ret`: The return value for the JAI camera frame capture.
            - `jai_frame`: The JAI camera frame for the given frame ID.
            - `rgb_ret`: The return value for the RGB JAI camera frame capture.
            - `rgb_jai_frame`: The RGB JAI camera frame for the given frame ID.
    """
    zed_frame, point_cloud = zed_cam.get_zed(f_id + sensor_aligner.zed_shift, exclude_depth=True)
    fsi_ret, jai_frame = jai_cam.get_frame()
    rgb_ret, rgb_jai_frame = rgb_jai_cam.get_frame()
    return zed_frame, point_cloud, fsi_ret, jai_frame, rgb_ret, rgb_jai_frame


def get_n_frames(max_cut_frame, jai_cam, metadata):
    """
    Returns the number of frames to run on.

    Args:
        max_cut_frame (float): The maximum frame to run on. If `max_cut_frame` is not finite,
         `n_frames` will be `max_cut_frame + 1`.
        jai_cam (JAI_Camera): The JAI camera object.
        metadata (dict): A dictionary containing metadata related to the camera setup and configuration.

    Returns:
        int: The number of frames to run on. If `metadata` contains a key "cut_frames",
         `n_frames` will be the number of frames adjusted by the cut ratio.
          If `max_cut_frame` is not finite, `n_frames` will be `max_cut_frame + 1`.

    Note:
        `n_frames` is the number of frames the program will run on.
                    If tree slicing was done will take n_frames as last frames siced
    """
    if np.isfinite(max_cut_frame):
        n_frames = max_cut_frame +1
    else:
        n_frames = jai_cam.get_number_of_frames()
        if "cut_frames" in metadata.keys():
            cut_frames = metadata["cut_frames"]
            n_frames = int(n_frames*cut_frames)+1
    return n_frames


def get_first_move(camera, thresh=0.01, debug=True, cam_type="zed"):
    translator = T(480, False, 'keypoints')
    frame = 1
    tx = 0
    counter = 0
    width = camera.get_width()
    while counter < 3:
        if cam_type == "zed":
            frame_img, point_cloud = camera.get_zed(frame, exclude_depth=True)
        else:
            _, frame_img = camera.get_frame(frame)
        tx, _ = translator.get_translation(frame_img, None)
        if np.abs(tx) > width*thresh:
            counter += 1
        else:
            counter = 0
        if debug:
            folder = "/media/fruitspec-lab/easystore/auto_zed_shift_testing"
            fig_name = f"{cam_type}_{frame}_{tx}"
            plt.imshow(frame_img)
            plt.title(f"frame: {frame}, tx: {tx}")
            plt.savefig(os.path.join(folder, fig_name))
            plt.show()
        frame += 1
    return frame


def get_zed_shift(args):
    # TODO this might not work on saturated images
    zed_cam = video_wrapper(args.zed.movie_path, args.zed.rotate, args.zed.depth_minimum, args.zed.depth_maximum)
    jai_cam = video_wrapper(args.jai.movie_path, args.jai.rotate)

    zed_move_frame = get_first_move(zed_cam)
    jai_move_frame = get_first_move(jai_cam, cam_type="jai")

    return zed_move_frame - jai_move_frame


def get_depth_to_bboxes(xyz_frame, jai_frmae, cut_coords, dets):
    """
    Retrives the depth to each bbox
    Args:
        xyz_frame (np.array): a Point cloud image
        jai_frmae (np.array): FSI image
        cut_coords (tuple): jai in zed coords
        dets (list): list of detections

    Returns:
        z_s (list): list with depth to each detection
    """
    cut_coords = dict(zip(["x1", "y1", "x2", "y2"], [[int(cord)] for cord in cut_coords]))
    xyz_frame_alligned = cut_zed_in_jai({"zed": xyz_frame}, cut_coords, rgb=False)["zed"]
    z_s = []
    for det in dets:
        r_h, r_w = xyz_frame_alligned.shape[0] / jai_frmae.shape[0], xyz_frame_alligned.shape[1]/ jai_frmae.shape[1]
        box = ((int(det[0]*r_w), int(det[1]*r_h)), (int(det[2]*r_w), int(det[3]*r_h)))
        _, _, z = xyz_center_of_box(xyz_frame_alligned, box, nir=jai_frmae[:, :, 0], swir_975=jai_frmae[:, :, 1])
        z_s.append(z)
    return z_s



def run(cfg, args):
    args = args.copy()
    # args.zed_shift = get_zed_shift(args)
    # print("zed shift:", args.zed_shift)
    slice_data_zed, slice_data_jai, args, metadata, frame_drop_jai, all_slices_path, max_cut_frame = load_logs(args)
    detector, results_collector, translation, sensor_aligner, zed_cam, rgb_jai_cam, jai_cam = init_run_objects(cfg, args)
    move_detector = FirstMoveDetector()
    f_id = 0
    align_detect_track = True
    if "align_detect_track" in metadata.keys():
        align_detect_track = metadata["align_detect_track"]
    tree_features = True
    if "tree_features" in metadata.keys():
        tree_features = metadata["tree_features"]
    n_frames = get_n_frames(max_cut_frame, jai_cam, metadata)

    # Read until video is completed
    print(f'Inferencing on {args.jai.movie_path}\n')
    pbar = tqdm(total=n_frames)
    s_frame = 0
    if sensor_aligner.zed_shift < 0:
        s_frame = np.abs(sensor_aligner.zed_shift)
    if sensor_aligner.zed_shift != 0 and not args.overwtire.zed_shift:
        args.auto_zed_shift = False
    if align_detect_track or args.overwtire.adt:
        while f_id < n_frames:
            pbar.update(1)
            zed_frame, point_cloud, fsi_ret, jai_frame, rgb_ret, rgb_jai_frame = get_frames(zed_cam, jai_cam,
                                                                                            rgb_jai_cam, f_id,
                                                                                            sensor_aligner)

            if not fsi_ret or not zed_cam.res or not rgb_ret:  # couldn't get frames, Break the loop
                break
            if is_sturated(rgb_jai_frame, 0.6) or is_sturated(zed_frame, 0.6):
                print(f'frame {f_id} is saturated, skipping')
                if f_id in frame_drop_jai:
                    sensor_aligner.zed_shift += 1
                f_id += 1
                continue
            if s_frame > f_id:
                print(f'frame {f_id} is pre sync, skipping')
                f_id += 1
                continue
            if args.auto_zed_shift:
                zed_shift, status = move_detector.update_state(zed_frame, jai_frame, f_id)
                if not status:
                    f_id += 1
                    continue
                metadata["zed_shift"] = zed_shift
                sensor_aligner.zed_shift = zed_shift
                args.auto_zed_shift = False
                print("auto zed shift detected zed shift of: ", zed_shift)
            if not args.no_sync:
            # align sensors
                corr, tx_a, ty_a, sx, sy = sensor_aligner.align_sensors(cv2.cvtColor(zed_frame, cv2.COLOR_BGR2RGB),
                                                                    jai_frame #rgb_jai_frame
                                                                    # cv2.cvtColor(jai_frame, cv2.COLOR_BGR2RGB),
                                                                    )# jai_drop=f_id in frame_drop_jai
            else:
                h, w = zed_frame.shape[:2]
                corr, tx_a, ty_a, sx, sy = (0, 0, h-1, w-1), 0, 0, 0, 0
            percent_seen = get_percent_seen(zed_frame, corr)
            # detect:
            det_outputs = detector.detect(jai_frame)
            z_s = get_depth_to_bboxes(point_cloud, jai_frame, corr, det_outputs)

            # find translation
            tx, ty = translation.get_translation(jai_frame, det_outputs)
            # if sensor_aligner.direction == "" and not isinstance(tx, type(None)):
            #     if np.abs(tx) > 15:
            #         sensor_aligner.direction = "right" if tx > 0 else "left"
            #         print(sensor_aligner.direction)
            # track:
            trk_outputs, trk_windows = detector.track(det_outputs, tx, ty, f_id)
            #collect results:
            results_collector.collect_detections(det_outputs, f_id, z_s)
            results_collector.collect_tracks(trk_outputs, z_s)
            results_collector.collect_alignment(corr, tx_a, ty_a, sx, sy, f_id, sensor_aligner.zed_shift)
            results_collector.collect_percent_seen(percent_seen, f_id)

            f_id += 1

        results_collector.dump_feature_extractor(args.output_folder)
        metadata["align_detect_track"] = False
        write_metadata(args, metadata)
    else:
        results_collector.set_self_params(args.output_folder, parmas=["alignment", "jai_zed", "detections", "tracks",
                                                                      "percent_seen"])
    pbar.close()
    if args.debug.align_graph and not args.no_sync:
        plot_alignmnt_graph(args, results_collector, frame_drop_jai)
    if slice_data_zed or slice_data_jai or os.path.exists(all_slices_path):
        if slice_data_jai:
            """this function depends on how we sliced (before or after slicing bug)"""
            slice_df = slice_to_trees(args.jai_slice_data_path, args.output_folder, resize_factor=3, h=2048, w=1536)
            slice_df.to_csv(os.path.join(args.output_folder, "slices.csv"))
        elif slice_data_zed:
            slice_data_zed = results_collector.converted_slice_data(slice_data_zed)# convert from zed coordinates to jai
            slice_df = post_process(slice_data_zed, args.output_folder, save_csv=True)
        else:
            slice_df = pd.read_csv(all_slices_path)
        slice_df = post_process_slice_df(slice_df)
        slice_df.to_csv(os.path.join(args.output_folder, "slices.csv"))
        results_collector.dump_to_trees(args.output_folder, slice_df)
        filtered_trees = results_collector.save_trees_sliced_track(args.output_folder, slice_df)
        if args.debug.trees:
            draw_on_tracked_imgaes(args, slice_df, filtered_trees, jai_cam, results_collector)

    if args.create_full_features and (slice_data_zed or slice_data_jai or os.path.exists(all_slices_path))\
            and (tree_features or args.overwtire.trees):
        df = create_row_features_fe_pipe(args.output_folder, zed_shift=args.zed_shift, max_x=600, max_y=900,
                                         save_csv=True, block_name=args.block_name, max_z=args.max_z,
                                         cameras={"zed_cam": zed_cam, "rgb_jai_cam": rgb_jai_cam, "jai_cam": jai_cam},
                                         debug=args.debug.features)
        metadata["tree_features"] = False
        write_metadata(args, metadata)
        if "cv" in df.columns:
            only_cv_df = df[["cv", "name"]]
            tree_ids = only_cv_df.loc[:, "name"].copy().apply(lambda x: x.split("_")[1][1:]).values
            only_cv_df["tree_id"] = tree_ids
            only_cv_df = only_cv_df[["name", "tree_id", "cv"]]
            only_cv_df.to_csv(os.path.join(args.output_folder, "trees_cv.csv"))
        else:
            print("problem with ", args.jai.movie_path)

    zed_cam.close()
    jai_cam.close()
    rgb_jai_cam.close()


def plot_alignmnt_graph(args, results_collector, frame_drop_jai):
    run_type = args.debug.run_type
    tx_data = [a["tx"] for a in results_collector.alignment]
    frames = [a["frame"] for a in results_collector.alignment]
    zed_shifts = np.array([a["zed_shift"] for a in results_collector.alignment])
    plt.figure(figsize=(15, 10))
    tx_data = np.clip(tx_data, -50, 200)
    plt.plot(frames, tx_data)
    graph_std = np.round(np.std(tx_data), 2)
    tx_conv = np.convolve(tx_data, np.ones(10) / 10, mode='same')
    conv_noise = np.round(np.mean(np.abs(tx_data[5:-5] - tx_conv[5:-5])), 2)
    conv_noise_med = np.round(np.median(np.abs(tx_data[5:-5] - tx_conv[5:-5])), 2)
    plt.plot(frames, tx_conv, color="orange")
    block = os.path.basename(os.path.dirname(args.output_folder))
    row = os.path.basename(args.output_folder)
    plt.title(f"{block}-{row}-{run_type}_std:{graph_std}_conv_noise:({conv_noise},{conv_noise_med})")
    for frame in frame_drop_jai[frame_drop_jai < np.max(frames)]:
        plt.vlines(frame, np.min(tx_data), np.max(tx_data), color="red", linestyles="dotted")
    for frame in np.array(frames)[np.where(zed_shifts[1:] - zed_shifts[:-1] == 1)[0]]:
        plt.vlines(frame, np.min(tx_data), np.max(tx_data), color="green", linestyles="dotted")
    plt.savefig(f"{args.output_folder}_{run_type}_graph.png")
    plt.show()


def draw_on_tracked_imgaes(args, slice_df, filtered_trees, jai_cam, results_collector):
    """
    this function draws tracking and slicing on an imgae, and saves each tree to args.debug_folder
    :param args: arguments conifg file
    :param slice_df: a slices data frame containing [frame_id, tree_id, start, end]
    :param filtered_trees: a dataframe of tracking results
    :param jai_cam: jai vamera video wrapper object
    :param results_collector: result collector object
    :return: None
    """
    slice_df["start"] = slice_df["start"].replace(-1, 0)
    slice_df["end"] = slice_df["end"].replace(-1, int(jai_cam.get_height() - 1))
    for tree_id in filtered_trees["tree_id"].unique():
        tree_slices = slice_df[slice_df["tree_id"] == tree_id]
        tree_tracks = filtered_trees[filtered_trees["tree_id"] == tree_id]
        unique_tracks = tree_tracks["track_id"].unique()
        new_ids = dict(zip(unique_tracks, range(len(unique_tracks))))
        tree_tracks.loc[:, "track_id"] = tree_tracks["track_id"].map(new_ids)
        for f_id in tree_tracks["frame_id"].unique():
            dets = tree_tracks[tree_tracks["frame_id"] == f_id]
            dets  = dets.values
            frame_slices = tree_slices[tree_slices["frame_id"] == f_id][["start", "end"]].astype(int).values[0]
            debug_outpath = os.path.join(args.debug_folder, f"{args.block_name}_{args.row_name}_T{tree_id}")
            validate_output_path(debug_outpath)
            ret, frame = jai_cam.get_frame(f_id)
            if ret:
                frame = cv2.line(frame, (frame_slices[0], 0), (frame_slices[0], int(jai_cam.get_width())),
                                 color=(255, 0, 0), thickness=2)
                frame = cv2.line(frame, (frame_slices[1], 0), (frame_slices[1], int(jai_cam.get_width())),
                                 color=(255, 0, 0), thickness=2)
                results_collector.draw_and_save(frame, dets, f_id, debug_outpath, t_index=6)


def get_metadata_path(args):
    """
    returns metadata path based on moviepath in args
    :param args: argumetns hash
    :return: meta_data_path
    """
    row_folder = os.path.dirname(args.jai.movie_path)
    meta_data_path = os.path.join(row_folder, "metadata.json")
    return meta_data_path


def write_metadata(args, metadata):
    """
    writes metadata json
    :param args: argumetns file
    :param metadata: metadata dict
    :return:
    """
    meta_data_path = get_metadata_path(args)
    write_json(meta_data_path, metadata)


def update_arg_with_metadata(args):
    """
    updates the arguments base on metadata json, this is for passsing specific argumetns for each row
    :param args: current args
    :return: updated args
    """
    meta_data_path = get_metadata_path(args)
    metadata = {}
    if os.path.exists(meta_data_path):
        with open(meta_data_path, 'r') as f:
            metadata = json.load(f)
            metadata_keys = metadata.keys()
            if "zed_shift" in metadata_keys:
                args.zed_shift = metadata["zed_shift"]
            if "block_name" in metadata_keys:
                args.block_name = metadata["block_name"]
            if "max_z" in metadata_keys:
                args.max_z = metadata["max_z"]
            if "zed_rotate" in metadata_keys:
                args.zed.rotate = metadata["zed_rotate"]
    return args, metadata


def validate_output_path(output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)


def zed_slicing_to_jai(slice_data_path, output_folder, rotate=False):
    slice_data = load_json(slice_data_path)
    slice_data = ResultsCollector().converted_slice_data(slice_data)
    slice_df = post_process(slice_data=slice_data)
    slice_df.to_csv(os.path.join(output_folder, 'all_slices.csv'))



if __name__ == "__main__":
    # reset_metadata("/media/fruitspec-lab/cam175/customers/PROPAL")
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/home/fruitspec-lab/FruitSpec/Code/fsCounter/vision/pipelines/config/dual_runtime_config.yaml"
    # config_file = "/config/pipeline_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(runtime_config)


    run(cfg, args)
