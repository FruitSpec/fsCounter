import os
import sys
import cv2
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pickle

from vision.tools.camera_calibration import undistort
from vision.misc.help_func import get_repo_dir, load_json, validate_output_path

# repo_dir = get_repo_dir()
# sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.data.results_collector import ResultsCollector
from vision.tools.translation import translation as T
from vision.data.fs_logger import Logger
from concurrent.futures import ThreadPoolExecutor

from vision.depth.slicer.slicer_flow import post_process
from vision.pipelines.ops.simulator import get_n_frames, write_metadata
from vision.pipelines.ops.frame_loader import FramesLoader
from vision.pipelines.ops.measure import stable_euclid_dist, get_pix_size
from vision.pipelines.ops.bboxes import depth_center_of_box, cut_zed_in_jai
from vision.pipelines.ops.bboxes import xyz_center_of_box, convert_dets, match_by_intersection
from vision.pipelines.ops.kp_matching.infer import lightglue_infer
from vision.pipelines.ops.fruit_cluster import FruitCluster
from vision.pipelines.misc.filters import batch_filter_by_distance
from vision.tools.color import get_tomato_color
from vision.pipelines.misc.debug_helpers import save_aligned





def debug_tracks_pc(jai_batch, trk_outputs, f_id):
    from vision.visualization.drawer import draw_rectangle, draw_text, draw_highlighted_test, get_color
    import matplotlib.pyplot as plt
    frame = jai_batch[0].copy().astype(np.uint8)
    for det in trk_outputs[0]:
        track_id = det[6]
        color_id = int(track_id) % 15  # 15 is the number of colors in list
        color = get_color(color_id)
        text_color = get_color(-1)
        frame = draw_rectangle(frame, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), color, 3)
        title = f'ID:{int(track_id)}:({det[8]:.2f}, {det[9]:.2f}, {det[10]:.2f})'
        frame = draw_highlighted_test(frame, title, (det[0], det[1]), frame.shape[1], color, text_color,
                                      True, 10, 3)
    cv2.imwrite(f"/media/fruitspec-lab/easystore/debug_pc/{f_id}.png", frame)

def run(cfg, args, metadata=None, n_frames=None):
    adt = Pipeline(cfg, args)
    results_collector_zed = ResultsCollector(rotate=args.rotate)

    zed_args = args.copy()
    #zed_args.output_folder = os.path.join(args.output_folder, 'zed')
    zed_args.output_folder =args.output_folder


    print(f'Inferencing on {args.jai.movie_path}\n')

    max_cut_frame = metadata['max_cut_frame'] if metadata is not None else np.inf
    if isinstance(max_cut_frame, str):
        if max_cut_frame == "inf":
            max_cut_frame = np.inf
        else:
            max_cut_frame = int(max_cut_frame)
    if n_frames is None:
        n_frames = min(len(adt.frames_loader.sync_jai_ids), max_cut_frame)

    n_frames = (n_frames//cfg.batch_size)*cfg.batch_size
    f_id = 0

    pbar = tqdm(total=n_frames)
    while f_id < n_frames:
        pbar.update(adt.batch_size)
        zed_batch, pc_batch, jai_batch, rgb_batch = adt.get_frames(f_id)

        # detect:
        # det: [x1, y1, x2, y2, obj_conf, class_conf, class_pred]
        zed_det_outputs = adt.detector_zed.detect(zed_batch)

        # find translation
        # debug
        zed_debug = []
        zed_output_path = os.path.join(zed_args.output_folder, 'translation')
        validate_output_path(zed_output_path)
        for id_ in range(len(zed_batch)):
            zed_debug.append({'output_path': zed_output_path, 'f_id': f_id + id_})

        zed_translation_results = adt.zed_translation.batch_translation(zed_batch, zed_det_outputs, zed_debug)

        # track:
        # trk: [x1, y1, x2, y2, conf, class_pred, track_id, track_depth, frame_id]
        zed_trk_outputs, zed_trk_windows = adt.detector_zed.track(zed_det_outputs, zed_translation_results, f_id)

        # measure:
        # zed trk: [x1, y1, x2, y2, conf, class_pred, track_id, track_depth, frame_id, x_ceter, y_center, depth, width, height]
        zed_trk_outputs = adt.get_xyzs_dims(pc_batch, zed_trk_outputs)

        # filter by distance:

        # filter zed detections:
        zed_filtered_outputs, zed_ranges, zed_dets_ids = batch_filter_by_distance(zed_trk_outputs, pc_batch, cfg.filters.distance.threshold)

        # cluster:

        # zed trk: [x1, y1, x2, y2, conf, class_pred, track_id, track_depth, frame_id, x_ceter, y_center, depth, width, height, cluster_id]
        zed_filtered_outputs, clusters = adt.cluster.cluster_batch(zed_filtered_outputs, zed_ranges)






        # color
        # zed trk: [x1, y1, x2, y2, conf, class_pred, track_id, track_depth, frame_id, x_ceter, y_center, depth, width, height, cluster_id, color]
        zed_filtered_outputs, zed_tomato_colors, zed_tomato_colors_viz = get_colors(zed_batch,
                                                                               zed_filtered_outputs)


        # collect results:
        results_collector_zed.collect_adt(zed_filtered_outputs, [], f_id, zed_translation_results)

        validate_output_path(zed_args.output_folder)
        results_collector_zed.debug_batch(batch_id=f_id, args=zed_args, trk_outputs=zed_filtered_outputs, det_outputs=zed_det_outputs,
                                          frames=zed_batch, depth=None, trk_windows=zed_trk_windows, det_colors=zed_tomato_colors_viz)


        f_id += adt.batch_size
        adt.logger.iterations += 1

    pbar.close()
    adt.frames_loader.close_cameras()
    adt.dump_log_stats(args)

    results_collector_zed.dump_to_csv(os.path.join(zed_args.output_folder, 'tracks.csv'), type="zed_syngenta")

    return results_collector_zed


class Pipeline():

    def __init__(self, cfg, args):
        self.logger = Logger(args)
        self.frames_loader = FramesLoader(cfg, args)
        #self.detector_jai = counter_detection(cfg.jai, args)
        self.detector_zed = counter_detection(cfg.zed, args)
        #self.undistort_jai = undistort(cfg.jai.calibration_path)
        #self.undistort_zed = undistort(cfg.zed.calibration_path)
        self.cluster = FruitCluster(cfg.clusters.max_single_fruit_dist,
                                    cfg.clusters.range_diff_threshold,
                                    cfg.clusters.max_losses)
        self.jai_translation = T(cfg.jai.batch_size, cfg.translation.translation_size, cfg.translation.dets_only,
                                 cfg.translation.mode)
        self.zed_translation = T(cfg.zed.batch_size, cfg.translation.translation_size, cfg.translation.dets_only,
                                 cfg.translation.mode)
        self.sensor_aligner = lightglue_infer(cfg, len_size=cfg.len_size)
        self.batch_size = cfg.batch_size
        self.len_size = cfg.len_size



    def get_frames(self, f_id, zed_shift=None):
        try:
            name = self.frames_loader.get_frames.__name__
            s = time.time()
            self.logger.debug(f"Function {name} started")
            output = self.frames_loader.get_frames(f_id, zed_shift)
            self.logger.debug(f"Function {name} ended")
            e = time.time()
            self.logger.statistics.append({'id': self.logger.iterations, 'func': name, 'time': e-s})

            return output
        except:
            self.logger.exception("Exception occurred")
            raise




    def detect(self, frames):
        try:
            name = self.detector.detect.__name__
            s = time.time()
            self.logger.debug(f"Function {name} started")
            output = self.detector.detect(frames)
            self.logger.debug(f"Function {name} ended")
            e = time.time()
            self.logger.debug(f"Function {name} execution time {e - s:.3f}")
            self.logger.statistics.append({'id': self.logger.iterations, 'func': name, 'time': e - s})

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

    def track(self, inputs, translations, f_id):
        try:
            name = self.detector.track.__name__
            s = time.time()
            self.logger.debug(f"Function {name} started")
            output = self.detector.track(inputs, translations, f_id)
            self.logger.debug(f"Function {name} ended")
            e = time.time()
            self.logger.debug(f"Function {name} execution time {e - s:.3f}")
            self.logger.statistics.append({'id': self.logger.iterations, 'func': name, 'time': e - s})

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

    def get_translation(self, frames, det_outputs, debug=None):
        try:
            name = self.translation.get_translation.__name__
            s = time.time()
            self.logger.debug(f"Function {name} started")
            output = self.translation.batch_translation(frames, det_outputs, debug)
            self.logger.debug(f"Function {name} ended")
            e = time.time()
            self.logger.debug(f"Function {name} execution time {e - s:.3f}")
            for res in output:
                self.logger.debug(f"JAI X translation {res[0]}, Y translation {res[1]}")
            self.logger.statistics.append({'id': self.logger.iterations, 'func': name, 'time': e - s})

            return output

        except:
            self.logger.exception("Exception occurred")
            raise




    def align_cameras(self, zed_batch, rgb_jai_batch):
        try:
            name = self.sensor_aligner.align_sensors.__name__
            s = time.time()
            self.logger.debug(f"Function {name} started")
            output = self.sensor_aligner.align_on_batch(zed_batch, rgb_jai_batch)
            # corr, tx_a, ty_a, sx, sy, kp_zed, kp_jai, match, st, M = self.sensor_aligner.align_sensors(cv2.cvtColor(zed_frame, cv2.COLOR_BGR2RGB),
            #                                                                                            rgb_jai_frame)
            self.logger.debug(f"Function {name} ended")
            e = time.time()
            self.logger.debug(f"Function {name} execution time {e - s:.3f}")
            self.logger.debug(f"Sensors frame shift {self.sensor_aligner.zed_shift}")
            self.logger.statistics.append({'id': self.logger.iterations, 'func': name, 'time': e - s})

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

    def dump_log_stats(self, args):

        dump = pd.DataFrame(self.logger.statistics, columns=['id', 'func', 'time'])
        dump.to_csv(os.path.join(args.output_folder, 'log_stats.csv'))

    def get_percent_seen(self, zed_batch, pc_batch, alignment_results):
        """
        Calculates the percentage of the scene that is visible in the jai for each input.
        Full field is the Zed

        Args:
            zed_batch (List): List of lists of input frames.
            alignment_results (List): List of lists of alignment results.

        Returns:
            List[float]: List of percentages of the scene that is visible for each input.
        """
        try:
            name = "percent_seen"
            s = time.time()
            self.logger.info(f"Function {name} started")
            cut_coords = [res[0] for res in alignment_results]
            with ThreadPoolExecutor(max_workers=len(cut_coords)) as executor:
                output = list(executor.map(get_percent_seen, zed_batch, pc_batch, cut_coords))
            # output = list(map(get_percent_seen, zed_batch, cut_coords))
            self.log_end_func(name, s)

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

    def get_xyzs_dims(self, xyz_batch, dets):
        """
        Computes the xyz values, width and height for each object detected in the image frames.

        Args:
            xyz_batch (list): A list of numpy arrays containing ZED camera Point Clouds frames.
            jai_batch (list): A list of numpy arrays containing RGB JAI camera frames.
            alignment_results (list): A list of alignment_ results, one for each frame in `xyz_batch`.
            dets (list): A list containing the object detection results.

        Returns:
            list: A list containing depth values, one for each object detected in the image frames.
        """
        try:
            name = "xyzs_to_outputs"
            s = time.time()
            self.logger.info(f"Function {name} started")
            output = get_xyz_to_bboxes_batch(xyz_batch, dets, True, True, True)
            self.log_end_func(name, s)

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

    def close_cams(self):
        """
        Closes the cameras.
        """
        self.zed_cam.close()
        self.jai_cam.close()
        self.rgb_jai_cam.close()

    def log_end_func(self, name, s):
        """
        Log the end of a function and its execution time.

        Args:
            name (str): The name of the function.
            s (float): The start time of the function.

        Returns:
            None
        """
        self.logger.info(f"Function {name} ended")
        e = time.time()
        self.logger.info(f"Function {name} execution time {e - s:.3f}")
        self.logger.statistics.append({'id': self.logger.iterations, 'func': name, 'time': e - s})

    def get_real_dims(self, xyz_batch, jai_batch, alignment_results, dets):
        """
        Computes the width and height values for each object detected in the image frames.

        Args:
            xyz_batch (list): A list of numpy arrays containing ZED camera Point Clouds frames.
            jai_batch (list): A list of numpy arrays containing RGB JAI camera frames.
            alignment_results (list): A list of alignment_ results, one for each frame in `xyz_batch`.
            dets (list): A list containing the object detection results.

        Returns:
            list: A list containing depth values, one for each object detected in the image frames.
        """
        try:
            name = "real_dims_to_outputs"
            s = time.time()
            self.logger.info(f"Function {name} started")
            cut_coords = tuple(res[0] for res in alignment_results)
            output = get_xyz_to_bboxes_batch(xyz_batch, jai_batch, cut_coords, dets, True)
            self.log_end_func(name, s)

            return output

        except:
            self.logger.exception("Exception occurred")
            raise


def zed_slicing_to_jai(slice_data_path, output_folder, rotate=False):
    slice_data = load_json(slice_data_path)
    slice_data = ResultsCollector().converted_slice_data(slice_data, mode=cfg.result_collector.mode)
    slice_df = post_process(slice_data=slice_data)
    slice_df.to_csv(os.path.join(output_folder, 'all_slices.csv'))


def get_number_of_frames(jai_max_frames, metadata=None):
    max_cut_frame = int(metadata['max_cut_frame']) if metadata is not None else np.inf
    n_frames = get_n_frames(max_cut_frame,jai_max_frames, metadata)

    return n_frames

def update_metadata(metadata, args):
    if metadata is None:
        metadata = {}
    metadata["align_detect_track"] = False
    write_metadata(args, metadata)


def get_xyz_to_bboxes(xyz_frame, dets, pc=False, dims=False,
                      euclid_dist=False, factor = 8/255):
    """
    Retrieves the depth to each bbox
    Args:
        xyz_frame (np.array): a Point cloud image
        jai_frame (np.array): FSI image
        cut_coords (tuple): jai in zed coords
        dets (list): list of detections
        pc (bool): flag for returning the entire x,y,z
        dims (bool): flag for returning the real width and height
        aligned (bool): flag indicating if the frames are already aligned
        resize_factors (Iterable): resize factors for an aligned imaged (r_h, r_w)

    Returns:
        z_s (list): list with depth to each detection
    """
    if not pc:
        xyz_frame = xyz_frame * factor
    for det in dets:
        box = ((int(det[0]), int(det[1])), (int(det[2]), int(det[3])))
        if pc:
            det += list(xyz_center_of_box(xyz_frame, box))
        else:
            det.append(xyz_center_of_box(xyz_frame, box)[2])
        if dims:
            if euclid_dist:
                det += list(stable_euclid_dist(xyz_frame, box))
            else: # pix size algorithm
                # extract pixel size
                size_pix_x, size_pix_y = get_pix_size(det[-1], [int(num) for num in det[:4]])
                det += [np.nanmedian(size_pix_x),
                        np.nanmedian(size_pix_y)]
    return dets


def get_xyz_to_bboxes_batch(xyz_batch, dets, pc=False, dims=False, euclid=False):
    """
    Retrieves the depth to each bbox in the batch
    Args:
        xyz_batch (np.array): batch a Point cloud image
        jai_batch (np.array): batch of FAI images
        cut_coords (tuple): batch of jai in zed coords
        dets (list): batch of list of detections per image
        pc (bool): flag for returning the entire x,y,z
        dims (bool): flag for returning the real width and height

    Returns:
        z_s (list): list of lists with depth to each detection
    """
    n = len(xyz_batch)

    return list(map(get_xyz_to_bboxes, xyz_batch, dets, [pc] * n, [dims] * n, [euclid] * n))

def get_depth_to_bboxes(depth_frame, jai_frame, cut_coords, dets, factor = 8 / 255):
    """
    Retrieves the depth to each bbox
    Args:
        xyz_frame (np.array): a Point cloud image
        jai_frame (np.array): FSI image
        cut_coords (tuple): jai in zed coords
        dets (list): list of detections
        pc (bool): flag for returning the entire x,y,z
        dims (bool): flag for returning the real width and height

    Returns:
        z_s (list): list with depth to each detection
    """
    depth_frame = depth_frame * factor
    cut_coords = dict(zip(["x1", "y1", "x2", "y2"], [[int(cord)] for cord in cut_coords]))
    depth_frame_aligned = cut_zed_in_jai({"zed": depth_frame}, cut_coords, rgb=False)["zed"]
    r_h, r_w = depth_frame_aligned.shape[0] / jai_frame.shape[0], depth_frame_aligned.shape[1] / jai_frame.shape[1]
    output = []
    for det in dets:
        box = ((int(det[0] * r_w), int(det[1] * r_h)), (int(det[2] * r_w), int(det[3] * r_h)))
        output.append(depth_center_of_box(depth_frame_aligned, box))
    return output

def get_depth_to_bboxes_batch(xyz_batch, jai_batch, batch_aligment, dets):
    """
    Retrieves the depth to each bbox in the batch
    Args:
        xyz_batch (np.array): batch a Point cloud image
        jai_batch (np.array): batch of FAI images
        dets (list): batch of list of detections per image

    Returns:
        z_s (list): list of lists with depth to each detection
    """
    cut_coords = []
    for a in batch_aligment:
        cut_coords.append(a[0])
    n = len(xyz_batch)
    return list(map(get_depth_to_bboxes, xyz_batch, jai_batch, cut_coords, dets))


def append_to_trk(trk_batch_res, results):
    for frame_res, frame_depth in zip(trk_batch_res, results):
        for trk, depth in zip(frame_res, frame_depth):
            trk.append(np.round(depth, 3))

    return trk_batch_res

def get_trks_colors(img, trk_outputs):
    colors = []

    for trk in trk_outputs:
        out_of_view = False
        h, w = img.shape[:2]
        trk_bbox = trk[:4]
        x1, y1, x2, y2 = trk_bbox
        if x1 < 0 or x1 > w:
            out_of_view = True
        if x2 < 0 or x2 > w:
            out_of_view = True
        if y1 < 0 or y1 > h:
            out_of_view = True
        if y2 < 0 or y2 > h:
            out_of_view = True

        if out_of_view:
            colors.append(-1)
        else:
            x1, y1 = int(max(x1, 0)), int(max(y1, 0))

            x2, y2 = int(min(x2, w - 1)), int(min(y2, h - 1))
            colors.append(get_tomato_color(img[y1:y2, x1:x2]))
    return colors


def add_result_to_track(trk_output, results):
    if trk_output:
        for id_, result in enumerate(results):
            trk_output[id_].append(result)
    return trk_output


def tomato_color_class_to_rgb(colors):
    colors_out = []
    colors_dict = {0:(255,255,255), # white
        1:(0,0,255), # red
    2:(0,125,255), # orange
    3:(128,0,128), # purpule
    4:(0,255,255), # yellow
    5:(0,255,0), # green
    -1:(0, 0 ,0)} # black - outside FOV
    for color in colors:
        colors_out.append(colors_dict[color])
    return colors_out

def batch_reduce_filterd(dets, keep_ids):

    return list(map(reduce_filtered, dets, keep_ids))

def reduce_filtered(dets, keep_ids):
    filtered_dets = []

    for id_, det in enumerate(dets):
        if id_ in keep_ids:
            filtered_dets.append(det)

    return filtered_dets


def get_colors(batch, trk_outputs):

    tomato_colors = list(map(get_trks_colors, batch, trk_outputs))
    tomato_colors_viz = list(map(tomato_color_class_to_rgb, tomato_colors))
    trk_outputs = list(map(add_result_to_track, trk_outputs, tomato_colors))

    return trk_outputs, tomato_colors, tomato_colors_viz

def add_clusters_to_zed_outputs(zed_filtered_outputs, jai_conv_filtered_outputs, clusters):
    outputs = []
    for zed_output, jai_output, frame_clusters in zip(zed_filtered_outputs, jai_conv_filtered_outputs, clusters):
        if zed_output and jai_output:
            zed_bboxes = np.array(zed_output)[:, :4]
            jai_bboxes = np.array(jai_output)[:, :4]

            matches = match_by_intersection(zed_bboxes, jai_bboxes)
            zed_bboxes_output = []
            for zed_bboxes_id in range(matches.shape[0]):
                if np.sum(matches[zed_bboxes_id, :]): # there is a match
                    cluster_index = np.argmax(matches[zed_bboxes_id, :])
                    cluster_id = frame_clusters[cluster_index]
                else:
                    cluster_id = -1
                zed_output[zed_bboxes_id].append(cluster_id)
            outputs.append(zed_output.copy())

    return outputs



def save_batch_aligned(zed_batch, jai_batch, output_folder, f_id, sub_folder='FOV', dets=None, lense=61, name=None, index_=-1):

    for i in range(len(zed_batch)):
        if dets is not None:
            c_dets = dets[i]
        else:
            c_dets = None

        save_aligned(zed_batch[i], jai_batch[i], output_folder, f_id + i, sub_folder=sub_folder, dets=c_dets, lense=lense, name=name,
                     index_=index_)


def get_color_masks(jai_image, dets, threshold=130):

    bboxes = np.array(dets)[:, :4]
    hsv = cv2.cvtColor(jai_image, cv2.COLOR_RGB2HSV)
    masks = []

    for bbox in bboxes:
        masks.append(seg_bbox(hsv, bbox, threshold))

    return masks

def seg_bbox(hsv_jai_image, bbox, threshold=130):

    cropped = hsv_jai_image[bbox[1]:bbox[3], bbox[0]:bbox[2], 0]
    cropped_mask = cropped.copy()
    cropped_mask[cropped < threshold] = 0
    cropped_mask[cropped >= threshold] = 1

    return cropped_mask



if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/syn_pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)


    zed_name = "ZED.svo"
    depth_name = "DEPTH.mkv"
    fsi_name = "Result_FSI.mkv"
    rgb_name = "Result_RGB.mkv"
    time_stamp = "jaized_timestamps.csv"

    output_path = "/home/matans/Documents/fruitspec/sandbox/syngenta/lean_flow_test_data_291123_5"
    validate_output_path(output_path)

    rows_dir = "/media/matans/My Book/FruitSpec/Syngenta/Calibration_data/141223"

    rows = os.listdir(rows_dir)
    failed = []
    for row in rows:
        row_folder = os.path.join(rows_dir, row)
        if not os.path.isdir(row_folder):
            continue
        sections = os.listdir(row_folder)
        for section in sections:
            row_folder = os.path.join(rows_dir, row, section)

            args.output_folder = row_folder # os.path.join(output_path, row)
            args.sync_data_log_path = os.path.join(row_folder, time_stamp)
            args.zed.movie_path = os.path.join(row_folder, zed_name)
            args.depth.movie_path = os.path.join(row_folder, depth_name)
            args.jai.movie_path = os.path.join(row_folder, fsi_name)
            args.rgb_jai.movie_path = os.path.join(row_folder, rgb_name)

            # if not os.path.exists(args.sync_data_log_path):
            #     continue

            validate_output_path(args.output_folder)

            try:
                rc_z = run(cfg, args)
            except:
                print(f'failed to analyze: {args.zed.movie_path}')
                failed.append(args.zed.movie_path)
        #rc_j.dump_feature_extractor(args.output_folder)

    print('failed:')
    print(failed)