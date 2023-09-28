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

from vision.misc.help_func import get_repo_dir, load_json, validate_output_path

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.data.results_collector import ResultsCollector
from vision.tools.translation import translation as T
from vision.depth.slicer.slicer_flow import post_process
from vision.tools.sensors_alignment import SensorAligner
from vision.tools.camera import batch_is_saturated
from vision.pipelines.ops.simulator import get_n_frames, write_metadata, init_cams, get_frame_drop
from vision.pipelines.ops.frame_loader import FramesLoader
from vision.data.fs_logger import Logger
from vision.pipelines.ops.bboxes import depth_center_of_box, cut_zed_in_jai
from vision.pipelines.ops.kp_matching.infer import lightglue_infer

def run(cfg, args, metadata=None, n_frames=None):

    adt = Pipeline(cfg, args)
    results_collector = ResultsCollector(rotate=args.rotate)

    print(f'Inferencing on {args.jai.movie_path}\n')

    frame_drop_jai = get_frame_drop(args)
    if n_frames is None:
        n_frames = len(adt.frames_loader.sync_jai_ids)

    f_id = 0

    pbar = tqdm(total=n_frames)
    while f_id < n_frames:
        pbar.update(adt.batch_size)
        zed_batch, depth_batch, jai_batch, rgb_batch = adt.get_frames(f_id)


        rgb_stauts, rgb_detailed = adt.is_saturated(rgb_batch, f_id)
        zed_stauts, zed_detailed = adt.is_saturated(zed_batch, f_id)
        if rgb_stauts or zed_stauts:
             for i in range(adt.batch_size):
                if f_id + i in frame_drop_jai:
                     adt.sensor_aligner.zed_shift += 1
             f_id += adt.batch_size
             adt.logger.iterations += 1
             continue


        s = time.time()
        alignment_results = adt.align_cameras(zed_batch, jai_batch)
        e = time.time()
        print(f'alignment time:{e-s:.4f}')

        # detect:
        det_outputs = adt.detect(jai_batch)

        # find translation
        translation_results = adt.get_translation(jai_batch, det_outputs)

        # depth:
        depth_results = get_depth_to_bboxes_batch(depth_batch, jai_batch, alignment_results, det_outputs)

        # track:
        s = time.time()
        trk_outputs, trk_windows = adt.track(det_outputs, translation_results, f_id, depth_results)
        e = time.time()
        print(f'tracking time:{e - s:.4f}')


        trk_outputs = append_to_trk(trk_outputs, depth_results)

        #collect results:
        results_collector.collect_detections(det_outputs, f_id)
        results_collector.collect_tracks(trk_outputs)
        results_collector.collect_alignment(alignment_results, f_id)
        results_collector.collect_jai_translation(translation_results, f_id)


        #results_collector.draw_and_save_batch(jai_batch, trk_outputs, f_id, args.output_folder)
        results_collector.debug_batch(f_id, args, trk_outputs, det_outputs, jai_batch, None, trk_windows)

        f_id += adt.batch_size
        adt.logger.iterations += 1

    pbar.close()
    adt.frames_loader.close_cameras()
    adt.dump_log_stats(args)

    update_metadata(metadata, args)

    return results_collector


class Pipeline():

    def __init__(self, cfg, args):
        self.logger = Logger(args)
        self.frames_loader = FramesLoader(cfg, args)
        self.detector = counter_detection(cfg, args)
        self.translation = T(cfg.batch_size, cfg.translation.translation_size, cfg.translation.dets_only, cfg.translation.mode)
        #self.sensor_aligner = SensorAligner(cfg=cfg.sensor_aligner, batch_size=cfg.batch_size)
        self.sensor_aligner = lightglue_infer(cfg)
        self.batch_size = cfg.batch_size


    def get_frames(self, f_id):
        try:
            name = self.frames_loader.get_frames.__name__
            s = time.time()
            self.logger.debug(f"Function {name} started")
            #output = self.frames_loader.get_frames(f_id, self.sensor_aligner.zed_shift)
            output = self.frames_loader.get_frames(f_id, 0)
            self.logger.debug(f"Function {name} ended")
            e = time.time()
            self.logger.statistics.append({'id': self.logger.iterations, 'func': name, 'time': e-s})

            return output
        except:
            self.logger.exception("Exception occurred")
            raise


    def is_saturated(self, frame, f_id, percentile=0.4, cam_name='JAI'):
        try:
            name = batch_is_saturated.__name__
            s = time.time()
            self.logger.debug(f"Function {name} started")
            status, detailed = batch_is_saturated(frame, percentile=percentile)
            self.logger.debug(f"Function {name} ended")
            e = time.time()
            self.logger.statistics.append({'id': self.logger.iterations, 'func': name, 'time': e-s})
            if status:
                if len(detailed) == 1:
                    self.logger.debug(f'Frame {f_id} is saturated, skipping')
                else:
                    self.logger.debug(f'Batch starting with frame {f_id} is saturated, skipping')

            return status, detailed
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

    def track(self, inputs, translations, f_id, dets_depth=None):
        try:
            name = self.detector.track.__name__
            s = time.time()
            self.logger.debug(f"Function {name} started")
            output = self.detector.track(inputs, translations, f_id, dets_depth)
            self.logger.debug(f"Function {name} ended")
            e = time.time()
            self.logger.debug(f"Function {name} execution time {e - s:.3f}")
            self.logger.statistics.append({'id': self.logger.iterations, 'func': name, 'time': e - s})

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

    def get_translation(self, frames, det_outputs):
        try:
            name = self.translation.get_translation.__name__
            s = time.time()
            self.logger.debug(f"Function {name} started")
            output = self.translation.batch_translation(frames, det_outputs)
            #output = self.sensor_aligner.batch_translation(frames)
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
            #self.logger.debug(f"Sensors frame shift {self.sensor_aligner.zed_shift}")
            self.logger.statistics.append({'id': self.logger.iterations, 'func': name, 'time': e - s})

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

    def dump_log_stats(self, args):

        dump = pd.DataFrame(self.logger.statistics, columns=['id', 'func', 'time'])
        dump.to_csv(os.path.join(args.output_folder, 'log_stats.csv'))



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
    logger = Logger(args)
    detector = counter_detection(cfg, args)
    results_collector = ResultsCollector(rotate=args.rotate)
    translation = T(cfg.translation.translation_size, cfg.translation.dets_only, cfg.translation.mode)
    sensor_aligner = SensorAligner(args=args.sensor_aligner, zed_shift=args.zed_shift)

    return detector, results_collector, translation, sensor_aligner, logger


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


def zed_slicing_to_jai(slice_data_path, output_folder, rotate=False):
    slice_data = load_json(slice_data_path)
    slice_data = ResultsCollector().converted_slice_data(slice_data)
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






if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)

    zed_name = "ZED.mkv"
    depth_name = "DEPTH.mkv"
    fsi_name = "Result_FSI.mkv"
    rgb_name = "Result_RGB.mkv"
    time_stamp = "jaized_timestamps.csv"

    output_path = "/home/matans/Documents/fruitspec/sandbox/tracker/depth/Fowler_BLOCK700_200723_row4_depth_ada5_ref1"
    #output_path = "/home/matans/Documents/fruitspec/sandbox/tracker/depth/Fowler_FREDIANI_210723_row7_depth_piv1"
    validate_output_path(output_path)

    rows_dir = "/media/matans/My Book/FruitSpec/Customers_data/Fowler/daily/BLOCK700/200723"
    #rows_dir = "/media/matans/My Book/FruitSpec/Customers_data/Fowler/daily/FREDIANI/210723"


    #rows_dir = "/media/matans/My Book/FruitSpec/WASHDE/June_29/"
    rows = os.listdir(rows_dir)
    rows = ["row_4"]
    #rows = ["row_7"]
    for row in rows:
        row_folder = os.path.join(rows_dir, row, '1')

        args.output_folder = os.path.join(output_path, row)
        args.sync_data_log_path = os.path.join(row_folder, time_stamp)
        args.zed.movie_path = os.path.join(row_folder, zed_name)
        args.depth.movie_path = os.path.join(row_folder, depth_name)
        args.jai.movie_path = os.path.join(row_folder, fsi_name)
        args.rgb_jai.movie_path = os.path.join(row_folder, rgb_name)

        validate_output_path(args.output_folder)

        rc = run(cfg, args, n_frames=150)
        rc.dump_results(args.output_folder)