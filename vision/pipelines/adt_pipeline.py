import os
import sys
import cv2
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

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


def run(cfg, args, metadata=None):

    adt = Pipeline(cfg, args)
    results_collector = ResultsCollector(rotate=args.rotate)

    print(f'Inferencing on {args.jai.movie_path}\n')

    frame_drop_jai = get_frame_drop(args)
    n_frames = get_number_of_frames(adt.jai_cam.get_number_of_frames(), metadata)

    f_id = 0

    pbar = tqdm(total=n_frames)
    while f_id < n_frames:
        pbar.update(adt.batch_size)
        zed_batch, jai_batch, rgb_batch = adt.get_frames(f_id)
        #zed_frame, point_cloud, fsi_ret, jai_frame, rgb_ret, rgb_jai_frame = adt.get_frames(f_id)

        #if not fsi_ret or not adt.zed_cam.res or not rgb_ret:  # couldn't get frames, Break the loop
             #break
        rgb_stauts, rgb_detailed = adt.is_saturated(rgb_batch, f_id)
        zed_stauts, zed_detailed = adt.is_saturated(zed_batch, f_id)
        if rgb_stauts or zed_stauts:
             for i in range(adt.batch_size):
                if f_id + i in frame_drop_jai:
                     adt.sensor_aligner.zed_shift += 1
             f_id += adt.batch_size
             continue

         # align sensors
        # corr, tx_a, ty_a, sx, sy = adt.align_cameras(cv2.cvtColor(zed_frame, cv2.COLOR_BGR2RGB),
        #                                                            rgb_jai_frame)
        if f_id >=218:
            a=1
        alignment_results = adt.align_cameras(zed_batch, rgb_batch)

        # detect:
        det_outputs = adt.detect(jai_batch)

        # find translation
        translation_results = adt.get_translation(jai_batch, det_outputs)

        # track:
        trk_outputs, trk_windows = adt.track(det_outputs, translation_results, f_id)

        #collect results:
        results_collector.collect_detections(det_outputs, f_id)
        results_collector.collect_tracks(trk_outputs)
        results_collector.collect_alignment(alignment_results, f_id)

#        results_collector.draw_and_save(jai_frame, trk_outputs, f_id, args.output_folder)

        f_id += adt.batch_size
        adt.logger.iterations += 1

    pbar.close()
    adt.zed_cam.close()
    adt.jai_cam.close()
    adt.rgb_jai_cam.close()
    adt.dump_log_stats(args)

    results_collector.dump_feature_extractor(args.output_folder)

    update_metadata(metadata)

    return results_collector


class Pipeline():

    def __init__(self, cfg, args):
        self.logger = Logger(args)
        self.frames_loader = FramesLoader(cfg, args)
        self.detector = counter_detection(cfg, args)
        self.translation = T(cfg.translation.translation_size, cfg.translation.dets_only, cfg.translation.mode)
        self.sensor_aligner = SensorAligner(args=args.sensor_aligner, zed_shift=args.zed_shift, batch_size=cfg.batch_size)
        self.zed_cam, self.rgb_jai_cam, self.jai_cam = init_cams(args)
        self.batch_size = cfg.batch_size


    def get_frames(self, f_id):
        try:
            name = self.frames_loader.get_frames.__name__
            s = time.time()
            self.logger.info(f"Function {name} started")
            output = self.frames_loader.get_frames(f_id, self.sensor_aligner.zed_shift)
            self.logger.info(f"Function {name} ended")
            e = time.time()
            self.logger.info(f"Function {name} execution time {e - s:.3f}")
            self.logger.statistics.append({'id': self.logger.iterations, 'func': name, 'time': e-s})

            return output
        except:
            self.logger.exception("Exception occurred")
            raise


    def is_saturated(self, frame, f_id, percentile=0.6, cam_name='JAI'):
        try:
            name = batch_is_saturated.__name__
            s = time.time()
            self.logger.info(f"Function {name} started")
            status, detailed = batch_is_saturated(frame, percentile=percentile)
            self.logger.info(f"Function {name} ended")
            e = time.time()
            self.logger.info(f"Function {name} execution time {e - s:.3f}")
            self.logger.statistics.append({'id': self.logger.iterations, 'func': name, 'time': e-s})
            if status:
                if len(detailed) == 1:
                    self.logger.info(f'Frame {f_id} is saturated, skipping')
                else:
                    self.logger.info(f'Batch starting with frame {f_id} is saturated, skipping')

            return status, detailed
        except:
            self.logger.exception("Exception occurred")
            raise


    def detect(self, frames):
        try:
            name = self.detector.detect.__name__
            s = time.time()
            self.logger.info(f"Function {name} started")
            output = self.detector.detect(frames)
            self.logger.info(f"Function {name} ended")
            e = time.time()
            self.logger.info(f"Function {name} execution time {e - s:.3f}")
            self.logger.statistics.append({'id': self.logger.iterations, 'func': name, 'time': e - s})

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

    def track(self, inputs, translations, f_id):
        try:
            name = self.detector.track.__name__
            s = time.time()
            self.logger.info(f"Function {name} started")
            output = self.detector.track(inputs, translations, f_id)
            self.logger.info(f"Function {name} ended")
            e = time.time()
            self.logger.info(f"Function {name} execution time {e - s:.3f}")
            self.logger.statistics.append({'id': self.logger.iterations, 'func': name, 'time': e - s})

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

    def get_translation(self, frames, det_outputs):
        try:
            name = self.translation.get_translation.__name__
            s = time.time()
            self.logger.info(f"Function {name} started")
            output = self.translation.batch_translation(frames, det_outputs)
            self.logger.info(f"Function {name} ended")
            e = time.time()
            self.logger.info(f"Function {name} execution time {e - s:.3f}")
            for res in output:
                self.logger.info(f"JAI X translation {res[0]}, Y translation {res[1]}")
            self.logger.statistics.append({'id': self.logger.iterations, 'func': name, 'time': e - s})

            return output

        except:
            self.logger.exception("Exception occurred")
            raise




    def align_cameras(self, zed_batch, rgb_jai_batch):
        try:
            name = self.sensor_aligner.align_sensors.__name__
            s = time.time()
            self.logger.info(f"Function {name} started")
            output = self.sensor_aligner.align_on_batch(zed_batch, rgb_jai_batch)
            # corr, tx_a, ty_a, sx, sy, kp_zed, kp_jai, match, st, M = self.sensor_aligner.align_sensors(cv2.cvtColor(zed_frame, cv2.COLOR_BGR2RGB),
            #                                                                                            rgb_jai_frame)
            self.logger.info(f"Function {name} ended")
            e = time.time()
            self.logger.info(f"Function {name} execution time {e - s:.3f}")
            self.logger.info(f"Sensors frame shift {self.sensor_aligner.zed_shift}")
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

def update_metadata(metadata):
    if metadata is None:
        metadata = {}
    metadata["align_detect_track"] = False
    write_metadata(args, metadata)



if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)

    validate_output_path(args.output_folder)
    #copy_configs(pipeline_config, runtime_config, args.output_folder)

    run(cfg, args)
