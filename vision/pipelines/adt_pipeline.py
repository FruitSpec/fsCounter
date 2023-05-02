import os
import sys
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
from vision.misc.help_func import get_repo_dir, load_json, validate_output_path
from vision.pipelines.detection_flow import counter_detection
from vision.data.results_collector import ResultsCollector
from vision.tools.translation import translation as translator
from vision.depth.slicer.slicer_flow import post_process
from vision.tools.sensors_alignment import SensorAligner, FirstMoveDetector
from vision.tools.camera import batch_is_saturated
from vision.pipelines.ops.simulator import get_n_frames, write_metadata, init_cams, get_frame_drop
from vision.pipelines.ops.frame_loader import FramesLoader
from vision.data.fs_logger import Logger
from vision.feature_extractor.boxing_tools import xyz_center_of_box, cut_zed_in_jai
from concurrent.futures import ThreadPoolExecutor
from vision.feature_extractor.image_processing import get_percent_seen
from vision.feature_extractor.tree_size_tools import stable_euclid_dist

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))


def run(cfg, args, metadata=None):

    adt = Pipeline(cfg, args, metadata=None)
    results_collector = ResultsCollector(rotate=args.rotate)

    print(f'Inferencing on {args.jai.movie_path}\n')

    frame_drop_jai = get_frame_drop(args)
    n_frames = get_number_of_frames(adt.jai_cam.get_number_of_frames(), metadata)

    f_id = 0

    pbar = tqdm(total=n_frames)
    while f_id < n_frames:
        pbar.update(adt.batch_size)
        zed_batch, jai_batch, rgb_batch, pc_batch = adt.get_frames(f_id)

        (rgb_status, _), (zed_status, _) = adt.is_saturated(rgb_batch, f_id), adt.is_saturated(zed_batch, f_id)
        if rgb_status or zed_status:
            for i in range(adt.batch_size):
                if f_id + i in frame_drop_jai:
                    adt.sensor_aligner.zed_shift += 1
            f_id += adt.batch_size
            continue

        if f_id < adt.s_frame:
            f_id += adt.batch_size
            continue

        # auto zed shift detection
        if adt.auto_shift:
            zed_shift, status = adt.update_moves(zed_batch, jai_batch, f_id)
            if not status:
                f_id += adt.batch_size
                continue
            adt.postprocess_auto_shift(zed_shift)

        # align sensors
        alignment_results = adt.align_cameras(zed_batch, rgb_batch)

        # detect:
        det_outputs = adt.detect(jai_batch)

        # find translation
        translation_results = adt.get_translation(jai_batch, det_outputs)

        # track:
        trk_outputs, trk_windows = adt.track(det_outputs, translation_results, f_id)

        # get depths
        xyz_dims_to_det = adt.get_xyzs_dims(pc_batch, jai_batch, alignment_results, det_outputs)
        det_outputs, trk_outputs = adt.stitch_xyzs(det_outputs, trk_outputs, xyz_dims_to_det)

        # percent of tree seen
        percent_seen = adt.get_percent_seen(zed_batch, alignment_results)

        # collect results:
        results_collector.collect_adt(det_outputs, trk_outputs, alignment_results, percent_seen, f_id)

        # results_collector.draw_and_save(jai_frame, trk_outputs, f_id, args.output_folder)

        f_id += adt.batch_size
        adt.logger.iterations += 1

    pbar.close()
    adt.close_cams()
    adt.dump_log_stats(args)

    results_collector.dump_feature_extractor(args.output_folder, args.max_z)

    update_metadata(metadata, adt.shift_val, args)

    return results_collector


class Pipeline:
    """
    This class contains functions for processing frames of video for alignment, object detection and tracking.
    """
    def __init__(self, cfg, args, metadata=None):
        """
        Initialize Pipeline object.

        Args:
        - cfg (object): Configuration object
        - args (object): Namespace object of command line arguments
        - metadata (dict): Dictionary containing metadata information
        """
        self.logger = Logger(args)
        self.frames_loader = FramesLoader(cfg, args)
        self.detector = counter_detection(cfg, args)
        self.translation = translator(cfg.translation.translation_size, cfg.translation.dets_only, cfg.translation.mode)
        if not isinstance(metadata, type(None)):
            if "zed_shift" in metadata.keys():
                args.zed_shift = metadata["zed_shift"]
        self.s_frame = -args.zed_shift if args.zed_shift < 0 else 0
        self.sensor_aligner = SensorAligner(args=args.sensor_aligner, zed_shift=args.zed_shift)
        self.move_detector = FirstMoveDetector(batch_size=cfg.batch_size)
        self.zed_cam, self.rgb_jai_cam, self.jai_cam = init_cams(args)
        self.batch_size = cfg.batch_size
        self.auto_shift = args.auto_zed_shift
        self.shift_val = 0

    def get_frames(self, f_id):
        """
        Get frames from all cameras given a frame ID and return it.

        Args:
        - f_id (int): Frame ID

        Returns:
        - output (list): a batch of numpy.array for each camera
        """
        try:
            name = self.frames_loader.get_frames.__name__
            s = time.time()
            self.logger.info(f"Function {name} started")
            output = self.frames_loader.get_frames(f_id, self.sensor_aligner.zed_shift)
            self.log_end_func(name, s)
            return output
        except:
            self.logger.exception("Exception occurred")
            raise

    def is_saturated(self, frame, f_id, percentile=0.6):
        """
        Check if a frame is saturated given a percentile threshold.

        Args:
        - frame (list): a batch of images (numpy.array)
        - f_id (int): Frame ID
        - percentile (float): Threshold percentile value for saturation

        Returns:
        - status (bool): Boolean indicating whether frame is saturated
        - detailed (list): List of saturated frames or empty list if frame is not saturated
        """
        try:
            name = batch_is_saturated.__name__
            s = time.time()
            self.logger.info(f"Function {name} started")
            status, detailed = batch_is_saturated(frame, percentile=percentile)
            self.log_end_func(name, s)
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
        """
        Detect objects in frames and return detection outputs.

        Args:
        - frames (list): List of images

        Returns:
        - output (list): List of detection outputs
        """
        try:
            name = self.detector.detect.__name__
            s = time.time()
            self.logger.info(f"Function {name} started")
            output = self.detector.detect(frames)
            self.log_end_func(name, s)

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

    def track(self, inputs, translations, f_id):
        """
        Tracks an object in a video frame.

        Args:
            inputs (list): A list of numpy arrays containing images frames.
            translations (list): A list of translations for each frame in `inputs`.
            f_id (int): The frame ID.

        Returns:
            list: A list containing the object tracking results.
        """
        try:
            name = self.detector.track.__name__
            s = time.time()
            self.logger.info(f"Function {name} started")
            output = self.detector.track(inputs, translations, f_id)
            self.log_end_func(name, s)

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

    def get_translation(self, frames, det_outputs):
        """
        Computes the translation (tx, ty) between the frames.

        Args:
            frames (list): A list of numpy arrays containing image frames.
            det_outputs (list): A list containing the detection outputs for each frame.

        Returns:
            list: A list of lists containing pixel translations for each frame in `frames`.
        """
        try:
            name = self.translation.get_translation.__name__
            s = time.time()
            self.logger.info(f"Function {name} started")
            output = self.translation.batch_translation(frames, det_outputs)
            self.log_end_func(name, s)
            for res in output:
                self.logger.info(f"JAI X translation {res[0]}, Y translation {res[1]}")

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

    def align_cameras(self, zed_batch, rgb_jai_batch):
        """
        Aligns the RGB and zed cameras using a translation matrix.

        Args:
            zed_batch (list): A list of numpy arrays containing ZED camera frames.
            rgb_jai_batch (list): A list of numpy arrays containing RGB JAI camera frames.

        Returns:
            list: A list of alignment results ((x1, y1, x2, y2), tx, ty, sx, sy), one for each frame in `zed_batch`.
        """
        try:
            name = self.sensor_aligner.align_sensors.__name__
            s = time.time()
            self.logger.info(f"Function {name} started")
            output = self.sensor_aligner.align_on_batch(zed_batch, rgb_jai_batch)
            self.log_end_func(name, s)
            self.logger.info(f"Sensors frame shift {self.sensor_aligner.zed_shift}")

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

    def get_depths(self, xyz_batch, jai_batch, alignment_results, dets):
        """
        Computes the depth values for each object detected in the image frames.

        Args:
            xyz_batch (list): A list of numpy arrays containing ZED camera Point Clouds frames.
            jai_batch (list): A list of numpy arrays containing RGB JAI camera frames.
            alignment_results (list): A list of alignment_ results, one for each frame in `xyz_batch`.
            dets (list): A list containing the object detection results.

        Returns:
            list: A list containing depth values, one for each object detected in the image frames.
        """
        try:
            name = "depth_to_outputs"
            s = time.time()
            self.logger.info(f"Function {name} started")
            cut_coords = tuple(res[0] for res in alignment_results)
            output = get_depth_to_bboxes_batch(xyz_batch, jai_batch, cut_coords, dets)
            self.log_end_func(name, s)

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

    def get_xyzs(self, xyz_batch, jai_batch, alignment_results, dets):
        """
        Computes the xyz values for each object detected in the image frames.

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
            cut_coords = tuple(res[0] for res in alignment_results)
            output = get_depth_to_bboxes_batch(xyz_batch, jai_batch, cut_coords, dets, True)
            self.log_end_func(name, s)

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

    def get_xyzs_dims(self, xyz_batch, jai_batch, alignment_results, dets):
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
            cut_coords = tuple(res[0] for res in alignment_results)
            output = get_depth_to_bboxes_batch(xyz_batch, jai_batch, cut_coords, dets, True, True)
            self.log_end_func(name, s)

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

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
            output = get_depth_to_bboxes_batch(xyz_batch, jai_batch, cut_coords, dets, True)
            self.log_end_func(name, s)

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

    def stitch_depths(self, det_outputs_batch, trk_outputs_batch, depth_to_det_batch):
        """
        Stitches depth information to the detected and tracked outputs.

        Args:
            det_outputs_batch (List): List of lists of detected outputs.
            trk_outputs_batch (List): List of lists of tracked outputs.
            depth_to_det_batch (List): List of lists of depths corresponding to each detection.

        Returns:
            List: List of the updated detected and tracked outputs.
        """
        try:
            name = "stitch_depths"
            s = time.time()
            self.logger.info(f"Function {name} started")
            for det_outputs, trk_outputs, depth_to_det in zip(det_outputs_batch, trk_outputs_batch, depth_to_det_batch):
                for i, depth in enumerate(depth_to_det):
                    det_outputs[i].append(depth)
                    trk_outputs[i].append(depth)
            output = (det_outputs_batch, trk_outputs_batch)
            self.log_end_func(name, s)

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

    def stitch_xyzs(self, det_outputs_batch, trk_outputs_batch, depth_to_det_batch):
        """
        Stitches xyz information to the detected and tracked outputs.

        Args:
            det_outputs_batch (List): List of lists of detected outputs.
            trk_outputs_batch (List): List of lists of tracked outputs.
            depth_to_det_batch (List): List of lists of depths corresponding to each detection.

        Returns:
            List: List of the updated detected and tracked outputs.
        """
        try:
            name = "stitch_xyzs"
            s = time.time()
            self.logger.info(f"Function {name} started")
            for det_outputs, trk_outputs, depth_to_det in zip(det_outputs_batch, trk_outputs_batch, depth_to_det_batch):
                for i, xyz in enumerate(depth_to_det):
                    det_outputs[i] += xyz
                    trk_outputs[i] += xyz
            output = (det_outputs_batch, trk_outputs_batch)
            self.log_end_func(name, s)

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

    def get_percent_seen(self, zed_batch, alignment_results):
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
                output = list(executor.map(get_percent_seen, zed_batch, cut_coords))
            # output = list(map(get_percent_seen, zed_batch, cut_coords))
            self.log_end_func(name, s)

            return output

        except:
            self.logger.exception("Exception occurred")
            raise

    def update_moves(self, zed_batch, jai_batch, f_id):
        """
        Updates the state of the move detector.

        Args:
            zed_batch (List[np.ndarray]): List of input frames.
            jai_batch (List[np.ndarray]): List of JAI camera frames.
            f_id (int): Frame ID.

        Returns:
            Tuple(int, bool): Tuple containing the updated state of the move detector.
        """
        try:
            name = "update_moves"
            s = time.time()
            self.logger.info(f"Function {name} started")
            output = self.move_detector.update_state(zed_batch, jai_batch, f_id)
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

    def postprocess_auto_shift(self, zed_shift):
        """
        Post-processes the auto shift.

        Args:
            zed_shift (float): The auto-detected ZED shift.
        """
        self.sensor_aligner.zed_shift = zed_shift
        self.auto_shift = False
        self.logger.info(f"auto zed shift detected zed shift of: {zed_shift}")
        self.shift_val = zed_shift

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
        
    def dump_log_stats(self, args):
        """
        Dump the logger statistics to a CSV file.

        Args:
            args (object): The arguments of the program.

        Returns:
            None
        """
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
    translation = translator(cfg.translation.translation_size, cfg.translation.dets_only, cfg.translation.mode)
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


def zed_slicing_to_jai(slice_data_path, output_folder):
    slice_data = load_json(slice_data_path)
    slice_data = ResultsCollector().converted_slice_data(slice_data)
    slice_df = post_process(slice_data=slice_data)
    slice_df.to_csv(os.path.join(output_folder, 'all_slices.csv'))


def get_number_of_frames(jai_max_frames, metadata=None):
    max_cut_frame = np.inf
    if metadata is not None:
        if metadata['max_cut_frame'] != 'inf':
            max_cut_frame = int(metadata['max_cut_frame'])
    n_frames = get_n_frames(max_cut_frame, jai_max_frames, metadata)
    return n_frames


def update_metadata(metadata, zed_shift_detected, args):
    if metadata is None:
        metadata = {}
    metadata["align_detect_track"] = False
    metadata["zed_shift"] = zed_shift_detected
    write_metadata(args, metadata)


def get_depth_to_bboxes(xyz_frame, jai_frame, cut_coords, dets, pc=False, dims=False):
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
    cut_coords = dict(zip(["x1", "y1", "x2", "y2"], [[int(cord)] for cord in cut_coords]))
    xyz_frame_aligned = cut_zed_in_jai({"zed": xyz_frame}, cut_coords, rgb=False)["zed"]
    r_h, r_w = xyz_frame_aligned.shape[0] / jai_frame.shape[0], xyz_frame_aligned.shape[1] / jai_frame.shape[1]
    output = []
    for det in dets:
        box = ((int(det[0] * r_w), int(det[1] * r_h)), (int(det[2] * r_w), int(det[3] * r_h)))
        det_output = []
        if pc:
            det_output += list(xyz_center_of_box(xyz_frame_aligned, box))
        else:
            det_output.append(xyz_center_of_box(xyz_frame_aligned, box)[2])
        if dims:
            det_output += list(stable_euclid_dist(xyz_frame_aligned, box))
        output.append(det_output)
    return output


def get_depth_to_bboxes_batch(xyz_batch, jai_batch, cut_coords, dets, pc=False, dims=False):
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
    return list(map(get_depth_to_bboxes, xyz_batch, jai_batch, cut_coords, dets, [pc]*n, [dims]*n))


if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)

    validate_output_path(args.output_folder)
    # copy_configs(pipeline_config, runtime_config, args.output_folder)

    run(cfg, args)
