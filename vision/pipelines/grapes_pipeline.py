import os
import sys
import cv2
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
import glob
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



def run(cfg, args, metadata=None):

    adt = Pipeline(cfg, args)

    results_collector = ResultsCollector(rotate=args.rotate)

    #print(f'Inferencing on {args.jai.movie_path}\n')

    relevant_frames_idx = adt.frames_loader.sync_jai_ids
    n_relevant_frames = len(relevant_frames_idx)

    index = 0

    pbar = tqdm(total=n_relevant_frames)
    while index < n_relevant_frames:
        pbar.update(adt.batch_size)

        f_idx = relevant_frames_idx[index]

        zed_batch, depth_batch, jai_batch, rgb_batch = adt.get_frames(index)

        # rgb_stauts, rgb_detailed = adt.is_saturated(rgb_batch, f_idx)
        # zed_stauts, zed_detailed = adt.is_saturated(zed_batch, f_idx)
        # if rgb_stauts or zed_stauts:
        #      for i in range(adt.batch_size):
        #         if f_idx + i in frame_drop_jai:
        #              adt.sensor_aligner.zed_shift += 1
        #      index += adt.batch_size
        #      adt.logger.iterations += 1
        #      continue

        alignment_results = adt.align_cameras(zed_batch, rgb_batch)

        # detect:
        det_outputs = adt.detect(jai_batch)

        # depth:
        depth_results = get_depth_to_bboxes_batch(depth_batch, jai_batch, alignment_results, det_outputs)
        det_outputs = append_to_trk(det_outputs, depth_results)

        # screen detections by depth (screen above 2 meters):
        det_outputs = [[det for det in det_outputs[0] if det[-1] < args.screen_detections_above_depth]]

        #collect results:
        results_collector.collect_detections(det_outputs, index, relevant_frames_idx)
        results_collector.collect_alignment(alignment_results, f_idx)
        results_collector.draw_and_save(jai_batch[0], det_outputs[0], f_idx, args.output_folder, args.row)


        index += adt.batch_size
        # f_idx = relevant_frames_idx[index]
        adt.logger.iterations += 1

    pbar.close()
    adt.frames_loader.zed_cam.close()
    adt.frames_loader.jai_cam.close()
    adt.frames_loader.rgb_jai_cam.close()
    adt.frames_loader.depth_cam.close()
    adt.dump_log_stats(args)

    update_metadata(metadata, args)
    path_detections = os.path.join(args.output_folder, 'detections.csv')
    path_gps_jai_zed = os.path.join(args.output_folder, 'gps_jai_zed.csv')
    results_collector.dump_to_csv(path_detections, 'detections')
    df_detections_count = save_detection_results(detection_csv_path=path_detections, gps_jai_zed_csv_path = path_gps_jai_zed, output_dir = args.output_folder, DEPTH_THRESHOLD = args.depth_to_grapes_in_meters, row = args.row, block = args.block, customer_code = args.customer_code, scan_date = args.scan_date )

    return df_detections_count, results_collector


class Pipeline():

    def __init__(self, cfg, args):
        self.logger = Logger(args)
        self.frames_loader = FramesLoader(cfg, args)
        self.detector = counter_detection(cfg, args)
        self.translation = T(cfg.translation.translation_size, cfg.translation.dets_only, cfg.translation.mode)
        self.sensor_aligner = SensorAligner(cfg=cfg.sensor_aligner, batch_size=cfg.batch_size)
        self.batch_size = cfg.batch_size


    def get_frames(self, f_id):
        try:
            name = self.frames_loader.get_frames.__name__
            s = time.time()
            self.logger.debug(f"Function {name} started")
            output = self.frames_loader.get_frames(f_id, self.sensor_aligner.zed_shift)
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

    def get_translation(self, frames, det_outputs):
        try:
            name = self.translation.get_translation.__name__
            s = time.time()
            self.logger.debug(f"Function {name} started")
            output = self.translation.batch_translation(frames, det_outputs)
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


def save_detection_results(detection_csv_path, gps_jai_zed_csv_path, output_dir, DEPTH_THRESHOLD, row, block, customer_code, scan_date):
    '''
    This function gets detection csv file, gps_jai_zed csv file, and DEPTH_THRESHOLD,
    and saves a csv file with the frame_id, number of detections in depth per frame, and
    longitude , latitude of the frame.
    '''

    # todo - the depth screen is preformed earlier, can remove this its redundant
    # read csv files:
    df_gps_jai_zed = pd.read_csv(gps_jai_zed_csv_path)
    df_gps_jai_zed['location'] = list(zip(df_gps_jai_zed['latitude'], df_gps_jai_zed['longitude']))
    df_gps_jai_zed.drop(columns=['latitude', 'longitude'], inplace=True)

    df_detection = pd.read_csv(detection_csv_path)
    df_detection['in_depth'] = df_detection['depth'] <= DEPTH_THRESHOLD
    df_detection = df_detection[df_detection['in_depth'] == True]

    # count detections per frame:
    df_detections_count = df_detection.groupby(['frame_id']).size().reset_index(name='count').set_index('frame_id')
    df_detections_count.insert(0, 'frame_id', df_detections_count.index)
    df_detections_count.insert(0,'row_num', int(row.split('_')[-1]))


    # get df_gps_jai_zed df where JAI_frame_number is in df_detections_count['frame_id']:
    df_gps_jai_zed = df_gps_jai_zed[df_gps_jai_zed['JAI_frame_number'].isin(df_detections_count.index.values)]

    df_detections_count['location'] = df_gps_jai_zed['location']
    df_detections_count['location'] = df_detections_count['location'].apply(lambda x: str(x).replace('(', '').replace(')', ''))

    # save df_detections_count to csv:
    output_path = os.path.join (output_dir, 'rows',f'{customer_code}_{block}_{scan_date}_row_{row}.csv')
    validate_output_path(os.path.dirname(output_path))
    #df_detections_count.to_csv(output_path, index = False)
    #print (f'Saved {output_path}')
    return df_detections_count

def run_rows (cfg, args):
    if args.row is not None:
        rows = [f'row_{args.row}']
    else:
        rows = os.listdir(args.input_block_folder)

    df_detections_all_rows = pd.DataFrame()

    for row in rows:
        print(f'************************ Row {row} *******************************')

        try:
            row_folder = os.path.join(args.input_block_folder, row, '1')
            if not os.path.exists(os.path.join(row_folder, "jaized_timestamps.csv")):
                print (f'jaized_timestamps.csv file not found')
                continue
            args.sync_data_log_path = os.path.join(row_folder, "jaized_timestamps.csv")
            args.zed.movie_path = os.path.join(row_folder, "ZED.mkv")
            args.depth.movie_path = os.path.join(row_folder, "DEPTH.mkv")
            args.jai.movie_path = os.path.join(row_folder, "Result_FSI.mkv")
            args.rgb_jai.movie_path = os.path.join(row_folder, "Result_RGB.mkv")
            args.screen_detections_above_depth = args.depth_to_grapes_in_meters + 1
            args.row = row
            validate_output_path(args.output_folder)

            df_detections_count, rc = run(cfg, args)
            df_detections_all_rows = pd.concat([df_detections_all_rows,df_detections_count], axis=0, ignore_index=True)
            output_file_path = os.path.join(args.output_folder, f'{args.customer_code}_{args.block}_{args.scan_date}.csv')
            df_detections_all_rows.to_csv(output_file_path, index=False)
            print (f'Saved {output_file_path}')

        except Exception as e:
            print(f"An error occurred: {e}")
            continue

    return df_detections_all_rows
def run_blocks(all_blocks_dir, path_to_distance_from_len, path_pipeline_config, path_runtime_config):
    distance_from_len = pd.read_excel(path_to_distance_from_len)
    distance_from_len = distance_from_len.set_index('Block code')

    cfg = OmegaConf.load(repo_dir + path_pipeline_config)
    args = OmegaConf.load(repo_dir + path_runtime_config)

    os.chdir(all_blocks_dir)
    blocks_paths = glob.glob('[0-9]*')  # get a list of files that start with a number
    blocks_paths = [os.path.join(all_blocks_dir, name) for name in blocks_paths if
                    os.path.isdir(name)]  # screen for subdirs

    for block_path in blocks_paths:
        date = [name for name in os.listdir(block_path) if os.path.isdir(os.path.join(block_path, name))][0]  # get the date from subdir (each block has one date)
        args.nav_path = os.path.join(all_blocks_dir, f'{date}.nav')
        args.block = os.path.basename(block_path)



        args.depth_to_grapes_in_meters = distance_from_len.loc[args.block]
        rows_dir = os.path.join(block_path, date)
        print(f'********* Starting block {rows_dir} ')
        run_rows(rows_dir, cfg, args)

def update_args(INPUT_FOLDER, CUSTOMER_CODE, BLOCK_CODE, SCAN_DATE, OUTPUT_FOLDER, DISTANCE_FROM_LEN, ROW=None):

    path_pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    path_runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"

    cfg = OmegaConf.load(repo_dir + path_pipeline_config)
    args = OmegaConf.load(repo_dir + path_runtime_config)

    df_distance_from_len = pd.read_excel(DISTANCE_FROM_LEN)
    df_distance_from_len = df_distance_from_len.set_index('Block code')
    args.depth_to_grapes_in_meters = (int(df_distance_from_len.loc[BLOCK_CODE][0])) / 100  # Convert cm to meter
    args.nav_path = os.path.join(INPUT_FOLDER, CUSTOMER_CODE, f'{SCAN_DATE}.nav')
    args.output_folder = os.path.join(OUTPUT_FOLDER, CUSTOMER_CODE, BLOCK_CODE, SCAN_DATE)
    args.scan_date = SCAN_DATE
    args.block = BLOCK_CODE
    args.customer_code = CUSTOMER_CODE
    args.input_block_folder = os.path.join(INPUT_FOLDER, CUSTOMER_CODE, BLOCK_CODE, SCAN_DATE)
    args.row = ROW

    return args, cfg


if __name__ == "__main__":

    INPUT_FOLDER = r'/media/fruitspec-lab-3/easystore'
    CUSTOMER_CODE = 'JACFAM'
    BLOCK_CODE = '204402XX'
    SCAN_DATE = '180723'
    OUTPUT_FOLDER = r'/media/fruitspec-lab-3/easystore/grapes_detector_output'
    DISTANCE_FROM_LEN = '/home/fruitspec-lab-3/FruitSpec/Data/grapes/USXXXX/GRAPES/dist_from_len_by_blocks.xlsx'
    ROW = None


    args, cfg = update_args(INPUT_FOLDER,CUSTOMER_CODE,BLOCK_CODE,SCAN_DATE, OUTPUT_FOLDER, DISTANCE_FROM_LEN, ROW)
    df_detections_all_rows = run_rows(cfg, args)














