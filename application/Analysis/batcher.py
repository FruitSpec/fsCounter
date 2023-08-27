import queue
import threading
from datetime import datetime

from application.utils.module_wrapper import ModuleTransferAction, ModulesEnum
from application.utils.settings import analysis_conf, consts
from vision.pipelines.ops.line_detection.rows_detector import RowDetector

from queue import Queue
import time

from vision.tools.camera import jai_to_channels


class Batcher:
    _batch_size = analysis_conf.batch_size
    _frames_queue = None
    _batches_queue = None
    _drop_next_zed = False
    _batch_push_event = threading.Event()
    _shutdown_event = threading.Event()
    _acquisition_start_event = threading.Event()
    _timestamp_log_dict = {}
    output_dir = ""

    def __init__(self, frames_queue, send_data):
        self._frames_queue = frames_queue
        self._send_data = send_data
        self._batches_queue = Queue(maxsize=analysis_conf.max_batches)

    def align(self, jai_frame, zed_frame):
        x1, x2, y1, y2 = 0, 0, 0, 0
        tx, ty = 0, 0
        self._drop_next_zed = False
        return (x1, y1, x2, y2), tx, ty

    def set_timestamp_log_dict(self, jai_frame_number, jai_timestamp, zed_frame_number, zed_timestamp,
                               imu_angular_velocity, imu_linear_acceleration, depth_img):
        depth_score = RowDetector.percent_far_pixels(depth_img)
        self._timestamp_log_dict = {
            consts.JAI_frame_number: jai_frame_number,
            consts.JAI_timestamp: jai_timestamp,
            consts.ZED_frame_number: zed_frame_number,
            consts.ZED_timestamp: zed_timestamp,
            consts.IMU_angular_velocity: imu_angular_velocity,
            consts.IMU_linear_acceleration: imu_linear_acceleration,
            consts.depth_score: depth_score
        }

    def prepare_batches(self):

        def get_zed_per_jai(jai_frame, current_zed=None):
            while True:
                previous_zed = current_zed
                current_zed = self._frames_queue.pop_zed()
                jai_timestamp = datetime.strptime(jai_frame.timestamp, '%Y-%m-%d %H:%M:%S.%f')
                current_zed_timestamp = datetime.strptime(current_zed.timestamp, '%Y-%m-%d %H:%M:%S.%f')
                if current_zed_timestamp > jai_timestamp:
                    try:
                        previous_zed_timestamp = datetime.strptime(previous_zed.timestamp, '%Y-%m-%d %H:%M:%S.%f')
                        curr_t_diff = (current_zed_timestamp - jai_timestamp).total_seconds()
                        prev_t_diff = (jai_timestamp - previous_zed_timestamp).total_seconds()
                        if curr_t_diff <= prev_t_diff:
                            return current_zed, None
                        else:
                            return previous_zed, current_zed
                    except AttributeError:
                        return current_zed, None

        batch = []
        batch_number = 0

        last_zed_frame = None
        while not self._shutdown_event.is_set():
            self._acquisition_start_event.wait()
            jai_frame = self._frames_queue.pop_jai()
            zed_frame, last_zed_frame = get_zed_per_jai(jai_frame, last_zed_frame)

            angular_velocity, linear_acceleration = zed_frame.imu.angular_velocity, zed_frame.imu.linear_acceleration
            angular_velocity = (angular_velocity.x, angular_velocity.y, angular_velocity.z)
            linear_acceleration = (linear_acceleration.x, linear_acceleration.y, linear_acceleration.z)

            self.set_timestamp_log_dict(
                jai_frame_number=jai_frame.frame_number,
                jai_timestamp=jai_frame.timestamp,
                zed_frame_number=zed_frame.frame_number,
                zed_timestamp=zed_frame.timestamp,
                imu_angular_velocity=angular_velocity,
                imu_linear_acceleration=linear_acceleration,
                depth_img=zed_frame.depth
            )

            self._send_data(
                ModuleTransferAction.JAIZED_TIMESTAMPS,
                self._timestamp_log_dict,
                ModulesEnum.GPS
            )

            self.align(jai_frame.rgb, zed_frame.rgb)
            batch.append((jai_frame, zed_frame))
            if len(batch) == self._batch_size:
                batch_number += 1
                while True:
                    try:
                        self._batches_queue.put_nowait((batch, batch_number, time.time()))
                        break
                    except queue.Full:
                        self._batches_queue.get()
                batch = []

    def pop_batch(self):
        return self._batches_queue.get(block=True)

    def start_acquisition(self):
        self._acquisition_start_event.set()

    def stop_acquisition(self):
        self._acquisition_start_event.clear()
        self._send_data(
            ModuleTransferAction.STOP_ACQUISITION,
            None,
            ModulesEnum.DataManager
        )
