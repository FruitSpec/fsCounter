import queue
import threading
from datetime import datetime

from application.utils.module_wrapper import ModuleTransferAction, ModulesEnum
from application.utils.settings import analysis_conf
from queue import Queue
import pandas as pd
import os
import time


class Batcher:
    _batch_size = analysis_conf.batch_size
    _frames_queue = None
    _batches_queue = None
    _drop_next_zed = False
    _lock = threading.Lock()
    _batch_push_event = threading.Event()
    _shutdown_event = threading.Event()
    _acquisition_start_event = threading.Event()
    _timestamps_log_dict = {}
    output_dir = ""

    def __init__(self, frames_queue, send_data):
        self._frames_queue = frames_queue
        self._send_data = send_data
        self._batches_queue = Queue(maxsize=analysis_conf.max_batches)
        self.init_timestamp_log_dict()

    def align(self, jai_frame, zed_frame):
        x1, x2, y1, y2 = 0, 0, 0, 0
        tx, ty = 0, 0
        self._drop_next_zed = False
        return (x1, y1, x2, y2), tx, ty

    def init_timestamp_log_dict(self):
        self._timestamps_log_dict = {
            "JAI_frame_number": [],
            "JAI_timestamp": [],
            "ZED_frame_number": [],
            "ZED_timestamp": [],
            "IMU_angular_velocity": [],
            "IMU_linear_acceleration": []
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
        self.init_timestamp_log_dict()
        while not self._shutdown_event.is_set():
            self._acquisition_start_event.wait()
            jai_frame = self._frames_queue.pop_jai()
            zed_frame, last_zed_frame = get_zed_per_jai(jai_frame, last_zed_frame)
            self._timestamps_log_dict["JAI_frame_number"].append(jai_frame.frame_number)
            self._timestamps_log_dict["JAI_timestamp"].append(jai_frame.timestamp)
            self._timestamps_log_dict["ZED_frame_number"].append(zed_frame.frame_number)
            self._timestamps_log_dict["ZED_timestamp"].append(zed_frame.timestamp)
            angular_velocity, linear_acceleration = zed_frame.imu.angular_velocity, zed_frame.imu.linear_acceleration
            angular_velocity = (angular_velocity.x, angular_velocity.y, angular_velocity.z)
            linear_acceleration = (linear_acceleration.x, linear_acceleration.y, linear_acceleration.z)
            self._timestamps_log_dict["IMU_angular_velocity"].append(angular_velocity)
            self._timestamps_log_dict["IMU_linear_acceleration"].append(linear_acceleration)

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
            if jai_frame.frame_number % 50 == 0:
                print("sending data from batcher")
                self._send_data(
                    ModuleTransferAction.JAIZED_TIMESTAMPS,
                    self._timestamps_log_dict, 
                    ModulesEnum.DataManager
                )
                self.init_timestamp_log_dict()

    def pop_batch(self):
        return self._batches_queue.get(block=True)

    def start_acquisition(self):
        self._acquisition_start_event.set()

    def stop_acquisition(self):
        self._acquisition_start_event.clear()
        print("BATCHER STOP ACQUISITION")
        if self._timestamps_log_dict["JAI_frame_number"]:
            print("BATCHER TIME + STOP")
            self._send_data(
                ModuleTransferAction.JAIZED_TIMESTAMPS_AND_STOP,
                self._timestamps_log_dict,
                ModulesEnum.DataManager
            )
        else:
            print("BATCHER STOP ONLY")
            self._send_data(
                ModuleTransferAction.STOP_ACQUISITION,
                None,
                ModulesEnum.DataManager
            )
        self.init_timestamp_log_dict()
