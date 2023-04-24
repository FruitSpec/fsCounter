import threading

from application.utils.settings import analysis_conf
from collections import deque
import cv2


class Batcher:
    _max_batches = analysis_conf["max batches"]
    _batch_size = analysis_conf["batch size"]
    _frames_queue = None
    _batches_deque = None
    _drop_next_zed = False
    _shutdown_event = threading.Event()
    _acquisition_start_event = threading.Event()

    def __init__(self, frames_queue):
        self._frames_queue = frames_queue
        self._batches_deque = deque()

    def align(self, jai_frame, zed_frame):
        x1, x2, y1, y2 = 0, 0, 0, 0
        tx, ty = 0, 0
        self._drop_next_zed = False
        return (x1, y1, x2, y2), tx, ty

    def prepare_batches(self):
        batch = []
        while not self._shutdown_event.is_set():
            self._acquisition_start_event.wait()
            jai_frame = self._frames_queue.pop_jai()
            if self._drop_next_zed:
                self._frames_queue.pop_zed()
            zed_frame = self._frames_queue.pop_zed()
            self.align(jai_frame.rgb, zed_frame)
            batch.append((jai_frame, zed_frame))
            if len(batch) == self._batch_size:
                if len(self._batches_deque) >= self._max_batches:
                    self._batches_deque.pop()
                self._batches_deque.appendleft(batch)
                batch = []
            if jai_frame.frame_number % 50 == 0:
                av = zed_frame.imu.angular_velocity
                la = zed_frame.imu.linear_acceleration
                print(f"angular_velocity: ({av.x}, {av.y}, {av.z})")
                print(f"linear acceleration: ({la.x}, {la.y}, {la.z})")
            # if jai_frame.frame_number % 50 == 0:
            #     cv2.destroyAllWindows()
            #     cv2.imshow("mat", jai_frame.rgb)
            #     cv2.waitKey(1000)

    def start_acquisition(self):
        self._acquisition_start_event.set()

    def stop_acquisition(self):
        self._acquisition_start_event.clear()
