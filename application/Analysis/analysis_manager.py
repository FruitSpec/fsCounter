from application.Analysis.batcher import Batcher
from multiprocessing import Process
from threading import Thread
import time
import numpy as np


class AnalysisManager:
    _batcher = None

    def __init__(self, frames_queue, send_data):

        self._batcher = Batcher(frames_queue, send_data)
        self._batch_thread = Thread(target=self.batch, daemon=True)
        self._detect_proc = Process(target=self.detect, daemon=True)
        self._track_proc = Process(target=self.track, daemon=True)

    def start_analysis(self):
        self._batch_thread.start()
        self._detect_proc.start()
        self._track_proc.start()

        self._batch_thread.join()
        self._detect_proc.join()
        self._track_proc.join()

    def set_output_dir(self, output_dir):
        self._batcher.output_dir = output_dir

    def start_acquisition(self):
        self._batcher.start_acquisition()

    def stop_acquisition(self):
        self._batcher.stop_acquisition()

    def batch(self):

        def share_batches():
            while True:
                batch, batch_number, batch_timestamp = self._batcher.pop_batch()
                del batch, batch_number, batch_timestamp
                frame_number_s, fsi_s, jai_rgb_s, p_cloud_s, zed_rgb_s = [], [], [], [], []
                # batch_metadata = np.asarray((batch_number, batch_timestamp), dtype=np.float128)
                # for i, frame in enumerate(batch):
                #     jai, zed = frame
                #     frame_number_s.append(np.asarray((jai.frame_number,), dtype=np.uint8))
                #     fsi_s.append(jai.fsi)
                #     jai_rgb_s.append(jai.rgb)
                #     p_cloud_s.append(zed.point_cloud)
                #     zed_rgb_s.append(zed.rgb)

        prepare_batches_t = Thread(target=self._batcher.prepare_batches, daemon=True)
        share_batches_t = Thread(target=share_batches, daemon=True)

        prepare_batches_t.start()
        share_batches_t.start()

        share_batches_t.join()
        prepare_batches_t.join()

    def detect(self):
        pass

    def track(self):
        pass
