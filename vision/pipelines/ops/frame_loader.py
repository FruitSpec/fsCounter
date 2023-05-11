import os
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from vision.tools.video_wrapper import video_wrapper
from vision.pipelines.ops.simulator import init_cams

class FramesLoader():

    def __init__(self, cfg, args):
        self.mode = cfg.frame_loader.mode
        self.zed_cam, self.rgb_jai_cam, self.jai_cam, self.depth_cam = init_cams(args, self.mode)
        self.batch_size = cfg.batch_size
        self.last_svo_fid = 0

        if self.mode in ['sync_svo', 'sync_mkv']:
            self.sync_zed_ids, self.sync_jai_ids = self.get_cameras_sync_data(args.frame_loader.sync_data_log_path)

    def get_frames(self, f_id, zed_shift):
        if self.mode == 'async':
            output = self.get_frames_batch_async(f_id, zed_shift)
        if self.mode == 'sync_svo':
            output = self.get_frames_batch_sync_svo(f_id)

        return output



    def get_frames_batch_async(self, f_id, zed_shift):
        cams = [self.zed_cam, self.jai_cam, self.rgb_jai_cam]
        batch_ids = self.get_batch_fids(f_id, zed_shift)

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(self.get_camera_frame_async, cams, batch_ids))

        return self.results_to_batch_async(results)


    @staticmethod
    def get_batch_fids(fid, zed_shift):
        zed_batch = fid + zed_shift
        jai_batch = fid
        return [zed_batch, jai_batch, jai_batch]


    def get_batch_fids_sync(self, fid):
        zed_batch = []
        jai_batch = []
        max_frame_id = len(self.sync_zed_ids) - 1
        for id_ in range(self.batch_size):
            if fid + id_ > max_frame_id:
                print('Not full batch')
                break
            zed_batch.append(self.sync_zed_ids[fid + id_])
            jai_batch.append(self.sync_jai_ids[fid + id_])

        if self.mode == 'sync_svo':
            return [zed_batch, jai_batch, jai_batch]
        elif self.mode == 'sync_mkv':
            return [zed_batch, zed_batch, jai_batch, jai_batch]



    @staticmethod
    def results_to_batch_async(results):
        zed_batch = results[0]
        jai_batch = results[1]
        rgb_batch = results[2]

        return zed_batch, jai_batch, rgb_batch


    def get_camera_frame_async(self, cam, f_id):
        batch = []
        max_frame_number = cam.get_number_of_frames()
        for id_ in range(self.batch_size):
            if f_id + id_ >= max_frame_number:
                break
            if cam.mode == 'svo':
                if id_ == 0 and f_id != self.last_svo_fid + self.batch_size:
                    cam.grab(f_id)
                else:
                    cam.grab()
                self.last_svo_fid = f_id
                _, frame = cam.get_frame()

            else:
                _, frame = cam.get_frame()
            batch.append(frame)

        return batch

    @staticmethod
    def get_camera_frame_sync(cam, f_ids, last_fid):
        batch = []
        max_frame_number = cam.get_number_of_frames()
        for f_id in f_ids:
            if f_id >= max_frame_number:
                break
            if cam.mode == 'svo':
                if f_id > last_fid + 1:
                    while f_id > last_fid + 1:
                        cam.grab()
                        last_fid += 1
                cam.grab()
                last_fid = f_id
                batch.append(cam.get_frame())
            else:
                if f_id > last_fid + 1:
                    while f_id > last_fid + 1:
                        _, _ = cam.get_frame()
                        last_fid += 1
                last_fid = f_id
                batch.append(cam.get_frame())

        return batch, last_fid

    @staticmethod
    def get_batch_results_sync(results):

        zed_data = results[0]
        jai_data = results[1]
        rgb_jai_data = results[2]



    def get_frames_batch_sync_svo(self, f_id):
        cams = [self.zed_cam, self.depth_cam, self.jai_cam, self.rgb_jai_cam]
        batch_ids = self.get_batch_fids_sync(f_id)

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(self.get_camera_frame_sync, cams, batch_ids))

        return self.results_to_batch(results)

    def get_frames_batch_sync_mkv(self, f_id):
        cams = [self.zed_cam, self.depth_cam, self.jai_cam, self.rgb_jai_cam]
        batch_ids = self.get_batch_fids_sync(f_id)

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(self.get_camera_frame_sync, cams, batch_ids))

        return self.results_to_batch(results)


    @staticmethod
    def get_cameras_sync_data(log_fp):
        zed_ids = []
        jai_ids = []
        log_df = pd.read_csv(log_fp)
        jai_frame_ids = list(log_df['JAI frame number'])
        zed_frame_ids = list(log_df['ZED frame number'])

        jai_first_id, zed_first_id = 0, 0
        p_zed_id = -1e5
        for jai_id, zed_id in zip(jai_frame_ids, zed_frame_ids):
            if p_zed_id > zed_id:
                jai_first_id = jai_id
            p_zed_id = zed_id

            zed_ids.append(zed_id)
            jai_ids.append(jai_id - jai_first_id)

        return zed_ids, jai_ids