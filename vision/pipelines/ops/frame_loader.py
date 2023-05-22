import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from vision.pipelines.ops.simulator import init_cams

class FramesLoader():

    def __init__(self, cfg, args):
        self.mode = cfg.frame_loader.mode
        self.zed_cam, self.rgb_jai_cam, self.jai_cam, self.depth_cam = init_cams(args, self.mode)
        self.batch_size = cfg.batch_size
        self.zed_last_id = 0
        self.depth_last_id = 0
        self.jai_last_id = 0
        self.rgb_jai_last_id = 0

        if self.mode in ['sync_svo', 'sync_mkv']:
            self.sync_zed_ids, self.sync_jai_ids = self.get_cameras_sync_data(args.sync_data_log_path)

    def get_frames(self, f_id, zed_shift):
        if self.mode == 'async':
            output = self.get_frames_batch_async(f_id, zed_shift)
        if self.mode == 'sync_svo':
            output = self.get_frames_batch_sync_svo(f_id)
        if self.mode == 'sync_mkv':
            output = self.get_frames_batch_sync_mkv(f_id)

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
                return [[], [], [], []]
            zed_batch.append(self.sync_zed_ids[fid + id_])
            jai_batch.append(fid + id_)  # jai is after frame drops - no frame jumps
        if self.mode == 'sync_svo':
            return [zed_batch, jai_batch, jai_batch]
        elif self.mode == 'sync_mkv':
            return [zed_batch, zed_batch, jai_batch, jai_batch]



    @staticmethod
    def results_to_batch_async(results):
        zed_batch = results[0][0]
        depth_batch = results[0][1]
        jai_batch = results[1][0]
        rgb_batch = results[2][0]

        return zed_batch, depth_batch, jai_batch, rgb_batch




    def get_camera_frame_async(self, cam, f_id):
        batch = []
        depth_batch = []
        max_frame_number = cam.get_number_of_frames()
        for id_ in range(self.batch_size):
            if f_id + id_ >= max_frame_number:
                break
            if cam.mode == 'svo':
                if id_ == 0 and f_id != self.zed_last_id + self.batch_size:
                    cam.grab(f_id)
                else:
                    cam.grab()
                self.zed_last_id = f_id
                _, frame = cam.get_frame()
                depth = cam.get_depth()
                batch.append(frame)
                depth_batch.append(depth)
            else:
                _, frame = cam.get_frame()
                batch.append(frame)

        return batch, depth_batch

    @staticmethod
    def get_camera_frame_sync(cam, f_ids, last_fid):
        batch = []
        depth_batch = []
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
                _, frame = cam.get_frame()
                batch.append(frame)
                depth_batch.append(cam.get_depth())
            else:
                if f_id > last_fid + 1:
                    while f_id > last_fid + 1:
                        _, _ = cam.get_frame()
                        last_fid += 1
                last_fid = f_id
                _, frame = cam.get_frame()
                batch.append(frame)

        return batch, depth_batch, last_fid

    @staticmethod
    def get_batch_results_sync(results, mode):

        if mode == 'sync_svo':
            zed_data = results[0]
            jai_data = results[1]
            rgb_jai_data = results[2]
        else:
            zed_data = results[0]
            depth_data = results[1]
            jai_data = results[2]
            rgb_jai_data = results[3]

        zed_batch = zed_data[0]
        depth_batch = zed_data[1]
        zed_last_id = zed_data[2]

        jai_batch = jai_data[0]
        jai_last_id = jai_data[2]

        rgb_jai_batch = rgb_jai_data[0]
        rgb_jai_last_id = rgb_jai_data[2]

        if mode == 'sync_mkv':
            depth_batch = depth_data[0]
            depth_last_id = depth_data[2]
        else:
            depth_last_id = zed_last_id

        return zed_batch, zed_last_id, depth_batch, depth_last_id, jai_batch, jai_last_id, rgb_jai_batch, rgb_jai_last_id


    def get_frames_batch_sync_svo(self, f_id):
        cams = [self.zed_cam, self.jai_cam, self.rgb_jai_cam]
        batch_ids = self.get_batch_fids_sync(f_id)
        last_ids = [self.zed_last_id, self.jai_last_id, self.rgb_jai_last_id]

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(self.get_camera_frame_sync, cams, batch_ids, last_ids))

        zed_batch, zed_last_id, depth_batch, depth_last_id, jai_batch, jai_last_id, rgb_jai_batch, rgb_jai_last_id = self.get_batch_results_sync(
            results, self.mode)
        self.zed_last_id = zed_last_id
        self.jai_last_id = jai_last_id
        self.rgb_jai_last_id = rgb_jai_last_id

        return zed_batch, depth_batch, jai_batch, rgb_jai_batch

    def get_frames_batch_sync_mkv(self, f_id):
        cams = [self.zed_cam, self.depth_cam, self.jai_cam, self.rgb_jai_cam]
        batch_ids = self.get_batch_fids_sync(f_id)
        # debug
        #p = "/home/matans/Documents/fruitspec/sandbox/VALENCIA/row_1A/SA4/frame_loader_ids.csv"
        #d = np.array(batch_ids).T
        #df = pd.DataFrame(data=d.tolist(), columns=['zed_id', 'depth_id', 'jai_id', 'rgb_id'])
        #df.to_csv(p, mode='a', index=False, header=False)

        ######
        last_ids = [self.zed_last_id, self.depth_last_id, self.jai_last_id, self.rgb_jai_last_id]

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(self.get_camera_frame_sync, cams, batch_ids, last_ids))

        zed_batch, zed_last_id, depth_batch, depth_last_id, jai_batch, jai_last_id, rgb_jai_batch, rgb_jai_last_id = self.get_batch_results_sync(
            results, self.mode)

        self.zed_last_id = zed_last_id
        self.jai_last_id = jai_last_id
        self.rgb_jai_last_id = rgb_jai_last_id
        self.depth_last_id = depth_last_id

        return zed_batch, depth_batch, jai_batch, rgb_jai_batch

    def close_cameras(self):
        self.zed_cam.close()
        self.jai_cam.close()
        self.rgb_jai_cam.close()
        if self.depth_cam is not None:
            self.depth_cam.close()


    @staticmethod
    def get_cameras_sync_data(log_fp):
        zed_ids = []
        jai_ids = []
        log_df = pd.read_csv(log_fp)
        jai_frame_ids = list(log_df['JAI_frame_number'])
        zed_frame_ids = list(log_df['ZED_frame_number'])

        zed_ids, jai_ids = arrange_ids(jai_frame_ids, zed_frame_ids)

        return zed_ids, jai_ids

    @staticmethod
    def arrange_ids(jai_frame_ids, zed_frame_ids):

        z = np.array(zed_frame_ids)
        j = np.array(jai_frame_ids)
        # find start index
        diff = z[1:] - z[:-1]
        start_index = np.argmin(diff)

        jai_offset = j[start_index]
        j -= jai_offset

        return z[start_index:].tolist(), j[start_index:].tolist()

