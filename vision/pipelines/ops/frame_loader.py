import os
from concurrent.futures import ThreadPoolExecutor

from vision.tools.video_wrapper import video_wrapper
from vision.pipelines.ops.simulator import init_cams

class FramesLoader():

    def __init__(self, cfg, args):
        self.zed_cam, self.rgb_jai_cam, self.jai_cam = init_cams(args)
        self.batch_size = cfg.batch_size
        self.last_svo_fid = 0

    def get_frames(self, f_id, zed_shift):
        if self.batch_size > 0: #1:
            return self.get_frames_batch(f_id, zed_shift)

        else:
            zed_frame, point_cloud = self.zed_cam.get_zed(f_id + zed_shift, exclude_depth=True)
            fsi_ret, jai_frame = self.jai_cam.get_frame()
            rgb_ret, rgb_jai_frame = self.rgb_jai_cam.get_frame()
            return zed_frame, point_cloud, fsi_ret, jai_frame, rgb_ret, rgb_jai_frame

    def get_frames_batch(self, f_id, zed_shift):
        cams = [self.zed_cam, self.jai_cam, self.rgb_jai_cam]
        batch_ids = self.get_batch_fids(f_id, zed_shift)
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(self.get_camera_frame, cams, batch_ids))

        return self.results_to_batch(results)

    def get_batch_fids(self, fid, zed_shift=0):
        zed_batch = fid + zed_shift
        jai_batch = fid

        return [zed_batch, jai_batch, jai_batch]



    @staticmethod
    def results_to_batch(results):
        zed_data = results[0]
        jai_data = results[1]
        rgb_data = results[2]
        zed_batch = []
        jai_batch = []
        rgb_batch = []
        for f_data in zed_data:
            zed_batch.append(f_data[1])
        for f_data in jai_data:
            jai_batch.append(f_data[1])
        for f_data in rgb_data:
            rgb_batch.append(f_data[1])

        return zed_batch, jai_batch, rgb_batch

    def get_camera_frame(self, cam, f_id):
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
                batch.append(cam.get_frame())
            else:
                #if id_ == 0:
                #    batch.append(cam.get_frame(f_id))
                #else:
                #    batch.append(cam.get_frame())
                batch.append(cam.get_frame())

        return batch