import os
from concurrent.futures import ThreadPoolExecutor

from vision.tools.video_wrapper import video_wrapper
from vision.pipelines.ops.simulator import init_cams

class FramesLoader():

    def __init__(self, cfg, args):
        """
        Constructor of FramesLoader class.

        Parameters:
        cfg (config object): configuration object
        args (argparse object): arguments object
        """
        self.zed_cam, self.rgb_jai_cam, self.jai_cam = init_cams(args)
        self.batch_size = cfg.batch_size

    def get_frames(self, f_id, zed_shift):
        """
        Function to get the frames for a given frame id and a ZED shift value.

        Parameters:
        f_id (int): the frame id
        zed_shift (int): the ZED shift value

        Returns:
        if self. batch_size > 0:
        zed_batch (list): list of ZED frames
        jai_batch (list): list of JAI frames
        rgb_batch (list): list of RGB frames
        pc_batch (list): list of point clouds
        else:
        zed_frame (numpy array): the ZED frame
        point_cloud (numpy array): the point cloud
        fsi_ret (int): the FSI return value
        jai_frame (numpy array): the JAI frame
        rgb_ret (int): the RGB return value
        rgb_jai_frame (numpy array): the RGB JAI frame
        """
        if self.batch_size > 0: #1:
            return self.get_frames_batch(f_id, zed_shift)

        else:
            zed_frame, point_cloud = self.zed_cam.get_zed(f_id + zed_shift, exclude_depth=True)
            fsi_ret, jai_frame = self.jai_cam.get_frame()
            rgb_ret, rgb_jai_frame = self.rgb_jai_cam.get_frame()
            return zed_frame, point_cloud, fsi_ret, jai_frame, rgb_ret, rgb_jai_frame

    def get_frames_batch(self, f_id, zed_shift):
        """
        Function to get the frames in batch mode for a given frame id and a ZED shift value.

        Parameters:
        f_id (int): the frame id
        zed_shift (int): the ZED shift value

        Returns:
        zed_batch (list): list of ZED frames
        jai_batch (list): list of JAI frames
        rgb_batch (list): list of RGB frames
        pc_batch (list): list of point clouds
        """
        cams = [self.zed_cam, self.jai_cam, self.rgb_jai_cam]
        batch_ids = self.get_batch_fids(f_id, zed_shift)
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(self.get_camera_frame, cams, batch_ids))

        return self.results_to_batch(results)

    def get_batch_fids(self, fid, zed_shift=0):
        """
        Function to get the batch frame ids for a given frame id and a ZED shift value.

        Parameters:
        fid (int): the frame id
        zed_shift (int): the ZED shift value

        Returns:
        batch_ids (list): list of batch frame ids
        """
        zed_batch = fid + zed_shift
        jai_batch = fid

        return [zed_batch, jai_batch, jai_batch]

    @staticmethod
    def results_to_batch(results):
        """
        Converts a list of camera frames and returns them in separate batches.

        Parameters:
        results (list): A list containing camera frames.

        Returns:
        zed_batch (list): A list containing ZED camera frames.
        jai_batch (list): A list containing JAI camera frames.
        rgb_batch (list): A list containing RGB camera frames.
        pc_batch (list): A list containing point clouds.
        """
        zed_data = results[0]
        jai_data = results[1]
        rgb_data = results[2]
        zed_batch = []
        jai_batch = []
        rgb_batch = []
        pc_batch = []
        for f_data in zed_data:
            zed_batch.append(f_data[0])
            pc_batch.append(f_data[1])
        for f_data in jai_data:
            jai_batch.append(f_data[1])
        for f_data in rgb_data:
            rgb_batch.append(f_data[1])

        return zed_batch, jai_batch, rgb_batch, pc_batch

    def get_camera_frame(self, cam, f_id):
        """
        Returns a batch of frames for a given camera and frame ID.

        Parameters:
        cam (Camera): The camera object.
        f_id (int): The frame ID.

        Returns:
        batch (list): A list containing the batch of frames.
        """
        batch = []
        max_frame_number = cam.get_number_of_frames()
        for id_ in range(self.batch_size):
            if f_id + id_ >= max_frame_number:
                break
            if cam.mode == 'svo':
                if id_ == 0:
                    batch.append(cam.get_zed(f_id, exclude_depth=True))
                else:
                    batch.append(cam.get_zed(exclude_depth=True))
            else:
                if id_ == 0:
                    batch.append(cam.get_frame(f_id))
                else:
                    batch.append(cam.get_frame())

        return batch