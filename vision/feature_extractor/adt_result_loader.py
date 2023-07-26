import os
import pandas as pd
from vision.tools.video_wrapper import video_wrapper
from vision.misc.help_func import load_json
import numpy as np
from vision.pipelines.ops.frame_loader import FramesLoader
from vision.feature_extractor.boxing_tools import cut_zed_in_jai
from vision.feature_extractor.boxing_tools import xyz_center_of_box
from vision.feature_extractor.tree_size_tools import stable_euclid_dist


import time

class ADTSBatchLoader:
    """
    A class for loading data ADT and slicing batches.

    Args:
        args: Arguments object containing configuration parameters.
        block_name: Name of the data block.
        row_path: Path to the row.
        tree_id: ID of the tree.

    Attributes:
        block_name (str): Name of the data block.
        row_path (str): Path to the row.
        cameras (dict): Dictionary containing camera objects.
        tree_id (int): ID of the tree.
        slicer_results (pandas.DataFrame): Data frame containing slicer results.
        alignment (pandas.DataFrame): Data frame containing alignment results.
        tracker_results (pandas.DataFrame): Data frame containing tracker results.
        jai_zed (dict): Dictionary containing JAI and ZED information.

    Methods:
        get_movie_paths(row_path, side): Returns a dictionary of movie paths.
        init_cameras(row_path, args): Initializes the cameras.
        load_dfs(): Loads data frames from files.
        load_adts(frame_ids): Loads ADTs (Additional Data Types).
        load_batch(frame_ids): Loads a batch of data.

    """
    def __init__(self, cfg, args, block_name, row_path, tree_id=1):
        """
        Initialize the ADTSBatchLoader object.

        Args:
            args: Arguments object containing configuration parameters.
            block_name: Name of the data block.
            row_path: Path to the row.
            tree_id: ID of the tree.

        """
        self.block_name = block_name
        self.row_path = row_path
        # self.cameras = self.init_cameras(row_path, args)
        self.tree_id = tree_id
        self.slicer_results, self.alignment, self.tracker_results = [], [], {}
        self.jai_translation = pd.DataFrame({})
        self.load_dfs()
        #self.get_row_data(args)
        self.frame_loader = FramesLoader(cfg, args)

    @staticmethod
    def get_movie_paths(row_path, side):
        """
        Get the movie paths based on the row path and side.

        Args:
            row_path: Path to the row.
            side: Side value indicating the camera side.

        Returns:
            Dictionary of movie paths.

        """
        if isinstance(side, int):
            row_dict = {"zed_movie_path": os.path.join(row_path, f"ZED_{side}.svo"),
                        "jai_movie_path": os.path.join(row_path, f"Result_FSI_{side}.mkv"),
                        "rgb_jai_movie_path": os.path.join(row_path, f"Result_RGB_{side}.mkv")}
        else:
            row_dict = {"zed_movie_path": os.path.join(row_path, f"ZED.svo"),
                        "jai_movie_path": os.path.join(row_path, f"Result_FSI.mkv"),
                        "rgb_jai_movie_path": os.path.join(row_path, f"Result_RGB.mkv")}
        return row_dict

    @staticmethod
    def init_cameras(row_path, args):
        """
        Initialize the cameras.

        Args:
            row_path: Path to the row.
            args: Arguments object containing configuration parameters.

        Returns:
            Dictionary of initialized camera objects.

        """
        side = 1 if row_path.endswith("A") else 2 if row_path.endswith("B") else ""
        cameras = {}
        row_dict = ADTSBatchLoader.get_movie_paths(row_path, side)
        jai_movie_path = row_dict["jai_movie_path"]
        rgb_movie_path = row_dict["rgb_jai_movie_path"]
        zed_movie_path = row_dict["zed_movie_path"]
        cameras["zed_cam"] = video_wrapper(zed_movie_path, args.zed.rotate,
                                                args.zed.depth_minimum, args.zed.depth_maximum)
        cameras["rgb_jai_cam"] = video_wrapper(rgb_movie_path, args.rgb_jai.rotate)
        cameras["jai_cam"] = video_wrapper(jai_movie_path, args.jai.rotate)
        return cameras

    def load_dfs(self):
        """
        Load data frames from files.

        """
        slices_path = os.path.join(self.row_path, "slices.csv")
        if os.path.exists(slices_path):
            self.slicer_results = pd.read_csv(slices_path)
        else:
            self.slicer_results = pd.read_csv(os.path.join(self.row_path, "all_slices.csv"))
        self.slicer_results = self.slicer_results[self.slicer_results["tree_id"] == self.tree_id]
        self.tracker_results = pd.read_csv(os.path.join(self.row_path, "tracks.csv"))
        self.alignment = pd.read_csv(os.path.join(self.row_path, "alignment.csv"))
        if os.path.exists(os.path.join(self.row_path, "jai_translations.csv")):
            self.jai_translation = pd.read_csv(os.path.join(self.row_path, "jai_translations.csv"))
        elif os.path.exists(os.path.join(self.row_path, "jai_translation.csv")):
            self.jai_translation = pd.read_csv(os.path.join(self.row_path, "jai_translation.csv"))
        for df in [self.slicer_results, self.tracker_results, self.alignment, self.jai_translation]:
            if "Unnamed: 0" in df.columns:
                df.drop("Unnamed: 0", axis=1, inplace=True)
        if "frame_id" in self.tracker_results.columns:
            self.tracker_results.rename({"frame_id": "frame"}, axis=1, inplace=True)

    def get_row_data(self, args):

        self.tracker_results = pd.read_csv(args.tracker_results)
        self.alignment = pd.read_csv(args.alignment)
        self.jai_translation = pd.read_csv(args.jai_translations)
        for df in [self.tracker_results, self.alignment, self.jai_translation]:
            if "Unnamed: 0" in df.columns:
                df.drop("Unnamed: 0", axis=1, inplace=True)
        if "frame_id" in self.tracker_results.columns:
            self.tracker_results.rename({"frame_id": "frame"}, axis=1, inplace=True)


    @staticmethod
    def validate_align(b_align, frame_ids):
        valid_coords = [cut_cords[0][:6] for cut_cords in b_align if len(cut_cords)]
        if not len(valid_coords):
            return [[55, 365, 1480, 1625, -999, -999, int(frame_id), 1] for frame_id in frame_ids]
        x1, y1, x2, y2, tx_a, ty_a = np.mean(valid_coords, axis=0)
        b_align = [cut_cords[0] if len(cut_cords) else [x1, y1, x2, y2, tx_a, ty_a, int(frame_ids[i]), 1]
                        for i, cut_cords in enumerate(b_align)]
        return b_align

    @staticmethod
    def validate_jai_translation(b_jai_translation, frame_ids):
        valid_translation = [jai_translation[:2] for jai_translation in b_jai_translation if len(jai_translation)]
        if not len(valid_translation):
            return []
        tx, ty = np.mean(valid_translation, axis=0)[0][:2]
        b_jai_translation = [jai_translation[0] if len(jai_translation) else [tx, ty, int(frame_ids[i])]
                             for i, jai_translation in enumerate(b_jai_translation)]
        return b_jai_translation

    def load_adts(self, frame_ids):
        """
        Load ADT and Slice results

        Args:
            frame_ids: List of frame IDs (strings).

        Returns:
            Tuple containing batch slicer, batch tracker, and b_align.

        """
        if self.block_name == "":
            self.block_name = os.path.basename(os.path.dirname(self.row_path))
        cols_slicer = ["start", "end"]
        batch_slicer = [self.slicer_results[self.slicer_results["frame_id"] == int(f_id)][cols_slicer].values.tolist()[0]
                        for f_id in frame_ids]
        batch_tracker = [self.tracker_results[self.tracker_results["frame"] == int(f_id)].values.tolist()
                         for f_id in frame_ids]
        cols_align = ["x1", "y1", "x2", "y2", "tx", "ty", "frame", "zed_shift"]
        b_align = [(self.alignment[self.alignment["frame"] == int(frame)][cols_align]).values.tolist()
                        for frame in frame_ids]
        # TODO handle cases when  no alignment in FE
        b_align = self.validate_align(b_align, frame_ids)

        if len(self.jai_translation):
            b_jai_translation = [(self.jai_translation[self.jai_translation["frame"] == int(frame)]).values.tolist()
                        for frame in frame_ids]
            b_jai_translation = self.validate_jai_translation(b_jai_translation, frame_ids)
        else:
            b_jai_translation = []
        return batch_slicer, batch_tracker, b_align, b_jai_translation

    def tracker_postprocess(self, batch_tracker, b_align, batch_zed, batch_fsi):
        xyz_dims_cols = ["pc_x", "pc_y", "depth", "width", "height"]
        tracker_res_cols = self.tracker_results.columns
        if np.all([col in tracker_res_cols for col in xyz_dims_cols]):
            return batch_tracker
        cut_coords = tuple(res[:4] for res in b_align)
        batch_tracker = get_xyz_to_bboxes_batch([zed[:, :, (1, 0, 2)] for zed in batch_zed], batch_fsi, cut_coords,
                                                  batch_tracker, True, True)
        return batch_tracker

    def load_batch(self, frame_ids):
        """
        Load a batch of data.

        Args:
            frame_ids: List of frame IDs (strings).

        Returns:
            Tuple containing batch_fsi, batch_zed, batch_jai_rgb, batch_rgb_zed,
            batch_tracker, batch_slicer, frame_ids, and b_align.

        """
        try:

            batch_slicer, batch_tracker, b_align, b_jai_translation = self.load_adts(frame_ids)
            batch_fsi, batch_zed, batch_jai_rgb, batch_rgb_zed = [], [], [], []
            batch_rgb_zed, batch_zed, batch_fsi, batch_jai_rgb = self.frame_loader.get_frames(int(frame_ids[0]), 0)
            batch_zed = [zed[:, :, (1, 0, 2)] for zed in batch_zed]
            batch_fsi = [fsi[:, :, ::-1] for fsi in batch_fsi]
            batch_tracker = self.tracker_postprocess(batch_tracker, b_align, batch_zed, batch_fsi)
        except:
            print("debug")

        return (batch_fsi, batch_zed, batch_jai_rgb, batch_rgb_zed, batch_tracker, batch_slicer, frame_ids,
                b_align, b_jai_translation)




def get_xyz_to_bboxes(xyz_frame, jai_frame, cut_coords, dets, pc=False, dims=False, aligned=False,
                        resize_factors=(1, 1)):
    """
    Retrieves the depth to each bbox
    Args:
        xyz_frame (np.array): a Point cloud image
        jai_frame (np.array): FSI image
        cut_coords (tuple): jai in zed coords
        dets (list): list of detections
        pc (bool): flag for returning the entire x,y,z
        dims (bool): flag for returning the real width and height
        aligned (bool): flag indicating if the frames are already aligned
        resize_factors (Iterable): resize factors for an aligned imaged (r_h, r_w)

    Returns:
        z_s (list): list with depth to each detection
    """
    if not aligned:
        cut_coords = dict(zip(["x1", "y1", "x2", "y2"], [[int(cord)] for cord in cut_coords]))
        xyz_frame_aligned = cut_zed_in_jai({"zed": xyz_frame}, cut_coords, rgb=False)["zed"]
        r_h, r_w = xyz_frame_aligned.shape[0] / jai_frame.shape[0], xyz_frame_aligned.shape[1] / jai_frame.shape[1]
    else:
        xyz_frame_aligned, r_h, r_w = xyz_frame, resize_factors[0], resize_factors[1]
    for det in dets:
        box = ((int(det[0] * r_w), int(det[1] * r_h)), (int(det[2] * r_w), int(det[3] * r_h)))
        det_output = []
        if pc:
            det += list(xyz_center_of_box(xyz_frame_aligned, box))
        else:
            det.append(xyz_center_of_box(xyz_frame_aligned, box)[2])
        if dims:
            det += list(stable_euclid_dist(xyz_frame_aligned, box))
    return dets


def get_xyz_to_bboxes_batch(xyz_batch, jai_batch, cut_coords, dets, pc=False, dims=False):
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
    return list(map(get_xyz_to_bboxes, xyz_batch, jai_batch, cut_coords, dets, [pc] * n, [dims] * n))


def append_to_trk(trk_batch_res, results):
    for frame_res, frame_depth in zip(trk_batch_res, results):
        for i, depth in enumerate(frame_depth):
            frame_res[i] += [np.round(depth, 3)]

    return trk_batch_res