import os
import pandas as pd
from vision.tools.video_wrapper import video_wrapper
from vision.misc.help_func import load_json
import numpy as np
from vision.pipelines.ops.frame_loader import FramesLoader

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
        self.jai_translation, self.percent_seen = pd.DataFrame({}), pd.DataFrame({})
        self.load_dfs()
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
        if os.path.exists(os.path.join(self.row_path, "percen_seen.csv")):
            self.percent_seen = pd.read_csv(os.path.join(self.row_path, "percen_seen.csv"))
        else:
            self.percent_seen = {}
        if os.path.exists(os.path.join(self.row_path, "jai_translations.csv")):
            self.jai_translation = pd.read_csv(os.path.join(self.row_path, "jai_translations.csv"))
        elif os.path.exists(os.path.join(self.row_path, "jai_translation.csv")):
            self.jai_translation = pd.read_csv(os.path.join(self.row_path, "jai_translation.csv"))
        for df in [self.slicer_results, self.tracker_results, self.alignment, self.jai_translation]:
            if "Unnamed: 0" in df.columns:
                df.drop("Unnamed: 0", axis=1, inplace=True)
        tracker_res_cols = self.tracker_results.columns
        if "frame_id" in tracker_res_cols:
            self.tracker_results.rename({"frame_id": "frame"}, axis=1, inplace=True)
        xyz_dims_cols = ["pc_x", "pc_y", "depth", "width", "height"]
        # if ("depth" in tracker_res_cols) and (not np.all([col in tracker_res_cols for col in xyz_dims_cols])):
        #     self.tracker_results.drop("depth", axis=1, inplace=True)

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
        if not self.slicer_results.empty:
            batch_slicer = [self.slicer_results[self.slicer_results["frame_id"] == int(f_id)][cols_slicer].values.tolist()[0]
                            for f_id in frame_ids]
        else:
            batch_slicer = [(-1, -1) for _ in frame_ids]
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
            b_jai_translation = [[] for i in range(len(frame_ids))]
        if len(self.percent_seen):
            try:
                batch_ps = [self.percent_seen[self.percent_seen["frame"] == int(f_id)].values.tolist()
                                 for f_id in frame_ids]
                batch_ps = [ps[0] if len(ps) > 0 else ps for ps in batch_ps]
            except Exception as e:
                batch_ps = [[] for i in range(len(frame_ids))]
        else:
            batch_ps = [[] for i in range(len(frame_ids))]
        return batch_slicer, batch_tracker, b_align, b_jai_translation, batch_ps

    def tracker_postprocess(self, batch_tracker, b_align, batch_zed, batch_fsi):
        xyz_dims_cols = ["pc_x", "pc_y", "depth", "width", "height"]
        tracker_res_cols = self.tracker_results.columns
        if np.all([col in tracker_res_cols for col in xyz_dims_cols]):
            return batch_tracker
        if "depth" in tracker_res_cols:
            return batch_tracker
        # cut_coords = tuple(res[:4] for res in b_align)
        # batch_tracker = get_depth_to_bboxes_batch([zed[:, :, (1, 0, 2)] for zed in batch_zed], batch_fsi, cut_coords,
        #                                           batch_tracker, True, True)
        # return batch_tracker

    def load_batch(self, frame_ids, shift=0):
        """
        Load a batch of data.

        Args:
            frame_ids: List of frame IDs (strings).

        Returns:
            Tuple containing batch_fsi, batch_zed, batch_jai_rgb, batch_rgb_zed,
            batch_tracker, batch_slicer, frame_ids, and b_align.

        """
        try:
            batch_slicer, batch_tracker, b_align, b_jai_translation, batch_ps = self.load_adts(frame_ids)
            batch_fsi, batch_zed, batch_jai_rgb, batch_rgb_zed = [], [], [], []
            batch_rgb_zed, batch_zed, batch_fsi, batch_jai_rgb = self.frame_loader.get_frames(int(frame_ids[0]),
                                                                                              0)
            batch_zed = [zed[:, :, (1, 0, 2)] for zed in batch_zed]
            batch_fsi = [fsi[:, :, ::-1] for fsi in batch_fsi]
            batch_tracker = self.tracker_postprocess(batch_tracker, b_align, batch_zed, batch_fsi)
        except:
            print("debug")

        return (batch_fsi, batch_zed, batch_jai_rgb, batch_rgb_zed, batch_tracker, batch_slicer, frame_ids,
                b_align, b_jai_translation, batch_ps)

