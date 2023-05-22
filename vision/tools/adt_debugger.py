import cv2
import pandas as pd
import os
from vision.visualization.drawer import draw_rectangle, draw_text, draw_highlighted_test, get_color
import numpy as np
from omegaconf import OmegaConf
from vision.misc.help_func import (get_repo_dir, load_json, validate_output_path, read_json,
                                   safe_read_csv, pop_list_drom_dict, post_process_slice_df)
from vision.tools.video_wrapper import video_wrapper
from vision.tools.image_stitching import plot_2_imgs
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
#from vision.feature_extractor.boxing_tools import cut_zed_in_jai
from vision.tools.manual_slicer import slice_to_trees_df
from tqdm import tqdm
from time import time
from itertools import chain
from vision.pipelines.ops.frame_loader import FramesLoader


class ADTDebugger:
    def __init__(self, args):
        """Initialize the ADTDebugger object with the given arguments.

        Args:
            args: An object containing the input arguments.

        Attributes:
            row_path: The path to the row data.
            rotate: Whether to rotate the video.
            methods: The analyzing methods to use.
            alignment_methods: The alignment filtering methods to use.
            t_index: The index of the track ID in each detection.
            max_depth: The maximum depth of the tree.
            im_output_shape: The output shape of the images.
            min_samples: The minimum number of samples required to count a fruit as valid.
            filter_depth: flag for filter depth use.
            s_frame: The start frame of the video.
            max_frame: The maximum frame of the video.
            fps: The frames per second of the video.
            new_vid_name: The name of the new video to create.
            jai_fp: The file path of the JAI camera video.
            zed_fp: The file path of the ZED camera video.
            side: The side of the camera to use.
            outputs_dir: The directory to output the results to.
            jai_frame_w_dets_dir: The directory to output JAI frames with detections.
            frame_w_dets_vid_name: The name of the video of frames with detections.
            cap_fsi: The JAI camera video object.
            zed_cam: The ZED camera video object.
            tracks_df: A dataframe containing tracking results.
            slices_df: A dataframe containing slicer results.
            jai_cors_in_zed: The JAI coordinates in ZED coordinates.
            jai_zed_frames_dict: A dictionary containing mapping of JAI and ZED frames.
            tracker_results: The tracking results.
            tracker_d_f: The tracking results after depth filtering.
            tracker_ms_f: The tracking results min samples filtering.
            tracker_ms_d_f: The tracking results with depth and min samples filtering.

        """
        print(f"debugging row {args.row_path}")
        self.row_path = args.row_path
        self.rotate = args.rotate
        self.methods = args.methods
        self.alignment_methods = args.alignment_methods
        self.t_index = args.t_index
        self.max_depth = args.max_depth
        self.im_output_shape = args.im_output_shape
        self.min_samples = args.min_samples
        self.filter_depth = args.filter_depth
        if args.scan_type == "multi_scans":
            self.scan_id = os.path.basename(self.row_path)
            self.row_name = os.path.basename(os.path.dirname(self.row_path))
        else:
            self.row_name = os.path.basename(self.row_path)
            self.scan_id = None
        self.zed_style = "svo"
        self.scan_type = args.scan_type
        # vid writing arguments
        self.s_frame, self.max_frame, self.fps, self.new_vid_name = self.get_vid_args(args)
        # video pathing
        self.jai_fp, self.zed_fp, side = self.get_vid_paths()
        # init dirs
        self.outputs_dir, self.jai_frame_w_dets_dir, self.frame_w_dets_vid_name = self.get_dirs(args, side)
        self.validate_paths()
        # video object init
        self.cap_fsi, self.zed_cam = self.get_cams()
        # read results
        self.tracks_df, self.slices_df, self.jai_cors_in_zed, self.jai_zed_frames_dict = self.read_adt_results()
        # self.jai_cors_in_zed = self.jai_cors_in_zed_fixer(self.jai_cors_in_zed)
        self.tracker_results, self.tracker_d_f, self.tracker_ms_f, self.tracker_ms_d_f = self.get_tracker_results()
        self.slicer_results = self.get_slicer_results()

    @staticmethod
    def jai_cors_in_zed_fixer(jai_cors_in_zed):
        jai_cors_in_zed["frame"] += np.array(list(chain(*[[0, 1, 2, 3]
                                                          for i in range(len(jai_cors_in_zed["frame"])//4)])))
        return jai_cors_in_zed

    @staticmethod
    def draw_dets(frame, dets, t_index=6):
        """Draw bounding boxes around detections on the given frame.

        Args:
            frame(np.ndarry): The frame on which to draw the bounding boxes.
            dets(list): A list of detections to draw.
            t_index(int): The index of the track ID in each detection.

        Returns:
            The frame with bounding boxes drawn around the detections.

        """
        frame = frame.copy()
        for det in dets:
            track_id = det[t_index]
            color_id = int(track_id) % 15  # 15 is the number of colors in list
            color = get_color(color_id)
            text_color = get_color(-1)
            frame = draw_rectangle(frame, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), color, 3)
            frame = draw_highlighted_test(frame, f'ID:{int(track_id)}', (det[0], det[1]), frame.shape[1], color,
                                          text_color, True, 10, 3)
        return frame

    @staticmethod
    def get_vid_args(args):
        """Get video-related arguments from the input arguments.

        Args:
            args: An object containing the input arguments.

        Returns:
            A tuple containing the start frame, maximum frame, frames per second, and the new video name.

        """
        s_frame, max_frame = args.frames_limit
        if max_frame == -1:
            max_frame = np.inf
        return s_frame, max_frame, args.fps, args.new_vid_name

    @staticmethod
    def tracker_df_2_dict(tracker_results_frame):
        """Convert the tracker results dataframe to a dictionary.

        Args:
            tracker_results_frame(pd,DataFrame): The tracker results dataframe for a specific frame.

        Returns:
            A dictionary containing the track IDs as keys and the corresponding bounding box points as values.

        """
        point_1 = tuple(zip(tracker_results_frame["x1"], tracker_results_frame["y1"]))
        point_2 = tuple(zip(tracker_results_frame["x2"], tracker_results_frame["y2"]))
        points = tuple(zip(point_1, point_2))
        return dict(zip(tracker_results_frame["track_id"], points))

    def validate_paths(self):
        """Validate the output paths."""
        validate_output_path(self.outputs_dir)
        if not isinstance(self.jai_frame_w_dets_dir, type(None)):
            validate_output_path(self.jai_frame_w_dets_dir)
        if "alignment" in self.methods:
            for align_method in self.alignment_methods:
                if align_method == "":
                    validate_output_path(os.path.join(self.outputs_dir, "alignment"))
                else:
                    validate_output_path(os.path.join(self.outputs_dir, f"alignment_{align_method}"))

    def get_vid_paths(self):
        """Get the paths of the JAI and ZED camera videos.

        Returns:
            A tuple containing the JAI camera video path, ZED camera video path, and the side of the camera.

        """
        side = 1 if self.row_path.endswith("A") else 2
        if self.scan_type == "multi_scans":
            jai_fp = os.path.join(self.row_path, f'Result_FSI.mkv')
            zed_fp = os.path.join(self.row_path, f'ZED.svo')
            return jai_fp, zed_fp, side
        jai_fp = os.path.join(self.row_path, f'Result_FSI_{side}.mkv')
        zed_fp = os.path.join(self.row_path, f'ZED_{side}.svo')
        return jai_fp, zed_fp, side

    def get_slice_data_path(self):
        """
        formats the path to slicing data
        Returns:
            jai_slice_data_path (str): the path to slice data json.
        """
        side = 1 if self.row_path.endswith("A") else 2
        row = os.path.basename(self.row_path)
        if self.scan_type == "multi_scans":
            jai_slice_data_path = os.path.join(self.row_path, f"Result_FSI_slice_data_{row}.json")
            if not os.path.exists(jai_slice_data_path):
                jai_slice_data_path = os.path.join(self.row_path, f"Result_FSI_slice_data.json")
        else:
            jai_slice_data_path = os.path.join(self.row_path, f"Result_FSI_{side}_slice_data_{row}.json")
            if not os.path.exists(jai_slice_data_path):
                jai_slice_data_path = os.path.join(self.row_path, f"Result_FSI_{side}_slice_data.json")
        return jai_slice_data_path

    def read_slice_df_from_json(self):
        """Read the slice dataframe from a JSON file.

        Returns:
            The slice dataframe.

        """
        jai_slice_data_path = self.get_slice_data_path()
        if os.path.exists(jai_slice_data_path):
            slices_df = slice_to_trees_df(jai_slice_data_path, self.row_path)
        else:
            slices_df = pd.DataFrame({}, columns=["tree_id", "frame_id", "start", "end"])
        return slices_df

    def read_adt_results(self):
        """Read the ADT results (tracks, slices, jai cors in zed, jai-zed mapping).

        Returns:
            A tuple containing the tracks dataframe, slices dataframe, JAI coordinates in ZED dataframe,
            and JAI and ZED frames mapping dictionary.

        """
        print("reading results")
        tracks_path = os.path.join(self.row_path, f'tracks.csv')
        slices_path = os.path.join(self.row_path, f'slices.csv')
        jai_cors_in_zed_path = os.path.join(self.row_path, f'jai_cors_in_zed.csv')
        jai_zed_path = os.path.join(self.row_path, f'jai_zed.json')
        tracks_df, slices_df = safe_read_csv(tracks_path), safe_read_csv(slices_path)
        if "frame_id" in tracks_df.columns:
            tracks_df.rename({"frame_id": "frame"}, axis=1, inplace=True)
        if not len(slices_df):
            slices_df = self.read_slice_df_from_json()
        slices_df = post_process_slice_df(slices_df)
        slices_df["start"] = slices_df["start"].replace(-1, 0)
        slices_df["end"] = slices_df["end"].replace(-1, int(self.get_width_height_cam()[0] - 1))
        jai_cors_in_zed, jai_zed_frames_dict = safe_read_csv(jai_cors_in_zed_path), read_json(jai_zed_path)
        if not os.path.exists(jai_cors_in_zed_path):
            alignment_path = os.path.join(self.row_path, f'alignment.csv')
            jai_cors_in_zed = safe_read_csv(alignment_path)
        if not os.path.exists(jai_zed_path):
            jai_zed_path = os.path.join(self.row_path, f'jaized_timestamps.log')
            zed_ids, _ = FramesLoader.get_cameras_sync_data(jai_zed_path)
            jai_zed_frames_dict = dict(zip([str(item) for item in range(len(zed_ids))], zed_ids))
        print("finished reading results")
        return tracks_df, slices_df, jai_cors_in_zed, jai_zed_frames_dict

    def get_cams(self):
        """Get the camera objects.

        Returns:
            A tuple containing the JAI camera video object and ZED camera video object.

        """
        if os.path.exists(self.jai_fp):
            cap_fsi = cv2.VideoCapture(self.jai_fp)
        else:
            cap_fsi = None
            print("incorrect jai path")
        mkv_path = f"{self.zed_fp[:-3]}mkv"
        if os.path.exists(self.zed_fp):
            zed_cam = video_wrapper(self.zed_fp, args.zed.rotate, args.zed.depth_minimum, args.zed.depth_maximum)
        elif os.path.exists(mkv_path):
            gray_cam = video_wrapper(mkv_path, args.zed.rotate)
            depth_cam = video_wrapper(os.path.join(self.row_path, f'DEPTH.mkv'), args.zed.rotate)
            zed_cam = [gray_cam, depth_cam]
            self.zed_style = "mkv"
        else:
            zed_cam = None
            print("incorrect zed path")
        return cap_fsi, zed_cam

    def get_dirs(self, args, side):
        """Get the directories for output.

         Args:
             args: An object containing the input arguments.
             side(int): The side of the camera.

         Returns:
             A tuple containing the outputs directory, JAI frames with detections directory,
             and the video name of frames with detections.

         """
        if self.scan_type == "multi_scans":
            block_name = os.path.basename(os.path.dirname(os.path.dirname(self.row_path)))
        else:
            block_name = os.path.basename(os.path.dirname(self.row_path))
        outputs_dir = os.path.join(args.outputs_dir, f"{block_name}{args.block_suffix}")
        validate_output_path(outputs_dir)
        outputs_dir = os.path.join(outputs_dir, self.row_name)
        validate_output_path(outputs_dir)
        if self.scan_type == "multi_scans":
            outputs_dir = os.path.join(outputs_dir, self.scan_id)
        if outputs_dir != "" and "jai_frame_w_dets" in self.methods:
            jai_frame_w_dets_dir = os.path.join(outputs_dir, "Fsi_w_dets")
        else:
            jai_frame_w_dets_dir = None
        if self.new_vid_name == "":
            frame_w_dets_vid_name = os.path.join(outputs_dir, f'Result_FSI_{side}_with_dets.mkv')
            if self.scan_type == "multi_scans":
                frame_w_dets_vid_name = os.path.join(outputs_dir, f'Result_FSI_with_dets.mkv')
        else:
            frame_w_dets_vid_name = os.path.join(outputs_dir, self.new_vid_name)
        return outputs_dir, jai_frame_w_dets_dir, frame_w_dets_vid_name

    def read_next_fsi(self):
        """Read the next frame from the JAI camera video.

        Returns:
            A tuple containing a boolean value indicating if the frame was read successfully and the frame itself.

        """
        ret = True
        _, frame_fsi = self.cap_fsi.read()
        if isinstance(frame_fsi, type(None)):
            ret = False
        if self.rotate:
            frame_fsi = cv2.rotate(frame_fsi, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return ret, frame_fsi

    def write_files(self):
        """Write the output files.

        Writes frames with detections, alignment frames, and generates a video with detections if enabled.

        """
        fsi_vid = self.preprocess_fsi()
        i, n = self.s_frame, min(self.cap_fsi.get(cv2.CAP_PROP_FRAME_COUNT), self.max_frame)
        while self.cap_fsi.isOpened() and i < n:
            print(f"\r{i+1}/{n} ({(i-self.s_frame) / (n - self.s_frame) * 100: .2f}%) frames", end="")
            ret, frame_fsi = self.read_next_fsi()
            if not ret:
                break
            dets = self.tracks_df[self.tracks_df["frame"] == i].to_numpy()
            frame_fsi_w_dets = self.draw_dets(frame_fsi, dets, t_index=self.t_index)
            if self.jai_frame_w_dets_dir != "":
                cv2.imwrite(os.path.join(self.jai_frame_w_dets_dir, f"frame_{i}.jpg"), frame_fsi_w_dets)
            if "vid_with_dets" in self.methods:
                fsi_vid.write(frame_fsi_w_dets)
            if "alignment" in self.methods:
                self.save_translation_debug_frame(i, frame_fsi)
            i += 1
        if "vid_with_dets" in self.methods:
            fsi_vid.release()

    def get_width_height_cam(self):
        """Get the width and height of the JAI camera.

        Returns:
            A tuple containing the width and height of the camera.

        """
        width = int(self.cap_fsi.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap_fsi.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.rotate:
            width, height = height, width
        return width, height

    def preprocess_fsi(self):
        """Preprocess the JAI camera video.

        Returns:
            The video writer object if writing frames with detections is enabled, otherwise None.

        """
        width, height = self.get_width_height_cam()
        if not self.fps:
            self.fps = int(self.cap_fsi.get(cv2.CAP_PROP_FPS))
        if "vid_with_dets" in self.methods:
            fsi_vid = cv2.VideoWriter(self.frame_w_dets_vid_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                  self.fps, (width, height))
        else:
            fsi_vid = None
        self.cap_fsi.set(cv2.CAP_PROP_POS_FRAMES, self.s_frame)
        return fsi_vid

    def get_tracker_results(self):
        """Get the tracker results with multiple filtering types.

        Returns:
            A tuple containing dictionaries of tracker results, depth-filtered results,
            minimum samples filtered results, and minimum samples and depth-filtered results.

        """
        print("processing tracker results")
        if not len(self.tracks_df):
            return {}
        tracker_results, depth_filtered_results = {}, {}
        min_samp_filtered, min_samp_depth_filtered = {}, {}
        uniq, counts = np.unique(self.tracks_df["track_id"], return_counts=True)
        non_valid_tracks_ms = uniq[counts < self.min_samples].tolist()
        for frame in tqdm(self.tracks_df["frame"].unique()):
            tracker_results_frame = self.tracks_df[self.tracks_df["frame"] == frame]
            tracker_results[frame] = self.tracker_df_2_dict(tracker_results_frame)
            cur_results = tracker_results[frame].copy()
            to_pop_depth = tracker_results_frame["track_id"][tracker_results_frame["depth"] > self.max_depth].tolist()
            depth_filtered_results[frame] = pop_list_drom_dict(cur_results, to_pop_depth)
            min_samp_filtered[frame] = pop_list_drom_dict(cur_results, non_valid_tracks_ms)
            min_samp_depth_filtered[frame] = pop_list_drom_dict(cur_results, non_valid_tracks_ms + to_pop_depth)
        print("tracker results processes")
        return tracker_results, depth_filtered_results, min_samp_filtered, min_samp_depth_filtered

    def get_slicer_results(self):
        """Get the slicer results.

        Returns:
            A dictionary containing the frame IDs as keys and the corresponding start and end points as values.

        """
        if not len(self.slices_df):
            return {}
        slices = tuple(zip(self.slices_df["start"], self.slices_df["end"]))
        slicer_results = dict(zip(self.slices_df["frame_id"], slices))
        return slicer_results

    def resize_imgs_coords(self, zed, xyz, frame_fsi):
        """Resize the ZED images, XYZ coordinates, and JAI frame.

        Args:
            zed(np.ndarray): The ZED image.
            xyz(np.ndarray): The XYZ coordinates.
            frame_fsi(np.ndarray): The JAI frame.

        Returns:
            A tuple containing the resized ZED image, XYZ coordinates, JAI frame, and the resize factors.

        """
        jai_h, jai_w = frame_fsi.shape[:2]
        zed = cv2.resize(zed, self.im_output_shape)
        xyz = cv2.resize(xyz, self.im_output_shape)
        frame_fsi = cv2.resize(frame_fsi, self.im_output_shape)
        resize_factors = self.im_output_shape[1] / jai_h, self.im_output_shape[0] / jai_w
        return zed, xyz, frame_fsi, resize_factors

    def resize_tracker_slicer(self):
        print("hello")

    def get_f_id_images(self, frame_number, frame_fsi):
        """Get the images for a specific frame id.

        Args:
            frame_number(int): The frame number.
            frame_fsi(np.ndarray): The JAI frame.

        Returns:
            A tuple containing the ZED image, XYZ coordinates, JAI frame, and the resize factors.

        """
        zed_frame_number = self.jai_zed_frames_dict[str(frame_number)]
        if self.zed_style == "svo":
            zed, xyz = self.zed_cam.get_zed(zed_frame_number, exclude_depth=True)
        else:
            _, zed = self.zed_cam[0].get_frame(zed_frame_number)
            xyz = self.zed_cam[1].get_frame(zed_frame_number)[1]/255*8
            # xyz = np.where(xyz == 8, np.nan, xyz)
        cur_coords = self.jai_cors_in_zed[self.jai_cors_in_zed["frame"] == frame_number]
        if len(cur_coords):
            zed = cut_zed_in_jai(zed, cur_coords.astype(int).reset_index(drop=True), image_input=True)
            xyz = cut_zed_in_jai(xyz, cur_coords.astype(int).reset_index(drop=True), image_input=True)
        else:
            zed, xyz = np.zeros_like(frame_fsi), np.zeros_like(frame_fsi)
        zed, xyz, frame_fsi, resize_factors = self.resize_imgs_coords(zed, xyz, frame_fsi)
        return zed, xyz, frame_fsi, resize_factors

    def filter_depth_imgs(self, frame_fsi, zed, xyz):
        """Filter the depth images, if value is larger than 'self.max_depth' it will be painted black,
            if the value is nan it will be painted white.

        Args:
            frame_fsi(np.ndarray): The JAI frame.
            zed(np.ndarray): The ZED image.
            xyz(np.ndarray): The XYZ coordinates.

        Returns:
            The filtered JAI frame, ZED image, and XYZ coordinates.

        """
        na_depth, non_valid_depths = np.isnan(xyz[:, :, 2]), xyz[:, :, 2] > self.max_depth
        zed[na_depth] = 255
        frame_fsi[na_depth] = 255
        frame_fsi[non_valid_depths] = 0
        zed[non_valid_depths] = 0
        return frame_fsi, zed, xyz

    def save_translation_debug_frame(self, frame_number, frame_fsi):
        """
        Saves the debug frame for translation.

        Args:
            frame_number (int): The frame number.
            frame_fsi (np.ndarray): The FSI frame.

        Returns:
            None
        """
        if str(frame_number) not in self.jai_zed_frames_dict.keys():
            return
        zed, xyz, frame_fsi, resize_factors = self.get_f_id_images(frame_number, frame_fsi)
        if self.filter_depth:
            frame_fsi, zed, xyz = self.filter_depth_imgs(frame_fsi, zed, xyz)
        saving_folders, tracker_results, n_methods = self.get_translation_lists_mp()
        for i in range(n_methods):
            self.draw_translation(tracker_results[i], frame_number, frame_fsi, zed, saving_folders[i], resize_factors)

    def get_translation_lists_mp(self):
        """
        Creates lists for processing of draw_translation.

        Returns:
            tuple: A tuple containing the following lists:
                   - saving_folders (list): Lists of saving folders for draw_translation.
                   - tracker_results (list): Lists of tracker results for draw_translation.
                   - n_methods (int): The number of methods used for alignment.
        """
        saving_folders = []
        tracker_results = []
        if "" in self.alignment_methods:
            saving_folders.append(os.path.join(self.outputs_dir, "alignment"))
            tracker_results.append(self.tracker_results)
        if "depth" in self.alignment_methods:
            saving_folders.append(os.path.join(self.outputs_dir, "alignment_depth"))
            tracker_results.append(self.tracker_d_f)
        if "min_samp" in self.alignment_methods:
            saving_folders.append(os.path.join(self.outputs_dir, "alignment_min_samp"))
            tracker_results.append(self.tracker_ms_f)
        if "min_samp_depth" in self.alignment_methods:
            saving_folders.append(os.path.join(self.outputs_dir, "alignment_min_samp_depth"))
            tracker_results.append(self.tracker_ms_d_f)
        return saving_folders, tracker_results, len(saving_folders)

    def get_slice_start_end(self, slices, frame_number, frame_fsi, r_w):
        """
        Gets the start and end indices of a slice.

        Args:
            slices (tuple): The start and end indices of the slice.
            frame_number (int): The frame number.
            frame_fsi (np.ndarray): The FSI frame.
            r_w (float): The resize factor for the width.

        Returns:
            tuple: A tuple containing the start and end indices of the slice.
        """
        if not isinstance(slices, type(None)):
            slice_start, slice_end = int(max(0, slices[0]*r_w)), int(min(slices[1]*r_w, frame_fsi.shape[1]-1))
        elif frame_number in self.slicer_results:
            slice_start = int(self.slicer_results[frame_number][0]*r_w)
            slice_end = int(self.slicer_results[frame_number][1]*r_w)
        else:
            slice_start, slice_end = 0, frame_fsi.shape[1]-1
        return slice_start, slice_end

    def draw_translation(self, tracker_results, frame_number, frame_fsi, zed, alignment_save_folder, resize_factors,
                         slices=None):
        """
        Draws the translations on the FSI and ZED images.

        Args:
            tracker_results (dict): The tracker results.
            frame_number (int): The frame number.
            frame_fsi (np.ndarray): The FSI image.
            zed (np.ndarray): The ZED image.
            alignment_save_folder (str): The alignment save folder.
            resize_factors (tuple): The resize factors for the images.
            slices (tuple, optional): The start and end indices of the slices. Defaults to None.

        Returns:
            None
        """
        frame_fsi = frame_fsi.copy()
        zed = zed.copy()
        r_h, r_w = resize_factors
        if frame_number in tracker_results.keys():
            for track_id, track in tracker_results[frame_number].items():
                track = ((int(track[0][0] * r_w), int(track[0][1] * r_h)),
                         (int(track[1][0] * r_w), int(track[1][1] * r_h)))
                frame_fsi = cv2.rectangle(frame_fsi, (track[0][0], track[0][1]), (track[1][0], track[1][1]),
                                          color=(0, 0, 255), thickness=2)
                zed = cv2.rectangle(zed, (track[0][0], track[0][1]), (track[1][0], track[1][1]), color=(0, 0, 255),
                                    thickness=2)
        slice_start, slice_end = self.get_slice_start_end(slices, frame_number, frame_fsi, r_w)
        max_y = self.im_output_shape[1]
        frame_fsi = cv2.line(frame_fsi, (slice_start, 0), (slice_start, max_y), color=(255, 0, 0), thickness=2)
        frame_fsi = cv2.line(frame_fsi, (slice_end, 0), (slice_end, max_y), color=(255, 0, 0), thickness=2)
        zed = cv2.line(zed, (slice_start, 0), (slice_start, max_y), color=(255, 0, 0), thickness=2)
        zed = cv2.line(zed, (slice_end, 0), (slice_end, max_y), color=(255, 0, 0), thickness=2)
        save_to = os.path.join(alignment_save_folder, f"{frame_number}.jpg")
        plot_2_imgs(zed[:, :, ::-1], frame_fsi[:, :, ::-1], frame_number, save_to=save_to, quick_save=True)

    def filter_outside_tree_boxes(self, tree_slices, tree_tracks):
        """
        Filters the tree tracks that are outside of the given tree slices.

        Args:
            tree_slices (pd.DataFrame): The tree slices.
            tree_tracks (pd.DataFrame): The tree tracks.

        Returns:
            pd.DataFrame: The filtered tree tracks.
        """
        dfs = []
        for frame in tree_slices["frame_id"]:
            slices = tree_slices[tree_slices["frame_id"] == frame][["start", "end"]].values[0]
            tree_tracks_frame = tree_tracks[tree_tracks["frame"] == frame]
            val_1 = tree_tracks_frame["x1"].values > slices[0]
            val_2 = tree_tracks_frame["x2"].values < slices[1]
            dfs.append(tree_tracks_frame[val_1 & val_2])
        return pd.concat(dfs)

    def get_tree_slice_track(self, tree_id, depth_filter=False, min_samp_filter=False):
        """
        Gets the tree slice and track data for the given tree ID.

        Args:
            tree_id (int): The tree ID.
            depth_filter (bool, optional): Whether to apply depth filtering. Defaults to False.
            min_samp_filter (bool, optional): Whether to apply minimum sample filtering. Defaults to False.

        Returns:
            pd.DataFrame: The tree slice and track data.
        """
        tree_slices = self.slices_df[self.slices_df["tree_id"] == tree_id]
        tree_tracks = self.tracks_df[np.isin(self.tracks_df["frame"], tree_slices["frame_id"])]
        tree_tracks = self.filter_outside_tree_boxes(tree_slices, tree_tracks)
        if min_samp_filter:
            unique_tracks, counts = np.unique(tree_tracks["track_id"], return_counts=True)
            tree_tracks = tree_tracks[np.isin(tree_tracks["track_id"], unique_tracks[counts >= self.min_samples])]
        if depth_filter:
            tree_tracks = tree_tracks[tree_tracks["depth"] < self.max_depth]
        unique_tracks, counts = np.unique(tree_tracks["track_id"], return_counts=True)
        new_ids = dict(zip(unique_tracks, range(len(unique_tracks))))
        tree_tracks.loc[:, "track_id"] = tree_tracks["track_id"].map(new_ids)
        tracker_results = {}
        for frame in self.tracks_df["frame"].unique():
            tracker_results_frame = tree_tracks[tree_tracks["frame"] == frame]
            tracker_results[frame] = self.tracker_df_2_dict(tracker_results_frame)
        return tree_tracks, tracker_results, tree_slices

    def get_tree_save_dir(self, min_samp_filter, depth_filter, tree_id):
        """
        Get the save directories for a tree based on filtering options.

        Args:
            min_samp_filter (bool): Flag indicating whether to apply minimum sample filtering.
            depth_filter (bool): Flag indicating whether to apply depth filtering.
            tree_id (int): Tree ID.

        Returns:
            tuple: A tuple containing the tree folder, alignment save folder, and FSI with tracks folder.
        """
        align_folder = f"alignment"
        if min_samp_filter:
            align_folder += "_min_samp"
        if depth_filter:
            align_folder += "_depth"
        trees_folder = os.path.join(self.outputs_dir, "trees")
        tree_folder = os.path.join(trees_folder, f"T{tree_id}")
        fsi_w_tracks_folder = os.path.join(tree_folder, "FSI_w_tracks")
        alignment_save_folder = os.path.join(tree_folder, align_folder)
        validate_output_path(trees_folder)
        validate_output_path(tree_folder)
        validate_output_path(fsi_w_tracks_folder)
        validate_output_path(alignment_save_folder)
        return tree_folder, alignment_save_folder, fsi_w_tracks_folder

    def draw_on_tracked_imgaes(self, depth_filter=False, min_samp_filter=False, save_dets=False):
        """
        Draw tracked images for each tree.

        Args:
            depth_filter (bool, optional): Flag indicating whether to apply depth filtering. Default is False.
            min_samp_filter (bool, optional): Flag indicating whether to apply minimum sample filtering. Default is False.
            save_dets (bool, optional): Flag indicating whether to save the detections. Default is False.
        """
        uniq_trees = self.slices_df["tree_id"].unique()
        for i, tree_id in enumerate(uniq_trees):
            print(f"starting tree: {i+1}, out of {len(uniq_trees)}, ({(i+1)/len(uniq_trees)*100:.2f}%)")
            tree_tracks, tracker_results, tree_slices = self.get_tree_slice_track(tree_id, depth_filter, min_samp_filter)
            uniq_frames = tree_tracks["frame"].unique()
            self.cap_fsi.set(cv2.CAP_PROP_POS_FRAMES, uniq_frames[0])
            tree_folder, alignment_save_folder, fsi_w_tracks_folder = self.get_tree_save_dir(min_samp_filter,
                                                                                             depth_filter, tree_id)
            for f_id in tqdm(uniq_frames):
                ret, frame_fsi_org = self.read_next_fsi()
                if not ret:
                    break
                zed, xyz, frame_fsi, resize_factors = self.get_f_id_images(f_id, frame_fsi_org)
                if self.filter_depth:
                    frame_fsi, zed, xyz = self.filter_depth_imgs(frame_fsi, zed, xyz)
                slices = tree_slices[tree_slices["frame_id"] == f_id][["start", "end"]].values[0]
                self.draw_translation(tracker_results, f_id, frame_fsi, zed, alignment_save_folder,
                                      resize_factors, slices)
                dets = self.tracks_df[self.tracks_df["frame"] == f_id].to_numpy()
                frame_fsi_w_dets = self.draw_dets(frame_fsi_org, dets, t_index=self.t_index)
                if save_dets:
                    cv2.imwrite(os.path.join(fsi_w_tracks_folder, f"frame_{f_id}.jpg"), frame_fsi_w_dets)

    def debug_trees(self):
        """
        Perform debugging for each tree.
        """
        methods_dict = dict(zip(["", "depth", "min_samp", "min_samp_depth"],
                            [[False, False, True], [True, False, False], [False, True, False], [True, True, False]]))
        for method in self.alignment_methods:
            df, msf, sdets = methods_dict[method]
            naming = ""
            if df:
                naming += "depth_filter"
            if msf:
                if len(naming) > 0:
                    naming += "_"
                naming += "min_samp_filter"
            if sdets:
                naming += "org_results"
            print("starting trees: ", naming)
            self.draw_on_tracked_imgaes(df, msf, sdets)

    def extract_cv_by_row(self):
        """
        calculates the number of unique tracks per row for each filtering type
        Returns:

        """
        tracks = {"no_filter": self.tracker_results, "depth_filter": self.tracker_d_f,
                  "min_samp_filter": self.tracker_ms_f, "min_samp_depth_filter": self.tracker_ms_d_f}
        for filter_type, tracker_esults in tracks.items():
            track_ids = list(chain(*[list(tracker_esults[key].keys()) for key in tracker_esults.keys()]))
            n_tracks = len(np.unique(track_ids))
            csv_name = f"n_track_ids_{filter_type}.csv"
            save_path = os.path.join(self.outputs_dir, csv_name)
            if self.scan_type == "multi_scans":
                pd.DataFrame({"row": [self.row_name], "scan_id": [self.scan_id],
                              "n_unique_track_ids": n_tracks}).to_csv(save_path)
            else:
                pd.DataFrame({"row": [self.row_name], "n_unique_track_ids": n_tracks}).to_csv(save_path)

    def extract_cv_by_tree(self):
        """
        calculates the number of unique tracks per tree for each filtering type
        Returns:

        """
        methods_dict = dict(zip(["_no_filter", "_depth_filter", "_min_samp_filter", "_min_samp_depth_filter"],
                            [[False, False], [True, False], [False, True], [True, True]]))
        if not len(self.slices_df["tree_id"].unique()):
            return None
        for method in methods_dict:
            depth_filter, min_samp_filter = methods_dict[method]
            n_tracks_tree, trees = [], []
            for i, tree_id in enumerate(self.slices_df["tree_id"].unique()):
                tree_tracks, _, _ = self.get_tree_slice_track(tree_id, depth_filter, min_samp_filter)
                tree_folder, _, _ = self.get_tree_save_dir(min_samp_filter, depth_filter, tree_id)
                n_tracks_tree.append(len(np.unique(tree_tracks["track_id"])))
                if self.scan_type == "multi_scans":
                    trees.append(f"{self.row_name}_S{self.scan_id}_T{tree_id}")
                else:
                    trees.append(f"{self.row_name}_T{tree_id}")
            save_path = os.path.join(self.outputs_dir, "trees", f"n_track_ids_per_tree{method}.csv")
            df = pd.DataFrame({"tree": trees, "n_unique_track_ids": n_tracks_tree})
            df.to_csv(save_path)

    def run(self):
        """
        run the debugger
        """
        self.extract_cv_by_row()
        self.extract_cv_by_tree()
        if any(method in self.methods for method in ["vid_with_dets", "alignment", "jai_frame_w_dets"]):
            print("starting debug by video")
            self.write_files()
        if "trees" in self.methods:
            print("starting debug by tree")
            self.debug_trees()
        self.cap_fsi.release()
        if self.zed_style == "svo":
            self.zed_cam.close()
        else:
            for cam in self.zed_cam:
                cam.close()


def agg_res_iner_loop(dfs_rows, dfs_trees, row_path, filter_type):
    if not os.path.isdir(row_path):
        return dfs_rows, dfs_trees
    row_res_path = os.path.join(row_path, f"n_track_ids_{filter_type}.csv")
    trees_res_path = os.path.join(row_path, "trees", f"n_track_ids_per_tree_{filter_type}.csv")
    if os.path.exists(row_res_path):
        dfs_rows.append(pd.read_csv(row_res_path, index_col=None))
    if os.path.exists(trees_res_path):
        dfs_trees.append(pd.read_csv(trees_res_path, index_col=None))
    return dfs_rows, dfs_trees


def agg_results(block_save_path, multi_scans=False):
    filters = ["no_filter", "depth_filter", "min_samp_filter", "min_samp_depth_filter"]
    for filter_type in filters:
        dfs_rows, dfs_trees, scans = [], [], []
        for row in os.listdir(block_save_path):
            row_path = os.path.join(block_save_path, row)
            if multi_scans:
                if not os.path.isdir(row_path):
                    continue
                for scan_id in os.listdir(row_path):
                    path_2_scan = os.path.join(row_path, scan_id)
                    dfs_rows, dfs_trees = agg_res_iner_loop(dfs_rows, dfs_trees, path_2_scan, filter_type)
            else:
                dfs_rows, dfs_trees = agg_res_iner_loop(dfs_rows, dfs_trees, row_path, filter_type)
        pd.concat(dfs_rows).to_csv(os.path.join(block_save_path, f"n_track_ids_{filter_type}.csv"))
        if len(dfs_trees):
            pd.concat(dfs_trees).to_csv(os.path.join(block_save_path, f"n_track_ids_per_tree_{filter_type}.csv"))


def debugger_runner_il(row_path, args):
    """
    This function is the interloop for multiprocessing. it runs the adt debugger on a given config
    Args:
        row_path (str): path to row for debugging
        args (NameSpace): config

    Returns:
        None
    """
    args.row_path = row_path
    debugger = ADTDebugger(args)
    debugger.run()


def get_rows_paths(block_path, args, multi_scans=False):
    if multi_scans:
        if block_path != "":
            row_paths = [os.path.join(block_path, row) for row in os.listdir(block_path)
                         if row in args.include_rows or "all" in args.include_rows]
            scans_path = []
            for row_path in row_paths:
                scans_path += [os.path.join(row_path, scan_id) for scan_id in os.listdir(row_path)
                         if scan_id in args.include_scans or "all" in args.include_scans]
            scans_path = [scan_id_path for scan_id_path in scans_path if os.path.isdir(scan_id_path)]
        else:
            scans_path = [args.row_path]
        return scans_path
    if block_path != "":
        row_paths = [os.path.join(block_path, row) for row in os.listdir(block_path)
                     if row in args.include_rows or "all" in args.include_rows]
        row_paths = [row for row in row_paths if os.path.isdir(row)]
    else:
        row_paths = [args.row_path]
    return row_paths


def debugger_runner(args, multi_process=False, use_thread_pool=False, multi_scans=False):
    """
    Runs debugger in multiprocess mode/ queue mode
    Args:
        args (NameSpace): config
        multi (bool): flag for using multiprocess
        use_thread_pool (bool): flag for using threads, if False and 'multi' is True will use ProcessPool

    Returns:
        None
    """
    block_path = args.block_path
    row_paths = get_rows_paths(block_path, args, multi_scans)
    args_list = [args]*len(row_paths)
    if multi_process:
        if use_thread_pool:
            with ThreadPoolExecutor(max_workers=min(6, len(row_paths))) as pool:
                results = pool.map(debugger_runner_il, row_paths, args_list)
        else:
            with ProcessPoolExecutor(max_workers=min(6, len(row_paths))) as pool:
                results = pool.map(debugger_runner_il, row_paths, args_list)
    else:
        list(map(debugger_runner_il, row_paths, args_list))
    if block_path != "":
        block_name = os.path.basename(args.block_path)
        outputs_dir = os.path.join(args.outputs_dir, f"{block_name}{args.block_suffix}")
        agg_results(outputs_dir, multi_scans)

def cut_zed_in_jai(pictures_dict, cur_coords, rgb=True, image_input=False):
    """
    cut zed to the jai region
    :param pictures_dict: {"frame": {"fsi":fsi,"rgb":rgb,"zed":zed} for each frame}
    :param cur_coords: {"x1":((x1,y1),(x2,y2))}
    :param rgb: process zedrgb image
    :return: pictures_dict with zed and zed_rgb cut to the jai region
    """
    x1 = max(cur_coords["x1"][0], 0)
    y1 = max(cur_coords["y1"][0], 0)
    if image_input:
        x2 = min(cur_coords["x2"][0], pictures_dict.shape[1])
        y2 = min(cur_coords["y2"][0], pictures_dict.shape[0])
    else:
        x2 = min(cur_coords["x2"][0], pictures_dict["zed"].shape[1])
        y2 = min(cur_coords["y2"][0], pictures_dict["zed"].shape[0])
    # x1, x2 = 145, 1045
    # y1, y2 = 370, 1597
    if image_input:
        return pictures_dict[y1:y2, x1:x2, :]
    pictures_dict["zed"] = pictures_dict["zed"][y1:y2, x1:x2, :]
    if rgb:
        pictures_dict["zed_rgb"] = pictures_dict["zed_rgb"][y1:y2, x1:x2, :]
    return pictures_dict


if __name__ == "__main__":
    repo_dir = get_repo_dir()
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    args = OmegaConf.load(repo_dir + runtime_config)
    # s_t = time()
    # debugger_runner(args.adt_debugger)
    # print("mapping time: ", time() - s_t) #759 s
    validate_output_path(args.adt_debugger.outputs_dir)
    debugger_runner(args.adt_debugger, False, False, args.adt_debugger.scan_type == "multi_scans")  # 289
    # debugger_runner(args.adt_debugger, True, False) #289

