import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from vision.misc.help_func import load_json, get_repo_dir, validate_output_path
import numpy as np
#import cupy as cp
import pandas as pd
from scipy.stats import skew
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from tqdm import tqdm
from vision.feature_extractor.vegetation_indexes import *
from vision.feature_extractor.stat_tools import *
from vision.feature_extractor.image_processing import *
from vision.feature_extractor.boxing_tools import *
from vision.feature_extractor.tree_size_tools import *
from vision.tools.image_stitching import get_tx_mask
import time
import matplotlib.pyplot as plt
from vision.tools.image_stitching import plot_2_imgs
from vision.tools.video_wrapper import video_wrapper
from vision.feature_extractor.adt_result_loader import ADTSBatchLoader
from omegaconf import OmegaConf
from vision.visualization.drawer import get_color
from tqdm import tqdm


class DebuggerFE:
    """This class is a collection of debugging function for the FeatureExtractor class"""

    def __init__(self, debug_dict):
        self.debug_dict = debug_dict

    @staticmethod
    def debug_ndvi(rgb, fsi, ndvi_img, ndvi_binary, frame_number):
        rgb_debug = rgb.copy().astype(np.uint8)
        rgb_debug[(1 - ndvi_binary).astype(bool)] = 0
        cv2.imshow(frame_number, cv2.cvtColor(rgb.astype(np.uint8)[:, :, ::-1], cv2.COLOR_RGB2HLS))
        cv2.waitKey()
        cv2.imshow(frame_number, fsi.astype(np.uint8)[:, :, ::-1])
        cv2.waitKey()
        cv2.imshow(frame_number, ((ndvi_img + 1) * 255 / 2).astype(np.uint8))
        cv2.waitKey()
        cv2.destroyAllWindows()

    def translation_debugging(self, minimal_frames, masks, frame_number, tree_name, tree_images):
        """
        This function is used for debugging the translation of masks between frames.
         It plots the image of two consecutive frames, one with the mask applied and one without the mask.
        It also saves the images to a specified file path.

        :param minimal_frames: List of ints representing the minimal frames that were selected for the tree
        :param masks: List of numpy arrays representing the masks that were generated for each frame
        :param frame_number: str representing the frame number that is currently being processed
        :param tree_name: str representing the name of the tree that is currently being processed
        :param tree_images: Dict of ints to Dicts of strings to numpy arrays representing the tree images and their
         corresponding data
        """
        iter_on = range(1, len(minimal_frames))
        if self.debug_dict["translation"]["one_frame"]:
            iter_on = [len(minimal_frames) // 2]
        for i in iter_on:
            fsi_next = tree_images[minimal_frames[i]]["fsi"].copy()
            fsi_next[masks[i - 1].astype(bool)] = 0
            save_to = self.debug_dict["translation"]["folder_path"]
            save_only = self.debug_dict["translation"]["save_only"]
            last_img = tree_images[minimal_frames[i - 1]]["fsi"]
            cur_img = tree_images[minimal_frames[i]]["fsi"]
            plot_2_imgs(last_img, cur_img, title=frame_number,
                        save_to=os.path.join(save_to, f"{tree_name}_{i}_clean.png"),
                        save_only=save_only)
            plot_2_imgs(last_img, fsi_next, title=frame_number,
                        save_to=os.path.join(save_to, f"{tree_name}_{i}_masked.png"),
                        save_only=save_only)

    def preprocess_debug(self, keys, tree_images, tracker_results, slicer_results, max_y, prefix=""):
        """
        debugging function for preprocessing
        :param keys: the frames to iterate on
        :param tree_images: tree images dict
        :param tracker_results: tracker results dict
        :param slicer_results: slicer results dict
        :param max_y: image height
        :param prefix: prefix for naming
        :return:
        """
        # for frame in [keys[0], keys[len(keys)//2], keys[-1]]:
        for frame in keys:
            frame = str(frame)
            fsi = tree_images[frame]["fsi"].copy()
            zed = tree_images[frame]["zed_rgb"].copy()
            xyz = tree_images[frame]["zed"].copy()
            zed[np.isnan(xyz[:, :, 2])] = (0, 0, 255)

            for id_, track in tracker_results[frame].items():
                fsi = cv2.rectangle(fsi, (track[0][0], track[0][1]), (track[1][0], track[1][1]), color=(255, 0, 0),
                                    thickness=2)
                zed = cv2.rectangle(zed, (track[0][0], track[0][1]), (track[1][0], track[1][1]), color=(255, 0, 0),
                                    thickness=2)
            fsi = cv2.line(fsi, (slicer_results[frame][0], 0), (slicer_results[frame][0], max_y), color=(255, 0, 0),
                           thickness=2)
            fsi = cv2.line(fsi, (slicer_results[frame][1], 0), (slicer_results[frame][1], max_y), color=(255, 0, 0),
                           thickness=2)
            zed = cv2.line(zed, (slicer_results[frame][0], 0), (slicer_results[frame][0], max_y), color=(255, 0, 0),
                           thickness=2)
            zed = cv2.line(zed, (slicer_results[frame][1], 0), (slicer_results[frame][1], max_y), color=(255, 0, 0),
                           thickness=2)
            saving_folder = os.path.join(self.debug_dict["preprocess"]["preprocess_saving_folder"], prefix)
            if not os.path.exists(saving_folder):
                os.mkdir(saving_folder)
            save_to = os.path.join(saving_folder, f"{frame}_debug_fe.jpg")
            plot_2_imgs(zed, fsi, frame, save_to=save_to, quick_save=True)
    @staticmethod
    def plot_3d_cloud(fruit_3d_space, centers, c=None, title=""):
        """
        Plot the 3D point cloud of the fruit space
        :param fruit_3d_space: Dictionary containing the 3D coordinates of each fruit
        :param centers: array of the coordinates of each fruit
        :param c: color of the points in the point cloud
        :return: None
        """
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        ax.scatter3D(-centers[:, 2], -centers[:, 0], -centers[:, 1], c=c)
        for i, label in enumerate(fruit_3d_space.keys()):  # plot each point + it's index as text above
            ax.text(-centers[i, 2], -centers[i, 0], -centers[i, 1], '%s' % (str(label)), size=10, zorder=1,
                    color='k')
        ax.set_xlabel('z')
        ax.set_ylabel('x')
        ax.set_zlabel('y')
        ax.view_init(20, 20)
        ax.set_title(title)
        plt.show()

    def draw_alignment(self, tracker_results, frame_number, frame_fsi, zed, tree_folder, slices):
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
        frame_fsi = frame_fsi.copy()[:,:,::-1].astype(np.uint8)
        zed = zed.copy()[:,:,::-1].astype(np.uint8)
        if frame_number in tracker_results.keys():
            for track_id, track in tracker_results[frame_number].items():
                color_id = int(track_id) % 15  # 15 is the number of colors in list
                color = get_color(color_id)
                frame_fsi = cv2.rectangle(frame_fsi, (track[0][0], track[0][1]), (track[1][0], track[1][1]),
                                          color=color, thickness=2)
                zed = cv2.rectangle(zed, (track[0][0], track[0][1]), (track[1][0], track[1][1]), color=color,
                                    thickness=2)
        slice_start, slice_end = slices
        max_y = frame_fsi.shape[0]
        frame_fsi = cv2.line(frame_fsi, (slice_start, 0), (slice_start, max_y), color=(255, 0, 0), thickness=2)
        frame_fsi = cv2.line(frame_fsi, (slice_end, 0), (slice_end, max_y), color=(255, 0, 0), thickness=2)
        zed = cv2.line(zed, (slice_start, 0), (slice_start, max_y), color=(255, 0, 0), thickness=2)
        zed = cv2.line(zed, (slice_end, 0), (slice_end, max_y), color=(255, 0, 0), thickness=2)
        tree_align_folder = os.path.join(self.debug_dict["alignment"]["save_to"], tree_folder)
        validate_output_path(tree_align_folder)
        save_to = os.path.join(tree_align_folder, f"{frame_number}.jpg")
        plot_2_imgs(zed[:, :, ::-1], frame_fsi[:, :, ::-1], frame_number, save_to=save_to, quick_save=True)

def init_cams(args):
    """
    initiates all cameras based on arguments file
    :param args: arguments file
    :return: zed_cam, rgb_jai_cam, jai_cam
    """
    zed_cam = video_wrapper(args.zed.movie_path, args.zed.rotate, args.zed.depth_minimum,
                            args.zed.depth_maximum)
    rgb_jai_cam = video_wrapper(args.rgb_jai.movie_path, args.rgb_jai.rotate)
    jai_cam = video_wrapper(args.jai.movie_path, args.jai.rotate)
    return zed_cam, rgb_jai_cam, jai_cam


class FeatureExtractor:
    """This class extracts features from zed and jai videos"""

    def __init__(self, args, tree_id: int = -1, row: str = "", block: str = "",
                 tree_images: dict = {}, slicer_results: dict = {}, tracker_results: dict = {}):
        self.tree_id, self.tree_name, self.block = tree_id, f"{row}_T{tree_id}", block  # for debugging
        self.max_x_pix, self.max_y_pix = args["max_x_pix"], args["max_y_pix"]  # for resize
        self.max_z = args["max_z"]  # for filtering
        self.min_number_of_tracks = args["min_number_of_tracks"]  # for filtering
        self.tree_images, self.slicer_results, self.tracker_results = tree_images, slicer_results, tracker_results
        self.minimal_frames_params = args["minimal_frames_params"]  # precent of frame to start or end
        self.remove_high_blues = args["remove_high_blues"]
        self.red_green_filter = args["red_green_filter"]
        self.plot_localiztion_cloud = args["plot_localiztion_cloud"]
        # self.frame_numbers = list(self.tree_images.keys())
        # self.minimal_frames, self.n_min_frames = self.get_minimal_frames()
        self.min_y, self.max_y = 0, 0
        self.masks = None
        self.no_homography = False
        self.dets_only = args["dets_only"]
        # self.get_masks(args["dets_only"])
        # TODO make it robust for translation failures: skip the bad frame and edit the relevant dicts
        self.debugger = DebuggerFE(args["debug"])
        self.tree_physical_features = self.init_physical_parmas(np.nan)
        self.tree_fruit_params = self.init_fruit_params()
        self.vol_style = args["vol_style"]
        self.filter_nans = args["filter_nans"]
        self.cv_only = args["cv_only"]
        self.direction = args["direction"]
        self.last_frame = None
        self.ndvis_binary_um, self.ndvis_binary, self.binary_box_imgs = [], [], []
        self.b_nir, self.b_swir_975, self.false_masks, = [], [], []
        self.b_fsi, self.b_zed, self.b_jai_rgb, self.b_rgb_zed, self.b_frame_numbers = [], [], [], [], []
        self.b_jai_translation, self.b_slicer, self.b_align, self.b_tracker_results = [], [], [], {}
        self.tree_physical_params = self.init_physical_parmas([])
        self.vegetation_indexes_keys = args["vegetation_indexes"]
        self.tree_vi = {**self.get_additional_vegetation_indexes(0, 0, 0, fill=[],
                                                                 vegetation_indexes_keys=self.vegetation_indexes_keys)}
        self.cv_res = {}
        self.tree_start, self.tree_end = False, False
        self.tracker_format = args["tracker_format"]
        self.min_dets_3d_init = args["min_dets_3d_init"]
        self.fruit_3d_space = {}
        self.global_shift = np.array([0, 0, 0])
        self.max_depth_change = args["max_depth_change"]
        self.physical_features_region = [eval(arg) for arg in args["physical_features_region"]]
        self.verbosity = args["verbosity"]
        self.skip_counter = 0
        self.acc_tx = 0

    @staticmethod
    def fill_dict(features_list: list, value: object) -> dict:
        """
        Creates a dictionary with the specified keys from features_list and the specified value.

        Args:
            features_list (list): A list of keys for the dictionary.
            value (object): The value to assign to each key.

        Returns:
            dict: A dictionary with the specified keys and values.
        """
        features_dict = {}
        if isinstance(value, Iterable):
            for key in features_list:
                features_dict[key] = value.copy()
            return features_dict
        for key in features_list:
            features_dict[key] = value
        return features_dict

    @staticmethod
    def update_features_dict(features_dict: dict, new_values: dict,
                             keep_dict: bool = False, replace: bool = False) -> dict:
        """
        Updates a dictionary with new values. The new values can be a list or an array, and will be appended to the
        existing values for the corresponding key in the dictionary.

        Args:
            features_dict (dict): The dictionary to update.
            new_values (dict): A dictionary with new values to add to the existing dictionary.
            keep_dict (bool): If True, the new values will be added as a separate iterable for each key. If False, the
                new values will be appended to the existing values for the corresponding key in the dictionary.
                 Default is False.
            replace (bool): If True, the existing values for each key will be replaced with the new values.
             Default is False.

        Returns:
            dict: The updated dictionary.
        """
        for key in new_values:
            values = new_values[key]
            if replace:
                features_dict[key] = values
                continue
            if isinstance(values, list) and not keep_dict:
                features_dict[key] += values
            elif isinstance(features_dict[key], np.ndarray):
                features_dict[key] = np.append(features_dict[key], values)
            else:
                features_dict[key].append(values)
        return features_dict

    @staticmethod
    def tracker_df_2_dict(tracker_results_frame: list) -> dict:
        """Convert the tracker adt results list to a dictionary.

        Args:
            tracker_results_frame(list): The tracker results list for a specific frame.

        Returns:
            A dictionary containing the track IDs as keys and the corresponding bounding box points as values.

        """
        tracker_results_frame = np.array(tracker_results_frame)
        point_1 = tuple(zip(tracker_results_frame[:, 0], tracker_results_frame[:, 1]))
        point_2 = tuple(zip(tracker_results_frame[:, 2], tracker_results_frame[:, 3]))
        points = tuple(zip(point_1, point_2))
        return dict(zip(tracker_results_frame[:, 6], points))

    @staticmethod
    def init_physical_parmas(value: object) -> dict:
        """
        Initializes a dictionary of physical parameters with the specified value for each key.

        Args:
            value (object): The value to assign to each key.

        Returns:
            dict: A dictionary of physical parameters with the specified value for each key.
        """
        features_list = ["total_foliage", "total_orange", "width", "height", "volume", "surface_area", "perimeter",
                         "avg_width", "avg_height", "avg_volume", "avg_perimeter", "foliage_fullness", "cont",
                         "ndvi_bin"]
        return FeatureExtractor.fill_dict(features_list, value)

    @staticmethod
    def init_tree_loc_params():
        features_list = ["mst_sums_arr", "mst_mean_arr", "mst_skew_arr",
         "n_clust_mean_arr", "n_clust_med_arr", "clusters_area_mean_arr", "clusters_area_med_arr",
         "clusters_ch_area_mean_arr", "clusters_ch_area_med_arr",
         "n_clust_arr_2", "n_clust_arr_4", "n_clust_arr_6"]
        return FeatureExtractor.fill_dict(features_list, 0)

    @staticmethod
    def init_fruit_params():
        """
        :param value: values to initialize features with
        :return: an initialized features dictionary
        """
        features_list = ["frame", "w_h_ratio", "q1", "q3", "avg_intens_arr", "med_intens_arr"]
        tree_fruit_params = FeatureExtractor.fill_dict(features_list, [])
        tree_fruit_params["frame"] = 0
        tree_fruit_params["w_h_ratio"], tree_fruit_params["intensity"], tree_fruit_params[
            "fruit_foliage_ratio"] = {}, {}, {}
        return tree_fruit_params

    @staticmethod
    def num_unique(np_arr):
        return len(np.unique(np_arr)) - 1

    @staticmethod
    def num_groups_db_scan(eps, centers, min_samples=2):
        return FeatureExtractor.num_unique(DBSCAN(eps=eps, min_samples=min_samples).fit(centers).labels_)

    @staticmethod
    def get_n_clusters_metrics(centers, distances, avg_diam, min_samples=2):
        clustring_dict = {f"n_clust_arr_{i}": FeatureExtractor.num_groups_db_scan(avg_diam * i, centers, min_samples)
                          for i in range(2, 8, 2)}
        clustering_mean = DBSCAN(eps=np.nanmean(distances), min_samples=min_samples).fit(centers)
        clustering_med = DBSCAN(eps=np.nanmedian(distances), min_samples=min_samples).fit(centers)
        clustering_mean_labels = clustering_mean.labels_
        clustering_med_labels = clustering_med.labels_
        clustring_dict["n_clust_mean_arr"] = FeatureExtractor.num_unique(clustering_mean_labels)
        clustring_dict["n_clust_med_arr"] = FeatureExtractor.num_unique(clustering_med_labels)
        return clustring_dict, clustering_mean_labels, clustering_med_labels

    @staticmethod
    def get_clustring_results(centers, avg_diam):
        problem_w_center = False
        if len(centers) > 0 and len(centers.shape) > 1:
            centers = centers[np.isfinite(centers[:, 2])]
        else:
            problem_w_center = True
        if avg_diam > 0 and len(centers) > 2 and not problem_w_center:
            mst, distances = compute_density_mst(centers)
            clustring_dict, c_mean_labels, c_med_labels = FeatureExtractor.get_n_clusters_metrics(centers, distances,
                                                                                                  avg_diam)
        else:
            distances, clustring_dict, c_mean_labels, c_med_labels = [], [], [], []
            problem_w_center = True
        return centers, distances, clustring_dict, c_mean_labels, c_med_labels, problem_w_center

    @staticmethod
    def apply_depth_filter(zed, images, max_z):
        depth_mask = np.abs(zed[:, :, 2]) > max_z
        for image in images:
            image[depth_mask] = 0
        zed[depth_mask] = np.nan
        return zed, images

    @staticmethod
    def cut_single_image(images: list, align_res: list):
        return get_zed_in_jai(images, align_res)

    @staticmethod
    def elipse_convexhull_area(centers, labels):
        if len(labels) == 0:
            labels = [0] * len(centers)
        ellipse_area = 0
        convex_hull_area = 0
        for label in np.unique(labels)[1:]:
            cnt = np.unique(centers[labels == label], axis=0)
            if len(cnt) < 4:
                continue
            std_cnt = np.std(cnt.round(1), axis=0) < 1e-10
            if sum(std_cnt) >= 1:
                cnt[:, std_cnt] += np.random.randn(cnt.shape[0]).reshape(-1, 1)/100 # add random noise if a fixed axis exsits
            cnvx_hull = ConvexHull(cnt)
            convex_hull_area += cnvx_hull.volume
            a = cnt ** 2
            o = np.ones(len(cnt))
            b, resids, rank, s = np.linalg.lstsq(a, o, rcond=None)
            ellipse_area += np.product(np.sqrt(np.abs(1 / b))) * pi
        return ellipse_area, convex_hull_area

    @staticmethod
    def get_fruit_distribution_on_tree(centers, min_y, max_y, num_breaks=3):
        """
        Calculate the distribution of fruits on the tree according to the real y value
        :param centers: array of shape (n, 3) containing the x, y, z coordinates of each fruit
        :param min_y: minimum y value of the tree
        :param max_y: maximum y value of the tree
        :param num_breaks: number of intervals/breaks in which to divide the tree along the y-axis
        :return: dictionary containing the percentage of fruits in each interval/break
        """
        breaks = np.linspace(min_y, max_y, num_breaks + 1)
        y_s = centers[:, 1]
        return {f"q_{i + 1}_precent_fruits": np.mean(np.all([y_s >= breaks[i], y_s < breaks[i + 1]], axis=0))
                for i in range(num_breaks)}

    @staticmethod
    def get_additional_vegetation_indexes(rgb, nir, swir_975, fill=None, mask=None, vegetation_indexes_keys=[]):
        vi_functions = vegetation_functions()
        if not vegetation_indexes_keys:
            vegetation_indexes_keys = vi_functions.keys()
        if not isinstance(fill, type(None)):
            if isinstance(fill, list) or isinstance(fill, dict):
                return {**{key: fill.copy() for key in vegetation_indexes_keys},
                        **{f"{key}_skew": fill.copy() for key in vegetation_indexes_keys}}
            return {**{key: fill for key in vegetation_indexes_keys},
                    **{f"{key}_skew": fill for key in vegetation_indexes_keys}}
        if 0 in rgb.shape:
            return {**{key: np.array([np.nan]) for key in vegetation_indexes_keys},
                    **{f"{key}_skew": np.array([np.nan]) for key in vegetation_indexes_keys}}
        red, green, blue = cv2.split(rgb)
        # input_dict = {"nir": np.float32(nir), "red": np.float32(red), "green": np.float32(green),
        #               "blue": np.float32(blue), "swir_975": np.float32(swir_975)}
        input_dict = {"nir": cp.asarray(nir), "red": cp.asarray(red),
                      "green": cp.asarray(green),
                      "blue": cp.asarray(blue), "swir_975": cp.asarray(swir_975)}

        if not isinstance(mask, type(None)):
            mask = mask.astype(bool)
            input_dict = {key: item[mask] for key, item in input_dict.items()}

        #clean_arr = {key: np.array(vi_functions[key](**input_dict).flatten())
        #             for key in vegetation_indexes_keys}
        clean_arr = {}
        try:
            for key in vegetation_indexes_keys:
                clean_arr[key] = np.array(vi_functions[key](**input_dict).flatten())
                #print(key)
        except:
            a = 1
        return {**clean_arr}

    @staticmethod
    def clean_veg_input(flat_hist):
        if isinstance(flat_hist, list):
            if isinstance(flat_hist[0], np.ndarray):
                flat_hist = np.concatenate(flat_hist)
        if not isinstance(flat_hist, np.ndarray):
            flat_hist = np.array(flat_hist)
        flat_hist = flat_hist[~np.isnan(flat_hist)]
        quant_trimmed = quantile_trim(flat_hist, (0.025, 0.975), keep_size=False)
        return quant_trimmed

    def reformat_tracker(self):
        new_format = []
        if len(self.tracker_format) == 8:
            track_id, x1, y1, x2, y2, pc_x, pc_y, pc_z = self.tracker_format
            for track_res in self.b_tracker_results:
                new_format.append({int(row[track_id]): ((int(row[x1]), int(row[y1])), (int(row[x2]), int(row[y2])),
                                                        (row[pc_x], row[pc_y], row[pc_z]))
                                   for row in track_res})
        if len(self.tracker_format) == 10:
            track_id, x1, y1, x2, y2, pc_x, pc_y, pc_z, width, height = self.tracker_format
            for track_res in self.b_tracker_results:
                new_format.append({int(row[track_id]): ((int(row[x1]), int(row[y1])), (int(row[x2]), int(row[y2])),
                                                        (row[pc_x], row[pc_y], row[pc_z], row[width], row[height]))
                                   for row in track_res})
        else:
            track_id, x1, y1, x2, y2 = self.tracker_format
            for track_res in self.b_tracker_results:
                new_format.append({int(row[track_id]): ((int(row[x1]), int(row[y1])), (int(row[x2]), int(row[y2])))
                                   for row in track_res})
        self.b_tracker_results = new_format

    def validate_slice_ending(self):
        im_width = self.b_fsi[0].shape[1]
        for i in range(len(self.b_slicer)):
            im_width * self.minimal_frames_params[1] > self.b_slicer[i][1]
        data_lists = [self.b_fsi, self.b_zed, self.b_jai_rgb, self.b_rgb_zed, self.b_tracker_results,
             self.b_slicer, self.b_frame_numbers, self.b_jai_translation, self.b_align]
        for j, data_list in enumerate([self.b_fsi, self.b_zed, self.b_jai_rgb, self.b_rgb_zed, self.b_tracker_results,
                                       self.b_slicer, self.b_frame_numbers, self.b_jai_translation, self.b_align]):
            data_lists[j] = data_list[:i+1]
        (self.b_fsi, self.b_zed, self.b_jai_rgb, self.b_rgb_zed, self.b_tracker_results,
                self.b_slicer, self.b_frame_numbers, self.b_jai_translation, self.b_align) = data_lists

    def load_batch(self, b_fsi, b_zed, b_jai_rgb, b_rgb_zed, b_tracker_results, b_slicer, b_frame_numbers,
                   b_align, b_jai_translation):
        self.b_fsi, self.b_zed, self.b_jai_rgb = b_fsi, [zed[:, :, :3] for zed in b_zed], b_jai_rgb
        self.b_rgb_zed = [zed[:, :, ::-1] for zed in b_rgb_zed]
        self.b_tracker_results = b_tracker_results
        self.reformat_tracker()
        self.b_slicer = [[max(int(res[0]), 0) if res[0] > -1 else 0,
                          min(int(res[1]), b_fsi[0].shape[1]) if res[1] > -1 else b_fsi[0].shape[1]]
                         for res in b_slicer]
        self.b_frame_numbers = b_frame_numbers
        self.b_align = [[int(x) if i < 6 else x for i, x in enumerate(coords)] for coords in b_align]
        if not len(b_jai_translation):
            self.b_jai_translation = [[]*len(b_fsi)]
        else:
            self.b_jai_translation = b_jai_translation

    def resize_images(self, images):
        #TODO can become > 10X faster with cupy
        return [cv2.resize(image, (self.max_x_pix, self.max_y_pix)) for image in images]

    def resize_images_cuda(self, zed, zed_rgb, fsi, jai_rgb):

        # max_xs = [self.max_x_pix for _ in range(4)]
        # max_ys = [self.max_y_pix for _ in range(4)]
        # images = [zed, zed_rgb, fsi, jai_rgb]
        # with ThreadPoolExecutor(max_workers=4) as executor:
        #     res = list(executor.map(self.resize_cuda, images, max_xs, max_ys))

        # zed = res[0]
        # zed_rgb = res[1]
        # fsi = res[2]
        # jai_rgb = res[3]

        zed = self.resize_cuda(zed, self.max_x_pix, self.max_y_pix)
        zed_rgb = self.resize_cuda(zed_rgb, self.max_x_pix, self.max_y_pix)
        fsi = self.resize_cuda(fsi, self.max_x_pix, self.max_y_pix)
        jai_rgb = self.resize_cuda(jai_rgb, self.max_x_pix, self.max_y_pix)


        return zed, zed_rgb, fsi, jai_rgb

    @staticmethod
    def resize_cuda(image, x_size, y_size):
        stream = cv2.cuda_Stream()
        input_GPU = cv2.cuda_GpuMat()
        input_GPU.upload(image, stream)

        output_GPU = cv2.cuda_GpuMat(y_size, x_size, input_GPU.type())
        #output_GPU = cv2.cuda_GpuMat()

        # adjust zed scale to be the same as jai using calibrated scale x and y
        cv2.cuda.resize(input_GPU, (x_size, y_size), output_GPU, interpolation=cv2.INTER_CUBIC, stream=stream)
        stream.waitForCompletion()
        output = output_GPU.download()
        #image_GPU = cv2.cuda.resize(image_GPU,
        #                          (x_size, y_size),
        #                          stream=stream)

        return output
    def cut_jai_in_zed_batch(self, images):
        return list(map(self.cut_single_image, images, self.b_align))

    def align_and_scale(self, fsi, zed, jai_rgb, zed_rgb, tracker_results, slices, align_res, jai_translation):

        zed, zed_rgb = self.cut_single_image([zed, zed_rgb], align_res)

       # scale:
        jai_h, jai_w = fsi.shape[:2]
        r_h, r_w = self.max_y_pix / jai_h, self.max_x_pix / jai_w
        zed, zed_rgb, fsi, jai_rgb = self.resize_images([zed, zed_rgb, fsi, jai_rgb]) # most consuming

        if self.remove_high_blues:
            zed = remove_high_blues(zed, zed_rgb[:, :, 2])

        if self.red_green_filter:
            zed[zed_rgb[:, :, 0] > zed_rgb[:, :, 1] + 10] = np.nan

        if self.max_z > 0:
            zed, (zed_rgb, fsi, jai_rgb) = self.apply_depth_filter(zed, [zed_rgb, fsi, jai_rgb], self.max_z) # second_most

        slices = (int(slices[0] * r_w), int(slices[1] * r_w))
        if jai_translation:
            jai_translation = [jai_translation[0]*r_w, jai_translation[1]*r_h, jai_translation[2]]
        tracker_results = {t_id: resize_bbox(bbox, r_w, r_h) for t_id, bbox in tracker_results.items()}

        return fsi, zed, jai_rgb, zed_rgb, tracker_results, slices, jai_translation

    def scale_align_batch(self):
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                res = np.array(list(executor.map(self.align_and_scale, self.b_fsi, self.b_zed, self.b_jai_rgb,
                                    self.b_rgb_zed, self.b_tracker_results,
                                    self.b_slicer, self.b_align, self.b_jai_translation)), dtype=object)
            #res = np.array(list(map(self.align_and_scale, self.b_fsi, self.b_zed, self.b_jai_rgb,
                                    #self.b_rgb_zed, self.b_tracker_results,
                                    #self.b_slicer, self.b_align, self.b_jai_translation)), dtype=object)
            self.b_fsi, self.b_zed, self.b_jai_rgb, self.b_rgb_zed, self.b_tracker_results, self.b_slicer, \
                self.b_jai_translation = res[:, 0], res[:, 1], res[:, 2], res[:, 3], res[:, 4], res[:, 5], res[:, 6]
        except:
            print("bug")

    def accumulate_tracker_res(self):
        b_tracker_results = dict(zip(self.b_frame_numbers, self.b_tracker_results))
        slicer_res = dict(zip(self.b_frame_numbers, self.b_slicer))
        b_tracker_results = filter_outside_tree_boxes(b_tracker_results, slicer_res, self.direction)
        if self.max_z > 0:
            tree_images = {}
            if len(self.tracker_format) in [8, 10]:
                for frame_number, zed_image in zip(self.b_frame_numbers, self.b_zed):
                    tree_images[frame_number] = {"zed": zed_image, "nir": None, "swir_975": None}
                b_tracker_results = filter_outside_zed_boxes(b_tracker_results, tree_images, self.max_z,
                                                             self.filter_nans, True)
            else:
                for frame_number, zed_image in zip(self.b_frame_numbers, self.b_zed):
                    tree_images[frame_number] = {"zed": zed_image, "nir": None, "swir_975": None}
                b_tracker_results = filter_outside_zed_boxes(b_tracker_results, tree_images,
                                                             self.max_z, self.filter_nans)
        self.tracker_results = {**self.tracker_results, **b_tracker_results}

    def remove_starter_pics(self, b_fsi, b_zed, b_jai_rgb, b_rgb_zed, b_tracker_results, b_slicer,
                      b_frame_numbers, b_align, b_jai_translation):
        im_width = b_fsi[0].shape[1]
        found_start = False
        for i in range(len(b_slicer)):
            if im_width*self.minimal_frames_params[0] < b_slicer[i][0]:
                continue
            found_start = True
            break
        return found_start

    def update_tree_fruit_params(self, boxes, fsi, rgb):
        if not boxes:
            return
        w_h_ratios = get_w_h_ratio(boxes.values())
        w_h_keys = np.fromiter(self.tree_fruit_params["w_h_ratio"].keys(), int)
        for i, track_id in enumerate(boxes.keys()):
            cur_ratio = w_h_ratios[i]
            # 90% of the function time is  get intensity
            cur_intens, cur_foilage_ratio = get_intensity(fsi, rgb, boxes[track_id])
            if np.isin(track_id, w_h_keys):
                self.tree_fruit_params["w_h_ratio"][track_id] = np.append(self.tree_fruit_params["w_h_ratio"][track_id],
                                                                     cur_ratio)
                self.tree_fruit_params["intensity"][track_id] = np.append(self.tree_fruit_params["intensity"][track_id],
                                                                     cur_intens)
                self.tree_fruit_params["fruit_foliage_ratio"][track_id] = np.append(
                    self.tree_fruit_params["fruit_foliage_ratio"][track_id],
                    cur_foilage_ratio)
            else:
                self.tree_fruit_params["w_h_ratio"][track_id] = np.array([cur_ratio])
                self.tree_fruit_params["intensity"][track_id] = np.array([cur_intens])
                self.tree_fruit_params["fruit_foliage_ratio"][track_id] = np.array([cur_foilage_ratio])

    def calc_batch_fruit_features(self):
        for trakcer_res, fsi, rgb in zip(self.b_tracker_results, self.b_fsi, self.b_jai_rgb):
            self.update_tree_fruit_params(trakcer_res, fsi, rgb)
        self.tree_fruit_params["frame"] += len(self.b_frame_numbers)

    def tree_intensity_summary(self):
        self.tree_fruit_params["intensity"] = np.fromiter(
            (np.median(intensity) for intensity in self.tree_fruit_params["intensity"].values()), float)
        norm_intens = normalize_intensity(self.tree_fruit_params["intensity"], "")
        q1, med_intens_arr, q3 = np.nanquantile(norm_intens, [0.25, 0.5, 0.75])
        self.tree_fruit_params["q1"] = q1
        self.tree_fruit_params["q3"] = q3
        self.tree_fruit_params["med_intens_arr"] = med_intens_arr
        self.tree_fruit_params["avg_intens_arr"] = np.nanmean(norm_intens)
        self.tree_fruit_params.pop("intensity")

    def get_tree_fruit_features(self):
        n_samp_per_fruit = np.fromiter((len(w_h_ratios) for w_h_ratios in
                                        self.tree_fruit_params["w_h_ratio"].values()), float)
        wh_std_per_fruit = np.fromiter((np.std(w_h_ratios) for w_h_ratios in
                                        self.tree_fruit_params["w_h_ratio"].values()), float)
        self.tree_fruit_params["w_h_ratio"] = np.median(wh_std_per_fruit[n_samp_per_fruit > 2])
        self.tree_intensity_summary()
        fruit_foliage_ratio = np.fromiter((np.nanmean(fruit_foliage_ratio) for fruit_foliage_ratio in
                                           self.tree_fruit_params["fruit_foliage_ratio"].values()), float)
        self.tree_fruit_params["fruit_foliage_ratio"] = np.mean(fruit_foliage_ratio[np.isfinite(fruit_foliage_ratio)])
        return self.tree_fruit_params

    def init_3d_fruit_space(self):
        """
        initiates a 3D space for the fruits on a tree .
        :param tracker_results: {"frame": {"id": ((x0,y0),(x1,y1))} for each frame}
        :param tree_images: {"frame": {"fsi":fsi,"rgb":rgb,"zed":zed} for each frame}
        :return: 3D fruit space
        """
        n_tracks = {frame_id: len(tracks) for frame_id, tracks in self.tracker_results.items()}
        all_frames = list(n_tracks.keys())
        n_tracks_counts = np.array(list(n_tracks.values()))
        try:
            min_dets = max(np.quantile(n_tracks_counts, self.min_dets_3d_init), 5)
            if not np.any(np.where(n_tracks_counts > min_dets)):
                return {}, 0
        except:
            print("wierd bug")
        first_frame = all_frames[np.min(np.where(n_tracks_counts > min_dets))]
        first_tracker_results = self.tracker_results[first_frame]
        fruit_space = {t_id: (bbox[2][0], bbox[2][1], bbox[2][2]) for t_id, bbox in first_tracker_results.items()}
        fruits_keys = list(fruit_space.keys())
        for fruit in fruits_keys:
            if np.isnan(fruit_space[fruit][0]):
                fruit_space.pop(fruit)
        return fruit_space, first_frame

    def fruit_space_frame_iter(self, frame_number, last_frame_boxes, boxes_w, boxes_h):
        frame_tracker_results, new_boxes, old_boxes = self.tracker_results[frame_number], {}, {}
        space_keys = last_frame_boxes.keys()
        for track_id, bbox in frame_tracker_results.items():
            x_center, y_center, z_center, width, height = bbox[2]
            if track_id not in space_keys:
                if np.isnan(z_center):
                    continue
                new_boxes[track_id] = (x_center, y_center, z_center)
                boxes_w, boxes_h = np.append(boxes_w, width), np.append(boxes_h, height)
            else:
                old_boxes[track_id] = (x_center, y_center, z_center)
        return frame_tracker_results, new_boxes, old_boxes, boxes_w, boxes_h

    def fruit_space_frame_postprocess(self, frame_tracker_results, last_frame_boxes, old_boxes, new_boxes):
        self.fruit_3d_space, shift = project_boxes_to_fruit_space_global(self.fruit_3d_space, last_frame_boxes,
                                                                         old_boxes, new_boxes,
                                                                         self.max_depth_change, self.global_shift)
        self.global_shift = self.global_shift + shift
        last_frame_boxes = self.zip_track_id_xyz(frame_tracker_results)
        return last_frame_boxes

    @staticmethod
    def zip_track_id_xyz(frame_tracker_results):
        last_frame_boxes = dict(zip(frame_tracker_results.keys(),
                                         [(val[2][0], val[2][1], val[2][2]) for val in frame_tracker_results.values()]))
        return last_frame_boxes

    def create_3d_fruit_space(self):
        self.fruit_3d_space, first_frame = self.init_3d_fruit_space()
        if not self.fruit_3d_space:
            return 0, []
        boxes_w, boxes_h, last_frame_boxes = np.array([]), np.array([]), self.fruit_3d_space
        for i, frame_number in enumerate(self.tracker_results.keys()):
            if int(frame_number) <= int(first_frame):
                continue
            frame_tracker_results, new_boxes, old_boxes, boxes_w, boxes_h = self.fruit_space_frame_iter(frame_number,
                                                                                                        last_frame_boxes,
                                                                                                        boxes_w, boxes_h)
            if not len(old_boxes):
                print("no old boxes: ", frame_number)
                continue
            last_frame_boxes = self. fruit_space_frame_postprocess(frame_tracker_results, last_frame_boxes,
                                                                   old_boxes, new_boxes)
        keys_to_pop = []
        for key in self.fruit_3d_space.keys():
            if not np.all(np.isfinite(self.fruit_3d_space[key])):
                keys_to_pop.append(key)
        for key in keys_to_pop:
            self.fruit_3d_space.pop(key)

        centers = np.array(list(self.fruit_3d_space.values()))
        if self.debugger.debug_dict["3d_space"]:
            DebuggerFE.plot_3d_cloud(self.fruit_3d_space, centers, title=f"{self.block}_{self.tree_name}")
        med_diam = np.nanmedian(np.nanmax([boxes_w, boxes_h], axis=0))
        return med_diam, centers

    def calc_localization_features(self, min_y, max_y):
        med_diam, centers = self.create_3d_fruit_space()
        centers, dists, clustring_dict, mean_labels, med_labels, fail = self.get_clustring_results(centers, med_diam)
        if fail:
            return self.init_tree_loc_params()
        area_mean, convex_hull_area_mean = self.elipse_convexhull_area(centers, mean_labels)
        area_med, convex_hull_area_med = self.elipse_convexhull_area(centers, med_labels)
        tree_loc_params = {"mst_sums_arr": np.sum(dists), "mst_mean_arr": np.mean(dists), "mst_skew_arr": skew(dists),
                           "clusters_area_mean_arr": area_mean, "clusters_area_med_arr": area_med,
                           "clusters_ch_area_mean_arr": convex_hull_area_mean,
                           "clusters_ch_area_med_arr": convex_hull_area_med, **clustring_dict,
                           **self.get_fruit_distribution_on_tree(centers, min_y, max_y),
                           "fruit_dist_center": (np.nanmean(centers[:, 1]) - min_y) / max((max_y - min_y), 1E-6)}
        return tree_loc_params

    def verbosity_print(self, print_val):
        if self.verbosity == 1:
            if isinstance(print_val, tuple):
                print(*print_val)
            else:
                print(print_val)

    def process_batch(self, b_fsi, b_zed, b_jai_rgb, b_rgb_zed, b_tracker_results, b_slicer,
                      b_frame_numbers, b_align, b_jai_translation):
        s_t_b = time.time()
        adts_res = (b_fsi, b_zed, b_jai_rgb, b_rgb_zed, b_tracker_results, b_slicer, b_frame_numbers,
                    b_align, b_jai_translation)
        if not len(b_fsi):
            return
        if not self.tree_start:
            found_start = self.remove_starter_pics(*adts_res)
            if found_start:
                self.tree_start = True
        time_data = {}
        s_t = time.time()
        self.load_batch(*adts_res)
        self.validate_slice_ending()
        if not len(self.b_fsi):
            return
        time_data["batch_loading"] = time.time() - s_t
        s_t = time.time()
        self.scale_align_batch() # time consumption (0.16)
        time_data["scale_align_batch"] = time.time() - s_t
        s_t = time.time()
        self.accumulate_tracker_res() # time consumption (0.01)
        time_data["accumulate_tracker_res"] = time.time() - s_t
        if not self.tree_start:
            return
        s_t = time.time()
        self.calc_batch_fruit_features()
        time_data["calc_batch_fruit_features"] = time.time() - s_t
        s_t = time.time()
        # TODO add frame skipping, start/end frame by slicing location
        self.preprocess_batch_images() # time consumption (1.8)
        time_data["preprocess_batch_images"] = time.time() - s_t
        print(time_data["preprocess_batch_images"])
        # self.save_imgs()
        s_t = time.time()
        self.calc_physical_features_batch() # time consumption (0.5)
        time_data["calc_physical_features_batch"] = time.time() - s_t
        s_t = time.time()
        self.calc_vi_features_batch() # time consumption (0.5)
        time_data["calc_vi_features_batch"] = time.time() - s_t
        if self.debugger.debug_dict["alignment"]["apply"]:
            list(map(self.debugger.draw_alignment, [self.tracker_results]*len(self.b_zed), self.b_frame_numbers,
                     self.b_fsi, self.b_rgb_zed, [f"{self.block}_{self.tree_name}"]*len(self.b_zed), self.b_slicer))
        time_data["processed batch: "] = time.time() - s_t_b

        return time_data

    def save_imgs(self):
        for ndvi_binary,binary_box, false_mask, fsi, zed, rgb_zed, frame_number in zip(self.ndvis_binary,
                                                                                       self.binary_box_imgs,
                                                                                       self.false_masks, self.b_fsi,
                                                                                       self.b_zed, self.b_rgb_zed,
                                                                                       self.b_frame_numbers):
            m_path = "/media/fruitspec-lab/easystore/debugging/feature_extractor_class_comparison/class"
            cv2.imwrite(os.path.join(m_path, "ndvi_binary", f"frame_{frame_number}.png"), ndvi_binary.astype(np.uint8)*255)
            cv2.imwrite(os.path.join(m_path, "binary_box", f"frame_{frame_number}.png"), binary_box.astype(np.uint8)*255)
            cv2.imwrite(os.path.join(m_path, "false_mask", f"frame_{frame_number}.png"), false_mask.astype(np.uint8)*255)
            cv2.imwrite(os.path.join(m_path, "fsi", f"frame_{frame_number}.png"), fsi[:, :, ::-1])
            cv2.imwrite(os.path.join(m_path, "rgb_zed", f"frame_{frame_number}.png"), rgb_zed[:, :, ::-1])

    def get_false_mask(self, fsi, frame_number, tracker_result=None, translation_results=[]):
        if isinstance(self.last_frame, type(None)):
            self.last_frame = make_bbox_pic(fsi, tracker_result) if self.dets_only else fsi
            return np.zeros_like(fsi[:, :, 0], dtype=bool), True
        fsi = make_bbox_pic(fsi.copy(), tracker_result) if self.dets_only else fsi.copy()
        try:
            if len(translation_results):
                height, width = fsi.shape[:2]
                mask = [get_tx_mask(int(translation_results[0]), width, height)]
            else:
                mask = get_frames_overlap(file_list=[self.last_frame, fsi], method='at')
            self.last_frame = fsi
            return mask[0].astype(bool), True
        except:
            print("no homography for frame: ", frame_number)
            return None, False

    def remove_bad_masks_batch(self, valids):
        res = []
        for b_res in [self.b_fsi, self.b_zed, self.b_jai_rgb, self.b_rgb_zed,
                      self.b_tracker_results, self.b_frame_numbers, self.false_masks]:
            res.append([x for x, valid in zip(b_res, valids) if valid])
        self.b_fsi, self.b_zed, self.b_jai_rgb, self.b_rgb_zed, \
            self.b_tracker_results, self.b_frame_numbers, self.false_masks = res

    def preprocess_batch_images(self):
        nir_swir_res = np.array(list(map(get_nir_swir, self.b_fsi)))
        self.b_nir, self.b_swir_975 = nir_swir_res[:, 0], nir_swir_res[:, 1]
        self.b_jai_rgb = [img.astype(float) for img in self.b_jai_rgb]

        with ThreadPoolExecutor(max_workers=4) as executor:
            mask_map_res = list(executor.map(self.get_false_mask,
                                             self.b_fsi,
                                             self.b_frame_numbers,
                                             self.b_tracker_results,
                                             self.b_jai_translation))

        #mask_map_res = list(map(self.get_false_mask, self.b_fsi, self.b_frame_numbers, self.b_tracker_results,
        #                        self.b_jai_translation)) # 60%

        self.false_masks, valids = [x[0] for x in mask_map_res], [x[1] for x in mask_map_res]
        self.remove_bad_masks_batch(valids) # TODO stich the  valids with relevant masks
        self.get_ndvis_batch() # 40%

        self.update_min_max_y()

    def update_min_max_y(self):
        if self.min_y or self.max_y:
            return
        for i, slice in enumerate(self.b_slicer):
            if (slice[0] == 0 and slice[1] >= self.max_x_pix-1) or (slice[0] > 0 and slice[1] < self.max_x_pix):
                tmp_img = self.b_zed[i][:, :, 1].copy()
                tmp_img[1-np.nan_to_num(self.ndvis_binary_um[i], 1).astype(bool)] = np.nan
                self.min_y, self.max_y = np.nanquantile(tmp_img, (0.1, 0.9))
                return

    def get_tree_results(self):
        self.get_cv()
        if self.cv_only:
            if isinstance(self.min_number_of_tracks, Iterable):
                return {f"cv{i}": self.cv_res[f"cv{i}"] for i in self.min_number_of_tracks}
            return self.cv_res["cv"]
        physical_features = self.transform_to_tree_physical_features(self.tree_physical_params)
        tree_fruit_features = self.get_tree_fruit_features()
        localization_features = self.calc_localization_features(self.min_y, self.max_y)
        vi_features = self.transform_to_vi_features()
        return physical_features, tree_fruit_features, localization_features, vi_features, self.cv_res


    def get_ndvis(self, frame_number: int):
        """
        Computes NDVI (Normalized Difference Vegetation Index) images for a given frame number.

        Args:
            frame_number (int): The frame number for which to compute the NDVI images.

        Returns:
            None.
        """
        fsi, rgb, nir, swir_975, xyz_point_cloud, zed_rgb = get_pictures(self.tree_images, frame_number,
                                                                         with_zed=True)
        boxes = self.tracker_results[frame_number]
        ndvi_img, ndvi_binary, binary_box_img = get_ndvi_pictures(rgb, nir, fsi, boxes)
        false_mask = self.masks[frame_number]
        ndvi_binary[false_mask] = np.nan
        binary_box_img[false_mask] = np.nan
        self.tree_images[frame_number]["ndvi_img"] = ndvi_img
        self.tree_images[frame_number]["ndvi_binary"] = ndvi_binary
        self.tree_images[frame_number]["binary_box_img"] = binary_box_img
        if self.debugger.debug_dict["ndvi_images"]:
            self.debugger.debug_ndvi(rgb, fsi, ndvi_img, ndvi_binary, frame_number)

    def get_ndvi_single(self, fsi, jai_rgb, nir, tracker_result, false_mask, frame_number):
        ndvi_img, ndvi_binary, binary_box_img = get_ndvi_pictures(jai_rgb, nir, fsi, tracker_result)
        ndvi_binary_un_msaked = ndvi_binary.copy()
        ndvi_binary[false_mask] = np.nan
        binary_box_img[false_mask] = np.nan
        if self.debugger.debug_dict["ndvi_images"]:
            self.debugger.debug_ndvi(jai_rgb, fsi, ndvi_img, ndvi_binary, frame_number)
        return ndvi_binary, binary_box_img, ndvi_binary_un_msaked

    def get_ndvis_batch(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            ndvis = np.array(list(executor.map(self.get_ndvi_single,
                                               self.b_fsi,
                                               self.b_jai_rgb,
                                               self.b_nir,
                                               self.b_tracker_results,self.false_masks,
                                               self.b_frame_numbers)))

        #ndvis = np.array(list(map(self.get_ndvi_single, self.b_fsi, self.b_jai_rgb, self.b_nir, self.b_tracker_results,
        #                          self.false_masks, self.b_frame_numbers)))
        self.ndvis_binary, self.binary_box_imgs, self.ndvis_binary_um = ndvis[:, 0], ndvis[:, 1], ndvis[:, 2]

    def replace_bad_masks(self):
        """
        Replace missing masks in 'self.masks' with a new mask obtained from overlapping two frames.
        Will pop from 'self.minimal_frames' any replaced mask

        Returns:
        None

        """
        good_masks = [not isinstance(mask, type(None)) for mask in self.masks.values()]
        bad_masks_keys = [key for key, mask in self.masks.items() if isinstance(mask, type(None))]
        last_img_ind, replace, keys_list = 0, False, list(self.masks.keys())
        for cur_image_ind in range(1, len(good_masks)):
            if good_masks[cur_image_ind]:
                if not replace:
                    last_img_ind = cur_image_ind
                    continue
                if replace:
                    cur_key = keys_list[cur_image_ind]
                    last_key = keys_list[last_img_ind]
                    new_mask = get_frames_overlap(file_list=[self.tree_images[last_key]["fsi"],
                                                             self.tree_images[cur_key]["fsi"]], method='at')
                    if isinstance(new_mask, type(None)):
                        bad_masks_keys.append(cur_key)
                        continue
                    self.masks[cur_key] = new_mask
                    replace = False
            else:
                replace = True
        for key in bad_masks_keys:
            self.masks.pop(key)
        self.minimal_frames = [key for key in self.minimal_frames if key not in bad_masks_keys]

    def get_cv(self):
        """
        Calculates the total count of fruits (cv feature) across all frames that have more than
         the specified minimum number of tracks.

        Returns:
            None
        """
        uniq, counts = np.unique(
            [id for frame in set(self.tracker_results.keys()) - {"cv"} for id in self.tracker_results[frame].keys()],
            return_counts=True)
        if isinstance(self.min_number_of_tracks, Iterable):
            self.cv_res["cv"] = len(uniq)
            for i in self.min_number_of_tracks:
                self.cv_res[f"cv{i}"] = len(uniq[counts >= i])
        else:
            self.cv_res["cv"] = len(uniq[counts >= self.min_number_of_tracks])

    def calc_frame_physical_parmas(self, xyz_point_cloud: np.ndarray, binary_box_img: np.ndarray,
                                   ndvi_binary: np.ndarray):
        """
        Calculate physical parameters of a frame using its XYZ point cloud and image ndvi mask.

        Args:
            xyz_point_cloud (ndarray): 3D point cloud of a frame (height x width x 3)
            binary_box_img (ndarray): binary image of bboxes on the tree (height x width)
            ndvi_binary (ndarray): binary image of NDVI values, indicating where the foliage is (height x width)

        Returns:
            dict: a dictionary containing physical parameters of the frame
                - "total_orange" (float): total orange area within part of tree in the frame (in square meters)
                - "total_foliage" (float): total foliage area within part of tree in the frame (in square meters)
                - "width" (ndarray): width of the part of tree in the frame (in meters)
                - "height" (ndarray): height of the part of tree in the frame (in meters)
                - "perimeter" (ndarray): perimeter of the part of tree in the frame (in meters)
                - "surface_area" (float): total surface area of the part of tree in the frame (in square meters)
                - "ndvi_bin" (int): total number of binary NDVI pixels
                - "cont" (int): total number of binary contur pixels
        """
        frame_physical_parmas = self.init_physical_parmas([])
        im_shape = binary_box_img.shape
        if not np.prod(im_shape):
            return None
        pixel_size_x, pixel_size_y, pixel_area = get_real_world_dims_with_correction(xyz_point_cloud[:, :, 2])
        total_orange = np.nansum(binary_box_img * pixel_area)
        total_foliage = np.nansum(ndvi_binary * pixel_area)
        top, buttom = self.physical_features_region[0], self.physical_features_region[1]
        top = int(im_shape[0]*top)
        buttom = int(im_shape[0] - im_shape[0]*buttom)
        frame_physical_parmas["total_orange"] = total_orange  # scalar
        frame_physical_parmas["total_foliage"] = total_foliage  # scalar
        frame_physical_parmas["width"] = calc_tree_widths(xyz_point_cloud[top:buttom],
                                                          ndvi_binary[top:buttom])  # np array (scaler for each row)
        frame_physical_parmas["height"] = calc_tree_heights(xyz_point_cloud,
                                                            ndvi_binary)  # np array (scaler for each col)
        frame_physical_parmas["perimeter"] = calc_tree_perimeter(xyz_point_cloud[top:buttom],
                                                                 ndvi_binary[top:buttom])  # np array (scaler for each row)
        frame_physical_parmas["surface_area"] = get_surface_area(xyz_point_cloud, ndvi_binary)  # scalar
        frame_physical_parmas["ndvi_bin"], frame_physical_parmas["cont"] = get_foliage_fullness(ndvi_binary)
        return frame_physical_parmas

    def physical_features_batch_preprocess(self):
        list_of_images = list(zip(self.ndvis_binary, self.binary_box_imgs, self.b_zed))
        slice_out_res = slice_outside_trees_batch(list_of_images, self.b_slicer, self.false_masks)
        ndvi_binary = [np.nan_to_num(imgs[0], nan=0) for imgs in slice_out_res]
        binary_box_img = [np.nan_to_num(imgs[1], nan=0) for imgs in slice_out_res]
        xyz_point_cloud = [imgs[2] for imgs in slice_out_res]
        return xyz_point_cloud, ndvi_binary, binary_box_img

    def calc_physical_features_batch(self):
        xyz_point_cloud, ndvi_binary, binary_box_img = self.physical_features_batch_preprocess()
        frames_params = list(map(self.calc_frame_physical_parmas, xyz_point_cloud, binary_box_img, ndvi_binary))
        for frame_params in frames_params:
            if isinstance(frame_params, type(None)):
                continue
            self.tree_physical_params = self.update_features_dict(self.tree_physical_params, frame_params, True)

        # for i in range(4):
        #     try:
        #         plot_2_imgs(ndvi_binary[i], xyz_point_cloud[i], title=f"{self.b_frame_numbers[i]}_class",
        #                     save_to=f"/media/fruitspec-lab/easystore/debugging/feature_extractor_class_comparison/class/physical_params_imgs/image_slice_{self.b_frame_numbers[i]}.png")
        #         plt.hist(frames_params[i]["width"], bins=50)
        #         plt.vlines(np.nanmean(frames_params[i]["width"]), 0, 100, color="black")
        #         plt.title(f"width distribution, {self.b_frame_numbers[i]}_class")
        #         plt.savefig(
        #             f"/media/fruitspec-lab/easystore/debugging/feature_extractor_class_comparison/class/physical_params_imgs/hist_{self.b_frame_numbers[i]}.png")
        #         plt.show()
        #     except:
        #         print("debug")

    def vi_batch_preprocess(self):
        list_of_images = list(zip(self.b_jai_rgb, self.b_nir, self.b_swir_975, self.ndvis_binary))
        slice_out_res = slice_outside_trees_batch(list_of_images, self.b_slicer, self.false_masks)
        b_jai_rgb = [imgs[0] for imgs in slice_out_res]
        b_nir = [imgs[1] for imgs in slice_out_res]
        b_swir_975 = [imgs[2] for imgs in slice_out_res]
        ndvis_binary = [np.nan_to_num(imgs[3], nan=0) for imgs in slice_out_res]
        fill = [None]*len(ndvis_binary)
        return b_jai_rgb, b_nir, b_swir_975, fill, ndvis_binary

    def calc_vi_features_batch(self):
        b_jai_rgb, b_nir, b_swir_975, fill, ndvis_binary = self.vi_batch_preprocess()
        frames_params = list(map(self.get_additional_vegetation_indexes, b_jai_rgb, b_nir, b_swir_975,
                                 fill, ndvis_binary, [self.vegetation_indexes_keys]*len(b_jai_rgb)))
        for frame_params in frames_params:
            if isinstance(frame_params, type(None)):
                continue
            self.tree_vi = self.update_features_dict(self.tree_vi, frame_params)

    def transform_to_vi_features(self):
        """
        :param features_dict: features dictionary
        :return: vegetation indexes features transform for one tree
        """
        vi_functions = vegetation_functions()
        if not self.vegetation_indexes_keys:
            vegetation_indexes_keys = vi_functions.keys()
        else:
            vegetation_indexes_keys = self.vegetation_indexes_keys
        for key in vegetation_indexes_keys:
            clean_key = self.clean_veg_input(self.tree_vi[key])
            self.tree_vi[key] = np.nanmean(clean_key)
            self.tree_vi[f"{key}_skew"] = skew(clean_key)
        return self.tree_vi

    def transform_to_tree_physical_features(self, tree_physical_params: dict) -> dict:
        """
        Transform physical parameters dictionary to a tree physical features dictionary.

        The function calculates scalar values for the physical parameters, including total orange,
        total foliage, and surface area. It then calculates foliage fullness
        and square root of surface area using the calculated scalar values.
        The function calculates statistics for width, perimeter, and height, including interquartile range max,
        median, and center values. It also calculates tree volume using `get_tree_volume()`.

        Finally, the function removes "ndvi_bin" and "cont" keys from the dictionary.

        Args:
            tree_physical_params (dict): A dictionary containing physical parameters for a tree.

        Returns:
            A dictionary containing physical features of the tree.
        """
        sclars = ["total_orange", "total_foliage", "surface_area", "ndvi_bin", "cont"]
        for scalar_key in sclars:
            tree_physical_params[scalar_key] = np.sum(tree_physical_params[scalar_key])
        tree_physical_params["foliage_fullness"] = tree_physical_params["ndvi_bin"] / tree_physical_params["cont"]
        tree_physical_params["surface_area"] = np.sqrt(tree_physical_params["surface_area"])
        summed_width = np.nansum(np.array(tree_physical_params["width"]), axis=0)
        summed_width = summed_width[summed_width > 0.25]
        summed_perimeter = np.nansum(np.array(tree_physical_params["perimeter"]), axis=0)
        summed_perimeter = summed_perimeter[summed_perimeter > 0.25]
        try:
            heights = np.concatenate(tree_physical_params["height"])
        except:
            if not len(tree_physical_params["height"]):
                for key in tree_physical_params.keys():
                    tree_physical_params[key] = 0
                for key in ["ndvi_bin", "cont"]:
                    tree_physical_params.pop(key)
                    return tree_physical_params
        heights = heights[heights > 0.25]
        tree_physical_params["width"] = iqr_max(summed_width)
        tree_physical_params["perimeter"] = iqr_max(summed_perimeter)
        tree_physical_params["height"] = iqr_max(heights)
        tree_physical_params["avg_width"] = np.nanmedian(summed_width)
        tree_physical_params["avg_perimeter"] = np.nanmedian(summed_perimeter)
        tree_physical_params["avg_height"] = np.nanmedian(heights)
        tree_physical_params["center_width"] = clac_statistic_on_center(summed_width)
        tree_physical_params["center_perimeter"] = clac_statistic_on_center(summed_perimeter)
        tree_physical_params["center_height"] = clac_statistic_on_center(heights)
        volume, avg_volume = get_tree_volume(tree_physical_params, vol_style=self.vol_style)
        tree_physical_params["volume"] = volume
        tree_physical_params["avg_volume"] = avg_volume
        for key in ["ndvi_bin", "cont"]:
            tree_physical_params.pop(key)
        return tree_physical_params

    def reset(self, args, tree_id, row_id, block_name):

        self.__init__(args, tree_id, row_id, block_name)
        # self.cv_res = {}
        # self.tree_physical_features = self.init_physical_parmas(np.nan)
        # self.tree_fruit_params = self.init_fruit_params()
        # self.fruit_3d_space = {}
        # self.tree_physical_params = self.init_physical_parmas([])
        # self.tree_vi = {**self.get_additional_vegetation_indexes(0, 0, 0, fill=[],
        #                                                          vegetation_indexes_keys=self.vegetation_indexes_keys)}
        # self.tree_start, self.tree_end = False, False


def run_on_tree(tree_frames, fe, adts_loader, batch_size, print_fids=False):
    n_batchs = len(tree_frames) // batch_size
    time_data = []
    for i in tqdm(range(n_batchs)):
        frame_ids = tree_frames[i * batch_size: (i + 1) * batch_size]
        if print_fids:
            print(frame_ids)
        s = time.time()
        batch_res = adts_loader.load_batch(frame_ids)
        e = time.time()
        batch_time = fe.process_batch(*batch_res)

        if batch_time is not None:
            batch_time['load_batch'] = e - s
            time_data.append(batch_time)


    if n_batchs * batch_size < len(tree_frames):
        frame_ids = tree_frames[n_batchs * batch_size:]
        if print_fids:
            print(frame_ids)
        s = time.time()
        batch_res = adts_loader.load_batch(frame_ids)
        e = time.time()
        batch_time = fe.process_batch(*batch_res)
        if batch_time is not None:
            batch_time['load_batch'] = e - s
            time_data.append(batch_time)

    return time_data


if __name__ == '__main__':
    repo_dir = get_repo_dir()
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    args_adts = OmegaConf.load(repo_dir + runtime_config)
    args = OmegaConf.load(repo_dir + "/vision/feature_extractor/feature_extractor_config.yaml")

    #row_scan_path = args_adts.output_folder
    row_scan_path = "/media/matans/My Book/FruitSpec/Feature_Extraction/test_data/SHAMVATI_R10"
    norgb = False
    print_fids = False

    slices = pd.read_csv(os.path.join(row_scan_path, "slices.csv"))
    trees = slices["tree_id"].unique()
    tree_id = trees[0]
    row = os.path.basename(row_scan_path)
    tree_frames = slices["frame_id"][slices["tree_id"] == tree_id].apply(str).values

    block_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(row_scan_path))))
    fe = FeatureExtractor(args, tree_id, row_scan_path, block_name)
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    cfg.batch_size = args.batch_size

    cfg.frame_loader.mode = "sync_svo"
    adts_loader = ADTSBatchLoader(cfg, args_adts, block_name, row_scan_path, tree_id=1)
    batch_size = args.batch_size
    run_on_tree(tree_frames, fe, adts_loader, batch_size, print_fids)

    res = fe.get_tree_results()
    # feature_extracted, cv = fe.extract_features()
    print("physical features are: ", res[0])
    print("fruit features are: ", res[1])
    print("localization features are: ", res[2])
    print("vi features are: ", res[3])
    print("cv is: ", fe.cv_res)
    # print(feature_extracted)