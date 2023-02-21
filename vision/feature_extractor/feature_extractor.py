import os
from collections.abc import Iterable
from os import path
from vision.misc.help_func import load_json
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from tqdm import tqdm

try:
    from vegetation_indexes import *
except:
    from vision.feature_extractor.vegetation_indexes import *
try:
    from stat_tools import *
except:
    from vision.feature_extractor.stat_tools import *
try:
    from image_processing import *
except:
    from vision.feature_extractor.image_processing import *
try:
    from boxing_tools import *
except:
    from vision.feature_extractor.boxing_tools import *
try:
    from tree_size_tools import *
except:
    from vision.feature_extractor.tree_size_tools import *
import time
import matplotlib.pyplot as plt
from vision.tools.image_stitching import plot_2_imgs
from vision.pipelines.movies_to_trees_pipe import update_save_log
from vision.tools.video_wrapper import video_wrapper


global fsi_size
global nir_channel
global ir975_channel
global blue_channel
fsi_size = (900, 600)
nir_channel, ir975_channel, blue_channel = 0, 1, 2


def get_minimal_frames(tree_images, slicer_results, tree_name, tracker_results=None):
    """
    Get the minimal frames needed to represent the tree in the video
    :param tree_images: (dict) a dictionary containing the images of the tree for each frame
    :param slicer_results: (dict) a dictionary containing the results of the slicer algorithm for each frame
    :param tree_name: (str) the name of the tree
    :param tracker_results: (dict) a dictionary containing the results of the tracker algorithm for each frame
    :return: (list) a list of frame numbers that represent the tree, (list) the masks of the tree for each frame
    """
    # TODO check for large trees / close video
    frames_numbers = list(tree_images.keys())
    n_frames = len(frames_numbers)
    where_all_tree_in_pic = np.where(np.all([[slice[0] > 30, slice[1] < fsi_size[1]-30]
                                             for slice in slicer_results.values()], axis=1))[0]
    masks = None
    if len(where_all_tree_in_pic) > 0:
        return [frames_numbers[where_all_tree_in_pic[0]]], masks
    else:
        q1_frame = frames_numbers[int(n_frames * 0.35)]
        q3_frame = frames_numbers[int(n_frames * 0.65)]
        try:
            _, masks = get_fsi_and_masks(tree_images, [q1_frame, frames_numbers[int(n_frames * 0.5)], q3_frame], tracker_results)
            return [q1_frame, frames_numbers[int(n_frames * 0.5)], q3_frame], masks
        except:
            print(f"no homography for all pics: {tree_name}, added more picturss")
            return [frames_numbers[int(n_frames * i/10)] for i in range(3, 8)], masks


def get_additional_vegetation_indexes(rgb, nir, swir_975, fill=None, mask=None):
    """
    :param rgb: rgb image
    :param nir: nir channel (800)
    :param swir_975: 975 channel
    :param fill: what value to fill dictionary with
    :param mask: mask to use for images
    :return: a vegetation indexes dictionary
    """
    vi_functions = vegetation_functions()
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
    clean_arr = {key: cp.asnumpy(vi_functions[key](**input_dict).flatten())
                 for key in vegetation_indexes_keys}
    return {**clean_arr}


def get_width_height(ndvi_binary, x_start, x_end, mask, point_cloud):
    """
    :param ndvi_binary: binary ndvi
    :param point_cloud: point_cloud from zed
    :param x_start: x start of tree
    :param x_end: x end of tree
    :param mask: foliage mask
    :return: widths, heights of picture
    """
    ndvi_binary = slice_outside_trees([ndvi_binary], x_start, x_end)
    ndvi_binary[mask] = 0
    np.nan_to_num(ndvi_binary, copy=False, nan=0)
    point_cloud[1 - ndvi_binary] = np.nan
    widths = np.sum(ndvi_binary, axis=1)
    heights = np.sum(ndvi_binary, axis=0)
    return widths, heights


def update_tree_foli_fetures(features_dict, new_values, keep_dict=False, replace=False):
    """
    :param features_dict: dictionary of the features
    :param new_values: new values to add to dictionary
    :param keep_dict: flag on how to treat iterators,
     if false will add the values to the current features list, else will add the list as is (list of lists)
    :param replace: flag to replace current value
    :return: an updated dictionary
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


def clean_veg_input(flat_hist):
    """
    :param flat_hist: flalt histogram
    :return: removes na and trims the histogram
    """
    if isinstance(flat_hist, list):
        if isinstance(flat_hist[0], np.ndarray):
            flat_hist = np.concatenate(flat_hist)
    if not isinstance(flat_hist, np.ndarray):
        flat_hist = np.array(flat_hist)
    flat_hist = flat_hist[~np.isnan(flat_hist)]
    quant_trimmed = quantile_trim(flat_hist, (0.025, 0.975), keep_size=False)
    return quant_trimmed


def transform_to_vi_features(features_dict):
    """
    :param features_dict: features dictionary
    :return: vegetation indexes features transform for one tree
    """
    vi_functions = vegetation_functions()
    vegetation_indexes_keys = vi_functions.keys()
    for key in vegetation_indexes_keys:
        clean_key = clean_veg_input(features_dict[key])
        features_dict[key] = np.nanmedian(clean_key)
        features_dict[f"{key}_skew"] = skew(clean_key)
    return features_dict


def fill_dict(features_list, value):
    """
    :param features_list: list of features to fill
    :param value: values to initialize features with
    :return: an initialized features dictionary
    """
    features_dict = {}
    if isinstance(value, Iterable):
        for key in features_list:
            features_dict[key] = value.copy()
        return features_dict
    for key in features_list:
        features_dict[key] = value
    return features_dict


def init_physical_parmas(value):
    """
    :param value: values to initialize features with
    :return: an initialized features dictionary
    """
    features_list = ["total_foliage", "total_orange", "width", "height", "volume", "surface_area", "perimeter",
                     "avg_width", "avg_height", "avg_volume", "avg_perimeter", "foliage_fullness", "cont", "ndvi_bin"]
    return fill_dict(features_list, value)


def init_fruit_params(value):
    """
    :param value: values to initialize features with
    :return: an initialized features dictionary
    """
    features_list = ["cv", "frame", "w_h_ratio", "q1", "q3", "avg_intens_arr", "med_intens_arr"]
    return fill_dict(features_list, value)


def transform_to_tree_physical_features(tree_physical_params):
    """
    transform from frame format to tree (scaler) format
    :param tree_physical_params: (dict) containing the physical parameters for a single tree
    :return: (dict) containing the physical parameters in a simplified format
    """
    sclars = ["total_orange", "total_foliage", "surface_area", "ndvi_bin", "cont"]
    for scalar_key in sclars:
        tree_physical_params[scalar_key] = np.sum(tree_physical_params[scalar_key])
    tree_physical_params["foliage_fullness"] = tree_physical_params["ndvi_bin"]/tree_physical_params["cont"]
    tree_physical_params["surface_area"] = np.sqrt(tree_physical_params["surface_area"])
    summed_width = np.nansum(np.array(tree_physical_params["width"]), axis=0)
    summed_width = summed_width[summed_width > 0.25]
    summed_perimeter = np.nansum(np.array(tree_physical_params["perimeter"]), axis=0)
    summed_perimeter = summed_perimeter[summed_perimeter > 0.25]
    heights = np.concatenate(tree_physical_params["height"])
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
    ## this is what we need to use when aggregation the frames
    # widths = tree_physical_params["width"][tree_physical_params["width"] > 0.25]
    # heights = tree_physical_params["height"][tree_physical_params["height"] > 0.25]
    # perimeters = tree_physical_params["perimeter"][tree_physical_params["perimeter"] > 0.25]
    # tree_physical_params["avg_width_full"] = np.nanmedian(widths)
    # tree_physical_params["avg_perimeter_full"] = np.nanmedian(perimeters)
    # tree_physical_params["avg_height_full"] = np.nanmedian(heights)
    # tree_physical_params["center_width_full"] = clac_statistic_on_center(widths)
    # tree_physical_params["center_perimeter_full"] = clac_statistic_on_center(perimeters)
    # tree_physical_params["center_height_full"] = clac_statistic_on_center(heights)
    # tree_physical_params["width"] = iqr_max(widths)
    # tree_physical_params["perimeter"] = iqr_max(perimeters)
    # tree_physical_params["height"] = iqr_max(heights)

    volume, avg_volume = get_tree_volume(tree_physical_params, vol_style="cone")
    tree_physical_params["volume"] = volume
    tree_physical_params["avg_volume"] = avg_volume
    for key in ["ndvi_bin", "cont"]:
        tree_physical_params.pop(key)
    return tree_physical_params


def calc_frame_physical_parmas(xyz_point_cloud, binary_box_img, ndvi_binary):
    """
    Calculate the physical parameters of a single frame
    :param xyz_point_cloud: (np.ndarray) 3D point cloud of the frame
    :param binary_box_img: (np.ndarray) binary image of the frame indicating the presence of oranges
    :param ndvi_binary: (np.ndarray) binary image of the frame indicating the presence of foliage
    :return: (dict) containing the physical parameters of the frame
    """
    # TODO this function is rather long we can make in shorter by not going thourgh the entire image for calculations
    frame_physical_parmas = init_physical_parmas([])
    pixel_size = get_real_world_dims_with_correction(xyz_point_cloud[:, :, 2])
    total_orange = np.nansum(binary_box_img * pixel_size)
    total_foliage = np.nansum(ndvi_binary * pixel_size)
    frame_physical_parmas["total_orange"] = total_orange  # scalar
    frame_physical_parmas["total_foliage"] = total_foliage  # scalar
    frame_physical_parmas["width"] = calc_tree_widths(xyz_point_cloud, ndvi_binary)  # np array (scaler for each row)
    frame_physical_parmas["height"] = calc_tree_heights(xyz_point_cloud, ndvi_binary)  # np array (scaler for each col)
    frame_physical_parmas["perimeter"] = calc_tree_perimeter(xyz_point_cloud, ndvi_binary)  # np array (scaler for each row)
    frame_physical_parmas["surface_area"] = get_surface_area(xyz_point_cloud, ndvi_binary)  # scalar
    frame_physical_parmas["ndvi_bin"], frame_physical_parmas["cont"] = get_foliage_fullness(ndvi_binary)
    return frame_physical_parmas


def calc_physical_features(tree_images, slicer_results, minimal_frames, tracker_results, tree_name, masks=None, dets_only=False):
    """
    Calculate physical features of a tree using minimal frames from the tree's images.

    :param tree_images: (dict) {"frame": {"fsi":fsi,"rgb":rgb,"zed":zed} for each frame}
    :param slicer_results: (dict) {"frame": (x_start,x_end) for each frame}
    :param minimal_frames: (list) of frame numbers
    :param tracker_results: (dict) {"frame": {"id": ((x0,y0),(x1,y1))} for each frame}
    :param tree_name: (str) name of the tree (for logging)
    :param masks: (ndarray) mask to be used on the images
    :param dets_only: (bool) if True, will use only detections, not tracker
    :return: (tuple) physical features for tree, updated tree images with ndvi's, masks, minimal_frames, frame_rate
    """
    # 17% of time
    if len(minimal_frames) > 1 and isinstance(masks, type(None)):
        try:
            _, masks = get_fsi_and_masks(tree_images, minimal_frames, tracker_results if dets_only else None)
        except:
            print(f"no homography for all pics: {tree_name}")
            return init_physical_parmas(np.nan), tree_images, masks, 0, 0.1
    y_ranges = get_y_ranges(minimal_frames, masks)
    false_mask, n_min_frames, tree_physical_params = None, len(minimal_frames), init_physical_parmas([])
    middle_frame = minimal_frames[len(minimal_frames) // 2]
    # xyz_imgs, ndvi_binary_imgs, binary_box_imgs = [], [], []
    for i, frame_number in enumerate(minimal_frames):
        print(f"physical features - {tree_name}: {frame_number}")
        fsi, rgb, nir, swir_975, xyz_point_cloud, zed_rgb = get_pictures(tree_images, frame_number, with_zed=True)
        boxes = tracker_results[frame_number]
        # 53%
        ndvi_img, ndvi_binary, binary_box_img = get_ndvi_pictures(rgb, nir, fsi, boxes)
        tree_images[frame_number]["ndvi_img"] = ndvi_img
        tree_images[frame_number]["ndvi_binary"] = ndvi_binary
        tree_images[frame_number]["binary_box_img"] = binary_box_img
        if middle_frame == frame_number:
            min_y, max_y = get_min_max_real_y(tree_images, slicer_results, minimal_frames)
        # rgb_debug = rgb.copy().astype(np.uint8)
        # rgb_debug[(1-ndvi_binary).astype(bool)] = 0
        # cv2.imshow(frame_number, cv2.cvtColor(rgb.astype(np.uint8)[:, :, ::-1], cv2.COLOR_RGB2HLS))
        # cv2.waitKey()
        # cv2.imshow(frame_number, fsi.astype(np.uint8)[:, :, ::-1])
        # cv2.waitKey()
        # cv2.imshow(frame_number, ((ndvi_img+1)*255/2).astype(np.uint8))
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        if i > 0:
            false_mask = masks[i - 1].astype(bool)
            ndvi_binary[false_mask] = np.nan
            binary_box_img[false_mask] = np.nan
        images = [xyz_point_cloud, ndvi_binary, binary_box_img]
        xyz_point_cloud, ndvi_binary, binary_box_img = slice_outside_trees(images, slicer_results, frame_number,
                                                                           reduce_size=True, mask=false_mask, i=i,
                                                                           y_ranges=y_ranges, n_min_frames=n_min_frames)
        ndvi_binary, binary_box_img = np.nan_to_num(ndvi_binary, nan=0), np.nan_to_num(binary_box_img, nan=0)

        # plot_2_imgs(ndvi_img, ndvi_binary, title=frame_number)
        # 28%
        if 0 in ndvi_binary.shape:
            continue
        # xyz_imgs.append(xyz_point_cloud.copy())
        # ndvi_binary_imgs.append(ndvi_binary.copy())
        # binary_box_imgs.append(binary_box_img.copy())
        frame_physical_parmas = calc_frame_physical_parmas(xyz_point_cloud, binary_box_img, ndvi_binary)
        tree_physical_params = update_tree_foli_fetures(tree_physical_params, frame_physical_parmas, keep_dict=True)
    if len(minimal_frames) > 1:
        translation_debugging(minimal_frames, masks, frame_number, tree_name, tree_images)
    # xyz_full_img = np.concatenate(xyz_imgs, axis=1)
    # ndvi_binary_full_img = np.concatenate(ndvi_binary_imgs, axis=1)
    # binary_box_full_img = np.concatenate(binary_box_imgs, axis=1)
    # tree_physical_params = calc_frame_physical_parmas(xyz_full_img, ndvi_binary_full_img, binary_box_full_img)
    return transform_to_tree_physical_features(tree_physical_params), tree_images, masks, min_y, max_y


def translation_debugging(minimal_frames, masks, frame_number, tree_name, tree_images):
    """
    This function is used for debugging the translation of masks between frames. It plots the image of two consecutive frames,
    one with the mask applied and one without the mask.
    It also saves the images to a specified file path.

    :param minimal_frames: List of ints representing the minimal frames that were selected for the tree
    :param masks: List of numpy arrays representing the masks that were generated for each frame
    :param frame_number: str representing the frame number that is currently being processed
    :param tree_name: str representing the name of the tree that is currently being processed
    :param tree_images: Dict of ints to Dicts of strings to numpy arrays representing the tree images and their corresponding data
    """
    for i in range(1, len(minimal_frames)):
        fsi_next = tree_images[minimal_frames[i]]["fsi"].copy()
        fsi_next[masks[i-1].astype(bool)] = 0
        # plot_2_imgs(tree_images[minimal_frames[i-1]]["fsi"], tree_images[minimal_frames[i]]["fsi"], title=frame_number,
        #             save_to=f"/media/fruitspec-lab/easystore/translation keypoints dets only/{tree_name}_{i}_clean.png",
        #             save_only=True)
        # plot_2_imgs(tree_images[minimal_frames[i-1]]["fsi"], fsi_next, title=frame_number,
        #             save_to=f"/media/fruitspec-lab/easystore/translation keypoints dets only/{tree_name}_{i}_masked.png",
        #             save_only=True)
    # fsi_next = tree_images[minimal_frames[2]]["fsi"].copy()
    # fsi_next[masks[1].astype(bool)] = 0
    # plot_2_imgs(tree_images[minimal_frames[1]]["fsi"], tree_images[minimal_frames[2]]["fsi"], title=frame_number,
    #             save_to=f"/media/fruitspec-lab/easystore/translation match dets only/{tree_name}_2_clean.png",
    #             save_only=True)
    # plot_2_imgs(tree_images[minimal_frames[1]]["fsi"], fsi_next, title=frame_number,
    #             save_to=f"/media/fruitspec-lab/easystore/translation match dets only/{tree_name}_2_masked.png",
    #             save_only=True)


def update_fruit_features(tree_fruit_params, tracker_results, fsi, rgb, frame_number):
    """
    Update the fruit features for a given tree
    :param tree_fruit_params: dictionary containing w_h_ratio, intensity and fruit_foliage_ratio for each fruit
    :param tracker_results: dictionary containing the bounding box for each fruit for the current frame
    :param fsi: fsi for the current frame
    :param rgb: rgb image for the current frame
    :param frame_number: current frame number
    :return: updated tree_fruit_params
    """
    boxes = tracker_results[frame_number]
    w_h_ratios = get_w_h_ratio(boxes.values())
    w_h_keys = np.fromiter(tree_fruit_params["w_h_ratio"].keys(), int)
    for i, track_id in enumerate(boxes.keys()):
        cur_ratio = w_h_ratios[i]
        # 90% of the function time is  get intensity
        cur_intens, cur_foilage_ratio = get_intensity(fsi, rgb, boxes[track_id])
        if np.isin(track_id, w_h_keys):
            tree_fruit_params["w_h_ratio"][track_id] = np.append(tree_fruit_params["w_h_ratio"][track_id], cur_ratio)
            tree_fruit_params["intensity"][track_id] = np.append(tree_fruit_params["intensity"][track_id], cur_intens)
            tree_fruit_params["fruit_foliage_ratio"][track_id] = np.append(tree_fruit_params["fruit_foliage_ratio"][track_id],
                                                                           cur_foilage_ratio)
        else:
            tree_fruit_params["w_h_ratio"][track_id] = np.array([cur_ratio])
            tree_fruit_params["intensity"][track_id] = np.array([cur_intens])
            tree_fruit_params["fruit_foliage_ratio"][track_id] = np.array([cur_foilage_ratio])
    return tree_fruit_params


def tree_intensity_summary(tree_fruit_params, tree_name, frame_number):
    """
    Calculate the intensity summary for a given tree
    :param tree_fruit_params: dictionary containing w_h_ratio, intensity and fruit_foliage_ratio for each fruit
    :param tree_name: name of the tree for logging
    :param frame_number: current frame number
    :return: updated tree_fruit_params containing median, q1, q3, avg_intens_arr and med_intens_arr
    """
    tree_fruit_params["intensity"] = np.fromiter(
        (np.median(intensity) for intensity in tree_fruit_params["intensity"].values()), float)
    norm_intens = normalize_intensity(tree_fruit_params["intensity"], f"{tree_name}: {frame_number}")
    q1, med_intens_arr, q3 = np.nanquantile(norm_intens, [0.25, 0.5, 0.75])
    tree_fruit_params["q1"] = q1
    tree_fruit_params["q3"] = q3
    tree_fruit_params["med_intens_arr"] = med_intens_arr
    tree_fruit_params["avg_intens_arr"] = np.nanmean(norm_intens)
    tree_fruit_params.pop("intensity")
    return tree_fruit_params


def calc_vi(tree_images, slicer_results, minimal_frames, tree_name, masks=None):
    """
    calculates the vegetitional indexes for the tree
    :param tree_images: {"frame": {"fsi":fsi,"rgb":rgb,"zed":zed} for each frame}
    :param slicer_results: {"frame": (x_start,x_end) for each frame}
    :param minimal_frames: list of frame numbers
    :param tree_name: name of the tree (for logging)
    :param masks: maks for minimal frames
    :return: vegetation_indexes for tree
    """
    # if len(minimal_frames) == 1:
    #     tree_minimal_image = tree_images[minimal_frames[0]]
    #     fsi, rgb = tree_minimal_image["fsi"], tree_minimal_image["rgb"]
    #     nir, swir_975 = get_nir_swir(fsi)
    #     boxes = tracker_results[minimal_frames[0]]
    #     ndvi_img, ndvi_binary, binary_box_img = get_ndvi_pictures(rgb, nir, fsi, boxes)
    #     x_start, x_end = slicer_results[minimal_frames[0]]
    #     rgb, nir, swir_975 = slice_outside_trees([rgb, nir, swir_975], x_start, x_end)
    #     return get_additional_vegetation_indexes(rgb, nir, swir_975, mask=ndvi_binary)
    y_ranges = get_y_ranges(minimal_frames, masks)
    n_min_frames = len(minimal_frames)
    false_mask = None
    if isinstance(masks, type(None)) and len(minimal_frames) > 1:
        print(f"no homography for all pics: {tree_name}")
        return get_additional_vegetation_indexes(0, 0, 0, fill=np.nan)
    features_dict = {**get_additional_vegetation_indexes(0, 0, 0, fill=[])}
    for i, frame_number in enumerate(minimal_frames):
        print(f"\r vegetational features - {tree_name}: {frame_number}", end="")
        fsi, rgb, nir, swir_975 = get_pictures(tree_images, frame_number)
        # ndvi_img, ndvi_binary, binary_box_img = get_ndvi_pictures(rgb, nir, fsi, boxes)
        ndvi_img, ndvi_binary, binary_box_img = get_pictures(tree_images, frame_number,
                                                             specific_pics=["ndvi_img", "ndvi_binary", "binary_box_img"])
        if i > 0:
            false_mask = masks[i - 1].astype(bool)
            ndvi_binary[false_mask] = np.nan
        rgb, nir, swir_975, ndvi_binary = slice_outside_trees([rgb, nir, swir_975, ndvi_binary], slicer_results,
                                                              frame_number, reduce_size=True, mask=false_mask, i=i,
                                                              y_ranges=y_ranges, n_min_frames=n_min_frames)

        frame_features = get_additional_vegetation_indexes(rgb, nir, swir_975, mask=ndvi_binary)
        features_dict = update_tree_foli_fetures(features_dict, frame_features)
    return transform_to_vi_features(features_dict)


def calc_fruit_features(tree_images, tracker_results, tree_name):
    """
    calculates fruit features for a given tree
    :param tree_images: {"frame": {"fsi":fsi,"rgb":rgb,"zed":zed} for each frame}
    :param tracker_results: {"frame": {"id": ((x0,y0),(x1,y1))} for each frame}
    :param tree_name: name of the tree (for logging)
    :return: physical features for tree
    """
    # ["cv" V, "frame" V, "w_h_ratio" V, "q1" V, "q3" V, "avg_intens_arr" V, "med_intens_arr" V]
    tree_fruit_params = init_fruit_params([])
    tree_fruit_params["w_h_ratio"], tree_fruit_params["intensity"], tree_fruit_params["fruit_foliage_ratio"] = {}, {}, {}
    for i, frame_number in enumerate(tree_images.keys()):
        print(f"\r fruit features - {tree_name}: {frame_number}", end="")
        fsi, rgb, nir, swir_975, xyz_point_cloud, zed_rgb = get_pictures(tree_images, frame_number, with_zed=True)
        # consumer function 90% of the time is here: update_fruit_features
        tree_fruit_params = update_fruit_features(tree_fruit_params, tracker_results, fsi, rgb, frame_number)
    tree_fruit_params["cv"] = tracker_results["cv"]
    tree_fruit_params["frame"] = len(tree_images.keys())
    n_samp_per_fruit = np.fromiter((len(w_h_ratios) for w_h_ratios in
                                    tree_fruit_params["w_h_ratio"].values()), float)
    wh_std_per_fruit = np.fromiter((np.std(w_h_ratios) for w_h_ratios in
                                                            tree_fruit_params["w_h_ratio"].values()), float)
    tree_fruit_params["w_h_ratio"] = np.median(wh_std_per_fruit[n_samp_per_fruit > 2])
    tree_fruit_params = tree_intensity_summary(tree_fruit_params, tree_name, frame_number)
    fruit_foliage_ratio = np.fromiter((np.nanmean(fruit_foliage_ratio) for fruit_foliage_ratio in
                                                            tree_fruit_params["fruit_foliage_ratio"].values()), float)
    tree_fruit_params["fruit_foliage_ratio"] = np.mean(fruit_foliage_ratio[np.isfinite(fruit_foliage_ratio)])
    return tree_fruit_params


def init_3d_fruit_space(tracker_results, tree_images):
    """
    initiates a 3D space for the fruits on a tree .
    :param tracker_results: {"frame": {"id": ((x0,y0),(x1,y1))} for each frame}
    :param tree_images: {"frame": {"fsi":fsi,"rgb":rgb,"zed":zed} for each frame}
    :return: 3D fruit space
    """
    all_frames = [frame for frame in tracker_results.keys() if frame != "cv"]
    first_frame = all_frames[int(len(all_frames)/4)]
    first_tracker_results = tracker_results[first_frame]
    xyz_point_cloud, swir_975, nir = get_pictures(tree_images, first_frame, specific_pics=["zed", "swir_975", "nir"])
    fruit_space = {id: (xyz_center_of_box(xyz_point_cloud, box, nir, swir_975))
                   for id, box in first_tracker_results.items()}
    fruits_keys = list(fruit_space.keys())
    for fruit in fruits_keys:
        if np.isnan(fruit_space[fruit][0]):
            fruit_space.pop(fruit)
    return fruit_space


def create_3d_fruit_space(tracker_results, tree_images, tree_name):
    """
    Create a 3D space for the fruits on a tree by projecting the fruits on the XYZ point cloud of the first frame.
    :param tracker_results: {"frame": {"id": ((x0,y0),(x1,y1))} for each frame}
    :param tree_images: {"frame": {"fsi":fsi,"rgb":rgb,"zed":zed} for each frame}
    :param tree_name: name of the tree (for logging)
    :return: 3D fruit space projected on the first frame, mean size of fruits (diameter)
    """
    # 30%
    fruit_3d_space = init_3d_fruit_space(tracker_results, tree_images)
    boxes_w, boxes_h = np.array([]), np.array([])
    n_frames = len(tree_images.keys())
    for i, frame_number in enumerate(tree_images.keys()):
        if i < n_frames/4:
            continue
        print(f"\r fruit space - {tree_name}: {frame_number}", end="")
        xyz_point_cloud, swir_975, nir = get_pictures(tree_images, frame_number,
                                                      specific_pics=["zed", "swir_975", "nir"])
        boxes = tracker_results[frame_number]
        # consumer function when using nir,swir 50%
        new_boxes, old_boxes, boxes_w_frame, boxes_h_frame = get_new_old_boxes(xyz_point_cloud,
                                                                               fruit_3d_space, boxes, nir, swir_975)
        boxes_w = np.append(boxes_w, boxes_w_frame)
        boxes_h = np.append(boxes_h, boxes_h_frame)
        # 20%
        fruit_3d_space = project_boxes_to_fruit_space(fruit_3d_space, old_boxes, new_boxes)
    return fruit_3d_space, safe_nanmean(np.nanmax([boxes_w, boxes_h], axis=0))


def num_unique(np_arr):
    """
    :param np_arr:
    :return: number of unique elements
    """
    return len(np.unique(np_arr)) - 1


def num_groups_db_scan(eps, centers, min_samples=2):
    """
    :param eps: epsilon value for db scan
    :param centers: centers of points
    :param min_samples: min_samples paramater for db scan
    :return: number of unique clusters
    """
    return num_unique(DBSCAN(eps=eps, min_samples=min_samples).fit(centers).labels_)


def get_n_clusters_metrics(centers, distances, avg_diam, min_samples=2):
    """
    :param centers: centers of points
    :param distances: distances of points
    :param min_samples: min_samples paramater for db scan
    :param avg_diam: avg diameter of fruits
    :return: number of clusters per method dictionary, labels for DBSCAN using mean,labels for DBSCAN using median
    """
    clustring_dict = {f"n_clust_arr_{i}": num_groups_db_scan(avg_diam * i, centers, min_samples)
                      for i in range(2, 8, 2)}
    clustering_mean = DBSCAN(eps=np.nanmean(distances), min_samples=min_samples).fit(centers)
    clustering_med = DBSCAN(eps=np.nanmedian(distances), min_samples=min_samples).fit(centers)
    clustering_mean_labels = clustering_mean.labels_
    clustering_med_labels = clustering_med.labels_
    clustring_dict["n_clust_mean_arr"] = num_unique(clustering_mean_labels)
    clustring_dict["n_clust_med_arr"] = num_unique(clustering_med_labels)
    return clustring_dict, clustering_mean_labels, clustering_med_labels


def elipse_convexhull_area(centers, labels):
    """
    :param centers: centers of points
    :param labels: label for each point
    :return: ellipse area, convex hull area
    """
    if len(labels) == 0:
        labels = [0] * len(centers)
    ellipse_area = 0
    convex_hull_area = 0
    for label in np.unique(labels)[1:]:
        cnt = np.unique(centers[labels == label], axis=0)
        if len(cnt) < 4:
            continue
        cnvx_hull = ConvexHull(cnt)
        convex_hull_area += cnvx_hull.volume
        a = cnt ** 2
        o = np.ones(len(cnt))
        b, resids, rank, s = np.linalg.lstsq(a, o, rcond=None)
        ellipse_area += np.product(np.sqrt(np.abs(1 / b))) * pi

    return ellipse_area, convex_hull_area


def plot_3d_cloud(fruit_3d_space, centers, c=None):
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
    plt.show()


def plot_2d_cloud(fruit_3d_space, centers, c=None):
    """
    Plot the 2D point cloud of the fruit space
    :param fruit_3d_space: a dictionary of fruits with their 3D coordinates
    :param centers: a list of the centers of each fruit
    :param c: color of the points
    :return: None
    """
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    ax.scatter(-centers[:, 0], -centers[:, 1], c=c)
    for i, label in enumerate(fruit_3d_space.keys()):  # plot each point + it's index as text above
        ax.text(-centers[i, 0], -centers[i, 1], '%s' % (str(label)), size=10, zorder=1,
                color='k')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    plt.show()


def get_fruit_distribution_on_tree(centers, min_y, max_y, num_breaks=3):
    """
    Calculate the distribution of fruits on the tree according to the real y value
    :param centers: array of shape (n, 3) containing the x, y, z coordinates of each fruit
    :param min_y: minimum y value of the tree
    :param max_y: maximum y value of the tree
    :param num_breaks: number of intervals/breaks in which to divide the tree along the y-axis
    :return: dictionary containing the percentage of fruits in each interval/break
    """
    out_put = {}
    breaks = np.linspace(min_y, max_y, num_breaks+1)
    y_s = centers[:, 1]
    for i in range(num_breaks):
        out_put[f"q_{i+1}_precent_fruits"] = np.mean(np.all([y_s >= breaks[i], y_s < breaks[i+1]], axis=0))
    return out_put



def calc_localization_features(tree_images, tracker_results, tree_name, min_y, max_y, centers=None):
    """
    calculate the features based on fruit distribution on the tree
    :param tree_images: a dictionary containing the full spectrum, RGB and ZED images for each frame
    :param tracker_results: a dictionary containing the bounding box for each fruit for the current frame
    :param tree_name: name of the tree (for logging)
    :param min_y: minimum y value of the bounding box
    :param max_y: maximum y value of the bounding box
    :param centers: (Optional) array of the fruit centers in 3D space
    :return: a dictionary containing physical features for tree
    """
    # ["mst_sums_arr", "mst_mean_arr", "mst_skew_arr",
    #                 "n_clust_mean_arr", "n_clust_med_arr", "clusters_area_mean_arr", "clusters_area_med_arr",
    #                 "clusters_ch_area_mean_arr", "clusters_ch_area_med_arr",
    #                 "n_clust_arr_2", "n_clust_arr_4", "n_clust_arr_6", "n_clust_arr_8", "n_clust_arr_10"]
    # consumer 95 %
    if isinstance(centers, type(None)):
        fruit_3d_space, avg_diam = create_3d_fruit_space(tracker_results, tree_images, tree_name)
        centers = np.array(list(fruit_3d_space.values()))
    else:
        avg_diam = 1
    # plot_3d_cloud(fruit_3d_space, centers)
    # plot_2d_cloud(fruit_3d_space, centers)
    problem_w_center = False
    if len(centers) > 0 and len(centers.shape) > 1:
        centers = centers[np.isfinite(centers[:, 2])]
    else:
        problem_w_center = True
    if avg_diam > 0 and len(centers) > 2 and not problem_w_center:
        mst, distances = compute_density_mst(centers)
        clustring_dict, clustering_mean_labels, clustering_med_labels = get_n_clusters_metrics(centers, distances, avg_diam)
    else:
        problem_w_center = True
    if problem_w_center:
        tree_loc_params = {}
        for key in ["mst_sums_arr", "mst_mean_arr", "mst_skew_arr",
                    "n_clust_mean_arr", "n_clust_med_arr", "clusters_area_mean_arr", "clusters_area_med_arr",
                    "clusters_ch_area_mean_arr", "clusters_ch_area_med_arr",
                    "n_clust_arr_2", "n_clust_arr_4", "n_clust_arr_6"]:
            tree_loc_params[key] = 0
        return tree_loc_params
    area_mean, convex_hull_area_mean = elipse_convexhull_area(centers, clustering_mean_labels)
    area_med, convex_hull_area_med = elipse_convexhull_area(centers, clustering_med_labels)
    tree_loc_params = {"mst_sums_arr": np.sum(distances), "mst_mean_arr": np.mean(distances),
                       "mst_skew_arr": skew(distances),
                       "clusters_area_mean_arr": area_mean, "clusters_area_med_arr": area_med,
                       "clusters_ch_area_mean_arr": convex_hull_area_mean,
                       "clusters_ch_area_med_arr": convex_hull_area_med, **clustring_dict,
                       **get_fruit_distribution_on_tree(centers, min_y, max_y),
                       "fruit_dist_center": (np.nanmean(centers[:, 1])-min_y)/(max_y-min_y)}
    return tree_loc_params


def filter_tracker_results(tracker_results, slicer_results, tree_images, max_z):
    """
    filters the tracker results based on slicing and max depth allowed
    :param tracker_results: tracker results dictionary
    :param slicer_results: slicer results dictionary
    :param tree_images: tree images dictionary
    :param max_z: max depth allowed
    :return: filter dictionary
    """
    tracker_results = filter_outside_tree_boxes(tracker_results, slicer_results)
    if max_z > 0:
        tracker_results = filter_outside_zed_boxes(tracker_results, tree_images, max_z)
    tracker_results["cv"] = len(
        {id for frame in set(tracker_results.keys()) - {"cv"} for id in tracker_results[frame].keys()})
    return tracker_results


def extract_features_for_tree(tree_images, slicer_results, tracker_results, tree_name, max_z=0):
    """
    extract all features for a given tree
    :param tree_images: {"frame": {"fsi":fsi image,"rgb":rgb image,"zed":zed image} for each frame}
    :param slicer_results: {"frame": (x_start,x_end) for each frame}
    :param tracker_results: {"frame": {"id": ((x0,y0),(x1,y1))} for each frame}
    :param tree_name: name of the tree (for logging)
    :param max_z: the maximum depth allowed to use
    :return:
    """
    s_time_0 = time.time()
    for frame in tree_images.keys():
        frame_images = tree_images[frame]
        frame_images["nir"], frame_images["swir_975"] = get_nir_swir(frame_images["fsi"])
        frame_images["rgb"] = frame_images["rgb"].astype(float)
    tracker_results = filter_tracker_results(tracker_results, slicer_results, tree_images, max_z)
    print(f" filter_outside_tree_boxes: { time.time()-s_time_0 }")
    s_time = time.time()
    # minimal_frames, masks = get_minimal_frames(tree_images, slicer_results, tree_name, tracker_results)
    minimal_frames = list(tree_images.keys())
    n_mininal = len(minimal_frames)
    minimal_frames = minimal_frames[int(n_mininal*0.25): int(n_mininal*0.75):2]
    # minimal_frames = [minimal_frames[int(n_mininal*0.5)]]
    masks = None
    print(f" get_minimal_frames: { time.time()-s_time }")
    s_time = time.time()
    # masks bags
    tree_physical_features, tree_images, masks, min_y, max_y = calc_physical_features(tree_images, slicer_results,
                                                                                      minimal_frames, tracker_results,
                                                                                      tree_name, masks)
    print(tree_physical_features)
    print(f" calc_physical_features: { time.time()-s_time }")
    s_time = time.time()
    tree_loc_features = calc_localization_features(tree_images, tracker_results,
                                                   tree_name, min_y, max_y)  # check logic and fruit localization
    print(f" calc_localization_features: { time.time()-s_time }")
    s_time = time.time()
    tree_fruit_features = calc_fruit_features(tree_images, tracker_results, tree_name)
    print(f" calc_fruit_features: { time.time()-s_time }")
    s_time = time.time()
    tree_veg_features = calc_vi(tree_images, slicer_results, minimal_frames, tree_name, masks)
    print(f" calc_vi: { time.time()-s_time }")
    tree_features = {**tree_loc_features, **tree_fruit_features, **tree_veg_features, **tree_physical_features}
    print(f" total time: {time.time() - s_time_0}")
    print(tree_features)
    tree_features["total_time"] = time.time() - s_time_0
    tree_features["name"] = tree_name
    return tree_features


# def read_images():
#     zed_folder = "/home/fruitspec-lab/FruitSpec/Sandbox/merge_sensors/ZED_subset"
#     jai_folder = "/home/fruitspec-lab/FruitSpec/Sandbox/merge_sensors/FSI_subset"
#     rgb_folder = "/home/fruitspec-lab/FruitSpec/Sandbox/merge_sensors/RGB_subset"
#     tracker_full_results = pd.read_csv(
#         "/home/fruitspec-lab/FruitSpec/Sandbox/merge_sensors/det/FSI_subset/detections_csv.csv")
#     top_left_points = list(zip(tracker_full_results["x1"].values, tracker_full_results["y1"].values))
#     bottom_right_points = list(zip(tracker_full_results["x2"].values, tracker_full_results["y2"].values))
#     bboxed = list(zip(top_left_points, bottom_right_points))
#     tracker_full_results["bbox"] = bboxed
#     # zed_frames = [545, 548, 551, 554, 557, 560, 563, 566, 569, 572, 575, 578, 581] #, 584]
#     zed_frames = [550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562]  # , 584]
#     # jai_frames = [539, 541, 543, 545, 547, 549, 551, 553, 555, 557, 559, 561, 563]
#     jai_frames = [540, 541, 542, 543, 544, 545, 546, 546, 548, 549, 550, 551, 552]
#     sliced_data = [(48, 1383), (0, 1314), (0, 1272), (0, 1257), (0, 1209), (0, 1200), (0, 1164), (0, 1150), (0, 1104),
#                    (0, 1083), (0, 1050), (0, 1020), (0, 981)]
#     tree_images = {}
#     slicer_results = {}
#     tracker_results = {"cv": tracker_full_results["track_id"].nunique()}
#     all_coords = pd.read_csv(path.join(jai_folder, "jain_cors_in_zed.csv"))
#     for i, frame in enumerate(jai_frames):
#         pictures_dict = {}
#         pictures_dict["fsi"] = cv2.imread(path.join(jai_folder, f"frame_{frame}.jpg"))[:, :, ::-1]
#         pictures_dict["rgb"] = cv2.imread(path.join(rgb_folder, f"rgb_{frame}.jpg"))[:, :, ::-1]
#         pictures_dict["zed"] = np.load(path.join(zed_folder, f"xyz_frame_{zed_frames[i]}.npy"))
#         pictures_dict["zed_rgb"] = cv2.imread(path.join(zed_folder, f"frame_{zed_frames[i]}.jpg"))[:, :, ::-1]
#         cur_coords = all_coords[all_coords["frame"] == frame].reset_index().astype(int)
#         pictures_dict["zed"] = pictures_dict["zed"][cur_coords["y1"][0]:cur_coords["y2"][0],
#                                cur_coords["x1"][0]:cur_coords["x2"][0], :]
#         pictures_dict["zed"] = cv2.resize(pictures_dict["zed"], pictures_dict["rgb"].shape[:2][::-1])[:, :, :3]
#         pictures_dict["zed_rgb"] = pictures_dict["zed_rgb"][cur_coords["y1"][0]:cur_coords["y2"][0],
#                                    cur_coords["x1"][0]:cur_coords["x2"][0], :]
#         pictures_dict["zed_rgb"] = cv2.resize(pictures_dict["zed_rgb"], pictures_dict["rgb"].shape[:2][::-1])[:, :, :3]
#         tree_images[str(frame)] = pictures_dict
#         slicer_results[str(frame)] = sliced_data[i]
#         tracker_results[str(frame)] = {row[1]["track_id"]: row[1]["bbox"] for row in
#                                        tracker_full_results[tracker_full_results["frame_ids"] == frame][
#                                            ["track_id", "bbox"]].iterrows()}
#         # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
#         # ax1.imshow(pictures_dict["zed_rgb"])
#         # ax2.imshow(pictures_dict["rgb"])
#         # plt.show()
#     return tree_images, slicer_results, tracker_results
#
#
# def read_images_small():
#     zed_folder = "/home/fruitspec-lab/PycharmProjects/foliage/counter/T_32"
#     jai_folder = "/home/fruitspec-lab/PycharmProjects/foliage/counter/T_32"
#     rgb_folder = "/home/fruitspec-lab/PycharmProjects/foliage/counter/T_32"
#     tracker_full_results = pd.read_csv(
#         "/home/fruitspec-lab/PycharmProjects/foliage/counter/T_32/det/tracker.csv")
#     top_left_points = list(zip(tracker_full_results["x1"].values, tracker_full_results["y1"].values))
#     bottom_right_points = list(zip(tracker_full_results["x2"].values, tracker_full_results["y2"].values))
#     bboxed = list(zip(top_left_points, bottom_right_points))
#     tracker_full_results["bbox"] = bboxed
#     # zed_frames = [545, 548, 551, 554, 557, 560, 563, 566, 569, 572, 575, 578, 581] #, 584]
#     zed_frames = [550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562]  # , 584]
#     # jai_frames = [539, 541, 543, 545, 547, 549, 551, 553, 555, 557, 559, 561, 563]
#     jai_frames = list(range(1415, 1450))
#     zed_frames = jai_frames
#     sliced_data = [(1324, 1536),
#                    (1263, 1536),(1194, 1536),(1122, 1536),(1083, 1536),(993, 1536),(900, 1536),(837, 1536),(756, 1536),(690, 1536),(621, 1536),
#                    (564, 1536),(474, 1536),(372, 1536),(288, 1536),(147, 1482),(63, 1404),
#                    (0, 1353),(0, 1281), (0, 1197), (0, 1113), (0, 1047), (0,987 ), (0, 930), (0, 867), (0, 813),
#                    (0, 732), (0,696 ), (0, 618),(0, 525),(0, 462), (0, 342), (0, 255), (0,189 ), (0,84 )]
#     tree_images = {}
#     slicer_results = {}
#     tracker_results = {"cv": tracker_full_results["track_id"].nunique()}
#     all_coords = pd.read_csv(path.join(jai_folder, "jain_cors_in_zed_wtx.csv"))
#     shapes_x = []
#     shapes_y = []
#     for i, frame in enumerate(jai_frames):
#         pictures_dict = {}
#         pictures_dict["fsi"] = cv2.imread(path.join(jai_folder, f"channel_FSI_frame_{frame}.jpg"))[:, :, ::-1]
#         pictures_dict["rgb"] = cv2.imread(path.join(rgb_folder, f"channel_RGB_frame_{frame}.jpg"))[:, :, ::-1]
#         pictures_dict["zed"] = np.load(path.join(zed_folder, f"xyz_frame_{zed_frames[i]}.npy"))[:, :, [1, 0, 2]]
#         pictures_dict["zed_rgb"] = cv2.imread(path.join(zed_folder, f"frame_{zed_frames[i]}.jpg"))[:, :, ::-1]
#         ############################################################################################# draw lines
#         zed = pictures_dict["zed_rgb"].astype(np.uint8)
#         x_c = int(zed.shape[1]/2)
#         y_c = int(zed.shape[0] / 2)
#         zed = cv2.line(zed, (x_c,y_c-20), (x_c,y_c+20), (255, 0, 0),5)
#         zed = cv2.line(zed, (x_c - 20, y_c), (x_c + 20, y_c), (255, 0, 0), 5)
#         pictures_dict["zed_rgb"] = zed
#         ############################################################################################# draw lines
#         cur_coords = all_coords[all_coords["frame"] == frame].reset_index().astype(int)
#         # pictures_dict["zed"] = pictures_dict["zed"][cur_coords["y1"][0]:cur_coords["y2"][0],
#         #                        cur_coords["x1"][0]:cur_coords["x2"][0], :]
#         # pictures_dict["zed_rgb"] = pictures_dict["zed_rgb"][cur_coords["y1"][0]:cur_coords["y2"][0],
#         #                            cur_coords["x1"][0]:cur_coords["x2"][0], :]
#         # 80 - 1045
#         ########################################################################################## custom cords
#         pictures_dict["zed"] = pictures_dict["zed"][365:1600, 400:1365, :]
#         pictures_dict["zed_rgb"] = pictures_dict["zed_rgb"][365:1600, 400:1365, :]
#         ########################################################################################## custom cords
#         shapes_x.append(pictures_dict["zed"].shape[1])
#         shapes_y.append(pictures_dict["zed"].shape[0])
#         tree_images[str(frame)] = pictures_dict
#         tracker_results[str(frame)] = {row[1]["track_id"]: row[1]["bbox"] for row in
#                                        tracker_full_results[tracker_full_results["image_id"] == frame][
#                                            ["track_id", "bbox"]].iterrows()}
#     ## resizing all images(importnant)
#     max_x = np.max(shapes_x)
#     max_y = np.max(shapes_y)
#     for i, frame in enumerate(jai_frames):
#         zed_pic = tree_images[str(frame)]["zed"]
#         r_h, r_w = max_y / pictures_dict["fsi"].shape[0], max_x / pictures_dict["fsi"].shape[1]
#         tree_images[str(frame)]["fsi"] = cv2.resize(tree_images[str(frame)]["fsi"],
#                                                                   (max_x, max_y))
#         tree_images[str(frame)]["zed"] = cv2.resize(zed_pic,
#                                                                   (max_x, max_y))
#         tree_images[str(frame)]["zed_rgb"] = cv2.resize(tree_images[str(frame)]["zed_rgb"],
#                                                                   (max_x, max_y))
#         tree_images[str(frame)]["rgb"] = cv2.resize(tree_images[str(frame)]["rgb"],
#                                                                   (max_x, max_y))
#
#         slicer_results[str(frame)] = (int(sliced_data[i][0]*r_w),int(sliced_data[i][1]*r_w))
#         tracker_results[str(frame)] = {row[1]["track_id"]:
#                                            ((int(row[1]["bbox"][0][0]*r_w), int(row[1]["bbox"][0][1]*r_h)),
#                                             (int(row[1]["bbox"][1][0]*r_w), int(row[1]["bbox"][1][1]*r_h)))
#                                        for row in
#                                        tracker_full_results[tracker_full_results["image_id"] == frame][
#                                            ["track_id", "bbox"]].iterrows()}
#         # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
#         # ax1.imshow(pictures_dict["zed_rgb"])
#         # ax2.imshow(pictures_dict["rgb"])
#         # plt.show()
#     # plotting only
#     for frame in jai_frames:
#         frame = str(frame)
#         fsi = tree_images[frame]["fsi"]
#         zed = tree_images[frame]["zed_rgb"]
#         for id, track in tracker_results[frame].items():
#             fsi = cv2.rectangle(fsi, (track[0][0], track[0][1]), (track[1][0], track[1][1]), color=(255, 0, 0), thickness=2)
#         # plt.imshow(fsi)
#         # plt.show()
#         for id, track in tracker_results[frame].items():
#             zed = cv2.rectangle(zed, (track[0][0], track[0][1]), (track[1][0], track[1][1]), color=(255, 0, 0), thickness=2)
#         # plt.imshow(zed)
#         # plt.show()
#         # plot_2_imgs(zed, fsi, frame)
#     return tree_images, slicer_results, tracker_results


def cut_zed_in_jai(pictures_dict, cur_coords):
    """
    cut zed to the jai region
    :param pictures_dict: {"frame": {"fsi":fsi,"rgb":rgb,"zed":zed} for each frame}
    :param cur_coords: {"x1":((x1,y1),(x2,y2))}
    :return: pictures_dict with zed and zed_rgb cut to the jai region
    """
    x1 = max(cur_coords["x1"][0], 0)
    x2 = min(cur_coords["x2"][0], pictures_dict["zed"].shape[1])
    y1 = max(cur_coords["y1"][0], 0)
    y2 = min(cur_coords["y2"][0], pictures_dict["zed"].shape[0])
    # x1, x2 = 145, 1045
    # y1, y2 = 370, 1597
    pictures_dict["zed"] = pictures_dict["zed"][y1:y2, x1:x2, :]
    pictures_dict["zed_rgb"] = pictures_dict["zed_rgb"][y1:y2, x1:x2, :]
    return pictures_dict


def update_tree_images_tracker_results(tree_images, tracker_results, tree_folder, frame, zed_shift, all_coords,
                                       tracker_full_results, cameras={}):
    """
    This function updates the tree images and tracker results for a given frame.
    :param tree_images: a dictionary that holds the tree images for each frame
    :param tracker_results: a dictionary that holds the tracker results for each frame
    :param tree_folder: the tree folder path
    :param frame: the current frame
    :param zed_shift: the shift between the zed frames and the other frames
    :param all_coords: all coordinates of jai inside zed
    :param tracker_full_results: full tracker results
    :param cameras: dictionary of cameras (video wrapper objects), if empty will use old pipe
    :return: updated tree_images, tracker_results, shapes_x, shapes_y, a boolean indicating success or failure
    """
    pictures_dict = {}
    if not cameras:
        fsi_path = path.join(tree_folder, f"channel_FSI_frame_{frame}.jpg")
        zed_path = path.join(tree_folder, f"frame_{int(frame) + zed_shift}.jpg")
        rgb_path = path.join(tree_folder, f"channel_RGB_frame_{frame}.jpg")
        zed_npy = path.join(tree_folder, f"xyz_frame_{int(frame) + zed_shift}.npy")
        if not (os.path.exists(fsi_path) and os.path.exists(zed_path)):
            return tree_images, tracker_results, False
        pictures_dict["fsi"] = cv2.imread(fsi_path)[:, :, ::-1]
        pictures_dict["rgb"] = cv2.imread(rgb_path)[:, :, ::-1]
        pictures_dict["zed"] = np.load(zed_npy)[:, :, [1, 0, 2]].astype(np.float32)
        pictures_dict["zed_rgb"] = cv2.imread(zed_path)[:, :, ::-1]
    else:
        zed_frame, depth, point_cloud = cameras["zed_cam"].get_zed(int(frame) + zed_shift)
        pictures_dict["zed"] = point_cloud[:, :, [1, 0, 2]].astype(np.float32)
        pictures_dict["zed_rgb"] = zed_frame[:, :, ::-1]
        fsi_ret, pictures_dict["fsi"] = cameras["jai_cam"].get_frame(int(frame))
        pictures_dict["fsi"] = pictures_dict["fsi"][:, :, ::-1]
        rgb_ret, pictures_dict["rgb"] = cameras["rgb_jai_cam"].get_frame(int(frame))
        if not np.all([fsi_ret, rgb_ret, not zed_frame is None]):
            return tree_images, tracker_results, False
    pictures_dict["zed"] = remove_high_blues(pictures_dict["zed"], pictures_dict["zed_rgb"][:, :, 2])
    pictures_dict["zed"][pictures_dict["zed_rgb"][:, :, 0] > pictures_dict["zed_rgb"][:, :, 1] + 10] = np.nan
    cur_coords = all_coords[all_coords["frame"] == frame].reset_index()
    if cur_coords.shape[0] == 0:
        return tree_images, tracker_results, False
    if np.isfinite(cur_coords["tx"][0]):
        pictures_dict = cut_zed_in_jai(pictures_dict, cur_coords.astype(int))
    else:
        return tree_images, tracker_results, False
    tree_images[frame] = pictures_dict
    tracker_results[frame] = {row[1]["track_id"]: row[1]["bbox"] for row in
                                   tracker_full_results[tracker_full_results["frame_id"] == frame][
                                       ["track_id", "bbox"]].iterrows()}
    return tree_images, tracker_results, True


def get_slicer_data(slice_path, max_w, tree_id=-1):
    """
    reads slicer data
    :param slice_path: path to slices.csv
    :param max_w: max width of picture
    :param tree_id: if -1 will use old pipe, else will use new pipe
    :return: dataframe contaiting the slices data
    """
    if isinstance(slice_path, str):
        sliced_data = pd.read_csv(slice_path)
    else:
        sliced_data = slice_path
    if tree_id != -1:
        sliced_data = sliced_data[sliced_data["tree_id"] == tree_id]
    if "starts" in sliced_data.keys():
        sliced_data["start"] = sliced_data["starts"]
    if "ends" in sliced_data.keys():
        sliced_data["end"] = sliced_data["ends"]
    sliced_data["start"].replace(-1, 0, inplace=True)
    sliced_data["end"].replace(-1, max_w, inplace=True)
    sliced_data = dict(zip(sliced_data["frame_id"].apply(str), tuple(zip(sliced_data["start"], sliced_data["end"]))))
    return sliced_data


def read_preprocess_results(tracker_path, tree_id=-1):
    """
    read the tracker results and adjusts them to relevent names
    :param tracker_path: path to tracker csv
    :param tree_id: the id of the tree we want the results for, if tree_id=-1 will treat tracker_path
                    as path to tree tracks instead of row tracks
    :return: dict of tracker results
    """
    tracker_full_results = pd.read_csv(tracker_path)
    if tree_id != -1:
        tracker_full_results = tracker_full_results[tracker_full_results["tree_id"] == tree_id]
    tracker_full_results["bbox"] = get_trakcer_bboxes(tracker_full_results)
    if "class_pred" in tracker_full_results:
        tracker_full_results.rename({"class_pred": "track_id"}, axis=1, inplace=True)
    if "frame" in tracker_full_results:
        tracker_full_results.rename({"frame": "image_id"}, axis=1, inplace=True)
    return tracker_full_results


def read_sort_jai_frames(tree_folder, tree_id=-1):
    """
    gets the frames number and sorts them
    :param tree_folder: folder of the tree
    :param tree_id: if tree_id = -1 will use original pipe, else will use tree_id to filter relevent tree for feature
                    extraction pipe
    :return: the frames, total number of frames
    """
    if tree_id == -1:
        jai_frames = [file.split(".")[0].split("_")[-1] for file in os.listdir(tree_folder)
                  if "jpg" in file and "FSI" in file]
    else:
        slices = pd.read_csv(os.path.join(tree_folder, "slices.csv"))
        slices_sub = slices[slices["tree_id"] == tree_id]
        jai_frames = slices_sub["frame_id"].values.tolist()
    jai_frames.sort(key=lambda x: int(x))
    n_frames = len(jai_frames)
    if n_frames > 75:
        jai_frames = jai_frames[::2]
        n_frames = len(jai_frames)
    return jai_frames, n_frames


def resize_dicts(tree_images,slicer_results, tracker_results, slice_path, frame,max_y,max_x,max_z,tracker_full_results,
                 tree_id=-1):
    """
    This function resizes the tree images, slicer results and tracker results for a given frame to a fixed size.

    Inputs:
    tree_images: a dictionary containing the images (fsi, zed, rgb) for each frame
    slicer_results: a dictionary containing the slicer results for each frame
    tracker_results: a dictionary containing the tracker results for each frame
    slice_path: path to the slicer results file
    frame: the current frame being processed
    max_y: the desired height of the resized images
    max_x: the desired width of the resized images
    max_z: the maximum depth value for the zed image
    tracker_full_results: dataframe containing all the results from the tracker
    tree_id: if -1 will use old pipe, else will use new pipe

    Outputs:
    tracker_results: the updated tracker results dictionary
    slicer_results: the updated slicer results dictionary
    tree_images: the updated tree images dictionary
    """
    jai_h, jai_w = tree_images[frame]["fsi"].shape[:2]
    sliced_data = get_slicer_data(slice_path, jai_w, tree_id)
    r_h, r_w = max_y / jai_h, max_x / jai_w
    zed_pic = tree_images[frame]["zed"]
    if np.prod(tree_images[frame]["zed"].shape) == 0:
        tree_images.pop(frame)
        tracker_results.pop(frame)
        return tracker_results, slicer_results, tree_images
    tree_images[frame]["fsi"] = cv2.resize(tree_images[frame]["fsi"], (max_x, max_y))
    tree_images[frame]["zed"] = cv2.resize(zed_pic, (max_x, max_y))
    tree_images[frame]["zed_rgb"] = cv2.resize(tree_images[frame]["zed_rgb"], (max_x, max_y))
    tree_images[frame]["rgb"] = cv2.resize(tree_images[frame]["rgb"], (max_x, max_y))
    if max_z > 0:
        depth_mask = tree_images[frame]["zed"][:, :, 2] > max_z
        tree_images[frame]["fsi"][depth_mask] = 0
        tree_images[frame]["zed_rgb"][depth_mask] = 0
        tree_images[frame]["rgb"][depth_mask] = 0
        tree_images[frame]["zed"][depth_mask] = np.nan
    slicer_results[frame] = (int(sliced_data[frame][0] * r_w), int(sliced_data[frame][1] * r_w))
    if tree_id == -1:
        tracker_subset = tracker_full_results[tracker_full_results["image_id"] == int(frame)]
    else:
        tracker_subset = tracker_full_results[tracker_full_results["frame_id"] == int(frame)]
    tracker_results[frame] = {row[1]["track_id"]: row_resized_bbox(row, r_w, r_h)
                              for row in tracker_subset[["track_id", "bbox"]].iterrows()}
    return tracker_results, slicer_results, tree_images


def preprocess_debug(keys, tree_images, tracker_results, slicer_results, max_y, prefix=""):
    """
    debugging function for preprocessing
    :param keys: the frames to iterate on
    :param tree_images: tree images dict
    :param tracker_results: tracker results dict
    :param slicer_results: slicer resultss dict
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

        for id, track in tracker_results[frame].items():
            fsi = cv2.rectangle(fsi, (track[0][0], track[0][1]), (track[1][0], track[1][1]), color=(255, 0, 0),
                                thickness=2)
        fsi = cv2.line(fsi, (slicer_results[frame][0], 0), (slicer_results[frame][0], max_y), color=(255, 0, 0),
                       thickness=2)
        fsi = cv2.line(fsi, (slicer_results[frame][1], 0), (slicer_results[frame][1], max_y), color=(255, 0, 0),
                       thickness=2)
        # plt.imshow(fsi)
        # plt.show()
        for id, track in tracker_results[frame].items():
            zed = cv2.rectangle(zed, (track[0][0], track[0][1]), (track[1][0], track[1][1]), color=(255, 0, 0),
                                thickness=2)
        zed = cv2.line(zed, (slicer_results[frame][0], 0), (slicer_results[frame][0], max_y), color=(255, 0, 0),
                       thickness=2)
        zed = cv2.line(zed, (slicer_results[frame][1], 0), (slicer_results[frame][1], max_y), color=(255, 0, 0),
                       thickness=2)
        # plt.imshow(zed)
        # plt.show()
        saving_folder = "/media/fruitspec-lab/easystore/testing_playground"
        if os.path.exists(saving_folder):
            save_to = os.path.join(saving_folder, f"{prefix}_{frame}_debug_fe.jpg")
            plot_2_imgs(zed, fsi, frame, save_to=save_to, save_only=True)
        else:
            plot_2_imgs(zed, fsi, frame)
        # plot_2_imgs(xyz[:,:,2], zed, frame)


def preprocess_tree(tree_folder, tracker_path, slice_path, zed_shift=0, max_x=None, max_y=None, max_z=0, debug=False):
    """
    preprocess the tree from raw data to somthing we can work with later
    :param tree_folder: folder of the tree
    :param tracker_path: path for tracker csv
    :param slice_path: path for slicer csv
    :param zed_shift: the shift between the cameras
    :param max_x: width to resize to
    :param max_y: height to resize to
    :param max_z: maxsium depth allowed, anything farther will be turned to nan
    :param debug: flag for preforming debugging
    :return: tree_images, slicer_results, tracker_results - processed dictionaries
    """
    tracker_full_results = read_preprocess_results(tracker_path)
    jai_frames, n_frames = read_sort_jai_frames(tree_folder)
    tree_images, slicer_results, tracker_results = {}, {}, {}
    shapes_x, shapes_y, to_drop = np.array([]), np.array([]), np.array([])
    all_coords = pd.read_csv(path.join(tree_folder, "jai_cors_in_zed.csv"))
    all_coords["frame"] = all_coords["frame"].apply(str)
    if len(all_coords) == 0:
        return {}, {}, {}
    for i, frame in enumerate(jai_frames):
        print(f"\r{i + 1}/{n_frames} ({(i + 1) / (n_frames) * 100: .2f}%) pictures", end="")
        try:
            tree_images, tracker_results, ret = update_tree_images_tracker_results(tree_images,
                                                                                             tracker_results,
                                                                                             tree_folder, frame,
                                                                                             zed_shift, all_coords,
                                                                                             tracker_full_results)
        except:
            print("loading problem: ", frame)
            ret = False
        if not ret:
            to_drop = np.append(to_drop, frame)
        else:
            tracker_results, slicer_results, tree_images = resize_dicts(tree_images, slicer_results, tracker_results,
                                                                        slice_path, frame, max_y, max_x,
                                                                        max_z, tracker_full_results)
    print("done reading")
    for frame in to_drop:
        jai_frames.remove(frame)
    if len(tree_images.keys()) == 0:
        return {}, {}, {}
    keys = list(tree_images.keys())
    if debug:
        preprocess_debug(keys, tree_images, tracker_results, slicer_results, max_y)
    print("done preprocess")
    return tree_images, slicer_results, tracker_results



def preprocess_tree_fe_pipe(row_path, tree_id, zed_shift=0, max_x=None,
                            max_y=None, max_z=0, debug=False,
                            cameras={"zed_cam": None, "rgb_jai_cam": None, "jai_cam": None}):
    """
    preprocess the tree from raw data to somthing we can work with later, this function is the same as preprocess_tree
    but works for feature_extraction_pipeline
    :param cameras: dictionary of cameras (video wrapper objects)
    :param row_path: folder of the row
    :param tree_id: tree id
    :param zed_shift: the shift between the cameras
    :param max_x: width to resize to
    :param max_y: height to resize to
    :param max_z: maxsium depth allowed, anything farther will be turned to nan
    :param debug: flag for preforming debugging
    :return: tree_images, slicer_results, tracker_results - processed dictionaries
    """
    tracker_full_results = read_preprocess_results(path.join(row_path, "trees_sliced_track.csv"), tree_id)
    jai_frames, n_frames = read_sort_jai_frames(row_path, tree_id)
    tree_images, slicer_results, tracker_results = {}, {}, {}
    shapes_x, shapes_y, to_drop = [], [], []
    slice_path = path.join(row_path, "slices.csv")
    all_coords = pd.read_csv(path.join(row_path, "jai_cors_in_zed.csv"))
    all_coords["frame"] = all_coords["frame"].apply(str)
    jai_zed = load_json(path.join(row_path, "jai_zed.json"))
    if len(all_coords) == 0:
        return {}, {}, {}
    for i, frame in enumerate(jai_frames):
        print(f"\r{i + 1}/{n_frames} ({(i + 1) / (n_frames) * 100: .2f}%) pictures", end="")
        try:
            tree_images, tracker_results, ret = update_tree_images_tracker_results(tree_images,
                                                                                             tracker_results,
                                                                                             row_path, str(frame),
                                                                                             jai_zed[frame]-frame,
                                                                                                       all_coords,
                                                                                             tracker_full_results,
                                                                                                       cameras)
        except:
            print("loading problem: ", frame)
            ret = False
        if not ret:
            to_drop = np.append(to_drop, frame)
        else:
            tracker_results, slicer_results, tree_images = resize_dicts(tree_images, slicer_results, tracker_results,
                                                                        slice_path, str(frame), max_y, max_x,
                                                                        max_z, tracker_full_results, tree_id)
    print("done reading")
    for camera in cameras.values():
        camera.close()
    for frame in to_drop:
        jai_frames.remove(frame)
    if len(tree_images.keys()) == 0:
        return {}, {}, {}
    keys = list(tree_images.keys())
    if debug:
        prefix = f"{path.basename(path.dirname(row_path))}_{path.basename(row_path)}"
        preprocess_debug(keys, tree_images, tracker_results, slicer_results, max_y, prefix)
    print("done preprocess")
    return tree_images, slicer_results, tracker_results


def create_row_features(path_to_row, zed_shift=0, max_x=600, max_y=900, save_csv=True, block_name="", max_z=0,
                        save_name="row_features.csv", debug=False):
    """
    extracts features for each row
    :param path_to_row: path to row folder
    :param zed_shift: shift between zed and jai
    :param max_x: width for resizeing
    :param max_y: heihgt for resiing
    :param save_csv: flag for saving csv
    :param block_name: the name of the block we process
    :param max_z: maxisum depth
    :param save_name: name for saving the results
    :param debug: flag for debugging
    :return: dataframe with the features for each tree in the row
    """
    df = pd.DataFrame({})
    row = os.path.basename(os.path.dirname(path_to_row))
    trees = [file for file in os.listdir(path_to_row) if os.path.isdir(os.path.join(path_to_row, file))]
    trees.sort(key=lambda x: int(x[1:]))
    # start_ind = 1 if int(row[1:]) % 2 == 0 else 2
    for tree in trees:
        s_t = time.time()
        tree_folder = os.path.join(path_to_row, tree)
        if not os.path.isdir(tree_folder):
            continue
        tracker_path = os.path.join(tree_folder, "tracker.csv")
        slice_path = os.path.join(tree_folder, "slices.csv")
        tree_images, slicer_results, tracker_results = preprocess_tree(tree_folder, tracker_path,
                                                                       slice_path, zed_shift=zed_shift, debug=debug,
                                                                       max_x=max_x, max_y=max_y, max_z=max_z)
        if len(tree_images) == 0:
            df = df.append({"name": f"{row}_{tree}", "block_name": block_name}, ignore_index=True)
            continue
        tree_features = extract_features_for_tree(tree_images, slicer_results, tracker_results, f"{row}_{tree}", max_z)
        tree_features["block_name"] = block_name
        df = df.append(tree_features, ignore_index=True)
        print("total tree process time: ", time.time() - s_t)
    if save_csv:
        df.to_csv(os.path.join(path_to_row, save_name))
    return df


def create_row_features_fe_pipe(path_to_row, zed_shift=0, max_x=600, max_y=900, save_csv=True, block_name="", max_z=0,
                                save_name="row_features.csv", debug=False,
                                cameras={"zed_cam": None, "rgb_jai_cam": None, "jai_cam": None}):
    """
    extracts features for each row
    :param path_to_row: path to row folder
    :param zed_shift: shift between zed and jai
    :param max_x: width for resizeing
    :param max_y: heihgt for resiing
    :param save_csv: flag for saving csv
    :param block_name: the name of the block we process
    :param max_z: maxisum depth
    :param save_name: name for saving the results
    :param debug: flag for debugging
    :return: dataframe with the features for each tree in the row
    """
    print(f"genereating features for block:{block_name}, row:{path_to_row.split('/')[-1]}")

    df = pd.DataFrame({})
    if block_name == "":
        block_name = os.path.basename(os.path.dirname(path_to_row))
    row = os.path.basename(path_to_row)
    slices = pd.read_csv(os.path.join(path_to_row, "slices.csv"))
    trees = slices["tree_id"].unique()
    # start_ind = 1 if int(row[1:]) % 2 == 0 else 2
    for tree_id in tqdm(trees):
        s_t = time.time()
        tree_images, slicer_results, tracker_results = preprocess_tree_fe_pipe(path_to_row, tree_id, zed_shift, max_x,
                                                                                max_y, max_z=max_z, debug=debug,
                                                                                cameras=cameras)
        if len(tree_images) == 0:
            df = df.append({"name": f"{row}_{tree_id}", "block_name": block_name}, ignore_index=True)
            continue
        tree_features = extract_features_for_tree(tree_images, slicer_results, tracker_results,
                                                  f"{row}_T{tree_id}", max_z)
        tree_features["block_name"] = block_name
        total_time = time.time() - s_t
        tree_features["full_process_time"] = total_time
        df = df.append(tree_features, ignore_index=True)
        print("total tree process time: ", total_time)
    if save_csv:
        df.to_csv(os.path.join(path_to_row, save_name))
    return df


def create_plot_features(plot_path, zed_shift=0, max_x=600, max_y=900, save_csv=True, block_name="", skip_rows=[],
                         skip_no_load=[], max_z=0, suffix="", log={"skip_rows": []}, customer_name="", debug=False):
    """

    extracts features for the whole plot

    :param plot_path: path to plot folder
    :param zed_shift: shift between zed and jai
    :param max_x: width for resizeing
    :param max_y: heihgt for resiing
    :param save_csv: flag for saving csv
    :param block_name: the name of the block we process
    :param skip_rows: which rows to skip (will take last feautres dataframe saved for them if exsists)
    :param skip_no_load: which rows to skip completly (without taking their feautres)
    :param max_z: maxisum depth
    :param suffix: ending for saving file
    :param log: log file for when we last stopped in case we crashed
    :param customer_name: name of customer
    :param debug: flag for debugging
    :return: dataframe with all the features
    """
    df = pd.DataFrame({})
    log_path = os.path.join(plot_path, "plot_log.json")
    skip_rows = skip_rows+log["skip_rows"]
    for row in os.listdir(plot_path):
        row_path = os.path.join(plot_path, row, "trees")
        if row in skip_no_load:
            continue
        if row in skip_rows:
            row_features_path = os.path.join(row_path, "row_features.csv")
            if os.path.exists(row_features_path):
                df = pd.concat([df, pd.read_csv(row_features_path)])
            continue
        if os.path.isdir(row_path):
            df = pd.concat([df, create_row_features(row_path, zed_shift, max_x, max_y, save_csv,
                                                    block_name, max_z, debug=debug)])
            log = update_save_log(log_path, log, {"skip_rows": skip_rows + [row]})
    df["customer_name"] = customer_name
    if save_csv:
        df.to_csv(os.path.join(plot_path, f"plot_features{suffix}.csv"))
    return df


if __name__ == '__main__':
    output_folder = "/media/fruitspec-lab/easystore/testing_playground"
    zed_shift = 0
    block_name = "test"
    max_z = 8
    row_path = "/media/fruitspec-lab/cam175/DEWAGD/190123/DWDBLE33/R35A"
    row_paths = ["/media/fruitspec-lab/cam175/DEWAGD/190123/DWDBLE33/R11A",
                 "/media/fruitspec-lab/cam175/DEWAGD/190123/DWDBLE33/R23A",
                 "/media/fruitspec-lab/cam175/DEWAGD/190123/DWDBLE33/R35A",
                 "/media/fruitspec-lab/cam175/DEWAGD/190123/DWDBLE33/R47A",
                 "/media/fruitspec-lab/cam175/DEWAGD/190123/DWDBLE33/R59A"]
    for row_path in row_paths:
        row = os.path.basename(row_path)
        side = 1
        row_dict = {"zed_novie_path": os.path.join(row_path, f"ZED_{side}.svo"),
                    "jai_novie_path": os.path.join(row_path, f"Result_FSI_{side}.mkv"),
                    "rgb_jai_novie_path": os.path.join(row_path, f"Result_RGB_{side}.mkv"),
                    "slice_data_path": os.path.join(row_path, f"ZED_{side}_slice_data_{row}.json"),
                    "jai_slice_data_path": os.path.join(row_path, f"Result_FSI_{side}_slice_data_{row}.json"),}
        jai_movie_path = row_dict["jai_novie_path"]
        rgb_movie_path = row_dict["rgb_jai_novie_path"]
        zed_movie_path = row_dict["zed_novie_path"]
        zed_cam = video_wrapper(zed_movie_path, 2, 0, 10)
        rgb_jai_cam = video_wrapper(rgb_movie_path, 1)
        jai_cam = video_wrapper(jai_movie_path, 1)
        create_row_features_fe_pipe(row_path, zed_shift=zed_shift, max_x=600, max_y=900,
                                    save_csv=True, block_name=block_name, max_z=max_z, debug=True, save_name="test.csv",
                                    cameras={"zed_cam": zed_cam, "rgb_jai_cam": rgb_jai_cam, "jai_cam": jai_cam})
        zed_cam.close()
        jai_cam.close()
        rgb_jai_cam.close()



    path_to_plot = "/media/fruitspec-lab/cam175/DEWAGB_test/190123/DWDBLE33"
    path_to_row = "/media/fruitspec-lab/easystore/JAIZED_CaraCara_301122/R2/trees"
    row_path = "/media/fruitspec-lab/easystore/JAIZED_CaraCara_301122/R2"
    # create_row_features("/media/fruitspec-lab/easystore/nir_long_rows/trees", zed_shift=0, max_x=600, max_y=900,
    #                     save_csv=True, block_name="", max_z=5, save_name="row_features.csv")
    # create_row_features(path_to_row, save_name="row_features_trans.csv")
    # df = create_plot_features(path_to_plot, block_name="CaraCaraNir", skip_rows=["R10", "R11", "R2", "R3", "R4", "R5", "R6", "R7", "R8"])
    df = create_plot_features(path_to_plot, block_name="DEWAGB", max_z=10, skip_rows=[], suffix="_testing", debug=True)
    # df = create_plot_features(path_to_plot, block_name="CaraCaraNir", max_z=5,
    #                           skip_rows=[f"R{i}" for i in list(range(2, 8)) + [10, 11]])
    tree = "T4"
    tree_folder = os.path.join(path_to_row, tree)
    # tree_folder = "/media/fruitspec-lab/easystore/nir_long_rows/trees/T10"
    tracker_path = os.path.join(tree_folder, "tracker.csv")
    slice_path = os.path.join(tree_folder, "slices.csv")
    tree_images, slicer_results, tracker_results = preprocess_tree(tree_folder, tracker_path,
                                                                   slice_path, zed_shift=0,
                                                                   max_x=600, max_y=900, max_z=5)
    # tree_images, slicer_results, tracker_results = read_images_small()
    print("loaded pictures")
    extract_features_for_tree(tree_images, slicer_results, tracker_results, tree, max_z=5)
    print("finito")