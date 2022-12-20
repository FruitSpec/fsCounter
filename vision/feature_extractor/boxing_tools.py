import numpy as np
import cv2
from skimage.util import img_as_ubyte
from stat_tools import iqr_trim, quantile_trim
import image_processing as img_pro
from tree_size_tools import stable_euclid_dist
from cupyx.scipy import ndimage
import cupy as cp

global fsi_size
global nir_channel
global ir975_channel
global blue_channel
fsi_size = (2048, 1536)
nir_channel, ir975_channel, blue_channel = 0, 1, 2


def make_bbox_pic(img, boxes):
    """
    :param img: image to use
    :param boxes: boxes to use
    :return: make a picture of only the bboxes parts
    """
    new_img = np.zeros(img.shape, dtype=int)
    if len(boxes) == 0:
        return new_img
    for i, box in boxes.items():
        t, b, l, r = box[0][1], box[1][1], box[0][0], box[1][0]
        new_img[t:b, l:r, :] = img[t:b, l:r, :]
    return new_img


def get_mask_corners(mask, top_bot=False):
    """
    :param mask: mask to get corners for
    :param top_bot: flag to only return top and bottom values
    :return: top, left, bottom, right points of mask
    """
    if isinstance(mask, type(None)):
        return 0, 0, 0, 0
    top, left, bottom, right = 0, 0, mask.shape[0], mask.shape[1]
    x_sums = np.sum(mask, axis=1)
    x_sums_non_zero = np.where(x_sums != 0)[0]
    if len(x_sums_non_zero) > 0:
        top = np.min(x_sums_non_zero)
        bottom = np.max(x_sums_non_zero)
    if top_bot:
        return top, bottom
    y_sums = np.sum(mask, axis=0)
    y_sums_non_zero = np.where(y_sums != 0)[0]
    if len(y_sums_non_zero) > 0:
        left = np.min(y_sums_non_zero)
        right = np.max(y_sums_non_zero)
    return top, left, bottom, right


def get_global_top_bottom(masks, min_factor=0.2):
    tops, bottoms = [], []
    for mask in masks:
        top, bottom = get_mask_corners(mask, top_bot=True)
        tops.append(top)
        bottoms.append(bottom)
    min_y, max_y = int(mask.shape[0]*min_factor), int(mask.shape[0]*(1-min_factor))
    return min(np.min(tops), min_y), max(np.max(bottoms), max_y)


def x_start_end_resizing(x_start, x_end, i=0, n_min_frames=1, cut_val=0.1):
    x_range = x_end-x_start
    cut_size = int(x_range * cut_val)
    if n_min_frames == 1:
        x_start += cut_size
        x_end -= cut_size
    elif i == 0:
        x_start += cut_size
    elif i == n_min_frames-1:
        x_end -= cut_size
    else:
        cut_size = int(cut_size/2)
        x_start += cut_size
        x_end -= cut_size
    return x_start, x_end


def slice_outside_trees(pictures, slicer_results, frame_number, reduce_size = False, mask = None, y_ranges = (),
                        i=0, n_min_frames=1, cut_val=0.1):
    """
    :param pictures: list of pictures to edit
    :param slicer_results: {"frame": (x_start,x_end) for each frame}
    :param frame_number: number of frame to take images for
    :return: vegetation_indexes for tree
    """
    x_start, x_end = slicer_results[frame_number]
    x_start, x_end = x_start_end_resizing(x_start, x_end, i=i, n_min_frames=n_min_frames, cut_val=cut_val)
    if not isinstance(mask, type(None)):
        top, left, bottom, right = get_mask_corners(mask)
        if left == 0:
            x_start = max(x_start, right)
        if right == pictures[0].shape[1]:
            x_end = min(x_end, left)
    for i, pic in enumerate(pictures):
        if reduce_size:
            if len(y_ranges) == 2:
                pic = pic[y_ranges[0]:y_ranges[1], x_start: x_end]
            else:
                pic = pic[:, x_start: x_end]
        elif not isinstance(mask, type(None)):
            pic = pic[top:bottom, left: right]
        else:
            pic[:, : x_start] = np.nan
            pic[:, x_end:] = np.nan
        pictures[i] = pic
    return pictures


def get_w_h_ratio(boxes):
    """
    :param boxes: boxes for tree
    :return: ratio of width and height
    """
    if len(boxes) == 0:
        return []
    x_0 = np.fromiter((box[0][0] for box in boxes),int)
    x_1 = np.fromiter((box[1][0] for box in boxes),int)
    y_0 = np.fromiter((box[0][1] for box in boxes),int)
    y_1 = np.fromiter((box[1][1] for box in boxes),int)
    w_h_ratio = np.fromiter((x if x < 1 else 1/x for x in np.abs(x_1-x_0) / np.abs(y_0-y_1)),float)
    return w_h_ratio


def get_intensity(fsi, rgb, box, use_cuda=False):
    cropped_fsi, cropped_rgb = orange_cropper(fsi, rgb, box=box)
    if use_cuda:
        cropped_fsi = cp.asnumpy(ndimage.median_filter(cp.array(cropped_fsi), 5))
        cropped_rgb = cp.asnumpy(ndimage.median_filter(cp.array(cropped_rgb), 5))
    else:
        cropped_fsi = med_blur(cropped_fsi)
        cropped_rgb = med_blur(cropped_rgb)
    hls = cv2.cvtColor(img_as_ubyte(cropped_rgb), cv2.COLOR_RGB2HLS)
    # 95% of time
    ndri = img_pro.make_ndri(cropped_fsi)
    binary_ndri = img_pro.ndvi_to_binary(ndri, 0.05)
    relevant_v = hls[:, :, 1][binary_ndri.astype(bool)]
    rel_v_flat = relevant_v.flatten()
    rel_v_flat_trim = quantile_trim(rel_v_flat, keep_size=False)
    if len(rel_v_flat_trim) == 0:
        return np.nan
    return np.nanmedian(rel_v_flat_trim)


def normalize_intensity(intensity, path_to_tree_folder):
    if len(intensity) == 0:
        print(intensity)
        print(f"something wrong with intensity: {path_to_tree_folder}")
        intensity = [0]
    min_intens = np.nanmin(intensity)
    max_intens = np.nanmax(intensity)
    if max_intens == min_intens:
        return [0]
    intensity = (intensity - min_intens) / (max_intens - min_intens)
    return intensity


def med_blur(img, ksize=5):
    """
    :param img: iamge to blur
    :param ksize: kernal to use
    :return: blurred image
    """
    if img.dtype != "uint8":
        img = img.astype(np.uint8)
    blurred_image = cv2.medianBlur(img, ksize)
    return blurred_image


def orange_cropper(img_fsi, img_rgb, box):
    """
    img_fsi,img_rgb: images or paths to image
    box: box of orange foramtted(T,L,B,R)
    """
    t, b, l, r = get_box_corners(box)
    return img_fsi[t:b, l:r], img_rgb[t:b, l:r]


def med_x_med_y(x_0, y_0, x_1, y_1):
    """
    return the middle x and middle y of the given points, params are self expiantory
    :param x_0:
    :param y_0:
    :param x_1:
    :param y_1:
    :return:
    """
    med_y = int((y_0 + y_1) / 2)
    med_x = int((x_1 + x_0) / 2)
    return med_x, med_y


def get_box_corners(box):
    t, b, l, r = box[0][1], box[1][1], box[0][0], box[1][0]
    return t, b, l, r


def min_max_finites(point_cloud, bbox):
    """
    calculates the minimum and maximum finite indexes in the bounding box (in order to counter points with no data)
    :param point_cloud: point_cloud map produced by zed
    :param bbox: the bounding box
    :return: minimum and maximum finite x, minimum and maximum finite y, middle x, middle y
    """
    y_0, y_1, x_0, x_1 = get_box_corners(bbox)
    range_x = np.arange(x_0, x_1+1)
    range_y = np.arange(y_0, y_1+1)
    med_x, med_y = med_x_med_y(x_0, y_0, x_1, y_1)
    finite_y = np.where(np.isfinite(iqr_trim(point_cloud[y_0:y_1,med_x][:,0])))[0]
    finite_x = np.where(np.isfinite(iqr_trim(point_cloud[med_y, x_0:x_1][:, 1])))[0]
    min_x = range_x[np.min(finite_x)] if len(finite_x) > 0 else 0
    max_x = range_x[np.max(finite_x)] if len(finite_x) > 0 else 1
    min_y = range_y[np.min(finite_y)] if len(finite_y) > 0 else 0
    max_y = range_y[np.max(finite_y)] if len(finite_y) > 0 else 1
    return min_x, max_x, min_y, max_y, med_x, med_y


def get_box_center(box):
    t, b, l, r = get_box_corners(box)
    return med_x_med_y(l, t, r, b)


def filter_outside_tree_boxes(tracker_results, slicer_results):
    for frame in tracker_results.keys():
        if frame == "cv":
            continue
        x_start, x_end = slicer_results[frame]
        x_0 = np.array([box[0][0] for box in tracker_results[frame].values()])
        x_1 = np.array([box[1][0] for box in tracker_results[frame].values()])
        for id in np.array(list(tracker_results[frame].keys()))[np.all([x_0 > x_start, x_1< x_end],axis = 0) == False]:
            tracker_results[frame].pop(id)
    tracker_results["cv"] = len({id for frame in set(tracker_results.keys())-{"cv"} for id in tracker_results[frame].keys()})
    return tracker_results


def cut_center_of_box(image, box, nir=None, swir_975=None):
    t, b, l, r = get_box_corners(box)
    y_max, x_max = image.shape[:2]
    if not isinstance(nir, type(None)):
        x0, x1, y0, y1 = max(0, l), min(x_max, r), max(0, t), min(y_max, b)
        cut_box = image[y0: y1, x0: x1]
        nir_cut = nir[y0: y1, x0: x1]
        swir_cut = swir_975[y0: y1, x0: x1]
        binary_box = img_pro.ndvi_to_binary(img_pro.make_ndri(nir=nir_cut, swir_975=swir_cut), 0)
        cut_box[binary_box == 1] = np.nan
        if np.mean(binary_box) < 0.5:
            cut_box = np.full(image.shape, np.nan)
    else:
        h_025 = int((b-t)*0.25)
        w_025 = int((r-l)*0.25)
        cut_box = image[max(0, t+h_025):min(y_max, b-h_025), max(0, l+w_025):min(x_max, r-w_025)]
    return cut_box


def xyz_center_of_box(image, box, nir=None, swir_975=None):
    cut_box = cut_center_of_box(image, box, nir=nir, swir_975=swir_975)
    x_median = np.nanmedian(cut_box[:, :, 0])
    y_median = np.nanmedian(cut_box[:, :, 1])
    z_median = np.nanmedian(cut_box[:, :, 2])
    return x_median, y_median, z_median


def get_new_old_boxes(xyz_point_cloud, fruit_3d_space, boxes, nir=None, swir_975=None):
    new_boxes, old_boxes = {}, {}
    boxes_w, boxes_h = np.array([]), np.array([])
    for id, box in boxes.items():
        if id not in fruit_3d_space.keys():
            x_center, y_center, z_center = xyz_center_of_box(xyz_point_cloud, box, nir=nir, swir_975=swir_975)
            if np.isnan(z_center):
                continue
            new_boxes[id] = (x_center, y_center, z_center)
            width, height = stable_euclid_dist(xyz_point_cloud, box, buffer=1)
            boxes_w = np.append(boxes_w, width)
            boxes_h = np.append(boxes_h, height)
        else:
            x_center, y_center, z_center = xyz_center_of_box(xyz_point_cloud, box, nir=nir, swir_975=swir_975)
            old_boxes[id] = (x_center, y_center, z_center)
    return new_boxes, old_boxes, boxes_w, boxes_h


def project_boxes_to_fruit_space(fruit_3d_space, old_boxes, new_boxes, n_closest=5):
    z_old = np.array([box[2] for box in old_boxes.values()])
    old_keys = np.array(list(old_boxes.keys()))
    for id, box in new_boxes.items():
        box_z = box[2]
        if np.isnan(box_z):
            continue
        dist_vec = np.abs([box_z - z_old])[0]
        closest_zs = np.sort(dist_vec)[:n_closest]
        closest_key = old_keys[[dist in closest_zs for dist in dist_vec]]
        shifts = [np.array(old_boxes[close_key]) - np.array(fruit_3d_space[close_key]) for close_key in closest_key]
        box_projection = tuple(np.array(box) - np.nanmedian(shifts, axis=0))
        fruit_3d_space[id] = box_projection
    return fruit_3d_space


def get_trakcer_bboxes(tracker_full_results):
    top_left_points = list(zip(tracker_full_results["x1"].values, tracker_full_results["y1"].values))
    bottom_right_points = list(zip(tracker_full_results["x2"].values, tracker_full_results["y2"].values))
    bboxed = list(zip(top_left_points, bottom_right_points))
    return bboxed


def row_resized_bbox(row, r_w, r_h):
    return ((int(row[1]["bbox"][0][0] * r_w), int(row[1]["bbox"][0][1] * r_h)),
            (int(row[1]["bbox"][1][0] * r_w), int(row[1]["bbox"][1][1] * r_h)))