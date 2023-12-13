import warnings

import numpy as np

def safe_nanmean(arr):
    """
    a wrapper for nanmean to avoid bad cases
    :param arr: array to apply nanmean on
    :return: the nanmean of the array
    """
    if not len(arr):
        return np.nan
    if sum(np.isfinite(arr)) == 0:
        return np.nan
    return np.mean(arr[np.isfinite(arr)])

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


def iqr_trim(arr, keep_size=True, n_std=3):
    """
    trims data based on the iqr method
    :param arr: array to work on
    :param keep_size: flag to keep size or not (if false will drop trimmed data, else will replace with nan)
    :param n_std: number of stds to keep
    :return: trimmed array
    """
    if len(arr) < 2:
        return []
    qauntiles = np.nanquantile(arr, (0.25, 0.75))
    iqr_val = qauntiles[1] - qauntiles[0]
    valid_range = qauntiles + np.array([-n_std*iqr_val, n_std*iqr_val])
    if keep_size:
        arr[arr < valid_range[0]] = np.nan
        arr[arr > valid_range[1]] = np.nan
    else:
        arr = arr[np.all([arr > valid_range[0], arr < valid_range[1]], axis=0)]
    return arr

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
    """
    return the cornes of the box
    :param box: box object
    :return: top, buttom, left, right
    """
    t, b, l, r = box[0][1], box[1][1], box[0][0], box[1][0]
    return t, b, l, r


def stable_euclid_dist(point_cloud, bbox, measure_percentile=0.1, ):
    """
    calculates euclidian width and height of the bounfing box
    :param point_cloud: point_cloud map produced by zed
    :param bbox: the bounding box
    :param buffer: how many pixels above and below the middle to take in order to estimate the distance
    :return: width, height
    """
    min_x, max_x, min_y, max_y, med_x, med_y = min_max_finites(point_cloud, bbox)
    x_all_nonfinite = False
    y_all_nonfinite = False
    if min_x == 0 and max_x == 1:
        x_all_nonfinite = True
    if min_y == 0 and max_y == 1:
        y_all_nonfinite = True

    if not x_all_nonfinite:
        height = bbox[1][1] - bbox[0][1]
        y_range = max(int(height * measure_percentile) // 2, 1)

        point_left = point_cloud[med_y - y_range:med_y + y_range + 1, min_x:min_x + 2, :3]
        point_right = point_cloud[med_y - y_range:med_y + y_range + 1, max_x - 2:max_x, :3]

        is_left_valid = validate_point(point_left)
        is_right_valid = validate_point(point_right)

        if is_left_valid and is_right_valid:
            mean_point_left = np.nanmean(point_left.reshape(-1, 3), axis=0)
            mean_point_right = np.nanmean(point_right.reshape(-1, 3), axis=0)

            h_dist = np.sqrt(np.nansum(np.power(mean_point_left - mean_point_right, 2)))

        else:
            h_dist = np.nan
    else:
        h_dist = np.nan

    if not y_all_nonfinite:
        width = bbox[1][0] - bbox[0][0]
        x_range = max(int(width * measure_percentile) // 2, 1)
        point_top = point_cloud[min_y: min_y + 2, med_x-x_range:med_x + x_range+1, :3]
        point_bottom = point_cloud[max_y - 2: max_y, med_x-x_range:med_x + x_range+1, :3]
        is_top_valid = validate_point(point_top)
        is_bottom_valid = validate_point(point_bottom)

        if is_top_valid and is_bottom_valid:
            mean_point_top = np.nanmean(point_top.reshape(-1, 3), axis=0)
            mean_point_bottom = np.nanmean(point_bottom.reshape(-1, 3), axis=0)

            v_dist = np.sqrt(np.nansum(np.power(mean_point_top - mean_point_bottom, 2)))

        else:
            v_dist = np.nan
    else:
        v_dist = np.nan

    return h_dist, v_dist

def validate_point(point):
    valid = False

    if point.size != 0: # empty

        point = point.reshape(-1, 3)

        how_many_nan_vec = np.sum(np.isnan(point), axis=1)

        if min(how_many_nan_vec) == 0: # at least 1 row is valid
            valid = True

    return valid


def get_pix_size_of_box(depth, box, fx=1065.98388671875, fy=1065.98388671875,
                 pixel_mm=0.0002, org_size=np.array([900, 600])):
    """
    Calculates the size of a pixel in millimeters given a distance from the camera and the intrinsic parameters of the camera.

    Args:
        depth (float): The depth from the camera to the object in meters.
        box (list): ROI for pixel size int hte following format: x1,y1,x2,y2.
        fx (float): The focal length of the camera in the x direction in pixels. Default is 1065.98388671875.
        fy (float): The focal length of the camera in the y direction in pixels. Default is 1065.98388671875.
        pixel_mm (float): The size of a pixel in millimeters. Default is 0.002.
        org_size (ndarray): The size of the image in pixels. Default is np.array([1920, 1080]).

    Returns:
        size_pix_x (ndarray): The size of a pixel
        size_pix_y (ndarray): The size of a pixel
    """
    x1, y1, x2, y2 = box
    y0, x0 = org_size / 2
    focal_len = (fx + fy) / 2 * pixel_mm
    x_range = np.arange(x1, x2 + 1)
    x_pix_dist_from_center = np.abs(np.array([x_range for i in range(y2 - y1)]) - x0)
    x_mm_dist_from_center = (x_pix_dist_from_center * (x_pix_dist_from_center + 1) * (pixel_mm ** 2))
    beta = np.arctan(0.001 / (focal_len + (x_mm_dist_from_center / focal_len)))
    gamma = np.arctan((x_mm_dist_from_center + 1) * pixel_mm / focal_len)
    size_pix_x = (np.tan(gamma) - np.tan(gamma - beta)) * depth * 2

    y_range = np.arange(y1, y2 + 1)
    y_pix_dist_from_center = np.abs(np.array([y_range for i in range(x2 - x1)]) - y0).T
    y_mm_dist_from_center = (y_pix_dist_from_center * (y_pix_dist_from_center + 1) * (pixel_mm ** 2))
    beta = np.arctan(0.001 / (focal_len + (y_mm_dist_from_center / focal_len)))
    gamma = np.arctan((y_mm_dist_from_center + 1) * pixel_mm / focal_len)
    size_pix_y = (np.tan(gamma) - np.tan(gamma - beta)) * depth * 2
    return size_pix_x, size_pix_y



def get_pix_size(depth, box = [0, 0, 1080, 1920], fx=1149.666015625, fy=1149.666015625, cx=545, cy=959,
                 pixel_mm=0.0002, org_size=np.array([1920, 1080]), normalize_factor=0, extreme_x_factor=0):
    x1, y1, x2, y2 = box
    y0, x0 = cy, cx
    extreme_xs = (org_size[1]*extreme_x_factor, org_size[1]*(1-extreme_x_factor))
    focal_len = (fx + fy) / 2 * pixel_mm
    x_range = np.arange(x1, x2)
    x_pix_dist_from_center = (np.abs(np.array([x_range for i in range(y2 - y1)]) - x0))

    x_1 = x_pix_dist_from_center * pixel_mm
    x_2 = (x_pix_dist_from_center + 1) * pixel_mm
    gamma = np.arctan(x_2/focal_len)
    beta = gamma - np.arctan(x_1/focal_len)
    if normalize_factor != 0:
        dist_from_center_penalty = np.abs((x_range - x0)*normalize_factor*pixel_mm)*depth
        very_far_unpenlty = (x_range < extreme_xs[0])*dist_from_center_penalty*0.75+\
                            (x_range > extreme_xs[1])*dist_from_center_penalty*0.5
    else:
        dist_from_center_penalty = 0
        very_far_unpenlty = 0
    size_pix_x = (np.tan(gamma) - np.tan(gamma - beta)) * depth*10 - dist_from_center_penalty + very_far_unpenlty
    y_range = np.arange(y1, y2)
    y_pix_dist_from_center = np.abs(np.array([y_range for i in range(x2 - x1)]) - y0).T
    y_mm_dist_from_center = (y_pix_dist_from_center * (y_pix_dist_from_center + 1) * (pixel_mm ** 2))
    beta = np.arctan(0.001 / (focal_len + (y_mm_dist_from_center / focal_len)))
    gamma = np.arctan((y_mm_dist_from_center + 1) * pixel_mm / focal_len)
    size_pix_y = (np.tan(gamma) - np.tan(gamma - beta)) * depth * 2
    return size_pix_x, size_pix_y