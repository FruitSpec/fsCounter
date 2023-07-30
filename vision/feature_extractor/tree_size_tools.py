import numpy as np
from math import pi
try:
    from stat_tools import smooth_data_np_average, iqr_trim, quantile_trim
except:
    from vision.feature_extractor.stat_tools import smooth_data_np_average, iqr_trim, quantile_trim
try:
    import boxing_tools as box_t
except:
    from vision.feature_extractor import boxing_tools as box_t
import cv2
import numba as nb


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


def stable_euclid_dist(point_cloud, bbox, buffer=1):
    """
    calculates euclidian width and height of the bounfing box
    :param point_cloud: point_cloud map produced by zed
    :param bbox: the bounding box
    :param buffer: how many pixels above and below the middle to take in order to estimate the distance
    :return: width, height
    """
    min_x, max_x, min_y, max_y, med_x, med_y = box_t.min_max_finites(point_cloud, bbox)
    if med_x + buffer+1 > point_cloud.shape[1]:
        buffer = point_cloud.shape[1] - med_x-1
    if med_x-buffer < 0:
        buffer = med_x
    if med_y + buffer+1 > point_cloud.shape[0]:
        buffer = point_cloud.shape[0] - med_y-1
    if med_y-buffer < 0:
        buffer = med_y
    point_left = point_cloud[med_y-buffer:med_y+buffer+1, min_x][:, :3]
    point_right = point_cloud[med_y-buffer:med_y+buffer+1, max_x][:, :3]
    point_top = point_cloud[min_y, med_x-buffer:med_x + buffer+1][:, :3]
    point_bottom = point_cloud[max_y, med_x-buffer:med_x + buffer+1][:, :3]
    axis = 0 if buffer == 0 else 1
    h_dist = np.sqrt(np.nansum(np.power(point_left - point_right, 2), axis=axis))
    v_dist = np.sqrt(np.nansum(np.power(point_top - point_bottom, 2), axis=axis))
    valid_h_dist = h_dist[np.where(h_dist > 0)]
    valid_v_dist = v_dist[np.where(v_dist > 0)]
    return safe_nanmean(valid_h_dist), safe_nanmean(valid_v_dist)


def calc_width_per_row(xyz_point_cloud, ndvi_binary, y, buffer=0, x_only=True):
    row_ndvi = ndvi_binary[y, :]
    row_pc = xyz_point_cloud[y, :, 2]
    foliage_row = np.where(np.all([np.isfinite(row_pc), row_ndvi == 1], axis=0))
    if np.sum(foliage_row) <= buffer * 2:
        return np.nan
    x_0 = np.min(foliage_row)
    x_1 = np.max(foliage_row)
    if x_0 + buffer > xyz_point_cloud.shape[1]:
        buffer = xyz_point_cloud.shape[1] - x_0
    if x_1-buffer < 0:
        buffer = x_1
    if x_only:
        points_left = xyz_point_cloud[y, x_0:x_0+buffer, 0]
        points_right = xyz_point_cloud[y, x_1-buffer:x_1, 0]
        h_dist = np.abs(points_left - points_right)
    else:
        points_left = xyz_point_cloud[y, x_0:x_0+buffer]
        points_right = xyz_point_cloud[y, x_1-buffer:x_1]
        h_dist = np.sqrt(np.nansum(np.power(points_left - points_right, 2), axis=1))
    h_dist_above_zero = h_dist[h_dist > 0]
    return np.median(h_dist_above_zero) if len(h_dist_above_zero) > 0 else np.nan


def calc_height_per_col(xyz_point_cloud, ndvi_binary, x, buffer=0):
    col_ndvi = ndvi_binary[:, x]
    col_pc = xyz_point_cloud[:, x, 2]
    foliage_row = np.where(np.all([np.isfinite(col_pc), col_ndvi == 1], axis=0))
    if np.sum(foliage_row) < buffer * 2:
        return np.nan
    y_0 = np.min(foliage_row)
    y_1 = np.max(foliage_row)
    if y_0 + buffer > xyz_point_cloud.shape[0]:
        buffer = xyz_point_cloud.shape[0] - y_0
    if y_1-buffer < 0:
        buffer = y_1
    points_left = xyz_point_cloud[y_0:y_0 + buffer, x]
    points_right = xyz_point_cloud[y_1-buffer:y_1, x]
    v_dist = np.sqrt(np.nansum(np.power(points_left - points_right, 2), axis=1))
    v_dist_above_zero = v_dist[v_dist > 0]
    return np.nanmedian(v_dist_above_zero) if np.nansum(v_dist_above_zero) > 0 else np.nan


def calc_tree_widths(xyz_point_cloud, ndvi_binary):
    widths = np.array([])
    n = ndvi_binary.shape[0]
    for y in range(n):
        # print(f"\r{y+1}/{n} ({(y+1) / n * 100: .2f}%) widths", end="")
        widths = np.append(widths, calc_width_per_row(xyz_point_cloud, ndvi_binary, y, buffer=1))
    # print()
    return widths

def calc_tree_heights(xyz_point_cloud, ndvi_binary):
    heights = np.array([])
    n = ndvi_binary.shape[1]
    for x in range(int(n)):
        # print(f"\r{x+1}/{n} ({(x+1) / (n) * 100: .2f}%) heights", end="")
        heights = np.append(heights, calc_height_per_col(xyz_point_cloud, ndvi_binary, x, buffer=3))
    # print()
    return heights


def get_real_world_dims_with_correction(depth_map, fx = 1065.98388671875, fy = 1065.98388671875, resized = True):
    """
    calculates each pixel size based on trigo
    :param depth_map: distance_map to each point or an empty string
    :return: size for each pixel
    """
    pic_size = depth_map.shape
    if resized:
        resize_fator_x = 1080/pic_size[0]
        resize_fator_y = 1920/pic_size[1]
    else:
        resize_fator_x = 1
        resize_fator_y = 1
    x0 = pic_size[1] /2
    y0 = pic_size[0] /2
    pixel_mm = 0.002
    focal_len = (fx + fy) / 2 * pixel_mm
    x_range = np.arange(1, pic_size[1]+1)
    X_pix_dist_from_center = np.abs(np.array([x_range for i in range(pic_size[0])]) - x0)
    X_mm_dist_from_center = (X_pix_dist_from_center * (X_pix_dist_from_center+1)*(pixel_mm**2))
    beta = np.arctan(0.001/(focal_len + (X_mm_dist_from_center/focal_len)))
    gamma = np.arctan((X_pix_dist_from_center+1)*pixel_mm/focal_len)
    size_x = (np.tan(gamma) - np.tan(gamma-beta))*depth_map*2*resize_fator_x
    size_y = (np.tan(gamma) - np.tan(gamma-beta))*depth_map*2*resize_fator_y
    return size_x, size_y, size_x*size_y


def clean_non_perimeter_depths(points):
    """
    in order to overcome 'holes' in the perimeter depth estimation clean the points that are
    deeper then the linear interpolation between the first and last points
    :param points: points to use
    :return: an array with nan for outlier points
    """
    start_point = points[0]
    end_point = points[-1]
    x_0 = start_point[0]
    z_0 = start_point[-1]
    m_x = (end_point[-1]-z_0)/(end_point[0] - x_0) if (end_point[0] - x_0) != 0 else np.nan
    max_depth = (points[:,0]-x_0)*m_x + z_0
    points[:, 2][points[:, 2] > max_depth] = np.nan
    return points


def clean_and_smooth_perimeter(points):
    """
    run clean_non_perimeter_depths and smooth_data_np_average on the data
    :param points: points to use
    :return: cleaned array
    """
    points = clean_non_perimeter_depths(points)
    points[:, 2] = smooth_data_np_average(points[:, 2])
    return points


def liner_approxsimation(dist_between_points, trim_iqr=False):
    """
    cleans the linear distance between points
    :param dist_between_points: distance between points
    :param trim_iqr: flag to use iqr_trim if not will use quantile_trim
    :return: a clean array
    """
    if not len(dist_between_points):
        return dist_between_points
    if trim_iqr:
        dist_between_points = iqr_trim(dist_between_points.copy())
    else:
        dist_between_points = quantile_trim(dist_between_points.copy(), trim_vals=(0.01, 0.95))
    nan_dists = np.isnan(dist_between_points)
    dist_between_points[nan_dists] = 0
    if np.mean(nan_dists):
        med_dist = np.nan
    else:
        med_dist = np.median(dist_between_points[~nan_dists])
    dist_between_points[nan_dists] = med_dist
    return dist_between_points


def get_perimeter(xyz_point_cloud, ndvi_binary, y, clean_smooth=True):
    # TODO add robustness via global minima
    row_points = xyz_point_cloud[y, :]
    row_ndvi = ndvi_binary[y, :]
    foliage_points = row_points[np.all([np.isfinite(row_points[:, 2]), row_ndvi == 1], axis=0)]
    if len(foliage_points) < 10:
        return np.nan
    if clean_smooth:
        #points_for_x = clean_and_smooth_perimeter(foliage_points)
        points_for_x = clean_non_perimeter_depths(foliage_points)
        points_for_x = points_for_x[np.isfinite(points_for_x[:, 2])]
    dist_between_consec_points_x = np.sqrt(np.nansum(np.power(points_for_x[:-1] - points_for_x[1:], 2), axis=1))
    dist_between_consec_points_x = liner_approxsimation(dist_between_consec_points_x)
    return np.nansum(dist_between_consec_points_x)


def calc_tree_perimeter(xyz_point_cloud, ndvi_binary):
    perimeters = np.array([])
    n = ndvi_binary.shape[0]
    for y in range(n):
        # print(f"\r{y}/{n-1} ({y/(n-1)*100: .2f}%) perimeters", end="")
        perimeters = np.append(perimeters, get_perimeter(xyz_point_cloud, ndvi_binary, y))
    # print()
    return perimeters


def get_finite_points(points):
    """
    returns a subset of points with finite depth (still need a relevent implementation for a matrix data - more than one row)
    :param points: points to use
    :return: filtered array
    """
    if len(points.shape) == 2:
        return points[np.isfinite(points[:, 2])]
    else:
        return points


def get_points_for_surface(point_cloud, mask, clean_smooth, subset_factor=1):
    """
    cleans the points for surface area calculations, will subset the point cloud given subset_factor paramater
    :param point_cloud: zed point cloud
    :param bbox: bounding box
    :param clean_smooth: flag to use or not cleaning and smoothing
    :param subset_factor: sampling rate will reduce the shape of the matrix by subset_factor^2
    :return:
    """
    sub_mask = mask
    y_1, x_1, _ = point_cloud.shape
    sub_matrix = point_cloud
    if subset_factor > 1:
        sub_matrix = point_cloud[0:(y_1 + 1 + subset_factor):subset_factor, 0:(x_1 + 1 + subset_factor):subset_factor]
        sub_mask = mask[0:(y_1 + 1 + subset_factor):subset_factor, 0:(x_1 + 1 + subset_factor):subset_factor]
        sub_matrix = np.array([clean_non_perimeter_depths(sub_matrix[y, :]) for y in range(sub_matrix.shape[0])])
        #sub_matrix[:,:,2] = smooth_data_np_average(sub_matrix[:,:,2])
    if clean_smooth and subset_factor == 1:
        sub_matrix = np.array([clean_and_smooth_perimeter(sub_matrix[y, :]) for y in range(sub_matrix.shape[0])])
    sub_matrix_finite = get_finite_points(sub_matrix)
    points_for_x = get_finite_points(sub_matrix_finite[:-1, :-1])
    points_p1_for_x = get_finite_points(sub_matrix_finite[:-1, 1:])
    points_for_y = get_finite_points(sub_matrix_finite[:-1, :-1])
    points_p1_for_y = get_finite_points(sub_matrix_finite[1:, :-1])
    return points_for_x, points_p1_for_x, points_for_y, points_p1_for_y, sub_mask[:-1, :-1]


def get_foliage_fullness(ndvi_binary):
    contours, _ = cv2.findContours(ndvi_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_image = cv2.drawContours(ndvi_binary.copy(), contours, -1, (1, 0, 0), -1)
    return np.sum(ndvi_binary), (np.sum(cont_image))


def get_surface_area(xyz_point_cloud, mask=None, clean_smooth=False, subset_factor=5):
    """
    this function is O(n^2) and should be reconsidered for use
    returns the surface area linear approximation the bounding box
    :param xyz_point_cloud: zed point cloud
    :param mask: mask for object (foliage)
    :param clean_smooth: flag to use or not cleaning and smoothing
    :param subset_factor: sampling rate will reduce the shape of the matrix by subset_factor^2
    :return: linear approximation of the surface area
    """
    # TODO check correctness
    if isinstance(mask, type(None)):
        mask = np.ones(xyz_point_cloud.shape[:2])
    points_for_x, points_p1_for_x, points_for_y, points_p1_for_y, mask = get_points_for_surface(xyz_point_cloud, mask,
                                                                                          clean_smooth, subset_factor)
    dist_between_consec_points_x = np.sqrt(np.nansum(np.power(points_for_x - points_p1_for_x, 2), axis=2))
    dist_between_consec_points_y = np.sqrt(np.nansum(np.power(points_for_y - points_p1_for_y, 2), axis=2))
    if len(dist_between_consec_points_x) == 0 or len(dist_between_consec_points_y) == 0:
        return 0
    dist_between_consec_points_xy = np.apply_along_axis(liner_approxsimation, 1,
                                                        dist_between_consec_points_x*dist_between_consec_points_y)
    return np.nansum(dist_between_consec_points_xy)


def get_tree_volume(tree_physical_params, vol_style="cone"):
    width = tree_physical_params["width"]
    height = tree_physical_params["height"]
    avg_width = tree_physical_params["avg_width"]
    avg_height = tree_physical_params["avg_height"]
    r = width / 2
    avg_r = avg_width / 2
    if vol_style == "cone":
        volume = 1 / 3 * pi * r ** 2 * height
        avg_volume = 1 / 3 * pi * avg_r ** 2 * avg_height
    elif vol_style == "cylinder":
        volume = pi * r ** 2 * height
        avg_volume = pi * avg_r ** 2 * avg_height
    elif vol_style == "sphere":
        volume = 4 / 3 * pi * r ** 3
        avg_volume = 4 / 3 * pi * avg_r ** 2 * avg_height
    return volume, avg_volume


def get_min_max_real_y(tree_images, slicer_results, minimal_frames):
    middle_frame = minimal_frames[len(minimal_frames)//2]
    xyz = tree_images[middle_frame]["zed"]
    binary_ndvi = tree_images[middle_frame]["ndvi_binary"]
    binary_ndvi = box_t.slice_outside_trees([binary_ndvi], slicer_results, middle_frame, reduce_size=False,
                                            mask=None, y_ranges=(), i=0, n_min_frames=1, cut_val=0.05)[0]
    y_s = xyz[:, :, 1] * binary_ndvi
    return np.nanquantile(y_s, (0.05, 0.95))


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



def get_pix_size(depth, box, fx=1149.666015625, fy=1149.666015625, cx=545, cy=959,
                 pixel_mm=0.0002, org_size=np.array([1920, 1080]), normalize_factor=0.01, extreme_x_factor = 0.15):
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
    # TODO create a map of pixel size for the entire image then send it thoiugrh the entire pipe including resize and everything
    x1, y1, x2, y2 = box
    y0, x0 = cy, cx
    extreme_xs = (org_size[1]*extreme_x_factor, org_size[1]*(1-extreme_x_factor))
    focal_len = (fx + fy) / 2 * pixel_mm
    x_range = np.arange(x1, x2 + 1)
    x_pix_dist_from_center = (np.abs(np.array([x_range for i in range(y2 - y1)]) - x0))

    x_1 = x_pix_dist_from_center * pixel_mm
    x_2 = (x_pix_dist_from_center + 1) * pixel_mm
    gamma = np.arctan(x_2/focal_len)
    beta = gamma - np.arctan(x_1/focal_len)
    try:
        dist_from_center_penalty = np.abs((x_range - x0)*normalize_factor*pixel_mm)*depth
        very_far_unpenlty = (x_range < extreme_xs[0])*dist_from_center_penalty*0.75+\
                            (x_range > extreme_xs[1])*dist_from_center_penalty*0.5
        size_pix_x = (np.tan(gamma) - np.tan(gamma - beta)) * depth*10 - dist_from_center_penalty + very_far_unpenlty
    except Exception as e:
        print(e)
    y_range = np.arange(y1, y2 + 1)
    y_pix_dist_from_center = np.abs(np.array([y_range for i in range(x2 - x1)]) - y0).T
    y_mm_dist_from_center = (y_pix_dist_from_center * (y_pix_dist_from_center + 1) * (pixel_mm ** 2))
    beta = np.arctan(0.001 / (focal_len + (y_mm_dist_from_center / focal_len)))
    gamma = np.arctan((y_mm_dist_from_center + 1) * pixel_mm / focal_len)
    size_pix_y = (np.tan(gamma) - np.tan(gamma - beta)) * depth * 2
    return size_pix_x, size_pix_y, size_pix_x*size_pix_y