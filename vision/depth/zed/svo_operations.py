import pyzed.sl as sl
import cv2
import numpy as np
from scipy.stats import gaussian_kde
import kornia as K
from vision.tools.image_stitching import plot_2_imgs
def get_frame(frame_mat, cam):
    cam.retrieve_image(frame_mat, sl.VIEW.LEFT)
    frame = frame_mat.get_data()[:, :, : 3]
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def get_depth(depth_mat, cam):
    cam_run_p = cam.get_init_parameters()
    cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
    depth = depth_mat.get_data()
    depth = (cam_run_p.depth_maximum_distance - np.clip(depth, 0, cam_run_p.depth_maximum_distance)) * 255 / cam_run_p.depth_maximum_distance
    bool_mask = np.where(np.isnan(depth), True, False)
    depth[bool_mask] = 0

    depth = cv2.medianBlur(depth, 5)

    return depth


def get_point_cloud(point_cloud_mat, cam):
    cam.retrieve_measure(point_cloud_mat, sl.MEASURE.XYZRGBA)
    point_cloud = point_cloud_mat.get_data()

    return point_cloud


def get_dist_vec(pc_mat, bbox2d, dim):
    x1 = int(bbox2d[0])
    y1 = int(bbox2d[1])
    x2 = int(bbox2d[2])
    y2 = int(bbox2d[3])

    if dim == 'width':
        mid_h = int((y2 + y1) / 2)

        if mid_h <= 0 or (x2 - x1) <= 0:
            mat = None
        else:
            mat = pc_mat[mid_h - 1: mid_h + 2, x1:x2, :-1].copy()
        return mat

    if dim == 'height':
        mid_w = int((x2 + x1) / 2)

        if mid_w <= 0 or (y2 - y1) <= 0:
            mat = None
        else:
            mat = pc_mat[y1:y2, mid_w - 1: mid_w + 2, :-1].copy()
        return mat


def get_cropped_point_cloud(bbox, point_cloud, margin=0.2):
    # TODO

    crop = point_cloud[max(int(bbox[1]), 0):int(bbox[3]), max(int(bbox[0]), 0): int(bbox[2]), :-1].copy()
    return crop


def get_distance(crop):
    h, w = crop.shape
    filter_ = 0.2
    if h < 4 or w < 4 or h == 0 or w == 0:
        return np.inf
    crop = crop[int(h * filter_): h - int(h * filter_), int(w * filter_):w - int(w * filter_)]
    return np.nanmedian(crop)


def get_width(crop, margin=0.2, fixed_z=True, max_z=1):
    h, w, c = crop.shape
    marginy = np.round(margin / 2 * h).astype(np.int16)
    crop_marg = crop[marginy:-marginy, :, :]
    crop_marg[crop_marg[:, :, 2] > max_z] = np.nan
    vec = np.nanmean(crop_marg, axis=0)
    vec = vec[np.isfinite(vec[:, 2])]
    if len(vec) < 2:
        return np.nan
    if fixed_z:
        width = np.sqrt(np.sum((vec[0, :-1] - vec[-1, :-1]) ** 2)) * 1000
    else:
        width = np.sqrt(np.sum((vec[0, :] - vec[-1, :]) ** 2)) * 1000
    return width


def get_height(crop, margin=0.2, fixed_z=True, max_z=1):
    h, w, c = crop.shape
    marginx = np.round(margin / 2 * w).astype(np.int16)
    crop_marg = crop[:, marginx:-marginx, :]
    crop_marg[crop_marg[:, :, 2] > max_z] = np.nan
    vec = np.nanmean(crop_marg, axis=1)
    vec = vec[np.isfinite(vec[:, 2])]
    if len(vec) < 2:
        return np.nan
    if fixed_z:
        height = np.sqrt(np.sum((vec[0, :-1] - vec[-1, :-1]) ** 2)) * 1000
    else:
        height = np.sqrt(np.sum((vec[0, :] - vec[-1, :]) ** 2)) * 1000
    return height


def filter_xyz_outliers(crop, nstd=2, as_points=True):
    """
    Filters out the outliers from the 3D points in the given crop.

    Args:
        crop (ndarray): A numpy array of shape (height, width, 3) containing the 3D points.
        nstd (float): The number of standard deviations to consider for defining the range of valid values.
        as_points (bool): Whether to return the filtered 3D points as an array of points or as an array of the same shape
            as the input crop.

    Returns:
        ndarray: A numpy array of filtered 3D points. If as_points is True, this is a numpy array of shape (n, 3),
            where n is the number of valid 3D points. Otherwise, it is a numpy array of the same shape as the input crop.
    """
    centers = crop.reshape(-1, 3)
    filtered_center = centers.copy()
    for channel in [0, 1, 2]:
        channel_vals = centers[:, channel]
        max_val = np.abs(np.nanmedian(channel_vals) + nstd * np.nanstd(channel_vals))
        filtered_center[np.abs(channel_vals) > max_val] = np.nan
    if as_points:
        return filtered_center
    return filtered_center.reshape(crop.shape)


def ellipsoid_fit(filtered_center):
    """
    Fits an ellipsoid to a set of 3D points using least squares estimation.

    Args:
    - filtered_center (numpy array): An N x 3 array of N 3D points in the form (x, y, z).

    Returns:
    - radius (float): The radius of the fitted ellipsoid.
    - semi_axis_1 (float): The length of the semi-major axis of the ellipsoid.
    - semi_axis_2 (float): The length of the semi-intermediate axis of the ellipsoid.
    - semi_axis_3 (float): The length of the semi-minor axis of the ellipsoid.
    """
    filtered_center[np.abs(filtered_center) > 10] = np.nan # to handle very large values
    A = np.column_stack([filtered_center, np.ones(len(filtered_center))])
    good_rows = np.all(np.isfinite(A), axis=1)
    A = A[good_rows]

    #   Assemble the f matrix
    f = np.zeros((len(A), 1))
    f[:, 0] = np.sum(A[:, :3] ** 2, axis=1)
    C, residules, rank, singval = np.linalg.lstsq(A, f)
    C = np.abs(C)
    #   solve for the radius
    t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
    radius = np.sqrt(t)
    # channels are switched
    return radius, np.sqrt(C[1]), np.sqrt(C[0]), np.sqrt(C[2])


def get_dimensions(point_cloud, dets, dist_max, method="reg"):
    dims = []
    for det in dets:
        # in case that is not a full fruit
        if det[-3] == 1:
            continue
        crop = get_cropped_point_cloud(det[:4], point_cloud)
        if method == "reg":
            width = get_width(crop, fixed_z=True, max_z=dist_max)
            height = get_height(crop, fixed_z=True, max_z=dist_max)
        elif method == "ellipsoid":
            filtered_center = filter_xyz_outliers(crop, nstd=2, as_points=True)
            _, width, height, _ = ellipsoid_fit(filtered_center)
            if width > 200 or height > 200: #if fruits are too big its probably due to noise
                width, height = np.nan, np.nan
        elif "pix_size" in method:
            width, height = get_dims_w_pixel_size(crop, det[:4], method.split("_")[-1])
        distance = get_distance(crop[:, :, 2])

        dims.append([height, width, distance])

    return dims


def sl_get_dimensions(dets, wrapper):
    import matplotlib.pyplot as plt
    dims = []
    objects_in = []
    for det in dets:
        x1, x2, y1, y2 = max(int(det[1]), 0), int(det[3]), 1080 - min(int(det[2]), 1080), 1080 - max(int(det[0]), 0)
        mat = sl.Mat()
        wrapper.cam.retrieve_image(mat, sl.VIEW.LEFT)

        # img = mat.get_data()
        # img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 5)
        # plt.imshow(img)
        # plt.show()

        # try:
        bounding_box_2d = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        tmp = sl.CustomBoxObjectData()
        tmp.unique_object_id = sl.generate_unique_id()
        tmp.label = -1
        tmp.probability = det[4]
        tmp.bounding_box_2d = bounding_box_2d
        tmp.is_grounded = False
        objects_in.append(tmp)
        # except OverflowError:
        #     print("***************************************", det)
        #     print(x1, x2, y1, y2)

    wrapper.cam.ingest_custom_box_objects(objects_in)
    objects_out = sl.Objects()
    wrapper.cam.retrieve_objects(objects_out)
    for obj in objects_out.object_list:
        dims.append([obj.dimensions[1] * 1000, obj.dimensions[0] * 1000, obj.position[2]])

    return dims


def average_det_depth(crop, margin=0.2):
    h, w, c = crop.shape
    marginx = np.round(margin / 2 * w).astype(np.int16)
    marginy = np.round(margin / 2 * h).astype(np.int16)

    return np.nanmean(crop[marginy:-marginy, marginx:-marginx, 2])


def get_dets_ranges(point_cloud, dets):
    ranges = []
    for det in dets:
        crop = get_cropped_point_cloud(det[:4], point_cloud)
        ranges.append(average_det_depth(crop))

    return ranges


def filter_by_kde(crop_rgb, crop_pc, threshold=0.001):
    """
    filters the rgb image based on point cloud density
    the genereal assumprion is that the close pixels are the fruit.
    :param crop_rgb: a cropped rgb image
    :param crop_pc: a cropped point cloud
    :param threshold: for
    :return:
    """
    crop_rgb = crop_rgb.copy()
    flat_pc = crop_pc[:, :, 2].flatten()
    kernel = gaussian_kde(flat_pc[np.isfinite(flat_pc)])
    distances = np.arange(np.nanmin(crop_pc[:, :, 2]), min(np.nanmax(crop_pc[:, :, 2]), 1), 0.0001)
    density = kernel(distances)
    density = density / sum(density)
    picks = []
    bottoms = []
    for ind, val in enumerate(density[1:-1]):
        if density[ind] > val and density[ind + 2] > val and val > threshold:
            bottoms.append(ind + 1)
        if density[ind] < val and density[ind + 2] < val and val > threshold:
            picks.append(ind + 1)
    min_dist = distances[picks[0]]
    thresh_dist = np.min(np.where(np.all([density < threshold, distances> min_dist ], axis=0)))
    crop_rgb[crop_pc[:, :, 2] > distances[thresh_dist]] = np.nan
    return crop_rgb


def apply_sobol(det_crop, plot_change=False):
    """
    applies sobol filterning on image
    :param det_crop: image to apply filter on
    :param plot_change: flag to show the image after applying sobol
    :return: image after sobol filtering
    """
    torch_img = K.utils.image_to_tensor(det_crop)
    torch_img = torch_img[None, ...].float() / 255.
    torch_img = K.enhance.adjust_contrast(torch_img, 0.5)
    torch_img_gray = K.color.rgb_to_grayscale(torch_img)
    processed_img = K.filters.sobel(torch_img_gray, True, 1e-3)  # BxCx2xHxW
    if plot_change:
        plot_2_imgs(det_crop, processed_img.detach().numpy()[0, 0] > 0.05)
    return processed_img


def get_pix_size(dist, fx=1065.98388671875, fy=1065.98388671875,
                 pixel_mm=0.0002, org_size=np.array([1920, 1080])):
    """
    Calculates the size of a pixel in millimeters given a distance from the camera and the intrinsic parameters of the camera.

    Args:
        dist (float): The distance from the camera to the object in millimeters.
        fx (float): The focal length of the camera in the x direction in pixels. Default is 1065.98388671875.
        fy (float): The focal length of the camera in the y direction in pixels. Default is 1065.98388671875.
        pixel_mm (float): The size of a pixel in millimeters. Default is 0.002.
        org_size (ndarray): The size of the image in pixels. Default is np.array([1920, 1080]).

    Returns:
        size_pix (float): The size of a pixel in the x direction in millimeters.
    """
    y0, x0 = org_size/2
    focal_len = (fx + fy) / 2 * pixel_mm
    x_range = np.arange(1, org_size[1]+1)
    x_pix_dist_from_center = np.abs(np.array([x_range for i in range(org_size[0])]) - x0)
    x_mm_dist_from_center = (x_pix_dist_from_center * (x_pix_dist_from_center+1)*(pixel_mm**2))
    beta = np.arctan(0.001/(focal_len + (x_mm_dist_from_center/focal_len)))
    gamma = np.arctan((x_mm_dist_from_center+1)*pixel_mm/focal_len)
    size_pix = (np.tan(gamma) - np.tan(gamma-beta))*dist*2
    return size_pix


def cut_center_of_box(image, margin=0.2):
    """
    Cuts the center of the box if NIR is provided, else fills pixels with no fruit with NaN values.

    Args:
    - image: A 3D Numpy array representing a cropped xyz image.
    - margin: A float representing the percentage of margin to add to the center of the image.

    Returns:
    - A 3D Numpy array representing the cropped image with the center of the box removed.
    """
    t, l, (b, r) = 0, 0, image.shape[:2]
    y_max, x_max = image.shape[:2]
    h_m = int((b - t) * margin)
    w_m = int((r - l) * margin)
    cut_box = image[max(0, t + h_m):min(y_max, b - h_m), max(0, l + w_m):min(x_max, r - w_m)]
    return cut_box


def xyz_center_of_box(image, method="median"):
    """
    Calculates the median or mean x, y, and z coordinates of the fruit.

    Args:
    - image: A 3D Numpy array representing a cropped xyz image.
    - method: A string representing the method to use to calculate the center of the fruit.

    Returns:
    - A tuple of floats representing the x, y, and z coordinates of the center of the fruit.
    """
    cut_box = cut_center_of_box(image)
    if method == "median":
        x_median = np.nanmedian(cut_box[:, :, 0])
        y_median = np.nanmedian(cut_box[:, :, 1])
        z_median = np.nanmedian(cut_box[:, :, 2])
    else:
        x_median = np.nanmean(cut_box[:, :, 0])
        y_median = np.nanmean(cut_box[:, :, 1])
        z_median = np.nanmean(cut_box[:, :, 2])
    return x_median, y_median, z_median


def dist_to_box_center(image, method="median"):
    """
    Calculates the distance from the camera to the center of the fruit.

    Args:
    - image: A 3D Numpy array representing a cropped xyz image.
    - method: A string representing the method to use to calculate the center of the fruit.

    Returns:
    - A float representing the distance from the camera to the center of the fruit.
    """
    return np.sum(np.array(list(xyz_center_of_box(image, method))) ** 2)


def get_dims_w_pixel_size(pc_img, box, center_method="median"):
    """
    Calculates the width and height of a 2D bounding box in millimeters, based on the pixel size of the image.

    Args:
    - pc_img: A 3D Numpy array representing a point cloud image.
    - box: A tuple of integers representing the (x1, y1, x2, y2) coordinates of the bounding box.
    - center_method: A string representing the method to use to calculate the center of the fruit.

    Returns:
    - A tuple of floats representing the width and height of the bounding box in millimeters.
    """
    dist = dist_to_box_center(pc_img, center_method)
    size_pix = get_pix_size(dist)
    x1, y1, x2, y2 = box
    width = np.mean(np.sum(size_pix[y1:y2, x1:x2], axis=0))
    height = np.mean(np.sum(size_pix[y1:y2, x1:x2], axis=1))
    return width, height