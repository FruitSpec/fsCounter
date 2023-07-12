import numpy as np
import cv2
from skimage.util import img_as_ubyte
try:
    from stat_tools import iqr_trim, quantile_trim
except:
    from vision.feature_extractor.stat_tools import iqr_trim, quantile_trim
try:
    import image_processing as img_pro
except:
    from vision.feature_extractor import image_processing as img_pro
try:
    from tree_size_tools import stable_euclid_dist
except:
    from vision.feature_extractor.tree_size_tools import stable_euclid_dist
#from cupyx.scipy import ndimage
from scipy import ndimage
#import cupy as cp
import numpy as cp
from scipy.optimize import minimize
global fsi_size
global nir_channel
global ir975_channel
global blue_channel
fsi_size = (2048, 1536)
nir_channel, ir975_channel, blue_channel = 0, 1, 2
import warnings


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
        if top_bot:
            return 0, 0
        return 0, 0, 0, 0
    if not np.sum(mask):
        if top_bot:
            return 0, 0
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
    """
    extract the top pixel and bouttom pixel of foliage, where the top and button are taken globally across al frames
    :param masks: masks between frames
    :param min_factor: y_button,y_top has to be in (hieght_in_pixels*min_factor, hieght_in_pixels*(1-min_factor))
    :return: min pixel and max pixel to use
    """
    tops, bottoms = [], []
    min_y, max_y = np.inf, 0
    for mask in masks:
        top, bottom = img_pro.get_mask_corners(mask, top_bot=True)
        tops.append(top)
        bottoms.append(bottom)
        if not np.isfinite(min_y) and not isinstance(mask, type(None)):
            min_y, max_y = int(mask.shape[0] * min_factor), int(mask.shape[0] * (1 - min_factor))
    return min(np.min(tops), min_y), max(np.max(bottoms), max_y)


def get_y_ranges(minimal_frames, masks):
    """
    gets the y ramges of all frames
    :param minimal_frames: the minimal frames for the tree
    :param masks: masking between frames
    :return: ymin, ymax
    """
    if len(minimal_frames) > 1:
        min_top, max_bottom = get_global_top_bottom(masks)
        y_ranges = (min_top, max_bottom)
    else:
        y_ranges = ()
    return y_ranges


def x_start_end_resizing(x_start, x_end, i=0, n_min_frames=1, cut_val=0.1):
    """
    gets the xpixel for starting and xpixel for ending each tree with a reduciong factor (to counter some edge cases)
    :param x_start: current x_pixel start
    :param x_end: current x_pixel end
    :param i: number of frame
    :param n_min_frames: number of minial frames
    :param cut_val: the resizing factor
    :return:
    """
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


def adjust_x_start_end_to_mask(mask, x_start, x_end, ret_tblr=False, im_shape=(0, 0, 2048, 1536)):
    if not isinstance(mask, type(None)):
        top, left, bottom, right = get_mask_corners(mask)
        if left == 0:
            x_start = max(x_start, right)
        if right == mask.shape[1]:
            x_end = min(x_end, left)
    else:
        top, left, bottom, right = (0, 0, *im_shape[:2])
    if ret_tblr:
        return x_start, x_end, top, left, bottom, right
    return x_start, x_end

def slice_outside_trees(pictures, slicer_results, frame_number, reduce_size=False, mask=None, y_ranges=(),
                        i=0, n_min_frames=1, cut_val=0.05):
    """
    slices out non tree pixels
    :param pictures: list of pictures to edit
    :param slicer_results: {"frame": (x_start,x_end) for each frame}
    :param frame_number: number of frame to take images for
    :param reduce_size: flag for resucing the pictures or filling non trees space  with nan
    :param mask: mask between frames
    :param y_ranges: top and buttom y_s
    :param i:current frame number
    :param n_min_frames:number of mininal frammes
    :param cut_val: cut param for x_start_end_resizing
    :return: sliced pictures
    """
    x_start, x_end = slicer_results[frame_number]
    x_start, x_end = x_start_end_resizing(x_start, x_end, i=i, n_min_frames=n_min_frames, cut_val=cut_val)
    x_start, x_end, top, left, bottom, right = adjust_x_start_end_to_mask(mask, x_start, x_end, True,
                                                                          pictures[0].shape[:2])
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

def slice_outside_trees_il(images, slicer_results, mask):
    x_start, x_end = slicer_results
    x_start, x_end = adjust_x_start_end_to_mask(mask, x_start, x_end)
    return [pic[:, x_start: x_end] for pic in images]


def slice_outside_trees_batch(list_of_images, slicer_results, masks):
    return list(map(slice_outside_trees_il, list_of_images, slicer_results, masks))


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
    """
    gets the intensity and fruit_foilage ratio of 1 fruit
    :param fsi: fsi image
    :param rgb: rgb image
    :param box: bounding box
    :param use_cuda: flag for using cuda (it might be slower)_
    :return: median intensity value, ndri/ndvi
    """
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
    ndvi = img_pro.make_ndvi(cropped_rgb, cropped_fsi[:, :, 0])
    binary_ndri = img_pro.ndvi_to_binary(ndri, 0.05)
    binary_ndvi = img_pro.ndvi_to_binary(ndvi, 0.05)
    relevant_v = hls[:, :, 1][binary_ndri.astype(bool)]
    rel_v_flat = relevant_v.flatten()
    rel_v_flat_trim = quantile_trim(rel_v_flat, keep_size=False)
    if len(rel_v_flat_trim) == 0:
        return np.nan, np.nanmean(binary_ndvi)/np.nanmean(binary_ndri) if np.nanmean(binary_ndri) else np.inf
    return np.nanmedian(rel_v_flat_trim), np.nanmean(binary_ndvi)/np.nanmean(binary_ndri)


def normalize_intensity(intensity, path_to_tree_folder):
    """
    normalizes the intensity of all samples of a given fruit
    :param intensity: array of intensity per fruit
    :param path_to_tree_folder: path to the tree folder for prinint errors
    :return: the normalized intensity
    """
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
    """
    return the cornes of the box
    :param box: box object
    :return: top, buttom, left, right
    """
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
    """
    rreturn the center of the box
    :param box:
    :return: med_x, med_y
    """
    t, b, l, r = get_box_corners(box)
    return med_x_med_y(l, t, r, b)


def filter_outside_tree_boxes(tracker_results, slicer_results, direction = "right"):
    """
    removes detections that are not on the tree
    :param tracker_results: dict of the tracker results
    :param slicer_results: dict of the slicer results
    :return: updated tracker_results
    """
    for frame in tracker_results.keys():
        if frame == "cv":
            continue
        x_start, x_end = slicer_results[frame]
        if direction == "left":
            x_end, x_start = slicer_results[frame]
            if x_start == 600:
                x_start = 0
            if x_end == 0:
                x_end = 600
        x_0 = np.array([box[0][0] for box in tracker_results[frame].values()])
        x_1 = np.array([box[1][0] for box in tracker_results[frame].values()])
        for id in np.array(list(tracker_results[frame].keys()))[np.all([x_0 > x_start, x_1 < x_end], axis=0) == False]:
            tracker_results[frame].pop(id)
    return tracker_results


def filter_outside_zed_boxes(tracker_results, tree_images = {}, max_z=0, filter_nans=True, use_box=False):
    """
    removes detections that are too far
    :param tracker_results: dict of the tracker results
    :param slicer_results: dict of the slicer results
    :param max_z: maxsimum depth allowed
    :param filter_nans: flag for filtering fruits with no depth
    :return: updated tracker_results
    """
    for frame in tree_images.keys():
        frame_images = tree_images[frame]
        boxes = tracker_results[frame]
        to_pop = []
        for id, box in boxes.items():
            if not use_box:
                _, _, z = xyz_center_of_box(frame_images["zed"], box, nir=None, swir_975=None)
            else:
                z = box[2][2]
            if np.abs(z) > max_z or ((np.isnan(z) or not np.isfinite(z)) and filter_nans):
                to_pop.append(id)
        for id in to_pop:
            boxes.pop(id)
    return tracker_results


def cut_center_of_box(image, box, nir=None, swir_975=None, margin=0.25):
    """
    cuts the center of the box if nir is provided, else will turn to nan pixels with no fruit
    :param image: image to cut from
    :param box: box to cut
    :param nir: infra red image
    :param swir_975: 975 image
    :param margin: percentage to add to center
    :return:
    """
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
        h_m = int((b-t)*margin)
        w_m = int((r-l)*margin)
        cut_box = image[max(0, t+h_m):min(y_max, b-h_m), max(0, l+w_m):min(x_max, r-w_m)]
    return cut_box


def xyz_center_of_box(image, box, nir=None, swir_975=None):
    """
    returns xyz for the fruit
    :param image: xyz image
    :param box: box to cut
    :param nir: infra red image
    :param swir_975: 975 image
    :return:
    """
    cut_box = cut_center_of_box(image, box, nir=nir, swir_975=swir_975)
    if not np.nansum(cut_box):
        return np.nan, np.nan, np.nan
    x_median = np.nanmedian(cut_box[:, :, 0])
    y_median = np.nanmedian(cut_box[:, :, 1])
    z_median = np.nanmedian(cut_box[:, :, 2])
    return x_median, y_median, z_median


def get_new_old_boxes(xyz_point_cloud, fruit_3d_space, boxes, nir=None, swir_975=None):
    """
    gets boxes for 3d space
    :param xyz_point_cloud: xyz point cloud matrix
    :param fruit_3d_space: current 3d space
    :param boxes: boxes to ingest
    :param nir: 800 channel matrix
    :param swir_975: 975 channel matrix
    :return: the new boxes, the old boxes, widths, hiehgts
    """
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


def project_boxes_to_fruit_space(fruit_3d_space, old_boxes, new_boxes, n_closest=5, prefilter=0):
    """
    projects boxes on to exsisting 3d space
    :param fruit_3d_space: current 3d space
    :param old_boxes: old boxes
    :param new_boxes: new boxes to add
    :param n_closest: number of closest boxes to use for calculations
    :return: updated 3d space
    """
    if not prefilter:
        to_pop = []
        for key, box in old_boxes.items():
            if np.abs(fruit_3d_space[key][2] - box[2])/np.abs(fruit_3d_space[key][2]) > prefilter:
                to_pop.append(key)
        for key in to_pop:
            old_boxes.pop(key)
    old_keys = np.array(list(old_boxes.keys()))
    z_old = np.array([box[2] for box in old_boxes.values()])
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

def project_boxes_to_fruit_space_vector(fruit_3d_space, old_boxes, new_boxes, n_closest=5, prefilter=0):
    """
    projects boxes on to exsisting 3d space
    :param fruit_3d_space: current 3d space
    :param old_boxes: old boxes
    :param new_boxes: new boxes to add
    :param n_closest: number of closest boxes to use for calculations
    :return: updated 3d space
    """
    if prefilter:
        to_pop = []
        for key, box in old_boxes.items():
            if np.abs(fruit_3d_space[key][2] - box[2])/np.abs(fruit_3d_space[key][2]) > prefilter:
                to_pop.append(key)
        for key in to_pop:
            old_boxes.pop(key)
    old_keys = np.array(list(old_boxes.keys()))
    shifts = [np.array(old_boxes[close_key]) - np.array(fruit_3d_space[close_key]) for close_key in old_keys]
    shift = np.nanmedian(shifts, axis=0)
    for id, box in new_boxes.items():
        if np.isnan(box[2]):
            continue
        box_projection = tuple(np.array(box) - shift)
        fruit_3d_space[id] = box_projection
    return fruit_3d_space


def project_boxes_to_fruit_space_global(fruit_3d_space, last_frame_boxes, old_boxes, new_boxes,
                                        prefilter=0, cur_shift=np.array([0, 0, 0])):
    """
    projects boxes on to exsisting 3d space
    :param fruit_3d_space: current 3d space
    :param old_boxes: old boxes
    :param new_boxes: new boxes to add
    :param n_closest: number of closest boxes to use for calculations
    :return: updated 3d space
    """
    if prefilter:
        to_pop = []
        for key, box in old_boxes.items():
            if not np.isfinite(fruit_3d_space[key][2]):
                to_pop.append(key)
                continue
            if np.abs(fruit_3d_space[key][2] - box[2])/np.abs(fruit_3d_space[key][2]) > prefilter:
                to_pop.append(key)
        for key in to_pop:
            old_boxes.pop(key)
    old_keys = np.array(list(old_boxes.keys()))
    try:
        shifts = [np.array(old_boxes[key]) - np.array(last_frame_boxes[key]) for key in old_keys]
    except RuntimeWarning as warning:
        # Handle the warning
        print("Warning encountered:", warning)
    shift = np.nanmedian(shifts, axis=0)
    if not len(old_keys):
        shift = np.array([0, 0, 0])
    for id, box in new_boxes.items():
        if np.isnan(box[2]):
            continue
        box_projection = tuple(np.array(box) - shift - cur_shift)
        fruit_3d_space[id] = box_projection
    return fruit_3d_space, shift

def multilateration(coords, distances, ord=2):
    # Define the objective function to minimize
    def objective(x):
        obj = np.sum(np.square(np.linalg.norm(coords - x, ord=ord, axis=1) - distances))
        return obj

    # Use scipy.optimize.minimize to minimize the objective function
    result = minimize(objective, x0=np.mean(coords, axis=0))

    # Return the coordinates of the optimized point
    return result.x

def project_boxes_to_fruit_space_trilateration(fruit_3d_space , old_boxes, new_boxes, prefilter=0.2):
    """
    projects boxes on to exsisting 3d space
    :param fruit_3d_space: current 3d space
    :param old_boxes: old boxes
    :param new_boxes: new boxes to add
    :param prefilter: int indicating threshold for removing outlier depths
    :return: updated 3d space
    """
    if prefilter:
        to_pop = []
        for key, box in old_boxes.items():
            org_depth = fruit_3d_space[key][2]
            if np.abs((org_depth - box[2])/org_depth) > prefilter:
                to_pop.append(key)
        for key in to_pop:
            old_boxes.pop(key)
    old_keys = np.array(list(old_boxes.keys()))
    org_coords = np.array([fruit_3d_space[key] for key in old_keys])
    coords = np.array([old_boxes[key] for key in old_keys])
    if len(coords) < 3:
        return fruit_3d_space
    for id, box in new_boxes.items():
        box_z = box[2]
        if np.isnan(box_z):
            continue
        distances = np.linalg.norm(coords - np.array(box), axis=1)
        box_projection = multilateration(org_coords, distances)
        fruit_3d_space[id] = box_projection
    return fruit_3d_space


def get_trakcer_bboxes(tracker_full_results):
    """
    gets the bboxes from a trackers dict
    :param tracker_full_results: trakcer dict
    :return: bboxes
    """
    top_left_points = list(zip(tracker_full_results["x1"].values, tracker_full_results["y1"].values))
    bottom_right_points = list(zip(tracker_full_results["x2"].values, tracker_full_results["y2"].values))
    bboxed = list(zip(top_left_points, bottom_right_points))
    return bboxed


def row_resized_bbox(row, r_w, r_h):
    """
    turns a row in the dataframe to resized bounding box
    :param row: the row
    :param r_w: resize factor for width
    :param r_h: resize factor for height
    :return: resized box
    """
    return ((int(row[1]["bbox"][0][0] * r_w), int(row[1]["bbox"][0][1] * r_h)),
            (int(row[1]["bbox"][1][0] * r_w), int(row[1]["bbox"][1][1] * r_h)))

def resize_bbox(bbox, r_w, r_h):
    """
    turns a row in the dataframe to resized bounding box
    :param row: the row
    :param r_w: resize factor for width
    :param r_h: resize factor for height
    :return: resized box
    """
    if len(bbox) == 2:

        return ((int(bbox[0][0] * r_w), int(bbox[0][1] * r_h)),
                (int(bbox[1][0] * r_w), int(bbox[1][1] * r_h)))

    return ((int(bbox[0][0] * r_w), int(bbox[0][1] * r_h)),
            (int(bbox[1][0] * r_w), int(bbox[1][1] * r_h)), bbox[2])


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
        if isinstance(pictures_dict, list):
            x2 = min(cur_coords["x2"][0], pictures_dict[0].shape[1])
            y2 = min(cur_coords["y2"][0], pictures_dict[0].shape[0])
        else:
            x2 = min(cur_coords["x2"][0], pictures_dict.shape[1])
            y2 = min(cur_coords["y2"][0], pictures_dict.shape[0])
    else:
        x2 = min(cur_coords["x2"][0], pictures_dict["zed"].shape[1])
        y2 = min(cur_coords["y2"][0], pictures_dict["zed"].shape[0])
    if image_input:
        if isinstance(pictures_dict, list):
            return [pic[y1:y2, x1:x2, :] for pic in pictures_dict]
        return pictures_dict[y1:y2, x1:x2, :]
    pictures_dict["zed"] = pictures_dict["zed"][y1:y2, x1:x2, :]
    if rgb:
        pictures_dict["zed_rgb"] = pictures_dict["zed_rgb"][y1:y2, x1:x2, :]
    return pictures_dict


def get_zed_in_jai(images, cut_coords):
    """
    cut zed to the jai region
    :param images (list): list of images
    :param cut_coords (list): [x1,y1,x2,y2]
    :return: images with zed and zed_rgb cut to the jai region
    """
    x1, y1, x2, y2 = cut_coords[:4]
    return [img[y1:y2, x1:x2, :] for img in images]