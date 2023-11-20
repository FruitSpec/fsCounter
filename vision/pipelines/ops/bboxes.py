import numpy as np
from numba import jit

def depth_center_of_box(image, box, nir=None, swir_975=None):
    """
    returns xyz for the fruit
    :param image: xyz image
    :param box: box to cut
    :return:
    """
    cut_box = cut_center_of_box(image, box)
    depth_median = np.nanmedian(cut_box)
    return depth_median

def cut_center_of_box(image, box, margin=0.25):
    """
    cuts the center of the box if nir is provided, else will turn to nan pixels with no fruit
    :param image: image to cut from
    :param box: box to cut
    :param margin: percentage to add to center
    :return:
    """
    t, b, l, r = get_box_corners(box)
    y_max, x_max = image.shape[:2]

    h_m = int((b-t)*margin)
    w_m = int((r-l)*margin)
    cut_box = image[max(0, t+h_m):min(y_max, b-h_m), max(0, l+w_m):min(x_max, r-w_m)]
    return cut_box

def get_box_corners(box):
    """
    return the cornes of the box
    :param box: box object
    :return: top, buttom, left, right
    """
    t, b, l, r = box[0][1], box[1][1], box[0][0], box[1][0]
    return t, b, l, r


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

def convert_dets(dets, Ms):

    new_dets = []
    for f_bboxes, M in zip(dets, Ms): # dets is in length of batch
        new_f_bboxes = []
        for bbox in f_bboxes:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]

            ul = np.array([x1, y1, 1])
            ur = np.array([x2, y1, 1])
            bl = np.array([x1, y2, 1])
            br = np.array([x2, y2, 1])

            corners = np.array([ul, ur, bl, br])
            transformed_location = np.dot(M, corners.T).T

            min_x = np.min(transformed_location[:, 0])
            min_y = np.min(transformed_location[:, 1])
            max_x = np.max(transformed_location[:, 0])
            max_y = np.max(transformed_location[:, 1])

            new_bb = bbox.copy()
            new_bb[0] = int(min_x)
            new_bb[1] = int(min_y)
            new_bb[2] = int(max_x)
            new_bb[3] = int(max_y)

            new_f_bboxes.append(new_bb)

        new_dets.append(new_f_bboxes)

    return new_dets

def match_by_intersection(bboxes1, bboxes2):

    intersections = []

    inetr_area = get_intersection(bboxes1, bboxes2)
    if len(inetr_area) > 0:
        intersections = inetr_area > 0

    return intersections

def get_intersection(bboxes1: np.array, bboxes2: np.array):  # matches
    inter_aera = []

    if len(bboxes1) > 0 and len(bboxes2) > 0:
        inter_aera = calc_intersection(np.array(bboxes1)[:, :4], np.array(bboxes2)[:, :4])

    return inter_aera


def calc_intersection(bboxes1: np.array, bboxes2: np.array):
    x11, y11, x12, y12 = bbox_to_coordinate_vectors(bboxes1)
    x21, y21, x22, y22 = bbox_to_coordinate_vectors(bboxes2)

    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    inter_aera = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

    return inter_aera

def bbox_to_coordinate_vectors(bboxes):
    x1_vec = bboxes[:, 0].copy()
    y1_vec = bboxes[:, 1].copy()
    x2_vec = bboxes[:, 2].copy()
    y2_vec = bboxes[:, 3].copy()

    x1_vec = np.expand_dims(x1_vec, axis=1)
    y1_vec = np.expand_dims(y1_vec, axis=1)
    x2_vec = np.expand_dims(x2_vec, axis=1)
    y2_vec = np.expand_dims(y2_vec, axis=1)

    return x1_vec, y1_vec, x2_vec, y2_vec


def xyz_center_of_box(image, box):
    """
    returns xyz for the fruit
    :param image: xyz image
    :param box: box to cut
    :param nir: infra red image
    :param swir_975: 975 image
    :return:
    """
    cut_box = cut_center_of_box(image, box)
    if not np.nansum(cut_box):
        return np.nan, np.nan, np.nan
    x_median = np.nanmedian(cut_box[:, :, 0])
    y_median = np.nanmedian(cut_box[:, :, 1])
    z_median = np.nanmedian(cut_box[:, :, 2])
    return x_median, y_median, z_median