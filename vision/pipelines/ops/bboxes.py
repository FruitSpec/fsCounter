import numpy as np

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