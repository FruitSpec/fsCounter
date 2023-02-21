import numpy as np
import cv2
from skimage.morphology import area_opening, area_closing
try:
    from vegetation_indexes import num_deno_nan_divide, num_deno_nan_divide_np, ndvi_cuda
except:
    from vision.feature_extractor.vegetation_indexes import num_deno_nan_divide, num_deno_nan_divide_np, ndvi_cuda
from vision.tools.image_stitching import get_frames_overlap, plot_2_imgs, keep_dets_only
try:
    from boxing_tools import get_mask_corners, make_bbox_pic
except:
    from vision.feature_extractor.boxing_tools import get_mask_corners, make_bbox_pic
from cupyx.scipy import ndimage
import cupy as cp

global fsi_size
global nir_channel
global ir975_channel
global blue_channel
fsi_size = (2048, 1536)
nir_channel, ir975_channel, blue_channel = 0, 1, 2


def multi_convert_gray(list_imgs):
    """
    this function takes a list of images and converts each of them to geay scale
    :param list_imgs: list contatining images
    :return: list of images in grayscale
    """
    return [cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY) for img in list_imgs]


def remove_high_blues(pic, blue, thresh=200, value=np.nan):
    """
    subtitutes pixels with high blue value
    :param pic: the picture
    :param blue: blue channel
    :param thresh: threshold for replacing
    :param value: the value to input insted
    :return:
    """
    pic[blue > thresh] = value
    return pic


def make_ndvi(rgb, nir):
    """
    rgb: rgb image
    nir: nir image
    return: ndvi image
    """
    red = rgb[:, :, 0]
    ndvi_img = num_deno_nan_divide(np.subtract(nir, red), np.add(nir, red))
    np.nan_to_num(ndvi_img, copy=False, nan=-1)
    return ndvi_img


def get_nir_swir(fsi):
    """
    returns the 800, 975 channels
    :param fsi: fsi image
    :return: nir, swir
    """
    return fsi[:, :, nir_channel].astype(float), fsi[:, :, ir975_channel].astype(float)


def make_ndri(fsi=None, nir=None, swir_975=None):
    """
    makes ndri image
    :param fsi: fsi image
    :param nir: nor iamge
    :param swir_975: channel 975 image
    :return: ndri image
    """
    if isinstance(nir, type(None)) or isinstance(swir_975, type(None)):
        nir, swir_975 = get_nir_swir(fsi)
    numerator = np.subtract(nir, swir_975)
    denominator = np.add(nir, swir_975)
    ndri = num_deno_nan_divide_np(numerator, denominator)
    ndri_out = np.nan_to_num(ndri, nan=-1)
    return ndri_out


def ndvi_to_binary(ndvi, thresh, adpt="", ker=35, const=125):
    """
    turns an image to binary given a threshold
    :param ndvi: ndvi image
    :param thresh: threshold to use for binary transformation
    :param adpt: specify adaptive threshold
    :param ker: kernal for adaptive threshold
    :param const: constant for adaptive threshold
    :return: a binary image
    """
    if adpt == "mean":
        return (
            cv2.adaptiveThreshold(cv2.normalize(ndvi, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, ker, const))
    if adpt == "gaussian":
        return (
            cv2.adaptiveThreshold(cv2.normalize(ndvi, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, ker, const))
    return cv2.threshold(ndvi, thresh, 1, cv2.THRESH_BINARY)[1]


def remove_floor(ndvi_binary, fsi, rgb, mask_size=0):
    """
    removes pixels which are suspected to be floor from ndvi binary image
    :param ndvi_binary: binary ndvi picture
    :param fsi: fsi picture
    :param rgb: rgb picture
    :param mask_size: total size of msak
    :return: extra operations to remove floor classifed as foliage from picture
    """
    fs_diff = ndvi_binary * fsi[:, :, 1] - ndvi_binary * fsi[:, :, 0]
    fs_diff_larger_30 = fs_diff > 30
    max_y = fs_diff_larger_30.shape[0]
    t_y = int(np.floor(max_y * 0.75))
    if np.sum(fs_diff_larger_30[t_y:max_y, :]) / np.sum(fs_diff_larger_30) > 0.4:
        ndvi_binary[fs_diff_larger_30] = 0
    ndvi_binary = remove_high_blues(ndvi_binary, rgb[:, :, 2], 200, 0)
    ndvi_binary[rgb[:, :, 0] > rgb[:, :, 1] + 25] = 0
    # not sure if this part is correct with the new camera settings
    # if np.sum(rgb[:, :, 2].flatten() == 255)/(rgb.shape[0]*rgb.shape[1] - mask_size) < 0.18:
    #     hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    #     h_channel = ndvi_binary * hls[:, :, 0]
    #     h_channel[h_channel > 30] = 0
    #     ndvi_binary = ndvi_binary - np.int32(cv2.blur(np.float32(h_channel > 0), (5, 5)) > 0.2)
    return ndvi_binary


def open_ndvi(ndvi_binary, mask):
    """
    :param ndvi_binary: binary ndvi picture
    :param mask: mask of picture
    :return: binary ndvi after applying morpological operations of opening and closing
    """
    area_opening_ndvi_128 = np.zeros(ndvi_binary.shape)
    top, left, bottom, right = get_mask_corners(mask)
    if left == 0 and right - 256 > 0:
        area_opening_ndvi_128[:, right - 256:] = area_closing(area_opening(ndvi_binary[:, right - 256:], 128), 256, 1)
    elif right == ndvi_binary.shape[1] and left + 256 < ndvi_binary.shape[1]:
        area_opening_ndvi_128[:, : left + 256] = area_closing(area_opening(ndvi_binary[:, : left + 256], 128), 256, 1)
    else:
        return area_closing(area_opening(ndvi_binary, 128), 256, 1)
    if top == 0 and bottom - 256 > 0:
        y_strip = area_closing(area_opening(ndvi_binary[bottom - 256:, :], 128), 256, 1)
        if left == 0:
            area_opening_ndvi_128[bottom:, :right] = y_strip[256:, :right]
        else:
            area_opening_ndvi_128[bottom:, left:] = y_strip[256:, left:]
    elif top + 256 < ndvi_binary.shape[0] and bottom == ndvi_binary.shape[0]:
        y_strip = area_closing(area_opening(ndvi_binary[:top + 256, :], 128), 256, 1)
        if left == 0:
            area_opening_ndvi_128[bottom:, :right] = y_strip[:top, :right]
        else:
            area_opening_ndvi_128[bottom:, left:] = y_strip[:top, left:]
    return area_opening_ndvi_128


def get_foliage(ndvi, fsi, rgb, thresh=0.05, use_floor_removal=True, mask=None, open_close=True):
    """
    :param ndvi: ndvi image
    :param fsi: fsi image
    :param rgb: rgb image
    :param thresh: binary threshold
    :param use_floor_removal: flag to use floor removal logic
    :param mask: mask of picture
    :param open_close: flag for using open and closing operations
    :return: foliage_mask
    """
    mask_size = 0 if isinstance(mask, type(None)) else np.sum(mask)
    ndvi_binary = ndvi_to_binary(ndvi, thresh)
    # plot_2_imgs(ndvi, ndvi_binary)
    if use_floor_removal:
        ndvi_binary = remove_floor(ndvi_binary, fsi, rgb, mask_size)
    if open_close:
        area_opening_ndvi_128 = open_ndvi(ndvi_binary, mask)
    else:
        return ndvi_binary
    return area_opening_ndvi_128


def get_ndvi_pictures(rgb, nir, fsi, boxes, mask=None, use_cuda=False,
                      use_floor_removal=True, open_close=False):
    """
    :param rgb: rgb image
    :param nir: infra red channel
    :param fsi: fsi image
    :param boxes: boxes for image
    :param mask: mask for image
    :param use_cuda: flag for using cuda
    :param use_floor_removal: flag for using floor removal
    :param open_close: falg for using open_close operations
    :return: ndvi image, total foilage, binary ndvi image, binary box image
    """
    # 92 % of time
    ndvi_img = make_ndvi(rgb, nir)
    box_img = make_bbox_pic(fsi, boxes)
    ndri_img = make_ndri(box_img)
    binary_box_img = ndvi_to_binary(ndri_img, 0.05)
    # plot_2_imgs(make_ndri(fsi), ndvi_to_binary(make_ndri(fsi), 0.05))
    if not use_cuda:
        #binary_box_img = area_closing(area_opening(binary_box_img, 25), 50)
        ndvi_binary = get_foliage(ndvi_img * (1 - binary_box_img), fsi, rgb, mask=mask,
                                  use_floor_removal=use_floor_removal, open_close=open_close)
    else:
        binary_box_img = cp.asnumpy(ndimage.grey_closing(ndimage.grey_opening(
                                        cp.array(binary_box_img), 25), 50))
        mask_size = 0 if isinstance(mask, type(None)) else np.sum(mask)
        ndvi_binary = ndvi_to_binary(ndvi_img * (1 - binary_box_img), 0)
        ndvi_binary = remove_floor(ndvi_binary, fsi, rgb, mask_size)
        ndvi_binary =cp.asnumpy(ndimage.grey_closing(ndimage.grey_opening(
                                            cp.array(ndvi_binary), 128), 256))
    return ndvi_img, ndvi_binary, binary_box_img


def get_pictures(tree_images, frame_number, with_zed=False, specific_pics=None):
    """
    :param tree_images: {"frame": {"fsi":fsi,"rgb":rgb,"zed":zed} for each frame}
    :param frame_number: number of frame to take images for
    :param with_zed: include zed picture
    :param specific_pics: specific pictures to pull
    :return: all relevant pictures
    """
    tree_image_frame = tree_images[frame_number]
    if not isinstance(specific_pics, type(None)):
        out_pics = []
        for image in specific_pics:
                out_pics.append(tree_image_frame[image])
        return out_pics
    fsi, rgb = tree_image_frame["fsi"], tree_image_frame["rgb"]
    nir, swir_975 = tree_image_frame["nir"], tree_image_frame["swir_975"]
    if with_zed:
        return fsi, rgb, nir.copy(), swir_975.copy(), tree_image_frame["zed"], tree_image_frame["zed_rgb"]
    return fsi, rgb, nir.copy(), swir_975.copy()


def get_fsi_and_masks(tree_images, minimal_frames, dets=None):
    """
    :param tree_images: {"frame": {"fsi":fsi,"rgb":rgb,"zed":zed} for each frame}
    :param minimal_frames: list of frame numbers
    :param dets: if not nan will use it the detections for the translation calculations
    :return: returns list of fsis and corresponding masks
    """
    if isinstance(dets, type(None)):
        fsi_list = [tree_images[frame]["rgb"].astype(np.uint8) for frame in minimal_frames]
    else:
        fsi_list = [make_bbox_pic(tree_images[frame]["rgb"].astype(np.uint8), dets[frame]) for frame in minimal_frames]
    masks = []
    if len(minimal_frames) > 1:
        masks = get_frames_overlap(file_list=fsi_list, method='at')
    return fsi_list, masks