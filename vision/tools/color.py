import numpy as np
import cv2


def get_color(det_crop, saturation_threshold=100, precentile=0.5):
    hsv = cv2.cvtColor(det_crop, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].copy()
    hue *= 2
    h, b = np.histogram(hue[hsv[:, :, 1] > saturation_threshold].flatten(), 360)
    p = np.cumsum(h) / sum(h)
    for i, percent in enumerate(p):
        if percent > precentile:
            break

    return b[i]


def get_hue(det_crop, saturation_threshold=50):
    hsv = cv2.cvtColor(det_crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].copy()
    h, b = np.histogram(hue[hsv[:, :, 1] > saturation_threshold].flatten(), 360)

    return h, b


def hue_filtering(rgb_crop, nstds=1):
    """
    Apply hue filtering to an RGB image crop.

    Parameters:
        rgb_crop (numpy.ndarray): Input RGB image crop as a numpy array.
        nstds (float): Number of standard deviations used to determine the upper and lower hue thresholds.

    Returns:
        numpy.ndarray: mask where True value indicated what was filtered out
    """
    rgb_c = rgb_crop.copy()
    hsv = cv2.cvtColor(rgb_c, cv2.COLOR_BGR2HSV)
    hue, sat, v = cv2.split(hsv.copy())
    hist_vals, hist_bins = np.histogram(hue, bins=50)
    mode = hist_bins[np.argmax(hist_vals)]
    if mode > 35:  # greener area
        nstds *= 1.5
    hue_std = np.std(hue)
    upper_limit = mode + nstds * hue_std
    lower_limit = mode - nstds * hue_std
    logical_vec = np.any([hue > upper_limit, hue < lower_limit], axis=0)
    return logical_vec