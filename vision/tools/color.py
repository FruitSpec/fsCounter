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


def get_tomato_color(rgb_crop):
    """
    Determines the color of a tomato based on an RGB image crop of the tomato.

    Parameters
    ----------
    rgb_crop : numpy.ndarray
        An BGR image crop of a tomato.

    Returns
    -------
    tuple
        A number representing the color of the tomato in RGB format.

    Notes
    -----
    This function assumes that the tomato is the main object in the input image crop.
    The color of the tomato is determined based on the hue values of the input image in HSV color space.
    The returned color tuple corresponds to the following color ranges in the HSV color space:
        - 1: hue < 10
        - 2: 10 <= hue < 17.5
        - 3: 17.5 <= hue < 45 and median hue * 1.15 <= mean hue of central 50% of the crop
        - 4: 17.5 <= hue < 45 and median hue * 1.15 > mean hue of central 50% of the crop
        - 5: hue >= 45

    """
    if not len(rgb_crop):
        return 0
    hsv = cv2.cvtColor(rgb_crop.astype(np.uint8), cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv.copy())
    hist_vals, hist_bins = np.histogram(hue, bins=100)
    mode_hue = hist_bins[np.argmax(hist_vals)]
    if mode_hue < 10:
        return 1
    if mode_hue < 17.5:
        return 2
    if mode_hue < 45:
        w, h = rgb_crop.shape[:2]
        w_025 = int(w/4)
        h_025 = int(h/4)
        hue_cut = hue[h_025:-h_025, w_025:-w_025]
        mean_hue_cut, median_hue_cut = np.nanmean(hue_cut), np.nanmedian(hue_cut)
        if median_hue_cut * 1.15 > mean_hue_cut: # low "skew"
            return 4
        else:
            return 3
    return 5


def get_hue(det_crop, saturation_threshold=50):
    hsv = cv2.cvtColor(det_crop, cv2.COLOR_RGB2HSV)
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