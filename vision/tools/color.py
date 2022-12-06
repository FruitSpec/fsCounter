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

