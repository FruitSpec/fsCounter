import cv2
import numpy as np

from vision.tools.image_stitching import resize_img, find_keypoints, match_descriptors
from vision.tools.image_stitching import calc_affine_transform, calc_homography

def align_sensors(zed_rgb, jai_rgb, zed_angles=[110, 70], jai_angles=[62, 62]):

    grey_zed = cv2.cvtColor(zed_rgb, cv2.COLOR_RGB2GRAY)
    grey_jai = cv2.cvtColor(jai_rgb, cv2.COLOR_RGB2GRAY)

    zed_mid_h = grey_zed.shape[0] // 2
    #zed_mid_w = grey_zed.shape[1] // 2
    zed_half_height = int(zed_mid_h / (zed_angles[0] / 2) * (jai_angles[0] / 2))
    y_s = zed_mid_h - zed_half_height
    y_e = zed_mid_h + zed_half_height
    cropped_zed = grey_zed[y_s: y_e]

    im_zed, r_zed = resize_img(cropped_zed, 960)
    im_jai, r_jai = resize_img(grey_jai, 960)

    kp_zed, des_zed = find_keypoints(im_zed)
    kp_jai, des_jai = find_keypoints(im_jai)

    M, st = get_affine_matrix(kp_zed, kp_jai, des_zed, des_jai)
    tx, ty, sx, sy = affine_to_values(M)

    x1, y1, x2, y2 = get_coordinates_in_zed(im_zed, im_jai, tx, ty, sx, sy)
    coordinates = convert_coordinates_to_orig(x1, y1, x2, y2, y_s, r_zed)

    return coordinates

def convert_coordinates_to_orig(x1, y1, x2, y2, y_s, r_zed):

    arr = np.array([x1, y1, x2, y2])
    arr = arr / r_zed

    arr[1] += y_s
    arr[3] += y_s

    return arr

def get_affine_matrix(kp_zed, kp_jai, des_zed, des_jai):
    match = match_descriptors(des_zed, des_jai)
    M, st = calc_affine_transform(kp_zed, kp_jai, match)

    return M, st

def affine_to_values(M):
    sx = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
    sy = np.sqrt(M[1, 0] ** 2 + M[1, 1] ** 2)

    tx = np.round(M[0, 2]).astype(np.int)
    ty = np.round(M[1, 2]).astype(np.int)

    return tx, ty, sx, sy

def get_coordinates_in_zed(grey_zed, grey_jai, tx, ty, sx, sy):
    jai_in_zed_height = np.round(grey_jai.shape[0] * sy).astype(np.int)
    jai_in_zed_width = np.round(grey_jai.shape[1] * sx).astype(np.int)

    z_h = grey_zed.shape[0]
    z_w = grey_zed.shape[1]
    if tx > 0:
        if ty > 0:
            x1 = tx
            y1 = ty
            x2 = tx + jai_in_zed_width
            y2 = ty + jai_in_zed_height
        else:
            x1 = z_w - tx - jai_in_zed_width
            y1 = z_h - ty - jai_in_zed_height
            x2 = z_w - tx
            y2 = z_h - ty
    else:
        if ty > 0:
            x1 = z_w - tx - jai_in_zed_width
            y1 = z_h - ty - jai_in_zed_height
            x2 = tx + jai_in_zed_width
            y2 = ty + jai_in_zed_height
        else:
            x1 = z_w - tx - jai_in_zed_width
            y1 = z_h - ty - jai_in_zed_height
            x2 = z_w - tx
            y2 = z_h - ty

    return x1, y1, x2, y2


if __name__ == "__main__":
    zed_frame = "/home/fruitspec-lab/FruitSpec/Sandbox/merge_sensors/ZED/frame_548.jpg"
    depth_frame = "/home/fruitspec-lab/FruitSpec/Sandbox/merge_sensors/ZED/depth_frame_548.jpg"
    jai_frame = "/home/fruitspec-lab/FruitSpec/Sandbox/merge_sensors/FSI_2_30_720_30/frame_539.jpg"
    rgb_jai_frame = "/home/fruitspec-lab/FruitSpec/Sandbox/merge_sensors/FSI_2_30_720_30/rgb_539.jpg"

    jai = cv2.imread(jai_frame)
    jai = cv2.cvtColor(jai, cv2.COLOR_BGR2RGB)
    rgb_jai = cv2.imread(rgb_jai_frame)
    rgb_jai = cv2.cvtColor(rgb_jai, cv2.COLOR_BGR2RGB)
    zed = cv2.imread(zed_frame)
    zed = cv2.rotate(zed, cv2.ROTATE_90_CLOCKWISE)
    zed = cv2.cvtColor(zed, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_frame)
    depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)

    corr = align_sensors(zed, rgb_jai)
    print(corr)