import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from vision.tools.image_stitching import (resize_img, find_keypoints,get_affine_homography,
                                          get_fine_keypoints, get_fine_translation, get_affine_matrix)
from vision.tools.image_stitching import calc_affine_transform, calc_homography


# zed_angles = (84.1, 53.8)

def multi_convert_gray(list_imgs):
    return [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in list_imgs]


def crop_zed_roi(grey_zed, zed_angles, jai_angles, y_s=None, y_e=None, x_s=0, x_e=None):
    zed_mid_h = grey_zed.shape[0] // 2
    zed_half_height = int(zed_mid_h / (zed_angles[0] / 2) * (jai_angles[0] / 2))
    if isinstance(y_s, type(None)):
        y_s = zed_mid_h - zed_half_height-100
    if isinstance(y_e, type(None)):
        y_e = zed_mid_h + zed_half_height+100
    if isinstance(x_e, type(None)):
        x_e = grey_zed.shape[1]
    if isinstance(x_s, type(None)):
        x_s = 0
    cropped_zed = grey_zed[y_s: y_e, x_s:x_e]
    return cropped_zed, y_s, y_e, x_s, x_e


def plot_kp(zed_rgb, kp_zed, jai_rgb, kp_jai, y_s, y_e):
    zed_rgb_kp = resize_img(zed_rgb[y_s: y_e], 960)[0].astype(np.uint8)
    for curKey in kp_zed:
        x = np.int(curKey.pt[0])
        y = np.int(curKey.pt[1])
        size = np.int(curKey.size)
        cv2.circle(zed_rgb_kp, (x, y), size, (0, 255, 0), thickness=5, lineType=8, shift=0)

    jai_rgb_kp = resize_img(jai_rgb, 960)[0].astype(np.uint8)
    for curKey in kp_jai:
        x = np.int(curKey.pt[0])
        y = np.int(curKey.pt[1])
        size = np.int(curKey.size)
        cv2.circle(jai_rgb_kp, (x, y), size, (0, 255, 0), thickness=5, lineType=8, shift=0)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    ax1.imshow(jai_rgb_kp)
    ax2.imshow(zed_rgb_kp)
    plt.show()


def cut_black(mat):
    if len(mat.shape) > 2:
        gray_mat = multi_convert_gray([mat])[0]
    else:
        gray_mat = mat
    org_size = gray_mat.shape[:2][::-1]
    mat = mat[:, np.where(np.sum(gray_mat, axis=0) > 0)[0]]
    mat = mat[np.where(np.sum(gray_mat, axis=1) > 0)[0], :]
    return cv2.resize(mat, org_size)


def plot_homography(zed_rgb, M_homography):
    wraped_img = cv2.warpPerspective(zed_rgb, M_homography, zed_rgb.shape[:2][::-1])
    non_black_img = cut_black(wraped_img)
    plt.imshow(non_black_img)
    plt.show()
    print(M_homography)


def first_translation(cropped_zed, grey_jai, zed_rgb, jai_rgb, y_s, y_e):
    im_zed, r_zed = resize_img(cropped_zed, 960)
    im_jai, r_jai = resize_img(grey_jai, 960)
    kp_zed, des_zed = find_keypoints(im_zed)
    kp_jai, des_jai = find_keypoints(im_jai)
    M, st = get_affine_matrix(kp_zed, kp_jai, des_zed, des_jai)
    tx, ty, sx, sy = affine_to_values(M)
    # M_homography, st_homography = get_affine_homography(kp_zed, kp_jai, des_zed, des_jai)
    #plot_kp(zed_rgb, kp_zed, jai_rgb, kp_jai, y_s, y_e)
    # plot_homography(resize_img(zed_rgb[y_s: y_e], 960)[0], M_homography)
    return tx, ty, sx, sy, im_zed, im_jai, r_zed, kp_jai, des_jai


def get_fine_affine_translation(im_zed_stage2, im_jai):
    kp_des_zed, kp_des_jai = get_fine_keypoints(im_zed_stage2), get_fine_keypoints(im_jai)
    translation_res = get_fine_translation(kp_des_zed, kp_des_jai, max_workers=5)
    translation_res = np.array([list(affine_to_values(M)) for tx, ty, M in translation_res])
    tx, ty, sx, sy = np.nanmean(translation_res, axis=0)
    return tx, ty, sx, sy


def align_sensors(zed_rgb, jai_rgb, zed_angles=[110, 70], jai_angles=[62, 62],
                  use_fine=False, zed_roi_params=dict(y_s=None, y_e=None, x_s=0, x_e=None)):
    grey_zed, grey_jai = multi_convert_gray([zed_rgb, jai_rgb])
    cropped_zed, y_s, y_e, x_s, x_e = crop_zed_roi(grey_zed, zed_angles, jai_angles, **zed_roi_params)
    tx, ty, sx, sy, im_zed, im_jai, r_zed, kp_jai, des_jai = first_translation(cropped_zed, grey_jai, zed_rgb, jai_rgb,
                                                                               y_s, y_e)
    x1, y1, x2, y2 = get_coordinates_in_zed(im_zed, im_jai, tx, ty, sx, sy)
    x1, y1, x2, y2 = convert_coordinates_to_orig(x1, y1, x2, y2, y_s, x_s, r_zed)
    h_z = zed_rgb.shape[0]
    w_z = zed_rgb.shape[1]
    if use_fine:
        im_zed_stage2 = grey_zed[max(int(y1 - h_z / 20), 0): min(int(y2 + h_z / 20), h_z),
                                 max(int(x1 - w_z / 10), 0): min(int(x2 + w_z / 10), w_z)]
        im_zed_stage2, r_zed2 = resize_img(im_zed_stage2, 960)
        tx2, ty2, sx2, sy2 = get_fine_affine_translation(im_zed_stage2, im_jai)
        kp_zed, des_zed = find_keypoints(resize_img(grey_zed[max(int(y1), 0): min(int(y2), h_z),
                                                  max(int(x1), 0): min(int(x2), w_z)],960)[0])
        M_homography, st_homography = get_affine_homography(kp_zed, kp_jai, des_zed, des_jai)
        # plot_homography(zed_rgb[max(int(y1), 0): min(int(y2), h_z),
        #                         max(int(x1), 0): min(int(x2), w_z)], M_homography)
        s2_x1, s2_y1, s2_x2, s2_y2 = get_coordinates_in_zed(im_zed_stage2, im_jai, tx2, ty2, sx2, sy2)
        s2_x1, s2_y1, s2_x2, s2_y2 = convert_coordinates_to_orig(s2_x1, s2_y1, s2_x2, s2_y2, 0, r_zed2)
        x1 = max(int(x1-w_z/10), 0) + int(s2_x1)
        x2 = x1 + int((s2_x2-s2_x1))
        y1 = max(int(y1-h_z/20), 0) + int(s2_y1)
        y2 = y1 + int((s2_y2 - s2_y1))
        tx, ty = int(tx + tx2 - w_z / 10), int(ty + ty2 - h_z / 20)

    coordinates = x1, y1, x2, y2
    return coordinates, tx, ty, sx, sy


def convert_coordinates_to_orig(x1, y1, x2, y2, y_s, x_s, r_zed):

    arr = np.array([x1, y1, x2, y2])
    arr = arr / r_zed

    arr[1] += y_s
    arr[3] += y_s

    arr[0] += x_s
    arr[2] += x_s

    return arr


def affine_to_values(M):
    if isinstance(M, type(None)):
        return np.nan, np.nan, np.nan, np.nan
    if np.isnan(M[0, 2]) or np.isnan(M[1, 2]) :
        return np.nan, np.nan, np.nan, np.nan
    sx = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
    sy = np.sqrt(M[1, 0] ** 2 + M[1, 1] ** 2)

    tx = np.round(M[0, 2]).astype(np.int)
    ty = np.round(M[1, 2]).astype(np.int)

    return tx, ty, sx, sy


def get_coordinates_in_zed(grey_zed, grey_jai, tx, ty, sx, sy):
    jai_in_zed_height = np.round(grey_jai.shape[0] * sy).astype(np.int)
    jai_in_zed_width = np.round(grey_jai.shape[1] * sx).astype(np.int)
    # jai_in_zed_height = np.round(grey_zed.shape[0] * sy).astype(np.int)
    # jai_in_zed_width = np.round(grey_zed.shape[1] * sx).astype(np.int)

    z_h = grey_zed.shape[0]
    z_w = grey_zed.shape[1]
    if tx > 0:
        x1 = tx
        x2 = tx + jai_in_zed_width
        if ty > 0:
            y1 = ty
            y2 = ty + jai_in_zed_height
            # x1 = tx
            # x2 = x1 + jai_in_zed_width
            # y1 = z_h + ty - jai_in_zed_height# r
            # y2 = z_h + ty # r
        else:
            # x1 = z_w - tx - jai_in_zed_width
            # y1 = z_h - ty - jai_in_zed_height
            # x2 = tx + jai_in_zed_width
            # y2 = z_h - ty
            y1 = z_h + ty - jai_in_zed_height# r
            y2 = z_h + ty # r

    else:
        x2 = z_w + tx  # r
        x1 = x2 - jai_in_zed_width  # r
        if ty > 0:
            # x1 = z_w - tx - jai_in_zed_width
            # y1 = z_h - ty - jai_in_zed_height
            # x2 = tx + jai_in_zed_width
            # y2 = ty + jai_in_zed_height
            y1 = ty
            y2 = ty + jai_in_zed_height
        else:
            x1 = z_w - tx - jai_in_zed_width
            y1 = z_h - ty - jai_in_zed_height
            x2 = z_w - tx
            y2 = z_h - ty
            y1 = z_h + ty - jai_in_zed_height# r
            y2 = z_h + ty# r
    return x1, y1, x2, y2


def align_folder(folder_path, result_folder="", plot_res=True, use_fine=False, zed_shift=0,
                 zed_roi_params=dict(y_s=None, y_e=None, x_s=0, x_e=None)):
    if result_folder == "":
        result_folder = folder_path
    frames = [frame.split(".")[0].split("_")[-1] for frame in os.listdir(folder_path) if "FSI" in frame]
    df_out = pd.DataFrame({"x1": [], "x2": [], "y1": [], "y2": [],
                           "tx": [], "ty": [], "sx": [], "sy": [], "frame": [], "zed_shift": []})
    frames.sort(key=lambda x: int(x))
    consec_less_threshold = 0
    consec_more_threshold = 0
    for frame in tqdm(frames):
        zed_path = os.path.join(folder_path, f"frame_{int(frame)+zed_shift}.jpg")
        rgb_path = os.path.join(folder_path, f"channel_RGB_frame_{frame}.jpg")
        if not (os.path.exists(zed_path) and os.path.exists(rgb_path)):
            continue
        rgb_jai = cv2.imread(rgb_path)
        rgb_jai = cv2.cvtColor(rgb_jai, cv2.COLOR_BGR2RGB)
        zed = cv2.imread(zed_path)
        zed = cv2.cvtColor(zed, cv2.COLOR_BGR2RGB)
        corr, tx, ty, sx, sy = align_sensors(zed, rgb_jai, use_fine=use_fine, zed_roi_params=zed_roi_params)
        x1, y1, x2, y2 = corr
        # print(f"frame:{frame}, x1:{int(x1)}, tx: {tx}, zed_shift: {zed_shift}")
        df_out = df_out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                "tx": tx, "ty": ty, "sx": sx, "sy": sy, "frame": frame, "zed_shift": zed_shift}, ignore_index=True)
        if plot_res:
            try:
                img1 = rgb_jai
                img2 = zed
                jai_in_zed = img2[int(y1): min(int(y2), zed.shape[0]), max(int(x1), 0): min(int(x2), zed.shape[1])]
                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
                ax1.imshow(cv2.resize(img1, jai_in_zed.shape[:2][::-1]))
                ax2.imshow(jai_in_zed)
                plt.show()
            except:
                print("resize problem")
        if tx < 20:
            consec_less_threshold+=1
        else:
            consec_less_threshold = 0
        if consec_less_threshold > 5:
            zed_shift-=1
            consec_less_threshold = 0
            consec_more_threshold = 0
        if tx > 120:
            consec_more_threshold+=1
        else:
            consec_more_threshold = 0
        if consec_more_threshold > 5:
            zed_shift+=1
            consec_less_threshold = 0
            consec_more_threshold = 0
    df_out.to_csv(os.path.join(result_folder, "jai_cors_in_zed.csv"))
    plt.plot(df_out["tx"])
    plt.ylim(-200, 200)
    plt.show()
    print(f"aligned: {folder_path}")


if __name__ == "__main__":
    align_folder("/media/fruitspec-lab/easystore/JAIZED_CaraCara_301122/R6/frames", use_fine=False,
                 zed_roi_params=dict(x_s=0, x_e=1080, y_s=310, y_e=1670), zed_shift=0)
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