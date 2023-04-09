import os
import cv2
import numpy as np
np.int = np.int_
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

from vision.tools.image_stitching import (resize_img, find_keypoints, get_affine_homography, affine_to_values,
                                          get_fine_keypoints, get_fine_translation, get_affine_matrix,
                                          get_fine_affine_translation, find_loftr_translation)
from vision.tools.image_stitching import calc_affine_transform, calc_homography, plot_2_imgs
from vision.feature_extractor.image_processing import multi_convert_gray
import seaborn as sns
#import cupy as cp
np.random.seed(123)
cv2.setRNGSeed(123)

class SensorAligner:
    """
    this class is for aligning the zed and jai cameras
    """
    def __init__(self, args, zed_shift=0):
        self.zed_angles = args.zed_angles
        self.jai_angles = args.jai_angles
        self.use_fine = args.use_fine
        self.zed_roi_params = args.zed_roi_params
        self.median_thresh = args.median_thresh
        self.remove_high_blues = args.remove_high_blues
        self.y_s, self.y_e, self.x_s, self.x_e = self.zed_roi_params.values()
        self.debug = args.debug
        self.zed_shift, self.consec_less_threshold, self.consec_more_threshold = zed_shift, 0, 0
        self.r_jai, self.r_zed = 1, 1
        self.size = args.size
        self.apply_normalization = args.apply_normalization
        self.apply_equalization = args.apply_equalization
        self.fixed_scaling = args.fixed_scaling
        self.affine_method = args.affine_method
        self.ransac = args.ransac
        self.matcher = self.init_matcher(args)
        self.sx = 0.60546875 #0.6102498372395834
        self.sy =  0.6133919843597263#0.6136618198110134
        self.roix = 930 #937
        self.roiy = 1255

    def init_matcher(self, args):

        matcher = cv2.SIFT_create()
        # matcher.setNOctaveLayers(6)
        # matcher.setEdgeThreshold(20)
        # matcher.setSigma(1)
        # matcher.setContrastThreshold(0.03)

        return matcher

    def crop_zed_roi(self, gray_zed):
        """
        crops region of interest of jai image inside the zed image
        :param gray_zed: image of gray_scale_zed
        :return: cropped image
        """
        zed_mid_h = gray_zed.shape[0] // 2
        zed_half_height = int(zed_mid_h / (self.zed_angles[0] / 2) * (self.jai_angles[0] / 2))
        if isinstance(self.y_s, type(None)):
            self.y_s = zed_mid_h - zed_half_height - 100
        if isinstance(self.y_e, type(None)):
            self.y_e = zed_mid_h + zed_half_height + 100
        if isinstance(self.x_e, type(None)):
            self.x_e = gray_zed.shape[1]
        if isinstance(self.x_s, type(None)):
            self.x_s = 0
        cropped_zed = gray_zed[self.y_s: self.y_e, self.x_s:self.x_e]
        return cropped_zed

    def first_translation(self, im_zed, im_jai, zed_rgb=None, jai_rgb=None):
        """
        applies first translation on an image
        :param im_zed: zed gray image cropped for ROI
        :param gray_jai: grayscale jai image
        :param zed_rgb: original zed rgb image for debugging
        :param jai_rgb: original jai rgb image for debugging
        :return: translation, scaling, resized images, keypoints and destenations of jai
        """
        if self.size not in im_zed.shape:
            im_zed, self.r_zed = resize_img(im_zed, self.size)
        if self.size not in im_jai.shape:
            im_jai, self.r_jai = resize_img(im_jai, self.size)
        if self.affine_method == "keypoints":
            kp_zed, des_zed = find_keypoints(im_zed, self.matcher) # consumes 33% of time
            kp_jai, des_jai = find_keypoints(im_jai, self.matcher) # consumes 33% of time
            M, st, match = get_affine_matrix(kp_zed, kp_jai, des_zed, des_jai, self.ransac, self.fixed_scaling) # consumes 33% of time
            tx, ty, sx, sy = affine_to_values(M)
        elif self.affine_method == "loftr":
            M, st, kp_zed, kp_jai = find_loftr_translation(im_zed, im_jai, True)
            des_jai = None
            tx, ty, sx, sy = affine_to_values(M)

        if self.debug.keypoints and not isinstance(zed_rgb, type(None)) and not isinstance(jai_rgb, type(None)):
            M_homography, st_homography = get_affine_homography(kp_zed, kp_jai, des_zed, des_jai)
            plot_kp(zed_rgb, kp_zed, jai_rgb, kp_jai, self.y_s, self.y_e)
            plot_homography(resize_img(zed_rgb[self.y_s: self.y_e], self.size)[0], M_homography)
        return tx, ty, sx, sy, im_zed, im_jai, kp_jai, des_jai, kp_zed, des_zed, match, st, M

    def normalize_img(self, image, original_img):
        """
        normalizes channels of image if the median value is smaller then the threshold
        :param image: image to normalize
        :param original_img: original image for real median value and debugging purposes
        :return: normalized image
        """
        if np.median(original_img) < self.median_thresh: # picture is saturated with most signal around low values
            if len(image.shape) > 2:
                for i in range(3):
                    cur_channel = image[:, :, i]
                    in_range_mask = np.all([cur_channel > 0, cur_channel < 230], axis=0)
                    clipping_top_val = np.quantile(cur_channel[in_range_mask], 0.95)
                    clipped_channel = np.clip(cur_channel, 0, clipping_top_val)
                    normalized_data = cv2.normalize(clipped_channel, None, alpha=25, beta=255,
                                                                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    image[:, :, i][in_range_mask] = normalized_data[in_range_mask]
            else:
                in_range_mask = np.all([image > 0, image < 230], axis=0)
                clipping_top_val = np.quantile(image[in_range_mask], 0.95)
                clipped_channel = np.clip(image, 0, clipping_top_val)
                normalized_data = cv2.normalize(clipped_channel, None, alpha=25, beta=255,
                                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                image[in_range_mask] = normalized_data[in_range_mask]
            if self.debug.preprocess:
                plot_2_imgs(original_img, image)
        return image

    def preprocess(self, zed_rgb, jai_img):
        """
        preprocesses the images for alignment
        :param zed_rgb: zed imgae
        :param jai_img: jai image
        :return: processed images

        """
        jai_img_c, self.r_jai = resize_img(jai_img, self.size)
        zed_rgb_c, self.r_zed = resize_img(self.crop_zed_roi(zed_rgb), self.size)
        if self.remove_high_blues:
            jai_img_c[jai_img_c[:, :, 2] > 230] = 0
            zed_rgb_c[zed_rgb_c[:, :, 2] > 230] = 0
        if self.apply_normalization:
            jai_img_c = self.normalize_img(jai_img_c, jai_img)
            zed_rgb_c = self.normalize_img(zed_rgb_c, zed_rgb)
        gray_zed, gray_jai = cv2.cvtColor(zed_rgb_c, cv2.COLOR_RGB2GRAY), cv2.cvtColor(jai_img_c, cv2.COLOR_RGB2GRAY)
        if self.apply_equalization:
            gray_jai = cv2.equalizeHist(gray_jai)
            gray_zed = cv2.equalizeHist(gray_zed)
        return gray_zed, gray_jai, jai_img_c, zed_rgb_c

    def align_on_batch(self, zed_batch, jai_batch, workers=4):
        zed_input = []
        jai_input = []
        for z, j in zip(zed_batch, jai_batch):
            if z is not None and j is not None:
                zed_input.append(z)
                jai_input.append(j)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(self.align_sensors, zed_input, jai_input))

        output = []
        for r in results:
            self.update_zed_shift(r[1])
            output.append([r[0], r[1], r[2], r[3], r[4], self.zed_shift])

        return output



    def align_calib_sensors(self, zed_rgb, jai_img, jai_drop=False, zed_drop=False):
        """
        aligns both sensors and updates the zed shift
        :param zed_rgb: rgb image
        :param jai_img: jai imgae
        :param jai_drop: flag if jai had a frame drop
        :param zed_drop: flag if zed had a frame drop
        :return: jai_in_zed coors, tx,ty,sx,sy, if use fine is set to True also returns keypoints, des, jai_image
        """

        gray_zed = cv2.cvtColor(zed_rgb, cv2.COLOR_RGB2GRAY)
        gray_jai = cv2.cvtColor(jai_img, cv2.COLOR_RGB2GRAY)

        # adjust zed scale to be the same as jai using calibrated scale x and y
        gray_zed = self.crop_zed_roi(gray_zed)
        gray_zed = cv2.resize(gray_zed, (int(gray_zed.shape[1] / self.sx), int(gray_zed.shape[0] / self.sy)))

        gray_zed, rz = resize_img(gray_zed, gray_zed.shape[0] // 3)
        gray_jai, rj = resize_img(gray_jai, gray_jai.shape[0] // 3)

        kp_zed, des_zed = find_keypoints(gray_zed, self.matcher)  # consumes 33% of time
        kp_jai, des_jai = find_keypoints(gray_jai, self.matcher)  # consumes 33% of time
        M, st, match = get_affine_matrix(kp_zed, kp_jai, des_zed, des_jai, self.ransac,
                                         self.fixed_scaling)  # consumes 33% of time

        dst_pts = np.float32([kp_zed[m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
        dst_pts = dst_pts[st.reshape(-1).astype(np.bool_)]
        src_pts = np.float32([kp_jai[m.trainIdx].pt for m in match]).reshape(-1, 1, 2)
        src_pts = src_pts[st.reshape(-1).astype(np.bool_)]

        deltas = np.array(dst_pts) - np.array(src_pts)

        tx = np.mean(deltas[:,0,0]) / rz * self.sx
        ty = np.mean(deltas[:,0,1]) / rz * self.sy

        if tx < 0:
            x1 = 0
            x2 = self.roix
        elif tx + self.roix > zed_rgb.shape[1]:
            x2 = zed_rgb.shape[1]
            x1 = zed_rgb.shape[1] - self.roix
        else:
            x1 = tx
            x2 = tx + self.roix

        if ty < 0:
            y1 = self.y_s
            y2 = self.y_s + self.roiy
        elif ty + self.roiy > (self.y_e - self.y_s):
            y2 = self.y_e
            y1 = self.y_e - self.roiy
        else:
            y1 = self.y_s + ty
            y2 = self.y_s + ty + self.roiy

        #self.update_zed_shift(tx)

        return (x1, y1, x2, y2), tx, ty, kp_zed, kp_jai, gray_zed, gray_jai, match, st

    def align_sensors(self, zed_rgb, jai_img, jai_drop=False, zed_drop=False):
        """
        aligns both sensors and updates the zed shift
        :param zed_rgb: rgb image
        :param jai_img: jai imgae
        :param jai_drop: flag if jai had a frame drop
        :param zed_drop: flag if zed had a frame drop
        :return: jai_in_zed coors, tx,ty,sx,sy, if use fine is set to True also returns keypoints, des, jai_image
        """
        if jai_drop:
            self.zed_shift += 1
        if zed_drop:
            self.zed_shift -= 1
        gray_zed, gray_jai, jai_img_c, zed_rgb_c = self.preprocess(zed_rgb, jai_img)
        if self.debug.kde:
            sns.kdeplot(gray_zed.flatten(), color="green")
            sns.kdeplot(gray_jai.flatten(), color="blue")
            plt.show()
        tx, ty, sx, sy, im_zed, im_jai, kp_jai, des_jai, kp_zed, des_zed, match, st, M = self.first_translation(gray_zed,
                                                                                                                gray_jai,
                                                                                                                zed_rgb_c,
                                                                                                                jai_img_c) #consumes >90% of function time

        x1, y1, x2, y2 = self.convert_translation_to_coors(im_zed, im_jai, tx, ty, sx, sy)
        # plot_2_imgs(zed_rgb[int(y1):int(y2), int(x1):int(x2)], jai_img)
        if not self.use_fine:
            return (x1, y1, x2, y2), tx, ty, sx, sy, kp_zed, kp_jai, match, st, M
        return self.align_sensors_fine(x1, y1, x2, y2, tx, ty, sx, sy, kp_jai, des_jai, im_jai, zed_rgb, gray_zed)

    def convert_translation_to_coors(self, im_zed, im_jai, tx, ty, sx, sy):
        """
        converts translation output to x1, y1, x2, y2
        :param im_zed: gray zed cropped image
        :param im_jai: gray jai image
        :param tx: translation in x axis
        :param ty: translation in y axis
        :param sx: scale in x axis
        :param sy: scale in y axis
        :return: x1, y1, x2, y2 (where is the jai in zed image, scaled to zed image original image)
        """
        if np.any([np.isnan(val) for val in [tx, ty, sx, sy]]):
            x1, y1, x2, y2 = 0, 0, *im_zed.shape[::-1]
            x2 -= 1
            y1 -= 1
        else:
            x1, y1, x2, y2 = self.get_coordinates_in_zed(im_zed, im_jai, tx, ty, sx, sy)
            x1, y1, x2, y2 = self.convert_coordinates_to_orig(x1, y1, x2, y2)
        return x1, y1, x2, y2

    @staticmethod
    def get_coordinates_in_zed(gray_zed, gray_jai, tx, ty, sx, sy):
        """
        converts translation output to bbox (x1, y1, x2, y2)
        :param gray_zed: zed gray image
        :param gray_jai: jai gray image
        :param tx: translation in x axis
        :param ty: translation in y axis
        :param sx: scale in x axis
        :param sy: scale in y axis
        :return: x1, y1, x2, y2 of jai image inside zed
        """
        jai_in_zed_height = np.round(gray_jai.shape[0] * sy).astype(np.int)
        jai_in_zed_width = np.round(gray_jai.shape[1] * sx).astype(np.int)
        z_w = gray_zed.shape[1]
        x1 = tx
        x2 = tx + jai_in_zed_width
        # else:
        #     x2 = z_w + tx
        #     x1 = x2 - jai_in_zed_width
        if ty > 0:
            y1 = ty
            y2 = ty + jai_in_zed_height
        else:
            y1 = - ty
            y2 = jai_in_zed_height - ty
        return x1, y1, x2, y2

    def convert_coordinates_to_orig(self, x1, y1, x2, y2):
        """
        converts coordinates to original zed scale
        :param x1: top left x coordinate
        :param y1: top left y coordinate
        :param x2: bottom right x coordinate
        :param y2: bottom right y coordinate
        :return:
        """
        arr = np.array([x1, y1, x2, y2]) / self.r_zed
        arr[[1, 3]] += self.y_s
        arr[[0, 2]] += self.x_s
        return arr

    def align_sensors_fine(self, x1, y1, x2, y2, tx, ty, sx, sy, kp_jai, des_jai, im_jai, zed_rgb, gray_zed):
        """
        aligns both sensors and updates the zed shift using fine translation
        :param x1: top left x coordinate
        :param y1: top left y coordinate
        :param x2: bottom right x coordinate
        :param y2: bottom right y coordinate
        :param tx: translation in x axis
        :param ty: translation in y axis
        :param sx: scale in x axis
        :param sy: scale in y axis
        :param kp_jai: key points of jai
        :param des_jai: destination of jai
        :param im_jai: jai gray image
        :param zed_rgb: zed rgb image
        :param gray_zed: zed gray image
        :return: (x1, y1, x2, y2), tx, ty, sx, sy
        """
        # TODO this function is not tested yet!
        h_z, w_z = zed_rgb.shape[0], zed_rgb.shape[1]
        im_zed_stage2 = gray_zed[max(int(y1 - h_z / 20), 0): min(int(y2 + h_z / 20), h_z),
                        max(int(x1 - w_z / 10), 0): min(int(x2 + w_z / 10), w_z)]
        im_zed_stage2, r_zed2 = resize_img(im_zed_stage2, 960)
        tx2, ty2, sx2, sy2 = get_fine_affine_translation(im_zed_stage2, im_jai)
        kp_zed, des_zed = find_keypoints(resize_img(gray_zed[max(int(y1), 0): min(int(y2), h_z),
                                                    max(int(x1), 0): min(int(x2), w_z)], 960)[0])
        if self.debug.homography:
            M_homography, st_homography = get_affine_homography(kp_zed, kp_jai, des_zed, des_jai)
            plot_homography(zed_rgb[max(int(y1), 0): min(int(y2), h_z),
                                    max(int(x1), 0): min(int(x2), w_z)], M_homography)
        s2_x1, s2_y1, s2_x2, s2_y2 = self.convert_translation_to_coors(im_zed_stage2, im_jai, tx2, ty2, sx2, sy2, r_zed2)
        x1 = max(int(x1 - w_z / 10), 0) + int(s2_x1)
        x2 = x1 + int((s2_x2 - s2_x1))
        y1 = max(int(y1 - h_z / 20), 0) + int(s2_y1)
        y2 = y1 + int((s2_y2 - s2_y1))
        tx, ty = int(tx + tx2 - w_z / 10), int(ty + ty2 - h_z / 20)
        self.update_zed_shift(tx)
        return (x1, y1, x2, y2), tx, ty, sx, sy

    def update_zed_shift(self, tx):
        """
        updated zed shift params
        :param tx: translation in x axis
        :return:
        """
        if isinstance(tx, type(None)):
            return
        if np.isnan(tx):
            return
        if tx < 20:
            self.consec_less_threshold += 1
        elif tx > 75:  # 120:
            self.consec_more_threshold += 1
        else:
            self.consec_less_threshold = max(self.consec_less_threshold - 1, 0)
            self.consec_more_threshold = max(self.consec_more_threshold - 1, 0)
        if self.consec_more_threshold > 5:
            self.zed_shift += 1
            self.consec_less_threshold = 0
            self.consec_more_threshold = 0
        if self.consec_less_threshold > 5:
            self.zed_shift -= 1
            self.consec_less_threshold = 0
            self.consec_more_threshold = 0

def multi_convert_gray(list_imgs):
    return [cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY) for img in list_imgs]


def crop_zed_roi(gray_zed, zed_angles, jai_angles, y_s=None, y_e=None, x_s=0, x_e=None):
    zed_mid_h = gray_zed.shape[0] // 2
    zed_half_height = int(zed_mid_h / (zed_angles[0] / 2) * (jai_angles[0] / 2))
    if isinstance(y_s, type(None)):
        y_s = zed_mid_h - zed_half_height-100
    if isinstance(y_e, type(None)):
        y_e = zed_mid_h + zed_half_height+100
    if isinstance(x_e, type(None)):
        x_e = gray_zed.shape[1]
    if isinstance(x_s, type(None)):
        x_s = 0
    cropped_zed = gray_zed[y_s: y_e, x_s:x_e]
    return cropped_zed, y_s, y_e, x_s, x_e



def first_translation(cropped_zed, gray_jai, zed_rgb, jai_rgb, y_s, y_e):
    im_zed, r_zed = resize_img(cropped_zed, 960)
    im_jai, r_jai = resize_img(gray_jai, 960)
    kp_zed, des_zed = find_keypoints(im_zed)
    kp_jai, des_jai = find_keypoints(im_jai)
    M, st = get_affine_matrix(kp_zed, kp_jai, des_zed, des_jai)
    tx, ty, sx, sy = affine_to_values(M)
    # M_homography, st_homography = get_affine_homography(kp_zed, kp_jai, des_zed, des_jai)
    #plot_kp(zed_rgb, kp_zed, jai_rgb, kp_jai, y_s, y_e)
    # plot_homography(resize_img(zed_rgb[y_s: y_e], 960)[0], M_homography)
    return tx, ty, sx, sy, im_zed, im_jai, r_zed, kp_jai, des_jai


def align_sensors(zed_rgb, jai_rgb, zed_angles=[110, 70], jai_angles=[62, 62],
                  use_fine=False, zed_roi_params=dict(y_s=None, y_e=None, x_s=0, x_e=None),
                  whiteness_thresh=0.05, remove_high_blues=True):
    # cv2.threshold(np.mean(jai_rgb, axis=2).astype(np.uint8), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # cv2.normalize(np.clip(jai_rgb, 0, 70), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    jai_rgb, zed_rgb = jai_rgb.copy(), zed_rgb.copy()
    if remove_high_blues:
        jai_rgb[jai_rgb[:, :, 2] > 240] = 0
        zed_rgb[zed_rgb[:, :, 2] > 240] = 0
    flat_rgb = jai_rgb.flatten()
    pix_dist = np.histogram(flat_rgb, bins=255, density=True)[0]
    if np.sum(pix_dist[200:240]) < whiteness_thresh:
        rgb_old = jai_rgb.copy()
        clipped_rgb = np.clip(jai_rgb, 0, np.quantile(flat_rgb[np.all([flat_rgb > 0, flat_rgb < 230], axis=0)], 0.9))
        jai_rgb = cv2.normalize(clipped_rgb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #     # plot_2_imgs(rgb_old, jai_rgb)
    gray_zed, gray_jai = multi_convert_gray([zed_rgb, jai_rgb])
    # import seaborn as sns
    # sns.kdeplot(gray_zed.flatten(), color="green")
    # sns.kdeplot(gray_jai.flatten(), color="blue")
    # plt.show()
    cropped_zed, y_s, y_e, x_s, x_e = crop_zed_roi(gray_zed, zed_angles, jai_angles, **zed_roi_params)
    tx, ty, sx, sy, im_zed, im_jai, r_zed, kp_jai, des_jai = first_translation(cropped_zed, gray_jai, zed_rgb, jai_rgb,
                                                                               y_s, y_e)
    if np.any([np.isnan(val) for val in [tx, ty, sx, sy]]):
        x1, y1, x2, y2 = 0, 0, *im_zed.shape[::-1]
        x2 -= 1
        y1 -= 1
    else:
        x1, y1, x2, y2 = get_coordinates_in_zed(im_zed, im_jai, tx, ty, sx, sy)
        x1, y1, x2, y2 = convert_coordinates_to_orig(x1, y1, x2, y2, y_s, x_s, r_zed)
    h_z = zed_rgb.shape[0]
    w_z = zed_rgb.shape[1]
    if use_fine:
        im_zed_stage2 = gray_zed[max(int(y1 - h_z / 20), 0): min(int(y2 + h_z / 20), h_z),
                                 max(int(x1 - w_z / 10), 0): min(int(x2 + w_z / 10), w_z)]
        im_zed_stage2, r_zed2 = resize_img(im_zed_stage2, 960)
        tx2, ty2, sx2, sy2 = get_fine_affine_translation(im_zed_stage2, im_jai)
        kp_zed, des_zed = find_keypoints(resize_img(gray_zed[max(int(y1), 0): min(int(y2), h_z),
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

    return (x1, y1, x2, y2), tx, ty, sx, sy


def convert_coordinates_to_orig(x1, y1, x2, y2, y_s, x_s, r_zed):

    arr = np.array([x1, y1, x2, y2])
    arr = arr / r_zed

    arr[1] += y_s
    arr[3] += y_s

    arr[0] += x_s
    arr[2] += x_s

    return arr


def get_coordinates_in_zed(gray_zed, gray_jai, tx, ty, sx, sy):
    jai_in_zed_height = np.round(gray_jai.shape[0] * sy).astype(np.int)
    jai_in_zed_width = np.round(gray_jai.shape[1] * sx).astype(np.int)

    z_w = gray_zed.shape[1]
    if tx > 0:
        x1 = tx
        x2 = tx + jai_in_zed_width
    else:
        x2 = z_w + tx
        x1 = x2 - jai_in_zed_width
    if ty > 0:
        y1 = ty
        y2 = ty + jai_in_zed_height
    else:
        y1 = - ty
        y2 = jai_in_zed_height - ty
    return x1, y1, x2, y2


def update_zed_shift(tx, zed_shift, consec_less_threshold, consec_more_threshold):
    if isinstance(tx, type(None)):
        return zed_shift, consec_less_threshold, consec_more_threshold
    if np.isnan(tx):
        return zed_shift, consec_less_threshold, consec_more_threshold
    if tx < 20:
        consec_less_threshold += 1
    else:
        consec_less_threshold = 0
    if consec_less_threshold > 5:
        zed_shift -= 1
        consec_less_threshold = 0
        consec_more_threshold = 0
    if tx > 120:
        consec_more_threshold += 1
    else:
        consec_more_threshold = 0
    if consec_more_threshold > 5:
        zed_shift += 1
        consec_less_threshold = 0
        consec_more_threshold = 0
    return zed_shift, consec_less_threshold, consec_more_threshold


def align_folder(folder_path, result_folder="", plot_res=True, use_fine=False, zed_shift=0,
                 zed_roi_params=dict(y_s=None, y_e=None, x_s=0, x_e=None)):
    if result_folder == "":
        result_folder = folder_path
    frames = [frame.split(".")[0].split("_")[-1] for frame in os.listdir(folder_path) if "FSI" in frame]
    df_out_list = []
    frames.sort(key=lambda x: int(x))
    consec_less_threshold, consec_more_threshold = 0, 0
    for frame in tqdm(frames):
        zed_path = os.path.join(folder_path, f"frame_{int(frame)+zed_shift}.jpg")
        rgb_path = os.path.join(folder_path, f"channel_RGB_frame_{frame}.jpg")
        if not (os.path.exists(zed_path) and os.path.exists(rgb_path)):
            continue
        rgb_jai = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        zed = cv2.cvtColor(cv2.imread(zed_path), cv2.COLOR_BGR2RGB)
        if is_sturated(rgb_jai_frame, 0.5) or is_sturated(zed_frame, 0.5):
            print(f'frame {frame} is saturated, skipping')
            continue
        corr, tx, ty, sx, sy = align_sensors(zed, rgb_jai, use_fine=use_fine, zed_roi_params=zed_roi_params)
        x1, y1, x2, y2 = corr
        # print(f"frame:{frame}, x1:{int(x1)}, tx: {tx}, zed_shift: {zed_shift}")
        df_out_list.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                "tx": tx, "ty": ty, "sx": sx, "sy": sy, "frame": frame, "zed_shift": zed_shift})
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
        zed_shift, consec_less_threshold, consec_more_threshold = update_zed_shift(tx, zed_shift, consec_less_threshold,
                                                                               consec_more_threshold)
    df_out = pd.DataFrame.from_records(df_out_list)
    df_out.to_csv(os.path.join(result_folder, "jai_cors_in_zed.csv"))
    plt.plot(df_out["tx"])
    plt.ylim(-200, 200)
    plt.show()
    print(f"aligned: {folder_path}")


def plot_kp(zed_rgb, kp_zed, jai_rgb, kp_jai, y_s, y_e):
    """
    this function is a debugging tool for image keypoints
    :param zed_rgb: zed image
    :param kp_zed: zed keypoints
    :param jai_rgb: jai image
    :param kp_jai: jai keypoints
    :param y_s: y axis start value
    :param y_e: y axis end value
    :return: None
    """
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
    """
    cuts out the black part of the image after homograpy is applied
    :param mat: the image
    :return: an image with the black part cut out
    """
    if len(mat.shape) > 2:
        gray_mat = multi_convert_gray([mat])[0]
    else:
        gray_mat = mat
    org_size = gray_mat.shape[:2][::-1]
    mat = mat[:, np.where(np.sum(gray_mat, axis=0) > 0)[0]]
    mat = mat[np.where(np.sum(gray_mat, axis=1) > 0)[0], :]
    return cv2.resize(mat, org_size)


def plot_homography(zed_rgb, M_homography):
    """
    this is a tool for validating the homography
    :param zed_rgb: rgb image
    :param M_homography: homograhpy matix
    :return: None
    """
    wraped_img = cv2.warpPerspective(zed_rgb, M_homography, zed_rgb.shape[:2][::-1])
    non_black_img = cut_black(wraped_img)
    plt.imshow(non_black_img)
    plt.show()
    print(M_homography)


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

    # this is for validating the matching between the images clean version
    # des1 = des_zed
    # des2 = des_jai
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # if isinstance(des1, type(None)) or isinstance(des2, type(None)):
    #     print(f'match descriptor des is non')
    # matches = flann.knnMatch(des1, des2, k=2)
    # # store all the good matches as per Lowe's ratio test.
    # match = []
    # for m, n in matches:
    #     if m.distance < 0.7 * n.distance:
    #         match.append(m)
    #
    # dst_pts = np.float32([kp_zed[m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
    # src_pts = np.float32([kp_jai[m.trainIdx].pt for m in match]).reshape(-1, 1, 2)
    # plt.imshow(
    #     cv2.drawMatches(im_zed, kp_zed, im_jai, kp_jai, match, None, (255, 0, 0), (0, 0, 255)))
    # plt.show()

    # this is for validating the matching between the images
    # des1 = des_zed
    # des2 = des_jai
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # if isinstance(des1, type(None)) or isinstance(des2, type(None)):
    #     print(f'match descriptor des is non')
    # matches = flann.knnMatch(des1, des2, k=2)
    # # store all the good matches as per Lowe's ratio test.
    # match = []
    # for m, n in matches:
    #     if m.distance < 0.7 * n.distance:
    #         match.append(m)
    #
    # dst_pts = np.float32([kp_zed[m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
    # src_pts = np.float32([kp_jai[m.trainIdx].pt for m in match]).reshape(-1, 1, 2)
    # dists = dst_pts - src_pts
    # abs_dy = np.abs(dists[:, 0, 1])
    # valid_logical = abs_dy < np.quantile(abs_dy, 0.5)
    # plt.imshow(
    #     cv2.drawMatches(im_zed, kp_zed, im_jai, kp_jai, np.array(match)[valid_logical], None, (255, 0, 0), (0, 0, 255)))
    # plt.show()

    # sift = cv2.SIFT_create()
    # # find key points
    # kp_zed, des1 = sift.detectAndCompute(im_zed, None)
    # sift = cv2.SIFT_create()
    # # find key points
    # kp_jai, des2 = sift.detectAndCompute(im_jai, None)
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
    # search_params = dict(checks=100)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # if isinstance(des1, type(None)) or isinstance(des2, type(None)):
    #     print(f'match descriptor des is non')
    # matches = flann.knnMatch(des1, des2, k=2)
    #
    # match = []
    # for m, n in matches:
    #     if m.distance < 0.8 * n.distance:
    #         match.append(m)
    #
    # dst_pts = np.float32([kp_zed[m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
    # src_pts = np.float32([kp_jai[m.trainIdx].pt for m in match]).reshape(-1, 1, 2)
    # M, status = cv2.estimateAffine2D(src_pts, dst_pts,
    #                                  ransacReprojThreshold=2.5, maxIters=5000)
    # out_img = cv2.drawMatches(im_zed, kp_zed, im_jai, kp_jai, np.array(match), None, (255, 0, 0), (0, 0, 255))
    # plt.imshow(out_img)
    # plt.show()
    # out_img = cv2.drawMatches(im_zed, kp_zed, im_jai, kp_jai, np.array(match)[status.reshape(-1).astype(np.bool_)],
    #                           None, (255, 0, 0), (0, 0, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(out_img)
    # plt.show()