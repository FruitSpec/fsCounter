import os
import cv2
import numpy as np
np.int = np.int_
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

from vision.tools.image_stitching import (resize_img, find_keypoints, get_affine_homography, affine_to_values,
                                          get_fine_keypoints, get_fine_translation, get_affine_matrix)
                                          #get_fine_affine_translation, find_loftr_translation)
from vision.tools.image_stitching import plot_2_imgs
from vision.tools.camera import stretch_rgb, is_saturated
from vision.tools.translation import translation as T
#from vision.feature_extractor.image_processing import multi_convert_gray
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
        self.apply_clahe = args.apply_clahe
        self.calib = args.calib
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
        if self.calib:
            self.y_s = 350
        self.sx = 0.60546875  # 0.6102498372395834
        self.sy = 0.6133919843597263  # 0.6136618198110134
        self.roix = 930  # 937
        self.roiy = 1255
        self.direction = args.direction
        self.update_shift_dict = {"": 0, "right": 1, "left": -1}

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
        if self.apply_clahe:
            jai_img_c = stretch_rgb(jai_img_c)
            zed_rgb_c = stretch_rgb(zed_rgb_c)
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

        if jai_drop:
            self.zed_shift += 1
        if zed_drop:
            self.zed_shift -= 1
        gray_zed, gray_jai, _, _ = self.preprocess(zed_rgb, jai_img)

        # adjust zed scale to be the same as jai using calibrated scale x and y
        gray_zed = self.crop_zed_roi(gray_zed)
        gray_zed = cv2.resize(gray_zed, (int(gray_zed.shape[1] / self.sx), int(gray_zed.shape[0] / self.sy)))

        gray_zed, rz = resize_img(gray_zed, gray_zed.shape[0] // 3)
        gray_jai, rj = resize_img(gray_jai, gray_jai.shape[0] // 3)

        kp_zed, des_zed = find_keypoints(gray_zed, self.matcher)  # consumes 33% of time
        kp_jai, des_jai = find_keypoints(gray_jai, self.matcher)  # consumes 33% of time
        M, st, match = get_affine_matrix(kp_zed, kp_jai, des_zed, des_jai, self.ransac,
                                         self.fixed_scaling)  # consumes 33% of time

        if not len(match):
            x1, y1, x2, y2 = 0, 0, *zed_rgb.shape[::-1][1:]
            x2 -= 1
            y1 -= 1
            return (x1, y1, x2, y2), np.nan, np.nan, self.sx, self.sy

        dst_pts = np.float32([kp_zed[m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
        dst_pts = dst_pts[st.reshape(-1).astype(np.bool_)]
        src_pts = np.float32([kp_jai[m.trainIdx].pt for m in match]).reshape(-1, 1, 2)
        src_pts = src_pts[st.reshape(-1).astype(np.bool_)]

        deltas = np.array(dst_pts) - np.array(src_pts)
        delta_x = np.mean(deltas[:, 0, 0])
        delta_y = np.mean(deltas[:, 0, 1])
        tx = int(delta_x / rz * self.sx) if np.isfinite(delta_x) else np.nan
        ty = int(delta_y / rz * self.sy) if np.isfinite(delta_y) else np.nan

        if np.isnan(delta_x) or np.isnan(delta_y):
            x1, y1, x2, y2 = 0, 0, *zed_rgb.shape[::-1][1:]
            x2 -= 1
            y1 -= 1
            return (x1, y1, x2, y2), tx, ty, self.sx, self.sy

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

        self.update_zed_shift(tx)

        return (x1, y1, x2, y2), tx, ty, self.sx, self.sy

    def align_sensors(self, zed_rgb, jai_img, jai_drop=False, zed_drop=False):
        """
        aligns both sensors and updates the zed shift
        :param zed_rgb: rgb image
        :param jai_img: jai imgae
        :param jai_drop: flag if jai had a frame drop
        :param zed_drop: flag if zed had a frame drop
        :return: jai_in_zed coors, tx,ty,sx,sy, if use fine is set to True also returns keypoints, des, jai_image
        """
        if self.calib:
            return self.align_calib_sensors(zed_rgb, jai_img, jai_drop, zed_drop)
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
        elif tx > 150:
            self.consec_less_threshold = 0
            self.consec_more_threshold += 1
        else:
            self.consec_less_threshold = max(self.consec_less_threshold - 1, 0)
            self.consec_more_threshold = max(self.consec_more_threshold - 1, 0)
        if self.consec_more_threshold > 5:
            self.zed_shift += self.update_shift_dict[self.direction]
            self.consec_less_threshold = 0
            self.consec_more_threshold = 0
        if self.consec_less_threshold > 5:
            self.zed_shift -= self.update_shift_dict[self.direction]
            self.consec_less_threshold = 0
            self.consec_more_threshold = 0


class FirstMoveDetector:
    """
    this class is used to detect when each camera starts the drive
    """
    def __init__(self, cameras: dict = {"cam_jai": "", "cam_zed": ""}, mode: str = "frames",
                translator_mode: str = "keypoints", thresh: float = 0.01, counter_thresh: int = 5,
                 debug: bool = False, batch_size: int = 0) -> None:
        """
        Args:
            cameras (dict): dictionery of cameras, needs to be full only if using camera mode
            mode (str): what mode to use (camera / frames), camera will read the images from camera
                frames will excpect to get frames from outside source
            translator_mode (str): translation mode to pass to translator
            thresh (float): minimum percentegre of imaghe width to count as movement
            counter_thresh (int): minimum number of detected frames with movement to count camera as moving
            debug (bool): flag for using debbuger
            batch_size (int): size of batch to be passed
        """
        self.mode = mode
        self.cameras = cameras
        if mode == "camera" and (isinstance(cameras["cam_jai"], str) or isinstance(cameras["cam_zed"], str)):
            raise ValueError("no cameras were passes")
        self.translator_zed, self.translator_jai = T(480, False, translator_mode), T(480, False, translator_mode)
        self.thresh, self.counter_thresh = thresh, counter_thresh
        self.zed_move, self.jai_move = 0, 0
        self.zed_width, self.jai_width = 0, 0
        self.counter_zed, self.counter_jai = 0, 0
        self.zed_first_move, self.jai_first_move = 0, 0
        self.debug = debug
        self.batch_size = batch_size

    def update_state(self, zed_frame: np.array = np.array([]), jai_frame: np.array = np.array([]),
                     frame_id: int = 0, sat: bool = False) -> tuple:
        """
        This function updates the 'zed_first_move' and 'jai_first_move', if both are greater then 0 it will return the
        zed_shift
        Args:
            zed_frame (np.array): zed_rgb frame to apply translation to
            jai_frame (np.array ): jai (preferred FSI) frame to apply translation to
            frame_id (int): current frame number
            sat (bool): flag for indicating saturated image, will not change zed frame counter if true

        Returns:
            zed_shift(int)
            boolean value if both cameras are already moving
        """
        if self.batch_size == 0:
            zed_frames, jai_frames = [zed_frame], [jai_frame]
        else:
            zed_frames, jai_frames = zed_frame, jai_frame
        for i, (zed_frame, jai_frame) in enumerate(zip(zed_frames, jai_frames)):
            if frame_id and self.mode == "camera":
                zed_frame, jai_frame, sat = self.read_frames_from_cam(frame_id)
            self.update_widths(zed_frame, jai_frame)
            tx_zed, tx_jai = self.get_txs(zed_frame, jai_frame, sat)
            if self.debug:
                self.debug_function("zed", frame_id + i, tx_zed, zed_frame)
                self.debug_function("jai", frame_id + i, tx_jai, jai_frame)
            self.update_counters(tx_zed, tx_jai)
            self.update_moves(frame_id + i)
            if self.zed_first_move and self.jai_first_move:
                return self.zed_first_move - self.jai_first_move, True
        return 0, False

    def read_frames_from_cam(self, frame_id: int = 0) -> tuple:
        """
        reads frames from the 2 cameras
        Args:
            frame_id (int): frame number

        Returns:
            zed_frame (np.array): rgb frame from zed camera
            jai_frame (np.array): FSI frame from jai camera
            sat (bool): flag indicating if frame is saturated
        """
        zed_frame, point_cloud = self.cameras["cam_zed"].get_zed(frame_id, exclude_depth=True)
        _, jai_frame = self.cameras["cam_jai"].get_frame(frame_id)
        sat = False
        if is_saturated(zed_frame, 0.6) or is_saturated(jai_frame, 0.6):
            print(frame_id, ": saturated")
            sat = True
        return zed_frame, jai_frame, sat

    def update_counters(self, tx_zed: int, tx_jai: int) -> None:
        """
        updated the counter values for zed and jai cam if there was movement detected
        Args:
            tx_zed (int): the translanlation in x axis for zed camera
            tx_jai (int): the translanlation in x axis for jai camera

        Returns:
            None
        """
        if np.abs(tx_zed) > self.zed_width * self.thresh:
            self.counter_zed += 1
        if np.abs(tx_jai) > self.jai_width * self.thresh:
            self.counter_jai += 1

    def update_moves(self, frame_id: int = 0) -> None:
        """
        update zed and jai first move if thier counter is larger then the threshold
        Args:
            frame_id (int): frame number

        Returns:
            None
        """
        if self.counter_zed > self.counter_thresh and not self.zed_first_move:
            self.zed_first_move = frame_id
        if self.counter_jai > self.counter_thresh and not self.jai_first_move:
            self.jai_first_move = frame_id

    def get_txs(self, zed_frame: np.array = np.array([]), jai_frame: np.array = np.array([]),
                sat: bool = False) -> tuple:
        """
        calculates translation in x axis for both cameras
        Args:
            zed_frame (np.array): zed_rgb frame to apply translation to
            jai_frame (np.array ): jai (preferred FSI) frame to apply translation to
            sat (bool): flag for indicating saturated image, will not change zed frame counter if true

        Returns:
            tx_zed (int): the translanlation in x axis for zed camera
            tx_jai (int): the translanlation in x axis for jai camera
        """
        if not sat:
            tx_zed, _ = self.translator_zed.get_translation(zed_frame, None)
        else:
            tx_zed = 0
        tx_jai, _ = self.translator_jai.get_translation(jai_frame, None)
        if isinstance(tx_jai, type(None)):
            tx_jai = np.nan
        if isinstance(tx_zed, type(None)):
            tx_zed = np.nan
        return tx_zed, tx_jai

    @staticmethod
    def debug_function(cam_type: str, frame: int, tx: int, img: np.array,
                 folder: str = "/media/fruitspec-lab/easystore/auto_zed_shift_testing"):
        """
        saves the images with translation values in the title
        Args:
            cam_type (str): jai or zed according to the camera that was used
            frame (int): frame number
            tx (int): translation value
            img (np.array): the curent image
            folder (str): where to save to

        Returns:
            None
        """
        fig_name = f"{cam_type}_{frame}_{tx}"
        plt.imshow(img)
        plt.title(f"frame: {frame}, tx: {tx}")
        if not os.path.exists(folder):
            os.mkdir(folder)
        plt.savefig(os.path.join(folder, fig_name))
        # plt.show()
        plt.close()

    def update_widths(self, zed_frame: np.array = np.array([]), jai_frame: np.array = np.array([])) -> None:
        """
        Updates the widths of the images if they are 0
        Args:
            zed_frame (np.array): zed_rgb frame to apply translation to
            jai_frame (np.array ): jai (preferred FSI) frame to apply translation to

        Returns:
            None
        """
        if not self.zed_width:
            self.zed_width = zed_frame.shape[1]
        if not self.jai_width:
            self.jai_width = jai_frame.shape[1]


def multi_convert_gray(list_imgs):
    return [cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY) for img in list_imgs]


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