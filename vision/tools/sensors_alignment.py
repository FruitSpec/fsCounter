import os
import cv2
import numpy as np
np.int = np.int_
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

from vision.tools.image_stitching import (resize_img, find_keypoints, find_keypoints_cuda, match_descriptors_cuda, get_affine_homography, affine_to_values,
                                          calc_affine_transform, get_affine_matrix, resize_img_cuda, draw_matches)
                                          #from vision.tools.image_stitching import calc_affine_transform, calc_homography, plot_2_imgs
#from vision.feature_extractor.image_processing import multi_convert_gray
# import seaborn as sns
#import cupy as cp
np.random.seed(123)
cv2.setRNGSeed(123)

class SensorAligner:
    """
    this class is for aligning the zed and jai cameras
    """
    def __init__(self, cfg, zed_shift=0, batch_size=1):
        self.zed_roi_params = cfg.zed_roi_params
        self.y_s, self.y_e, self.x_s, self.x_e = self.zed_roi_params.values()
        self.debug = cfg.debug
        self.zed_shift, self.consec_less_threshold, self.consec_more_threshold = zed_shift, 0, 0
        self.r_jai, self.r_zed = 1, 1
        self.size = cfg.size
        self.ransac = cfg.ransac
        self.matcher = self.init_matcher(cfg)
        self.batch_size = batch_size
        self.sx = cfg.sx #0.6102498372395834
        self.sy = cfg.sy#0.6136618198110134
        self.roix = cfg.roix#930 #937
        self.roiy = cfg.roiy
        self.use_cuda = True if cv2.cuda.getCudaEnabledDeviceCount() > 0 else False

        if self.batch_size > -1:
            self.init_for_batch()

    def init_for_batch(self):
        sxs = []
        sys = []
        origins = []
        rois = []
        ransacs = []
        for i in range(self.batch_size):
            sxs.append(self.sx)
            sys.append(self.sy)
            origins.append([self.x_s, self.y_s, self.x_e, self.y_e])
            rois.append([self.roix, self.roiy])
            ransacs.append(self.ransac)

        self.sxs = sxs
        self.sys = sys
        self.origins = origins # x1, y1, x2, y2
        self.rois = rois
        self.ransacs = ransacs

    def init_matcher(self, args):

        matcher = cv2.SIFT_create()

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

    def crop_zed_roi_cuda(self, zed_GPU):
        """
        crops region of interest of jai image inside the zed image
        :param gray_zed: image of gray_scale_zed
        :return: cropped image
        """
        zed_mid_h = zed_GPU.size()[1] // 2
        zed_half_height = int(zed_mid_h / (self.zed_angles[0] / 2) * (self.jai_angles[0] / 2))
        if isinstance(self.y_s, type(None)):
            self.y_s = zed_mid_h - zed_half_height - 100
        if isinstance(self.y_e, type(None)):
            self.y_e = zed_mid_h + zed_half_height + 100
        if isinstance(self.x_e, type(None)):
            self.x_e = zed_GPU.size()[0]
        if isinstance(self.x_s, type(None)):
            self.x_s = 0
        cropped_zed_GPU = zed_GPU.adjustROI(self.y_s, self.x_s, self.y_e, self.x_e)
        return cropped_zed_GPU


    def align_on_batch(self, zed_batch, jai_batch, workers=4, debug=None):
        if len(zed_batch) < 1:
            corr, tx, ty = align_sensors_cuda(zed_batch[0], jai_batch[0])
            results = [[corr, tx, ty]]
        else:
            zed_input = []
            jai_input = []
            streams = []
            for z, j in zip(zed_batch, jai_batch):
                if z is not None and j is not None:
                    zed_input.append(z)
                    jai_input.append(j)
                    #streams.append(cv2.cuda_Stream())
            debug = [None] * self.batch_size
            with ThreadPoolExecutor(max_workers=workers) as executor:
                #sx, sy, origin, roi, ransac
                results = list(executor.map(align_sensors_cuda,
                                            zed_input,
                                            jai_input,
                                            self.sxs,
                                            self.sys,
                                            self.origins,
                                            self.rois,
                                            self.ransacs,
                                            debug))
                #

        output = []
        for r in results:
        #    self.update_zed_shift(r[1])
            output.append([r[0], r[1], r[2], self.zed_shift])

        return output



    def align_sensors(self, zed_rgb, jai_img, jai_drop=False, zed_drop=False):
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

        self.update_zed_shift(tx)

        return (x1, y1, x2, y2), tx, ty, kp_zed, kp_jai, gray_zed, gray_jai, match, st


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
        elif tx > 120:
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

def align_sensors_cuda(zed_rgb, jai_img, sx, sy, origin, roi, ransac, debug=None, scale_factor=4 ):
    """
    aligns both sensors and updates the zed shift
    :param zed_rgb: rgb image
    :param jai_img: jai imgae
    :param sx: calibrated coefficient of x scale
    :param sy: calibrated coefficient of y scale
    :param origin: rough calibration [x1, y1, x2, y2]
    :param roi: size of roi - [widthX, witdhY]
    :return: jai_in_zed coors, tx,ty, if use fine is set to True also returns keypoints, des, jai_image
    """

    # transfer to grayscale
    jai_gray = cv2.cvtColor(jai_img, cv2.COLOR_BGR2GRAY)
    zed_gray = cv2.cvtColor(zed_rgb, cv2.COLOR_RGB2GRAY)
    h, w = zed_gray.shape
    zed_size = [w, h]
    zed_cpu = zed_gray[origin[1]: origin[3], origin[0]: origin[2]] # origin: x1, y1, x2, y2
    stream1 = cv2.cuda_Stream()
    stream2 = cv2.cuda_Stream()

    # upload images to gpu
    jai_GPU = cv2.cuda_GpuMat()
    jai_GPU.upload(jai_gray, stream1)
    zed_GPU = cv2.cuda_GpuMat()
    zed_GPU.upload(zed_cpu, stream2)

    # adjust zed scale to be the same as jai using calibrated scale x and y
    zed_GPU = cv2.cuda.resize(zed_GPU, (int(zed_GPU.size()[0] / sx),
                                        int(zed_GPU.size()[1] / sy)),
                                        stream=stream2)

    zed_GPU, rz = resize_img_cuda(zed_GPU, zed_GPU.size()[1] // scale_factor, stream2)
    jai_GPU, rz = resize_img_cuda(jai_GPU, jai_GPU.size()[1] // scale_factor, stream1)

    kp_zed, des_zed_GPU = find_keypoints_cuda(zed_GPU)  # consumes 33% of time
    kp_jai, des_jai_GPU = find_keypoints_cuda(jai_GPU)  # consumes 33% of time

    match, matches, matchesMask = match_descriptors_cuda(des_zed_GPU, des_jai_GPU, stream1)
    stream1.waitForCompletion()
    stream2.waitForCompletion()

    M, st = calc_affine_transform(kp_zed, kp_jai, match, ransac)

    if debug is not None:
        zed_cpu = zed_GPU.download()
        jai_cpu = jai_GPU.download()
        draw_matches(zed_cpu, kp_zed, jai_cpu, kp_jai, match, st, debug['output_path'], debug['f_id'])

    stream1 = None
    stream2 = None
    zed_GPU = None
    jai_GPU = None
    des_jai_GPU = None
    des_zed_GPU = None


    # in case no matches or less than 5 matches
    if len(st) == 0 or np.sum(st) <= 5:
        print('failed to align, using center default')
        tx = -999
        ty = -999
        # roi in frame center
        mid_x = (origin[0] + origin[2]) // 2
        mid_y = (origin[1] + origin[3]) // 2
        x1 = mid_x - (roi[0] // 2)
        x2 = mid_y + (roi[0] // 2)
        y1 = mid_y - (roi[1] // 2)
        y2 = mid_y + (roi[1] // 2)
    else:
        dst_pts = np.float32([kp_zed[m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
        dst_pts = dst_pts[st.reshape(-1).astype(np.bool_)]
        src_pts = np.float32([kp_jai[m.trainIdx].pt for m in match]).reshape(-1, 1, 2)
        src_pts = src_pts[st.reshape(-1).astype(np.bool_)]

        deltas = np.array(dst_pts) - np.array(src_pts)

        tx = M[0, 2]
        ty = M[1, 2]
        tx = tx / rz * sx
        ty = ty / rz * sy
        #tx = np.mean(deltas[:, 0, 0]) / rz * sx
        #ty = np.mean(deltas[:, 0, 1]) / rz * sy

        x1, y1, x2, y2 = get_zed_roi(tx, ty, roi, origin, zed_size)

    return (x1, y1, x2, y2), tx, ty

def get_zed_roi(tx, ty, roi, origin, zed_size):

    if tx < 0:
        x1 = 0
        x2 = roi[0]
    elif tx + roi[0] > zed_size[0]:
        x2 = zed_size[0]
        x1 = zed_size[0] - roi[0]
    else:
        x1 = tx
        x2 = tx + roi[0]

    if ty < 0:
        y1 = origin[1]
        y2 = origin[1] + roi[1]
    elif ty + roi[1] > zed_size[1]:
        y2 = origin[3]
        y1 = origin[3] - roi[1]
    else:
        y1 = origin[1] + ty
        y2 = origin[1] + ty + roi[1]

    return x1, y1, x2, y2




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
