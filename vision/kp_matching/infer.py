import cv2
import time
import threading
import queue
import numpy as np

from vision.kp_matching.sp_lg import LightGlue, SuperPoint, DISK
from vision.kp_matching.sp_lg.utils import load_image, rbd, numpy_image_to_torch
from vision.tools.image_stitching import resize_img


class lightglue_infer():

    def __init__(self, cfg, type='superpoint'):
        """
        type can be 'superpoint' or 'disk'
        """
        if type == 'superpoint':
            self.extractor = SuperPoint(max_num_keypoints=512).eval().cuda()  # load the extractor
        elif type == 'disk':
            self.extractor = DISK(max_num_keypoints=512).eval().cuda()  # load the extractor

        self.matcher = LightGlue(features=type, depth_confidence=0.9, width_confidence=0.95).eval().cuda()  # load the matcher

        self.y_s, self.y_e, self.x_s, self.x_e = cfg.sensor_aligner.zed_roi_params.values()
        self.size = cfg.sensor_aligner.size
        self.sx = cfg.sensor_aligner.sx
        self.sy = cfg.sensor_aligner.sy
        self.roix = cfg.sensor_aligner.roix
        self.roiy = cfg.sensor_aligner.roiy
        self.zed_size = [1920, 1080]

    def to_tensor(self, image):

        return numpy_image_to_torch(image).cuda()


    def match(self, input0, input1):
        s = time.time()
        feats0 = self.extractor.extract(input0)
        feats1 = self.extractor.extract(input1)

        # match the features
        matches01 = self.matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)
        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]]

        return points0, points1, matches

    def preprocess_images(self, zed, jai_rgb, downscale=4):

        cropped_zed = zed[self.y_s: self.y_e, self.x_s:self.x_e, :]
        input_zed = cv2.resize(cropped_zed, (int(cropped_zed.shape[1] / self.sx), int(cropped_zed.shape[0] / self.sy)))

        input_zed, rz = resize_img(input_zed, input_zed.shape[0] // downscale)
        input_jai, rj = resize_img(jai_rgb, jai_rgb.shape[0] // downscale)

        input_zed = self.to_tensor(input_zed)
        input_jai = self.to_tensor(input_jai)

        return input_jai, rj, input_zed, rz


    @staticmethod
    def calcaffine(src_pts, dst_pts):
        if dst_pts.__len__() > 0 and src_pts.__len__() > 0:  # not empty - there was a match
            M, status = cv2.estimateAffine2D(src_pts, dst_pts)
        else:
            M = None
            status = []

        return M, status

    def get_tx_ty(self, M, st, rz):
        # in case no matches or less than 5 matches
        if len(st) == 0 or np.sum(st) <= 5:
            print('failed to align, using center default')
            tx = -999
            ty = -999
            # roi in frame center
            mid_x = (self.x_s + self.x_e) // 2
            mid_y = (self.y_s + self.y_e) // 2
            x1 = mid_x - (self.roix // 2)
            x2 = mid_y + (self.roix // 2)
            y1 = mid_y - (self.roiy // 2)
            y2 = mid_y + (self.roiy // 2)
        else:


            tx = M[0, 2]
            ty = M[1, 2]
            tx = tx / rz * self.sx
            ty = ty / rz * self.sy
            # tx = np.mean(deltas[:, 0, 0]) / rz * sx
            # ty = np.mean(deltas[:, 0, 1]) / rz * sy

            x1, y1, x2, y2 = self.get_zed_roi(tx, ty)

        return (x1, y1, x2, y2), tx, ty

    def get_zed_roi(self, tx, ty):

        if tx < 0:
            x1 = 0
            x2 = self.roix
        elif tx + self.roix > self.zed_size[1]:
            x2 = self.zed_size[1]
            x1 = self.zed_size[1] - self.roix
        else:
            x1 = tx
            x2 = tx + self.roix

        if ty < 0:
            y1 = self.y_s + ty
            if y1 < 0:
                y1 = self.y_s
            y2 = y1 + self.roiy
        elif ty + self.roiy > self.zed_size[0]:
            y2 = self.y_e
            y1 = self.y_e - self.roiy
        else:
            y1 = self.y_s + ty
            y2 = self.y_s + ty + self.roiy

        return x1, y1, x2, y2


    def align_sensors(self, zed, jai_rgb):

        zed_input, rz, jai_input, rj = self.preprocess_images(zed, jai_rgb)

        points0, points1, matches = self.match(zed_input, jai_input)

        points0 = points0.cpu().numpy()
        points1 = points1.cpu().numpy()

        M, st = self.calcaffine(points0, points1)

        return self.get_tx_ty(M, st, rz)


def inference(extractor, image, batch_queue):
    kp = extractor.extract(image)
    batch_queue.put(kp)
    return batch_queue

