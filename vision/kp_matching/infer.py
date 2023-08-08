import cv2
import time

from vision.kp_match.lightglue import LightGlue, SuperPoint, DISK
from vision.kp_match.lightglue.utils import load_image, rbd, numpy_image_to_torch
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

    def to_tensor(self, image):

        return numpy_image_to_torch(image).cuda()


    def match(self, input0, input1):
        s = time.time()
        feats0 = self.extractor.extract(input0)
        feats1 = self.extractor.extract(input1)
        e = time.time()
        print(f"fe: {e-s}")
        s = e
        # match the features
        matches01 = self.matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)
        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]]

        e = time.time()
        print(f"matching: {e - s}")
        return points0, points1, matches

    def align_sensors(self, image0, image1):
        input0, input1 = self.preprocess_images(image0, image1)
        points0, points1, matches = self.match(input0, input1)

        M, status = self.calcaffine(points0, points1)


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