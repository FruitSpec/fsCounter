import cv2
import numpy as np
import torch
import kornia.feature as KF
import kornia as K
from vision.tools.image_stitching import find_keypoints, find_translation, keep_dets_only, resize_img
from vision.visualization.loftr_drawer import draw_LAF_matches


class translation():

    def __init__(self, translation_size=480, dets_only=True, mode='match', debug_path=None):
        self.translation_size = translation_size
        self.dets_only = dets_only
        self.mode_init(mode)

        self.last_frame = None
        self.last_kp = None
        self.last_des = None
        self.debug_path = debug_path # in case no path given, no debug data is saved

    def mode_init(self, mode):
        self.mode = mode
        if mode == "LoFTR":
            self.matcher = KF.LoFTR(pretrained="outdoor")
        else:
            self.matcher = None


    def get_translation(self, frames, detections, batch_size=1):
        if batch_size == 1:
            tx, ty = self.single_frame_translation(frames, detections)
        else:
            pass

        return tx, ty

    def single_frame_translation(self, frame, detections):
        if self.dets_only:
            frame = keep_dets_only(frame, detections)
        frame, r = resize_img(frame, self.translation_size)
        tx, ty = self.find_frame_translation(frame, r)

        return tx, ty


    def find_frame_translation(self, frame, r):

        if self.mode == 'match':
            tx, ty = self.find_match_translation(frame, r)
        elif self.mode == 'keypoints':
            tx, ty = self.find_keypoint_translation(frame, r)
        elif self.mode == 'LoFTR':
            tx, ty = self.find_loftr_translation(frame, r)
        return tx, ty

    def find_keypoint_translation(self, frame, r):

        kp, des = find_keypoints(frame)
        if self.last_kp is not None:
            tx, ty, good, matches, matchesMask = find_translation(self.last_kp, self.last_des, kp, des, r)

        self.last_kp = kp
        self.last_des = des

        return tx, ty


    def find_match_translation(self, frame, r):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.last_frame is not None:
            try:
                # Apply template Matching
                res = cv2.matchTemplate(self.last_frame, frame[50:-50, 30:-30], cv2.TM_CCOEFF_NORMED)
                x_vec = np.mean(res, axis=0)
                y_vec = np.mean(res, axis=1)
                tx = (np.argmax(x_vec) - (res.shape[1] // 2 + 1)) / r
                ty = (np.argmax(y_vec) - (res.shape[0] // 2 + 1)) / r

            except:
                Warning('failed to match')
                tx = None
                ty = None

        else:
            tx, ty = None, None

        self.last_frame = frame

        return tx, ty

    def find_loftr_translation(self, frame, r):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = K.image_to_tensor(frame, True).float() / 255.

        if self.last_frame is not None:
            #try:
            input_dict = {"image0":  torch.unsqueeze(frame, dim=0),  # LofTR works on grayscale images only
                          "image1": torch.unsqueeze(self.last_frame, dim=0)}
            with torch.inference_mode():
                correspondences = self.matcher(input_dict)

            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()
            #Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
            M, status = cv2.estimateAffine2D(mkpts0, mkpts1, ransacReprojThreshold=3)

            tx = int(np.round(M[0, 2] / r))
            ty = int(np.round(M[1, 2] / r))
            #inliers = inliers > 0
            #if self.debug_path is not None:
            #    self.draw_loftr(frame, mkpts0, mkpts1, inliers, self.debug_path)


            #except:
            ##    Warning('failed to match')
            #    tx = None
            #    ty = None

        else:
            tx, ty = None, None

        self.last_frame = frame

        return tx, ty

    def draw_loftr(self, frame, mkpts0, mkpts1, inliers):

        draw_LAF_matches(
            KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1, -1, 2),
                                         torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                                         torch.ones(mkpts0.shape[0]).view(1, -1, 1)),

            KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1, -1, 2),
                                         torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                                         torch.ones(mkpts1.shape[0]).view(1, -1, 1)),
            torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
            K.tensor_to_image(self.last_frame),
            K.tensor_to_image(frame),
            inliers,
            debug_path = self.debug_path,
            draw_dict={'inlier_color': (0.2, 1, 0.2),
                       'tentative_color': None,
                       'feature_color': (0.2, 0.5, 1), 'vertical': False})

