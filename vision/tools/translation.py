import cv2
import numpy as np
from vision.tools.image_stitching import find_keypoints, find_translation, keep_dets_only, resize_img


class translation():

    def __init__(self, translation_size=480, dets_only=True, mode='match'):
        self.translation_size = translation_size
        self.dets_only = dets_only
        self.mode = mode

        self.last_frame = None
        self.last_kp = None
        self.last_des = None

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