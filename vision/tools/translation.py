import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from matplotlib import cm

from vision.tools.image_stitching import find_keypoints, find_translation, keep_dets_only, resize_img
from collections import deque


class translation():

    def __init__(self, batch_size, translation_size=480, dets_only=True, mode='match', debug_path=None, history_length=5, direction="right"):

        self.translation_size = translation_size
        self.dets_only = dets_only
        self.mode_init(mode)
        self.batch_size = batch_size
        self.last_frame = None
        self.last_kp = None
        self.last_des = None
        self.debug_path = debug_path # in case no path given, no debug data is saved
        self.history_length = history_length
        self.history_x = deque(maxlen=history_length)
        self.history_y = deque(maxlen=history_length)
        self.border_case_counter = 0


    def mode_init(self, mode):
        self.mode = mode


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

        self.last_frame = frame.copy()

        return tx, ty

    def batch_translation(self, batch, detections, debug=None):
        if self.mode == 'match':
            output = self.batch_match(batch, detections, debug=debug)
        elif self.mode == 'keypoints':
            output = self.batch_keypoints(batch)
        else:
            raise Exception(f'{self.mode} is not implemented')

        return output


    def batch_match(self, batch, detections, workers=4, debug=None):
        batch_preproc_, r_ = self.preprocess_batch(batch, detections, workers=workers)

        batch_last_frames = [self.last_frame]
        for i in range(len(batch) - 1):
            batch_last_frames.append(batch_preproc_[i])
        if debug is None:
            debug = [None] * self.batch_size

        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(self.find_match_translation, batch_last_frames, batch_preproc_, r_, debug))

        self.last_frame = batch_preproc_[-1].copy()

        results = self.validate_results(results)

        return results

    def validate_results(self, results, margin=5, true_border_case_switch=False):
        """
            true_border_case: if true is allowing a border value to be used.
                              happen when the border events in sequence is above
                              2 * history length
        """
        final_results = []
        for result in results:
            if result[0] is None:
                final_results.append((result[0], result[1]))
                continue
            tx = result[0]
            ty = result[1]
            max_width = result[2]
            if (tx > max_width - margin) or (tx < margin): # border case
                if true_border_case_switch:
                    if self.border_case_counter > 2 * self.history_length: # true border case
                        final_results.append((tx, ty))
                        continue
                if len(self.history_x) == self.history_length:
                    tx = np.mean(self.history_x)
                if len(self.history_y) == self.history_length:
                    ty = np.mean(self.history_y)
                self.border_case_counter += 1
            else: # valid result
                self.history_x.append(tx)
                self.history_y.append(ty)
                self.border_case_counter = 0

            final_results.append((tx, ty))

        return final_results





        if isinstance(tx, type(None)):
            return tx
        if not self.maxlen:
            return tx
        if self.direction == "right":
            if tx > 0:
                self.memory.append(tx)
            else:
                tx = np.mean(self.memory)
        else:
            if tx < 0:
                self.memory.append(tx)
            else:
                tx = np.mean(self.memory)
        return tx





    def preprocess_batch(self, batch, detections, workers=4):
        is_dets_only = [self.dets_only for i in range(len(batch))]
        sizes = [self.translation_size for i in range(len(batch))]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(self.frame_preprocess, batch, detections, sizes, is_dets_only))

        output_frames = []
        rs = []
        for res in results:
            output_frames.append(res[0])
            rs.append(res[1])
        return output_frames, rs

    @staticmethod
    def frame_preprocess(frame, detections, size, dets_only=True):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if dets_only:
            frame = keep_dets_only(frame, detections)
        frame, r = resize_img(frame, size)

        return frame, r



    def find_frame_translation(self, frame, r):

        if self.mode == 'match':
            tx, ty = self.find_match_translation(self.last_frame, frame, r)
            self.last_frame = frame.copy()
        elif self.mode == 'keypoints':
            tx, ty, kp, des = self.find_keypoint_translation(frame, r, last_kp=self.last_kp, last_des=self.last_des)
            self.last_kp = kp
            self.last_des = des

        return tx, ty

    @staticmethod
    def find_keypoint_translation(frame, r, last_frame=None, last_kp=None, last_des=None):

        kp, des = find_keypoints(frame)
        # preferred - better runtime
        if last_kp is not None and last_des is not None:
            tx, ty, _, _, _ = find_translation(last_kp, last_des, kp, des, r)
        elif last_frame is not None:
            last_kp, last_des = find_keypoints(last_frame)
            tx, ty, _, _, _ = find_translation(last_kp, last_des, kp, des, r)
        else:
            raise Exception("Input data is not enough to perform translation")

        return tx, ty, kp, des

    @staticmethod
    def find_match_translation(last_frame, frame, r, debug=None):

        if last_frame is not None:
            try:
                # Apply template Matching
                res = cv2.matchTemplate(last_frame, frame[50:-50, 50:-50], cv2.TM_CCOEFF_NORMED)

                # max point method
                max_pt = cv2.minMaxLoc(res)[3]
                tx = (max_pt[0] - (res.shape[1] // 2 + 1)) / r
                ty = (max_pt[1] - (res.shape[0] // 2 + 1)) / r
                max_width = (res.shape[1] // 2) / r

                if debug is not None:
                    save_results(res, max_pt, debug)


            except:
                Warning('failed to match')
                tx = None
                ty = None
                max_width = None
        else:
            tx = None
            ty = None
            max_width = None

        return tx, ty, max_width







def save_results(res, max_pt, debug):

    file_name = os.path.join(debug['output_path'], f"matching_result_f{debug['f_id']}.jpg")

    # selected_cmap = cm.YlGn
    selected_cmap = cm.YlOrRd
    res = cv2.circle(res * 255, max_pt, 3, (255, 0, 0))
    plt.imsave(file_name, res, cmap=selected_cmap)

