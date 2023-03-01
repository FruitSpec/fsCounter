import os.path

import cv2
import numpy as np

from vision.tracker.fsTracker.base_track import Track
from vision.tracker.fsTracker.score_func import compute_ratios, dist, confidence_score, get_intersection, relativity_score
from vision.tracker.fsTracker.score_func import z_score
from vision.tracker.fsTracker.base_track import TrackState
from vision.misc.help_func import validate_output_path

class FsTracker():
    """
    A FruitSpec objects tracker.
    Designed to track high-speed moving objects
    """
    def __init__(self, frame_size=[2048, 1536], frame_id=0, track_id=0,
                 score_weights=[0.5, 1, 0.5, 1, 1], match_type='center', det_area=1, max_losses=10, major=3, minor=2.5, debug_folder=None):
        """
        tracker class init method:

        :param frame_size: list with following values: [height, width] of the original frame size
        :param frame_id: indicate what is the starting frame number. to use in case of reset
        :param track_id: indicate what is the starting track number. to use in case of reset
        :param score_weights: list of score weights - [ratio score, distance score, confidence score, depth score, relativity score]
        :param match_type: method to use to determine what is window - box match: 'center' - if the center of the bbox is in window,
                                                                                  'inter' - if the bbox intersect with window
        :param det_area: value between (0-1] - what percentage of bbox to use as margin for intersection
        :param max_losses: how many times a tracklet is kept alive before delete in case it is lost
        :param major: factor on translation - how much to increase window far side movement on top of translation value
        :param minor: factor on translation - how much to increase window close side movement on top of translation value
        :param debug_folder: path to debug folder
        """


        self.tracklets = []
        self.track_id = track_id
        self.frame_id = frame_id
        self.max_distance = frame_size[1]
        self.x_distance = 0
        self.y_distance = 0
        self.major = major
        self.minor = minor
        self.match_type = match_type
        self.det_area = det_area
        self.max_losses = max_losses

        self.score_weights = score_weights
        self.frame_size = frame_size

        self.last_center_x = None


        if debug_folder is not None:
            self.last_frame = None
            self.f_id = 0
            validate_output_path(debug_folder)
            self.debug_folder = debug_folder


    def reset_state(self):
        pass

    def update(self, detections, tx, ty, frame_id=None, dets_depth=None):
        """
        tracker update method - assign track ids to detections

        :param detections: list of detected objects to assign track id
        :param tx: x translation between current frame and last
        :param ty: y translation between current frame and last
        :param frame_id: [optional] frame number
        :param dets_depth: [optional] list of depth value for each det
        """


        tx, ty = self.update_frame_meta(frame_id, tx, ty)
        self.update_max_distance()
        self.is_extreme_shift()
        self.deactivate_tracks()

        """ find dets that match tracks"""
        tracklets_bboxes, tracklets_scores, track_acc_dist, tracks_acc_height, track_depth = self.get_tracklets_data()
        track_windows = self.get_track_search_window_by_id([tx, ty], tracklets_bboxes)
        matches = self.match_detections_to_windows(track_windows, detections, self.match_type)
        not_coupled = self.calc_matches_score_and_update(matches,
                                                         detections,
                                                         track_windows,
                                                         tracklets_bboxes,
                                                         tracklets_scores,
                                                         dets_depth,
                                                         track_depth)

        """ add new dets tracks"""
        for det_id in not_coupled:
            self.add_track(detections, det_id, dets_depth)

        """ remove tracks exceeding max_range"""
        self.update_accumulated_dist()

        online_track = [t.output() for t in self.tracklets if t.is_activated]

        return online_track, track_windows

    def get_tracklets_data(self):
        tracklets_bboxes = []
        tracklets_scores = []
        track_depth = []

        tracklets = list(map(self.get_track, self.tracklets))
        track_acc_dist = list(map(self.get_acc_dist, self.tracklets))
        track_acc_height = list(map(self.get_acc_height, self.tracklets))
        track_depth = list(map(self.get_track_depth, self.tracklets))
        if len(tracklets) > 0:
            tracklets = np.vstack(tracklets)
            tracklets_bboxes = tracklets[:, :4]
            tracklets_scores = tracklets[:, 4]

        return tracklets_bboxes, tracklets_scores, track_acc_dist, track_acc_height, track_depth




    def update_frame_meta(self, frame_id, tx, ty):
        self.f_id = frame_id
        if tx is not None:
            self.x_distance = tx
        else:
            tx = 0

        if ty is not None:
            self.y_distance = tx
        else:
            ty = 0

        return tx, ty


    def get_track_search_window_by_id(self, search_window, tracklets_bboxes, margin=15):
        """

        :param search_window: list of [translation x, translation y]
        :param tracklets_bboxes: list of tracks bboxes
        :param margin: pixels to increase window

        :return:
            tracklets_windows: list of windows coordinates in format [x1, y1, x2, y2]
        """

        tx = search_window[0] * self.major
        ty = search_window[1] * self.major
        tx_m = search_window[0] * self.minor
        ty_m = search_window[1] * self.minor
        tracklets_windows = []

        if len(tracklets_bboxes) > 0:
            tracklets_bboxes = np.vstack(tracklets_bboxes)


            if tx > 0:  # moving right - fruits moving left
                x1 = tracklets_bboxes[:, 0] - tx
                x2 = tracklets_bboxes[:, 2] + margin - tx_m
            else:
                x1 = tracklets_bboxes[:, 0] - margin - tx_m
                x2 = tracklets_bboxes[:, 2] - tx


            x1[x1 < 0] = 0
            x2[x2 > self.frame_size[1]] = self.frame_size[1]

            if ty > 0:
                y1 = tracklets_bboxes[:, 1] - ty
                y2 = tracklets_bboxes[:, 3] + margin - ty_m
            else:
                y1 = tracklets_bboxes[:, 1] - margin - ty_m
                y2 = tracklets_bboxes[:, 3] - ty




            y1[y1 < 0] = 0
            y2[y2 > self.frame_size[0]] = self.frame_size[0]

            tracklets_windows = np.stack([x1, y1, x2, y2], axis=1)

        return tracklets_windows
    @staticmethod
    def get_track(track):

        return np.array(track.output())

    @staticmethod
    def get_acc_dist(track):

        if len(track.accumulated_dist) > 0:
            acc_dist = np.mean(track.accumulated_dist)
        else:
            acc_dist = 0
        return acc_dist

    @staticmethod
    def get_acc_height(track):

        if len(track.accumulated_height) > 0:
            acc_height = np.mean(track.accumulated_height)
        else:
            acc_height = 0
        return acc_height
    @staticmethod
    def get_track_depth(track):

        return track.depth

    def get_track_search_window_by_id_np(self, search_window):


        track_windows = []
        for track in self.tracklets:
            track_windows.append(track.get_track_search_window(search_window))


        return track_windows

    @staticmethod
    def get_detections_center(detections):
        det_centers = []
        if len(detections) > 0:
            dets = np.array(detections)[:, :4]
            center_x = (dets[:, 0] + dets[:, 2]) / 2
            center_y = (dets[:, 1] + dets[:, 3]) / 2
            det_centers = np.stack((center_x, center_y), axis=1)

        return det_centers

    def match_detections_to_windows(self, windows, detections, match_type='center'):
        """ match_type can be:
                1. inter to match by intersection
                2. center to match if center is in window"""
        if match_type == 'center':
            matches = self.match_by_center(windows, detections)
        elif match_type == 'inter':
            matches = self.match_by_intersection(windows, detections)
        else:
            print(f'unknown matching type: {match_type}')
            matches = []

        return matches

    def match_by_intersection(self, bboxes1, bboxes2):

        intersections = []

        if self.det_area < 1 and len(bboxes2) > 0:
            bboxes2 = np.array(bboxes2)[:, :4]
            margin_coef = (1 - self.det_area) / 2
            margin_x = (bboxes2[:, 2] - bboxes2[:, 0]) * margin_coef
            margin_y = (bboxes2[:, 3] - bboxes2[:, 1]) * margin_coef
            bboxes2[:, 0] += margin_x
            bboxes2[:, 2] -= margin_x
            bboxes2[:, 1] += margin_y
            bboxes2[:, 3] -= margin_y

        inetr_area = get_intersection(bboxes1, bboxes2)
        if len(inetr_area) > 0:
            intersections = inetr_area > 0

        return intersections


    def match_by_center(self, windows, detections):
        intersections = []

        if len(detections) > 0:
            centers = self.get_detections_center(detections)
            dets = np.zeros((centers.shape[0], 4))
            dets[:, 0] = centers[:, 0]
            dets[:, 1] = centers[:, 1]
            dets[:, 2] = centers[:, 0] + 1
            dets[:, 3] = centers[:, 1] + 1
            inetr_area = get_intersection(windows, dets)
            if len(inetr_area) > 0:
                intersections = inetr_area > 0

        return intersections

    @staticmethod
    def is_in_range(window, center):
        x_valid = False
        y_valid = False
        if window[0] <= center[0] and center[0] <= window[2]:
            x_valid = True

            if window[1] <= center[1] and center[1] <= window[3]:
                y_valid = True

        return x_valid & y_valid


    def calc_matches_score_and_update(self, matches, detections, track_windows, track_bboxes, trk_score, dets_depth=None, trk_depth=None):
        """
        The method calculate the score between detections and tracks and assign the best matches
        the method return list of detections that were not assigned

        :param matches: bool mask of tracks windows matches to detections
        :param detections: list of detections
        :param track_windows: list of track windows
        :param track_bboxes: list of track bboxes
        :param trk_score: list of track score
        :param dets_depth: list of detections depth
        :param trk_depth: list of tracks depth
        :return:
            not_coupled: list of detections without matching to track
        """
        if len(track_windows) == 0:
            return [i for i in range(len(detections))]
        elif len(detections) == 0:
            return []

        detections_arr = np.array(detections)
        dets = detections_arr[:, :4]
        dets_score = detections_arr[:, 4] * detections_arr[:, 5]


        # calculate scores
        weigthed_score = self.calculate_scores(track_windows,
                                               dets,
                                               track_bboxes,
                                               trk_score,
                                               dets_score,
                                               trk_depth,
                                               dets_depth,
                                               matches)

        # assign from tracks by score - high to low
        coupled_dets = []
        n_dets = weigthed_score.shape[1]
        for c in range(n_dets):
            mat_id = np.argmax(weigthed_score)
            trk_id = mat_id // n_dets
            det_id = mat_id % n_dets
            if weigthed_score[trk_id, det_id] == 0:
                break
            if dets_depth is not None:
                depth = dets_depth[det_id]
            else:
                depth = None

            self.tracklets[trk_id].update(detections[det_id], depth)
            coupled_dets.append(det_id)
            weigthed_score[:, det_id] = 0
            weigthed_score[trk_id, :] = 0


        not_coupled = []
        for det_id in range(n_dets):
            if det_id not in coupled_dets:
                not_coupled.append(det_id)

        return not_coupled

    def calculate_scores(self, track_windows, dets, track_bboxes, trk_score, dets_score, trk_depth, dets_depth, matches):
        # calculate scores
        dist_score = dist(track_windows, dets, self.x_distance * self.major, self.y_distance * self.major, np.abs(self.max_distance))
        ratio_score = compute_ratios(track_bboxes, dets)
        conf_score = confidence_score(trk_score, dets_score)
        rel_score = relativity_score(track_bboxes, dets)

        # support both cases - with depth and without
        if trk_depth is not None and dets_depth is not None:
            depth_score = z_score(trk_depth, dets_depth)
        else:
            depth_score = np.zeros(matches.shape)

        # remove values of no matches
        matches = self.assign_single_match(matches)
        no_match = np.logical_not(matches)
        dist_score[no_match] = 0  # no match
        ratio_score[no_match] = 0
        conf_score[no_match] = 0
        rel_score[no_match] = 0
        depth_score[no_match] = 0

        # weight scores
        weigthed_ratio = ratio_score * self.score_weights[0]
        weigthed_dist = dist_score * self.score_weights[1]
        weigthed_conf = conf_score * self.score_weights[2]
        weigthed_depth = depth_score * self.score_weights[3]
        weigthed_rel = rel_score * self.score_weights[4]

        # weigthed sum
        weigthed_score = (weigthed_ratio + weigthed_dist + weigthed_depth + weigthed_rel + weigthed_conf) / np.sum(self.score_weights)

        return weigthed_score

    def assign_single_match(self, matches):

        mask = np.zeros(matches.shape)
        v = np.sum(matches, axis=1)
        for i, m in enumerate(v):
            if m == 1:
                det_id = np.argmax(matches[i, :])
                mask[i, det_id] = 1
            else:
                mask[i, :] = 1
        matches = np.logical_and(matches, mask)


        return matches


    def get_matches(self, track_tf):
        temp_track_index = [track_id for track_id, tf in enumerate(track_tf) if tf]

        matched_tracks = []
        matched_track_index = []
        matched_tracks_acc_dist = []
        for index in temp_track_index:
            track = self.tracklets[index]
            matched_tracks.append(track.output())
            matched_track_index.append(index)
            if len(track.accumulated_dist) == 0:
                matched_tracks_acc_dist.append(self.x_distance)
            else:
                matched_tracks_acc_dist.append(np.mean(track.accumulated_dist))

        return matched_tracks, matched_track_index, matched_tracks_acc_dist

    #@staticmethod
    #def get_matches(detections, det_ids, det_tf, coupled):
    #    temp_det_index = [det_id for det_id, tf in zip(det_ids, det_tf) if tf]

    #    matched_dets = []
    #    matched_det_index = []
    #    for index in temp_det_index:
    #        if index not in coupled:
    #            matched_dets.append(detections[index])
    #            matched_det_index.append(index)

    #    return matched_dets, matched_det_index

    def add_track(self, detections, det_id, dets_depth):
        if dets_depth is not None:
            depth = dets_depth[det_id]
        else:
            depth = None

        t = Track()
        t.add(detections[det_id], depth, self.track_id, self.frame_size)
        self.track_id += 1
        self.tracklets.append(t)

    def deactivate_tracks(self):
        for t in self.tracklets:
            t.is_activated = False

    def update_accumulated_dist(self):

        new_tracklets_list = []
        for track in self.tracklets:
            if track.is_activated is False:  # not found in current frame
                valid = False
                if len(track.accumulated_dist) == 0:  # object found once
                    track.accumulated_dist.append(self.x_distance)
                    track.accumulated_height.append(self.y_distance)

                track.bbox[0] -= np.mean(track.accumulated_dist)
                track.bbox[2] -= np.mean(track.accumulated_dist)
                track.bbox[1] -= np.mean(track.accumulated_height)
                track.bbox[3] -= np.mean(track.accumulated_height)
                track.state = TrackState.Lost
                track.lost_counter += 1

                if int(track.bbox[0]) >= 0 & int(track.bbox[2]) >= 0 & int(track.bbox[0]) <= self.frame_size[1] & int(track.bbox[0]) <= self.frame_size[1]:
                    valid = True
                if track.lost_counter < self.max_losses and valid:
                    new_tracklets_list.append(track)
            else:
                new_tracklets_list.append(track)

        self.tracklets = new_tracklets_list

    def update_max_distance(self, mag=2):

        self.max_distance = self.frame_size[1]
        # self.max_distance = mag * self.x_distance
        # if np.abs(self.max_distance) < self.minimal_max_distance:
        #     self.max_distance = self.minimal_max_distance

    def is_extreme_shift(self):

        """in case of extreme shift all tracks
           will be deleted to prevenet extreme windows"""
        if np.abs(self.x_distance) > self.frame_size[1] // 3 or np.abs(self.y_distance) > self.frame_size[1] // 3:
            self.x_distance = 0
            self.y_distance = 0
            self.tracklets = []

    def save_debug(self, frame, good, matches, matchesMask, kp):

        # Need to draw only good matches, so create a mask

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=0)
        debug_img = cv2.drawMatchesKnn(self.last_frame, self.last_kp, frame, kp, matches, None, **draw_params)
        file_name = os.path.join(self.debug_folder, f'kp_f{self.f_id}.jpg')
        cv2.imwrite(file_name, debug_img)



def test_func(f_list):
    import pickle
    tracker = FsTracker()

    for f in f_list:
        with open(f,'rb') as fp:
            d = pickle.load(fp)
        frame = d["f"]
        dets = d["det"]

        tracker.update(dets, frame)

if __name__ == "__main__":
    f_list = ["/home/fruitspec-lab/FruitSpec/Sandbox/Sliced_data/RA_3_A_2/RA_3_A_2/res/f1.pkl",
              "/home/fruitspec-lab/FruitSpec/Sandbox/Sliced_data/RA_3_A_2/RA_3_A_2/res/f2.pkl",
              "/home/fruitspec-lab/FruitSpec/Sandbox/Sliced_data/RA_3_A_2/RA_3_A_2/res/f3.pkl"]

    test_func(f_list)



