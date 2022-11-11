import numpy as np

from vision.tracker.fsTracker.base_track import Track
from vision.tracker.fsTracker.score_func import compute_ratios, dist, confidence_score
from vision.tools.image_stitching import find_keypoints, find_translation, resize_img


class FsTracker():

    def __init__(self, frame_size=[2048, 1536], frame_id=0, track_id=0, minimal_max_distance=10, max_ranges=20, translation_size=640):
        self.tracklets = []
        self.track_id = track_id
        self.frame_id = frame_id
        self.max_distance = minimal_max_distance
        self.minimal_max_distance = minimal_max_distance
        self.max_ranges = max_ranges
        self.x_distance = 0
        self.y_distance = 0

        self.score_weights = [0.5, 1, 0.5]
        self.frame_size = frame_size

        self.translation_size = translation_size
        self.last_kp = None
        self.last_des = None
        self.last_center_x = None

    def reset_state(self):
        pass

    def update(self, detections, frame):
        frame, r = resize_img(frame, self.translation_size)
        search_window = self.get_search_ranges(detections, frame, r)
        self.update_max_distance()

        self.is_extreme_shift()
        self.deactivate_tracks()
        """ find dets that match tracks"""
        track_windows = self.get_track_search_window_by_id(search_window)
        matches = self.match_detections_to_windows(track_windows, detections)
        not_coupled = self.calc_matches_score_and_update(matches, detections)

        """ add new dets tracks"""
        for det_id in not_coupled:
            self.add_track(detections[det_id])

        """ remove tracks exceeding max_range"""
        self.update_accumulated_dist()

        online_track = [t.output() for t in self.tracklets if t.is_activated]

        return online_track, track_windows



    def get_search_ranges(self, detections, frame, r, percentile=10):
        kp, des = find_keypoints(frame)
        if self.last_kp is not None:
            tx, ty = find_translation(self.last_kp, self.last_des, kp, des, r)
            if tx is None:
                tx, ty = self.x_distance, self.y_distance
        else:
            tx, ty = 0, 0
        self.last_kp = kp
        self.last_des = des

        # if len(self.x_distance) >= self.max_ranges:
        #     new_x_dist = [r for r in self.x_distance[1:]]
        #     self.x_distance = new_x_dist

        self.x_distance = tx
        self.y_distance = ty

        return tx, ty

    def get_track_search_window_by_id(self, search_window):

        track_windows = []
        for track in self.tracklets:
            track_windows.append(track.get_track_search_window(search_window))


        return track_windows

    @staticmethod
    def get_detections_center(detections):

        det_centers = []
        for det in detections:
            center_x = (det[0] + det[2]) / 2
            center_y = (det[1] + det[3]) / 2
            det_centers.append((center_x, center_y))

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

        matches = {}

        if len(bboxes1) > 0 and len(bboxes2) > 0:
            x11, y11, x12, y12 = np.split(np.array(bboxes1), 4, axis=1)
            x21, y21, x22, y22 = np.split(np.array(bboxes2)[:, :4], 4, axis=1)
            xA = np.maximum(x11, np.transpose(x21))
            yA = np.maximum(y11, np.transpose(y21))
            xB = np.minimum(x12, np.transpose(x22))
            yB = np.minimum(y12, np.transpose(y22))
            inetr_area = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
            intersections = inetr_area > 0

            for i in range(intersections.shape[1]):
                matches[i] = list(intersections[:, i])

        return matches

    def match_by_center(self, windows, detections):
        centers = self.get_detections_center(detections)

        ids_list = []
        dets_list = []
        for det_id, center in enumerate(centers):
            t_list = [center for _ in windows]
            dets_list.append(t_list)
            ids_list.append(det_id)

        matches = dict()
        for det_id in ids_list:
            track_tf = list(map(self.is_in_range, windows, dets_list[det_id]))
            matches[det_id] = track_tf

        return matches

    @staticmethod
    def is_in_range(window, center):
        x_valid = False
        y_valid = False
        if window[0] <= center[0] and center[0] <= window[2]:
            x_valid = True

            if window[1] <= center[1] and center[1] <= window[3]:
                y_valid = True

        return x_valid & y_valid


    def calc_matches_score_and_update(self, matches, detections):
        det_ids = [i for i in range(len(detections))]
        coupled_dets = []

        for id_, track_tf in matches.items():

            matched_tracks, matched_track_index, matched_tracks_acc_dist = self.get_matches(track_tf)
            if len(matched_track_index) == 0:
                continue

            det = detections[id_]

            matches_bbox = [[track[0], track[1], track[2], track[3]] for track in matched_tracks]
            ratio_score = compute_ratios(det[:4], matches_bbox)

            dist_score = np.array(dist(det[:4], matches_bbox, matched_tracks_acc_dist, np.abs(self.max_distance)))
            conf_score = np.array(confidence_score(det[4] * det[5], matched_tracks))

            weigthed_iou = ratio_score * self.score_weights[0]
            weigthed_dist = dist_score * self.score_weights[1]
            weigthed_conf = conf_score * self.score_weights[2]

            weigthed_score = (weigthed_iou + weigthed_dist + weigthed_conf) / np.sum(self.score_weights)
            track_index = np.argmax(weigthed_score)

            self.tracklets[matched_track_index[track_index]].update(det)
            coupled_dets.append(id_)

        not_coupled = []
        for det_id in det_ids:
            if det_id not in coupled_dets:
                not_coupled.append(det_id)

        return not_coupled

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

    def add_track(self, det):
        t = Track()
        t.add(det, self.track_id, self.frame_size)
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
                if len(track.accumulated_dist) > 0:  # object found once
                    track.accumulated_dist.append(self.x_distance)
                track.bbox[0] -= self.x_distance
                track.bbox[2] -= self.x_distance
                track.bbox[1] -= self.y_distance
                track.bbox[3] -= self.y_distance

                if track.bbox[0] >= 0 & track.bbox[2] >= 0 & track.bbox[0] <= self.frame_size[1] & track.bbox[0] <= self.frame_size[1]:
                    valid = True
                if np.abs(np.sum(track.accumulated_dist)) <= self.max_distance and valid:
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



