import os.path
import cv2
import numpy as np
#from cython_bbox import bbox_overlaps
from numba import jit
import pickle
from collections import deque

from vision.tracker.fsTracker.base_track import Track
from vision.tracker.fsTracker.score_func import compute_ratios, dist, get_intersection, relativity_score
from vision.tracker.fsTracker.score_func import z_score
from vision.misc.help_func import validate_output_path


class FsTracker():

    def __init__(self, cfg, args, frame_id=0, track_id=0, debug_folder=None):

        self.tracklets = []
        self.track_id = track_id
        self.frame_id = frame_id
        self.max_distance = cfg.tracker.minimal_max_distance
        self.minimal_max_distance = cfg.tracker.minimal_max_distance
        self.x_distance = 0
        self.y_distance = 0
        self.major = cfg.tracker.major
        self.minor = cfg.tracker.minor
        self.match_type = cfg.tracker.match_type
        self.det_area = cfg.tracker.det_area
        self.max_losses = cfg.tracker.max_losses
        self.ranges = cfg.tracker.ranges
        self.close_frame = cfg.tracker.close_frame

        self.score_weights = cfg.tracker.score_weights
        self.frame_size = args.frame_size

        self.history_mid = deque(maxlen=10)
        self.history_mid_std = deque(maxlen=10)
        self.history_close = deque(maxlen=10)
        self.history_close_std = deque(maxlen=10)
        self.history_far = deque(maxlen=10)
        self.history_far_std = deque(maxlen=10)
        self.history_mid_not_updated = 0
        self.history_close_not_updated = 0
        self.history_far_not_updated = 0
        self.not_coupled_ratio = 1
        self.box_far_size_threshold = 250
        self.box_mid_size_threshold = 400


        if int(np.__version__.split('.')[-2]) > 21:
            self.match_by_intersection = self.match_by_intersection_lab
        else:
            from cython_bbox import bbox_overlaps
            self.match_by_intersection = self.match_by_intersection_edge

        compile_data = cfg.tracker.compile_data_path
        if compile_data is not None:
            self.run_compile(compile_data)

        if debug_folder is not None:
            self.last_frame = None
            self.f_id = 0
            validate_output_path(debug_folder)
            self.debug_folder = debug_folder


    def reset_state(self):
        self.tracklets = []
        self.frame_id = 0
        self.track_id = 0

    def run_compile(self, compile_data):
        with open(compile_data, 'rb') as f:
            copile_data = pickle.load(f)
        dets_inputs = copile_data['dets']
        translation_inputs = copile_data['translation']

        print('Start tracker init')
        for det_input, translation_input in zip(dets_inputs, translation_inputs):
            for frame_det, frame_trans in zip(det_input, translation_input):
                if frame_det is not None:
                    tx, ty = frame_trans
                    self.update(frame_det, tx, ty)

        self.reset_state()
        print('Done tracker init')



    def update(self, detections, tx, ty, frame_id=None, dets_depth=None):
        tx, ty = self.update_frame_meta(frame_id, tx, ty)
        self.update_max_distance()
        self.is_extreme_shift()

        self.deactivate_tracks()
        """ find dets that match tracks"""
        tracklets_bboxes, tracklets_scores, track_acc_dist, tracks_acc_height, track_depth = self.get_tracklets_data()
        track_windows = self.get_track_search_window_by_id([tx, ty], tracklets_bboxes, track_depth, dets_depth)
        matches = self.match_detections_to_windows(track_windows, detections, self.match_type)
        not_coupled_dets, trk_det_couples = self.calc_matches_score_and_update(matches,
                                                                          detections,
                                                                          track_windows,
                                                                          tracklets_bboxes,
                                                                          dets_depth,
                                                                          track_depth)

        """ update adapdive distance"""
        if len(detections) > 0:
            self.not_coupled_ratio = len(not_coupled_dets) / len(detections)
        self.update_adaptive_distance()

        """ remove lost tracks"""
        self.update_accumulated_dist()

        """ add new dets tracks"""
        for det_id in not_coupled_dets:
            self.add_track(detections, det_id, dets_depth)

        online_track = [t.output() for t in self.tracklets if t.is_activated]

        return online_track, track_windows

    def get_tracklets_data(self):
        tracklets_bboxes = []
        tracklets_scores = []

        results = list(map(self.get_data, self.tracklets))
        tracklets = []
        track_acc_dist = []
        track_acc_height = []
        track_depth = []
        for res in results:
            tracklets.append(res[0])
            track_acc_dist.append(res[1])
            track_acc_height.append(res[2])
            track_depth.append(res[3])
        if len(tracklets) > 0:
            tracklets = np.vstack(tracklets)
            tracklets_bboxes = tracklets[:, :4]
            tracklets_scores = tracklets[:, 4]

        return tracklets_bboxes, tracklets_scores, track_acc_dist, track_acc_height, track_depth

    def get_data(self, track):
        tracklet = self.get_track(track)
        acc_dist = self.get_acc_dist(track)
        acc_height = self.get_acc_height(track)
        track_depth = self.get_track_depth(track)

        return tracklet, acc_dist, acc_height, track_depth

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

    def remove_lost_tracks(self, tracklets, trk_det_couples, tracklets_bboxes):
        activated_trk = list(trk_det_couples.keys())
        bool_vec = np.ones((len(tracklets_bboxes)))
        lost_numbers = len(tracklets_bboxes) - len(activated_trk)
        # no lost tracklets
        if lost_numbers == 0:
            return tracklets

        lost_means = np.ones((lost_numbers, 2))
        lost_count = np.zeros((lost_numbers))
        lost_trk_ids = []
        count = 0
        for i in range(len(tracklets_bboxes)):
            if i in activated_trk:
                bool_vec[i] = 0
            else:
                lost_trk_ids.append(i)
                tracklets[i].lost_counter += 1
                if len(tracklets[i].accumulated_dist) == 0:  # object found once
                    tracklets[i].accumulated_dist.append(self.x_distance)
                    tracklets[i].accumulated_height.append(self.y_distance)
                lost_means[count, 0] = np.mean(tracklets[i].accumulated_dist)
                lost_means[count, 1] = np.mean(tracklets[i].accumulated_height)
                lost_count[count] = tracklets[i].lost_counter
                count += 1


        lost_tracklets_bboxes = tracklets_bboxes[bool_vec.astype(np.bool), :]
        valid = calc_and_validate(lost_tracklets_bboxes, lost_means, lost_count, np.array(self.frame_size), self.max_losses)

        valid_trk_ids = np.array(lost_trk_ids)[valid]
        orig_lost_ids = [i for i in range(len(lost_trk_ids))]
        valid_lost_ids = np.array(orig_lost_ids)[valid]


        for trk_id, lost_id in zip(valid_trk_ids, valid_lost_ids):
            tracklets[trk_id].bbox = lost_tracklets_bboxes[lost_id, :]

        new_tracklets = []
        for i, t in enumerate(tracklets):
            if (i in valid_trk_ids) or (i in activated_trk):
                new_tracklets.append(t)

        return new_tracklets


    def get_track_search_window_by_id(self, search_window, tracklets_bboxes, track_depth, dets_depth, margin=5):
        # tx = search_window[0] * self.major
        # ty = search_window[1] * self.major
        # tx_m = search_window[0] * self.minor
        # ty_m = search_window[1] * self.minor
        tracklets_windows = []

        tx, tx_m, ty, ty_m, type_ = self.get_ranges_vectors(search_window, len(tracklets_bboxes), tracklets_bboxes,  track_depth, dets_depth)

        if len(tracklets_bboxes) > 0:

            x1, x2, y1, y2 = tracklets_to_windows(tracklets_bboxes, tx, tx_m, ty, ty_m, margin, np.array(self.frame_size))

            tracklets_windows = np.stack([x1, y1, x2, y2, type_], axis=1)

        return tracklets_windows


    def get_ranges_vectors(self, search_window, n_windows, tracklets_bboxes, track_depth, dets_depth):

        if None in track_depth:
            txs, tx_ms, tys, ty_ms, type_ = self.ranges_without_depth(search_window, n_windows)
        else:
            print(f"not coupled ratio: {self.not_coupled_ratio}")
            #if self.not_coupled_ratio < 0.6 and len(self.history_mid) > 5 and  len(self.history_close) > 5:
            mean_mid, mean_close, mean_far, use_adaptive = self.get_adaptive_distances(search_window)
            range_, frame_type = self.is_close_scene(dets_depth, self.close_frame)

            txs = []
            tx_ms = []
            tys = []
            ty_ms = []
            type_ = []

            for bbox in tracklets_bboxes:
                box_range, box_type = self.update_box_type_in_close_scene(bbox, range_)
                # if range_ == 'close':
                #     box_range, box_type = self.update_box_type_in_close_scene(bbox, range_)
                # else:
                #     box_range = 'mid'
                #     box_type = 1

                if use_adaptive:
                    #print('using adaptive')
                    if box_type == 2:
                        tx = mean_far
                    if box_type == 1:
                        tx = mean_mid
                    elif box_type == 0:
                        tx = mean_close
                else:
                    tx = search_window[0]

                txs.append(tx * self.ranges[box_range]['major'])
                tx_ms.append(tx * self.ranges[box_range]['minor'])
                tys.append(search_window[1] * self.ranges[box_range]['major'])
                ty_ms.append(search_window[1] * self.ranges[box_range]['minor'])
                type_.append(box_type)
        return np.array(txs), np.array(tx_ms), np.array(tys), np.array(ty_ms), np.array(type_)

    def get_adaptive_distances(self, search_window, not_coupled_ratio_thrshold=0.8):
        if len(self.history_mid) > 5 and self.not_coupled_ratio < not_coupled_ratio_thrshold:
            use_adaptive = True
            mean_mid = np.mean(self.history_mid)
            if len(self.history_far) > 0:
                mean_far = np.mean(self.history_far)
            else:
                mean_far = mean_mid

            if len(self.history_close) > 0:
                mean_close = np.mean(self.history_close)
            else:
                mean_close = mean_mid
        else:
            use_adaptive = False
            mean_mid = search_window[0]
            mean_close = search_window[0]
            mean_far = search_window[0]

        return mean_mid, mean_close, mean_far, use_adaptive



    def update_box_type_in_close_scene(self, bbox, range_):
        size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if range_ == 'close':
            if size < self.box_far_size_threshold:
                box_range = 'far'
                box_type = 2
            elif size < self.box_mid_size_threshold:
                box_range = 'mid'
                box_type = 1
            else:
                box_range = 'close'
                box_type = 0
        else:
            box_range = 'mid'
            box_type = 1

        return box_range, box_type
    @staticmethod
    def is_close_scene(dets_depth, close_score=0.9):

        if dets_depth is None:
            range_ = 'mid'
            frame_type = 1
        else:
            if len(dets_depth) > 10:
                dets_depth.sort()
                min_ranges = dets_depth[:10]
            else:
                min_ranges = dets_depth

            scene_range = np.mean(min_ranges)
            if scene_range > close_score:
                range_ = 'mid'
                frame_type = 1
            else:
                range_ = 'close'
                frame_type = 0

        return range_, frame_type

    def ranges_without_depth(self, search_window, n_windows):
        txs = []
        tx_ms = []
        tys = []
        ty_ms = []
        type_ = []
        for _ in range(n_windows):
            txs.append(search_window[0] * self.ranges['mid']['major'])
            tx_ms.append(search_window[0] * self.ranges['mid']['minor'])
            tys.append(search_window[1] * self.ranges['mid']['major'])
            ty_ms.append(search_window[1] * self.ranges['mid']['minor'])
            type_.append(1)

        return txs, tx_ms, tys, ty_ms, type_


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

    # def get_track_search_window_by_id_np(self, search_window):
    #
    #
    #     track_windows = []
    #     for track in self.tracklets:
    #         track_windows.append(track.get_track_search_window(search_window))
    #
    #
    #     return track_windows

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

    def match_by_intersection_lab(self, bboxes1, bboxes2):

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

    def match_by_intersection_edge(self, bboxes1, bboxes2):
       intersections = []
       atlbrs = [box for box in bboxes1]
       btlbrs = [box[:4] for box in bboxes2]
       #bboxes2 = np.array(bboxes2).astype(np.float32)[:, :4]
       ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
       if ious.size > 0:
           ious = bbox_overlaps(
               np.ascontiguousarray(atlbrs, dtype=np.float32),
               np.ascontiguousarray(btlbrs, dtype=np.float32)
           )
           intersections = ious > 0

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


    def calc_matches_score_and_update(self, matches, detections, track_windows, track_bboxes, dets_depth=None, trk_depth=None):
        if len(track_windows) == 0:
            return [i for i in range(len(detections))], {}
        elif len(detections) == 0:
            return [], {}

        detections_arr = np.array(detections)
        dets = detections_arr[:, :4]

        dist_score = dist(track_windows, dets, np.abs(self.max_distance))
        ratio_score = compute_ratios(track_bboxes.astype(np.float32), dets)
        #rel_score = relativity_score(track_bboxes, dets)

        # remove values of no matches
        #matches = assign_single_match(matches)
        no_match = np.logical_not(matches)
        dist_score[no_match] = 0  # no match
        ratio_score[no_match] = 0
        #rel_score[no_match] = 0

        # calculate z score only if depth values exist
        if dets_depth is not None:
            no_match = np.logical_not(matches)
            depth_score = z_score(trk_depth, dets_depth)
            depth_score[no_match] = 0
            weigthed_depth = depth_score * self.score_weights[2]
        else:
            weigthed_depth = 0

        weigthed_ratio = ratio_score * self.score_weights[0]
        weigthed_dist = dist_score * self.score_weights[1]
        #weigthed_rel = rel_score * self.score_weights[3]

        #weigthed_score = (weigthed_ratio + weigthed_dist + weigthed_depth + weigthed_rel) / np.sum(self.score_weights)
        weigthed_score = (weigthed_ratio + weigthed_dist + weigthed_depth) / np.sum(self.score_weights)

        trk_det_couples, not_coupled = assign_det_by_score(weigthed_score)
        for trk_id, det_id in trk_det_couples.items():
            if dets_depth is not None:
                depth = dets_depth[det_id]
            else:
                depth = None
            self.tracklets[trk_id].update(detections[det_id], depth, track_windows[trk_id][4])

        return not_coupled, trk_det_couples


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


    def add_track(self, detections, det_id, dets_depth):
        if dets_depth is not None:
            depth = dets_depth[det_id]
        else:
            depth = None

        t = Track()
        t.add(detections[det_id], depth, self.track_id, self.frame_size, 1)
        self.track_id += 1
        self.tracklets.append(t)


    def deactivate_tracks(self):
        for t in self.tracklets:
            t.is_activated = False


    def update_accumulated_dist(self):

        new_tracklets_list = []
        for track in self.tracklets:
            if track.is_activated is False:  # not found in current frame
                track.lost_counter += 1
                if track.lost_counter < self.max_losses:
                    if track.type == 1 and len(self.history_mid) > 5:
                        mean_dist = np.mean(np.array(self.history_mid))
                    elif track.type == 0 and len(self.history_close) > 5:
                        mean_dist = np.mean(np.array(self.history_close))
                    elif track.type == 2 and len(self.history_far) > 5:
                        mean_dist = np.mean(np.array(self.history_far))
                    else:
                        mean_dist = self.x_distance

                    # if len(track.accumulated_dist) == 0:  # object found once
                    #     track.accumulated_dist.append(self.x_distance)
                    #     track.accumulated_height.append(self.y_distance)
                    # mean_dist = np.mean(track.accumulated_dist)
                    # mean_height = np.mean(track.accumulated_height)
                    track.bbox[0] -= mean_dist
                    track.bbox[2] -= mean_dist
                    #track.bbox[1] -= mean_height
                    #track.bbox[3] -= mean_height
                    #track.bbox[1] -= self.y_distance
                    #track.bbox[3] -= self.y_distance

                    valid = self.validate_track_location(track, self.frame_size[1])
                    if valid:
                        new_tracklets_list.append(track.copy())
            else:
                new_tracklets_list.append(track.copy())

        self.tracklets = new_tracklets_list


    @staticmethod
    def validate_track_location(track, frame_width):
        valid = False
        box_start = int(track.bbox[0])
        box_end = int(track.bbox[2])

        if (box_start >= 0) & (box_start < frame_width) & (box_end > 0) & (box_end <= frame_width):
            # inside frame
            valid = True
        elif box_start < 0:
            if box_end > 0: # on left edge
                valid = True
        elif box_end > frame_width: # on right edge
            if box_start < frame_width:
                valid = True

        return valid
    @staticmethod
    def validate_bbox_location(bboxes, frame_size):

        left_start_bool = bboxes[:, 0] >= 0
        left_end_bool = bboxes[:, 0] < frame_size[1]
        right_start_bool = bboxes[:, 2] > 0
        right_end_bool = bboxes[:, 2] <= frame_size[1]
        inside_bool = left_start_bool & left_end_bool & right_start_bool & right_end_bool

        left_out_bool = np.logical_not(left_start_bool)
        on_left_edge = left_out_bool & right_start_bool

        right_out_bool = bboxes[:, 2] > frame_size[1]
        on_right_edge = right_out_bool & left_end_bool

        inside_bool = np.logical_or(inside_bool, on_left_edge)
        inside_bool = np.logical_or(inside_bool, on_right_edge)
        return inside_bool

    def update_max_distance(self, mag=2):

        self.max_distance = self.frame_size[1]
        # self.max_distance = mag * self.x_distance
        # if np.abs(self.max_distance) < self.minimal_max_distance:
        #     self.max_distance = self.minimal_max_distance

    def update_adaptive_distance(self, min_samples=4):

        close = []
        mid = []
        far = []
        for tracklet in self.tracklets:
            if (tracklet._count > 3) and (tracklet.lost_counter == 0):
                last_distance = tracklet.accumulated_dist[-1]
                type_ = tracklet.type

                if type_ == 1:
                    mid.append(last_distance)
                elif type_ == 0:
                    close.append(last_distance)
                elif type_ == 2:
                    far.append(last_distance)

        if len(mid) > min_samples:
            self.history_mid.append(np.mean(mid))
            self.history_mid_std.append(np.std(mid))
            self.history_mid_not_updated = 0
        else:
            self.history_mid_not_updated += 1

        if len(close) > min_samples:
            self.history_close.append(np.mean(close))
            self.history_close_std.append(np.std(close))
            self.history_close_not_updated = 0
        else:
            self.history_close_not_updated += 1

        if len(far) > min_samples:
            self.history_far.append(np.mean(far))
            self.history_far_std.append(np.std(far))
            self.history_far_not_updated = 0
        else:
            self.history_far_not_updated += 1


        if self.history_mid_not_updated > 5:
            self.history_mid.clear()
            self.history_mid_std.clear()


        if self.history_close_not_updated > 5:
            self.history_close.clear()
            self.history_close_std.clear()

        if self.history_far_not_updated > 5:
            self.history_far.clear()
            self.history_far_std.clear()




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

@jit(nopython=True, cache=True, nogil=True)
def assign_single_match(matches):
    mask = np.zeros(matches.shape)
    for i in range(mask.shape[0]):
        if np.sum(matches[i, :]) == 1:
            det_id = np.argmax(matches[i, :])
            mask[i, det_id] = 1
        else:
            mask[i, :] = 1
    matches = np.logical_and(matches, mask)

    return matches

@jit(nopython=True, cache=True, nogil=True)
def assign_det_by_score(weigthed_score):
    trk_det_couples_dict = {}

    coupled_dets = []
    coupled_trk = []
    n_dets = weigthed_score.shape[1]
    mat_ids = np.argsort(weigthed_score.flatten())[::-1]
    # for c in range(n_dets):
    for mat_id in mat_ids:
        # mat_id = np.argmax(weigthed_score)
        trk_id = mat_id // n_dets
        det_id = mat_id % n_dets
        if weigthed_score[trk_id, det_id] == 0:
            break
        if len(coupled_dets) == n_dets:
            break
        if det_id in coupled_dets:
            continue
        if trk_id in coupled_trk:
            continue

        trk_det_couples_dict[trk_id] = det_id
        coupled_dets.append(det_id)
        coupled_trk.append(trk_id)

    not_coupled = []
    for det_id in range(n_dets):
        if det_id not in coupled_dets:
            not_coupled.append(det_id)

    return trk_det_couples_dict, not_coupled


#@jit(nopython=True, cache=True, nogil=True)
def tracklets_to_windows(tracklets_bboxes, tx, tx_m, ty, ty_m, margin, frame_size):
    # tracklets_bboxes = np.vstack(tracklets_bboxes)
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for i, tracklet_bbox in enumerate(tracklets_bboxes):
        if tx[i] > 0:  # moving right - fruits moving left
            x1.append(tracklet_bbox[0] - tx[i])
            x2.append(tracklet_bbox[2] - (tx_m[i] - margin))
        else:
            x1.append(tracklet_bbox[0] - (tx_m[i] + margin))
            x2.append(tracklet_bbox[2] - tx[i])



        if ty[i] > 0:
            y1.append(tracklet_bbox[1] - ty[i])
            y2.append(tracklet_bbox[3] - (ty_m[i] - margin))
        else:
            y1.append(tracklet_bbox[1] - (ty_m[i] + margin))
            y2.append(tracklet_bbox[3] - ty[i])

    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)

    x1[x1 < 0] = 0
    x2[x2 > frame_size[1]] = frame_size[1]

    y1[y1 < 0] = 0
    y2[y2 > frame_size[0]] = frame_size[0]

    return x1, x2, y1, y2




@jit(nopython=True)
def get_weighted_scores(matches, dist_score, ratio_score, rel_score, score_weights):

    matches = matches.astype(np.int32)
    dist_score = logical_not_assign_zero(dist_score, matches)
    ratio_score = logical_not_assign_zero(ratio_score, matches)
    rel_score = logical_not_assign_zero(rel_score, matches)
    weigthed_ratio = ratio_score * score_weights[0]
    weigthed_dist = dist_score * score_weights[1]
    weigthed_rel = rel_score * score_weights[3]

    return weigthed_ratio, weigthed_dist, weigthed_rel

@jit(nopython=True)
def logical_not_assign_zero(mat, mask):
    new_mat = np.zeros((mat.shape))
    for i in range(new_mat.shape[0]):
        for j in range(new_mat.shape[1]):
            if mask[i, j] > 0:
                new_mat[i, j] = mat[i, j]
    return new_mat


def thread_dist(trk_windows, detections, max_distance, result_queue):
    dist_score = dist(trk_windows, detections, max_distance)
    result_queue.put({"dist_score": dist_score})

def thread_compute_ratios(track_bboxes, dets, result_queue):
        ratio_score = compute_ratios(track_bboxes, dets)
        result_queue.put({"ratio_score": ratio_score})


def thread_relativity_score(track_bboxes, dets, result_queue):
    rel_score = relativity_score(track_bboxes, dets)
    result_queue.put({"rel_score": rel_score})


@jit(nopython=True)
def validate_bbox_location(bboxes, frame_size):

    left_start_bool = bboxes[:, 0] >= 0
    left_end_bool = bboxes[:, 0] < frame_size[1]
    right_start_bool = bboxes[:, 2] > 0
    right_end_bool = bboxes[:, 2] <= frame_size[1]
    inside_bool = left_start_bool & left_end_bool & right_start_bool & right_end_bool

    left_out_bool = np.logical_not(left_start_bool)
    on_left_edge = left_out_bool & right_start_bool

    right_out_bool = bboxes[:, 2] > frame_size[1]
    on_right_edge = right_out_bool & left_end_bool

    inside_bool = np.logical_or(inside_bool, on_left_edge)
    inside_bool = np.logical_or(inside_bool, on_right_edge)
    return inside_bool

@jit(nopython=True)
def calc_and_validate(lost_tracklets_bboxes, lost_means, lost_count, frame_size, max_losses):
    lost_tracklets_bboxes[:, 0] -= lost_means[:, 0]
    lost_tracklets_bboxes[:, 2] -= lost_means[:, 0]
    lost_tracklets_bboxes[:, 1] -= lost_means[:, 1]
    lost_tracklets_bboxes[:, 3] -= lost_means[:, 1]

    valid_loc = validate_bbox_location(lost_tracklets_bboxes, frame_size)
    valid_lost = lost_count < max_losses
    valid = np.logical_and(valid_lost, valid_loc)

    return valid

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



