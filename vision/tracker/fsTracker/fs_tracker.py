import os.path
import cv2
import numpy as np
# from cython_bbox import bbox_overlaps
from numba import jit
import pickle

from vision.tracker.fsTracker.base_track import Track
from vision.tracker.fsTracker.score_func import compute_ratios, dist, get_intersection, relativity_score
from vision.tracker.fsTracker.score_func import z_score
from vision.misc.help_func import validate_output_path


class FsTracker():

    def __init__(self, frame_size=[2048, 1536], frame_id=0, track_id=0, minimal_max_distance=10,
                 score_weights=[0.5, 1, 1, 0], match_type='center', det_area=1, translation_size=640, max_losses=10,
                 major=3, minor=2.5, compile_data=None, debug_folder=None):

        self.tracklets = []
        self.track_id = track_id
        self.frame_id = frame_id
        self.max_distance = minimal_max_distance
        self.minimal_max_distance = minimal_max_distance
        self.x_distance = 0
        self.y_distance = 0
        self.major = major
        self.minor = minor
        self.match_type = match_type
        self.det_area = det_area
        self.max_losses = max_losses

        self.score_weights = score_weights
        self.frame_size = frame_size

        self.translation_size = translation_size

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
        track_windows = self.get_track_search_window_by_id([tx, ty], tracklets_bboxes)
        matches = self.match_detections_to_windows(track_windows, detections, self.match_type)
        not_coupled_dets, trk_det_couples = self.calc_matches_score_and_update(matches,
                                                                          detections,
                                                                          track_windows,
                                                                          tracklets_bboxes,
                                                                          tracklets_scores,
                                                                          dets_depth,
                                                                          track_depth)

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


    def get_track_search_window_by_id(self, search_window, tracklets_bboxes, margin=15):
        tx = search_window[0] * self.major
        ty = search_window[1] * self.major
        tx_m = search_window[0] * self.minor
        ty_m = search_window[1] * self.minor
        tracklets_windows = []

        if len(tracklets_bboxes) > 0:

            x1, x2, y1, y2 = tracklets_to_windows(tracklets_bboxes, tx, tx_m, ty, ty_m, margin, np.array(self.frame_size))

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

    def match_by_intersection(self, bboxes1, bboxes2):

        intersections = []

        atlbrs = [box for box in bboxes1]
        btlbrs = [box[:4] for box in bboxes2]
        #bboxes2 = np.array(bboxes2).astype(float32)[:, :4]
        ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
        if ious.size > 0:
            ious = bbox_overlaps(
                np.ascontiguousarray(atlbrs, dtype=float),
                np.ascontiguousarray(btlbrs, dtype=float)
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


    def calc_matches_score_and_update(self, matches, detections, track_windows, track_bboxes, trk_score, dets_depth=None, trk_depth=None):
        if len(track_windows) == 0:
            return [i for i in range(len(detections))], {}
        elif len(detections) == 0:
            return [], {}

        detections_arr = np.array(detections)
        dets = detections_arr[:, :4]

        dist_score = dist(track_windows, dets, np.abs(self.max_distance))
        ratio_score = compute_ratios(track_bboxes, dets)
        rel_score = relativity_score(track_bboxes, dets)

        # no match get 0 score
        matches = assign_single_match(matches)
        no_match = np.logical_not(matches)
        dist_score[no_match] = 0  # no match
        ratio_score[no_match] = 0
        rel_score[no_match] = 0

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
        weigthed_rel = rel_score * self.score_weights[3]

        weigthed_score = (weigthed_ratio + weigthed_dist + weigthed_depth + weigthed_rel) / np.sum(self.score_weights)

        trk_det_couples, not_coupled = assign_det_by_score(weigthed_score)
        for trk_id, det_id in trk_det_couples.items():
            if dets_depth is not None:
                depth = dets_depth[det_id]
            else:
                depth = None
            self.tracklets[trk_id].update(detections[det_id], depth)

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
                track.lost_counter += 1
                if track.lost_counter < self.max_losses:
                    if len(track.accumulated_dist) == 0:  # object found once
                        track.accumulated_dist.append(self.x_distance)
                        track.accumulated_height.append(self.y_distance)
                    mean_dist = np.mean(track.accumulated_dist)
                    mean_height = np.mean(track.accumulated_height)
                    track.bbox[0] -= mean_dist
                    track.bbox[2] -= mean_dist
                    track.bbox[1] -= mean_height
                    track.bbox[3] -= mean_height

                    valid = self.validate_track_location(track, self.frame_size[1])
                    if valid:
                        new_tracklets_list.append(track)
            else:
                new_tracklets_list.append(track)

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


@jit(nopython=True, cache=True, nogil=True)
def tracklets_to_windows(tracklets_bboxes, tx:float, tx_m:float, ty:float, ty_m:float, margin, frame_size):
   # tracklets_bboxes = np.vstack(tracklets_bboxes)

    if tx > 0:  # moving right - fruits moving left
        x1 = tracklets_bboxes[:, 0] - tx
        x2 = tracklets_bboxes[:, 2] + margin - tx_m
    else:
        x1 = tracklets_bboxes[:, 0] - margin - tx_m
        x2 = tracklets_bboxes[:, 2] - tx

    x1[x1 < 0] = 0
    x2[x2 > frame_size[1]] = frame_size[1]

    if ty > 0:
        y1 = tracklets_bboxes[:, 1] - ty
        y2 = tracklets_bboxes[:, 3] + margin - ty_m
    else:
        y1 = tracklets_bboxes[:, 1] - margin - ty_m
        y2 = tracklets_bboxes[:, 3] - ty

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

def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)

    for k in range(K):
        box_area = (
                (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                    min(boxes[n, 2], query_boxes[k, 2]) -
                    max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                        min(boxes[n, 3], query_boxes[k, 3]) -
                        max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = (
                            (boxes[n, 2] - boxes[n, 0] + 1) *
                            (boxes[n, 3] - boxes[n, 1] + 1) +
                            box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


if __name__ == "__main__":
    f_list = ["/home/fruitspec-lab/FruitSpec/Sandbox/Sliced_data/RA_3_A_2/RA_3_A_2/res/f1.pkl",
              "/home/fruitspec-lab/FruitSpec/Sandbox/Sliced_data/RA_3_A_2/RA_3_A_2/res/f2.pkl",
              "/home/fruitspec-lab/FruitSpec/Sandbox/Sliced_data/RA_3_A_2/RA_3_A_2/res/f3.pkl"]

    test_func(f_list)



