import numpy as np
#from cython_bbox import bbox_overlaps as bbox_ious


def compute_ratios(trk_windows, dets):
    trk_area = (trk_windows[:, 2] - trk_windows[:, 0]) * (trk_windows[:, 3] - trk_windows[:, 1])
    det_area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])

    trk_area = np.expand_dims(trk_area, axis=0)
    det_area = np.expand_dims(det_area, axis=0)

    trk_det = trk_area.T / det_area
    det_trk = (1 / trk_area.T) * det_area

    return np.min(np.stack((trk_det, det_trk), axis=2), axis=2)


def ratio(trck_box, det_box):
    trck_area = (trck_box[2] - trck_box[0]) * (trck_box[3] - trck_box[1])
    det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])

    return min(trck_area / det_area, det_area / trck_area)

def dist(trk_windows, detections, mean_x, mean_y, max_distance):

    distances = compute_dist_on_vec(trk_windows, detections)
    distances = distances / max_distance
    distances[distances > 1] = 1  # distance higher than allowed

    return 1 - distances


def relativity_score(trk_boxes, detections, relatives=10, score_rel=3):
    relatives = min(min(len(trk_boxes), len(detections)) - 1, relatives)
    trk_mag, trk_ang, trk_args = get_features(trk_boxes, relatives)
    det_mag, det_ang, det_args = get_features(detections, relatives)

    best_trk_mag = trk_mag
    best_trk_ang = trk_ang
    best_det_mag = det_mag
    best_det_ang = det_ang

    if len(best_trk_mag) < score_rel:
        Warning("number of relatives is smaller than 'score_rel', using all relatives")
        score_rel = len(best_trk_mag)

    res = []
    for i in range(len(det_mag)):

        m = best_det_mag[i, 1:]
        a = best_det_ang[i, 1:]

        dm = best_trk_mag[:, 1:] - m
        da = best_trk_ang[:, 1:] - a

        #tf_m = dm < 20
        #tf_a = np.abs(da) < 10

        #tf = tf_m & tf_a
        #count = np.sum(tf, axis=1)
        #relative_match = count > 2
        relative_diff = np.sqrt(np.abs(dm * da))
        #relative_diff = da ** 2
        relative_diff = np.sum(relative_diff, axis=1)  # [:, :score_rel], axis=1)
        res.append(relative_diff)

    res = np.stack(res, axis=1)

    res /= max(res.max(), 1E-6)

    return 1 - res
    #return res







def get_features(bboxes, relatives):

    center_x = (bboxes[:, 2] + bboxes[:, 0]) / 2
    center_y = (bboxes[:, 3] + bboxes[:, 1]) / 2

    center_x = np.expand_dims(center_x, axis=0)
    center_y = np.expand_dims(center_y, axis=0)
    x_diffs = center_x.T - center_x
    y_diffs = center_y.T - center_y
    magnitudes = np.sqrt(x_diffs**2 + y_diffs**2)
    angles = np.arctan2(y_diffs, x_diffs) * 180 / np.pi

    args = np.argsort(magnitudes, axis=1)

    ordered_mag = np.take_along_axis(magnitudes, args, axis=1)
    ordered_ang = np.take_along_axis(angles, args, axis=1)

    return ordered_mag[:, :relatives], ordered_ang[:, :relatives], args
    #return magnitudes, angles





# def dist(tlbr, btlbrs, mean_movement, max_distance):
#
#     tlbrs = [tlbr for _ in btlbrs]
#
#     distances = np.array(list(map(compute_dist, tlbrs, btlbrs, mean_movement)))
#     distances = distances / max_distance
#
#     return 1 - distances
    #return distances

def compute_dist(atlbr, btlbr, mean_movment):

    # a center - detection
    a_x = (atlbr[0] + atlbr[2]) / 2
    a_y = (atlbr[1] + atlbr[3]) / 2

    # b center - tracks
    b_x = (btlbr[0] + btlbr[2]) / 2 - mean_movment
    b_y = (btlbr[1] + btlbr[3]) / 2

    # Euclidan distance
    return np.sqrt((a_x - b_x)**2 + (a_y - b_y)**2)
    # center - tracks

def compute_dist_on_vec(trk_windows, dets):

    trk_x = (trk_windows[:, 0] + trk_windows[:, 2]) / 2 # - mean_x_movment
    trk_y = (trk_windows[:, 1] + trk_windows[:, 3]) / 2 #- mean_y_movment


    det_x = (dets[:, 0] + dets[:, 2]) / 2
    det_y = (dets[:, 1] + dets[:, 3]) / 2

    trk_x = np.expand_dims(trk_x, axis=0)
    det_x = np.expand_dims(det_x, axis=0)

    sqaured_diff_x = np.power((trk_x.T - det_x), 2)

    trk_y = np.expand_dims(trk_y, axis=0)
    det_y = np.expand_dims(det_y, axis=0)

    sqaured_diff_y = np.power((trk_y.T - det_y), 2)

    return np.sqrt(sqaured_diff_x + sqaured_diff_y)



def confidence_score(trk_score, dets_score):
    trk_score = np.expand_dims(trk_score, axis=0)
    dets_score = np.expand_dims(dets_score, axis=0)

    return 1 - np.abs(trk_score.T - dets_score)

def z_score(trk_depth, dets_depth):

    trk_score = np.expand_dims(trk_depth, axis=0)
    dets_score = np.expand_dims(dets_depth, axis=0)

    z_diff = np.abs(trk_score.T - dets_score)
    z_diff = z_diff / (np.nanmax(z_diff) - np.nanmin(z_diff))
    z_diff[np.isnan(z_diff)] = 0.5  # in case of nan, the output score will be 0.5

    return 1 - z_diff
    # score = []
    # for det in dets:
    #     det_conf = det[4] * det[5]
    #     score.append(1 - np.abs(aconf - det_conf))
    #
    # return score

def get_intersection(bboxes1, bboxes2):  # matches
    inter_aera = []

    if len(bboxes1) > 0 and len(bboxes2) > 0:
        x11, y11, x12, y12 = np.split(np.array(bboxes1), 4, axis=1)
        x21, y21, x22, y22 = np.split(np.array(bboxes2)[:, :4], 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        inter_aera = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

    return inter_aera
