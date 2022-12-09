import numpy as np
#from cython_bbox import bbox_overlaps as bbox_ious

# def ious(atlbrs, btlbrs):
#     """
#     Compute cost based on IoU
#     :type atlbrs: list[tlbr] | np.ndarray
#     :type atlbrs: list[tlbr] | np.ndarray
#
#     :rtype ious np.ndarray
#     """
#     ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
#     if ious.size == 0:
#         return ious
#
#     ious = bbox_ious(
#         np.ascontiguousarray(atlbrs, dtype=np.float),
#         np.ascontiguousarray(btlbrs, dtype=np.float)
#     )
#
#     return ious
#def compute_ratios(trk_windows, dets):
    #trk_area = (trk_windows[:, 2] - trk_windows[:, 0]) * (trk_windows[:, 3] - trk_windows[:, 1])
    #det_area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])

#    trk_area = np.expand_dims(trk_area, axis=0)
#    det_area = np.expand_dims(det_area, axis=0)

#    trk_det = trk_area.T / det_area
#    det_trk = (1 / trk_area.T) * det_area

#    return np.min(np.stack((trk_det, det_trk), axis=2), axis=2)

def compute_ratios(trck_bbox, dets):
    trck_boxes = [trck_bbox for _ in dets]
    return np.array(list(map(ratio, trck_boxes, dets)))

def ratio(trck_box, det_box):
    trck_area = (trck_box[2] - trck_box[0]) * (trck_box[3] - trck_box[1])
    det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])

    return min(trck_area / det_area, det_area / trck_area)

#def dist(tlbr, btlbrs, max_distance):
def dist(tlbr, btlbrs, mean_movement, max_distance):

    tlbrs = [tlbr for _ in btlbrs]

    #distances = np.array(list(map(compute_dist, tlbrs, btlbrs)))
    distances = np.array(list(map(compute_dist, tlbrs, btlbrs, mean_movement)))
    distances = distances / max_distance

    return 1 - distances
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

def confidence_score(aconf, dets):

    score = []
    for det in dets:
        det_conf = det[4] * det[5]
        score.append(1 - np.abs(aconf - det_conf))

    return score
