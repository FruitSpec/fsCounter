import cv2
import numpy as np

from vision.tracker.fsTracker.score_func import get_intersection


def filter_by_distance(dets, depth, threshold=150, percentile=0.4, factor=2.5):
    filtered_dets = []
    range_ = []
    for det in dets:
        crop = depth[det[1]:det[3] - 1, det[0]:det[2] - 1]
        h, w = crop.shape
        if w == 0 or h == 0:
            range_.append(0)
        else:
            range_.append(np.nanmean(crop))

    if range_:  # not empty
        bool_vec = np.array(range_) > threshold

        for d_id, bool_val in enumerate(bool_vec):
            if bool_val:
                filtered_dets.append(dets[d_id])

    return filtered_dets

def filter_by_intersection(dets_outputs, threshold=0.8):
    filtered = []
    if len(dets_outputs) > 0:
        dets = np.array(dets_outputs)
        dets = dets[:, :4]

        inter = get_intersection(dets, dets)
        area = (dets[:, 3] - dets[:, 1]) * (dets[:, 2] - dets[:, 0])

        inter = inter / area

        tot_duplicants = []
        valid_dets = []
        for i in range(inter.shape[1]):
            #inter[i, :] = 0
            vec = inter[:, i]
            vec[i] = 0
            dup = vec > threshold

            if np.sum(dup) > 0:
                for j, tf in enumerate(dup):
                    if tf:
                        tot_duplicants.append(j)

    filtered = [det for i, det in enumerate(dets_outputs) if i not in tot_duplicants]
    return filtered



def filter_by_duplicates(dets_outputs, iou_threshold=0.9):
    dets = np.array(dets_outputs)
    scores = dets[:, 4] * dets[:, 5]
    dets = dets[:, :4]

    inter = get_intersection(dets, dets)





    area = (dets[:, 3] - dets[:, 1]) * (dets[:, 2] - dets[:, 0])

    area = np.expand_dims(area, axis=0)
    mat_area = area.T + area

    union = mat_area - inter

    iou_mat = inter / union

    tot_duplicants = []
    for i in range(iou_mat.shape[1]):
        iou_mat[i, :] = 0
        vec = iou_mat[:, i]
        dup = vec > iou_threshold

        if np.sum(dup) > 0:
            dup_index = []

            for j, tf in enumerate(dup):
                if tf:
                    dup_index.append(j)
            tot_duplicants += dup_index

    filtered = [det for i, det in enumerate(dets_outputs) if i not in tot_duplicants]
    return filtered


def filter_by_size(det_outputs, size_threshold=1000):
    filtered = []
    if len(det_outputs) > 0:
        dets = np.array(det_outputs)[:, :4]
        area = (dets[:, 3] - dets[:, 1]) * (dets[:, 2] - dets[:, 0])

        tf_vec = area > size_threshold
        for det, tf in zip(det_outputs, tf_vec):
            if tf:
                filtered.append(det)

    return filtered


def filter_by_height(det_outputs, depth, bias=0, y_crop=200):
    filtered = []

    if len(det_outputs) > 0:
        grad_y = cv2.Sobel(depth[y_crop:-y_crop, :], cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        grad_y[grad_y >= -300] = 0
        grad_y[grad_y < -300] = 1

        y_loc = np.argmax(grad_y, axis=0)
        y_threshold = np.mean(y_loc[y_loc > 0]) + y_crop + bias

        for det in det_outputs:
            if det[1] > y_threshold:
                filtered.append(det)

    return filtered
