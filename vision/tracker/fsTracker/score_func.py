import numpy as np
import cupy as cp
import torch
from numba import jit
import time
import threading
import queue
#from cython_bbox import bbox_overlaps as bbox_ious

@jit(nopython=True, cache=True, nogil=True)
def compute_ratios(trk_windows: np.array, dets: np.array) -> np.array:
    trk_area = (trk_windows[:, 2] - trk_windows[:, 0]) * (trk_windows[:, 3] - trk_windows[:, 1])
    det_area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])

    trk_area = np.expand_dims(trk_area, axis=0)
    det_area = np.expand_dims(det_area, axis=0)

    trk_det = trk_area.T / det_area
    det_trk = (1 / trk_area.T) * det_area

    help_mat = np.stack((trk_det, det_trk), axis=2)
    #result = np.min(help_mat, axis=2)
    result = np.empty(help_mat.shape[:2])
    for i in range(help_mat.shape[0]):
        for j in range(help_mat.shape[1]):
            result[i, j] = np.min(help_mat[i, j, :])

    #result = np.min(help_mat, 2)

    return result


def compute_ratios_GPU(trk_windows, dets):
    trk_area = (trk_windows[:, 2] - trk_windows[:, 0]) * (trk_windows[:, 3] - trk_windows[:, 1])
    det_area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])

    trk_area = torch.unsqueeze(trk_area, dim=0)
    det_area = torch.unsqueeze(det_area, dim=0)

    trk_det = trk_area.T / det_area
    det_trk = (1 / trk_area.T) * det_area

    result = torch.min(torch.stack((trk_det, det_trk), dim=2), dim=2)[0]

    return result.to('cpu').numpy()

@jit(nopython=True, cache=True, nogil=True)
def ratio(trck_box, det_box):
    trck_area = (trck_box[2] - trck_box[0]) * (trck_box[3] - trck_box[1])
    det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])

    return min(trck_area / det_area, det_area / trck_area)

#@jit(nopython=True)
def dist(trk_windows, detections, max_distance):

    distances = compute_dist_on_vec(trk_windows, detections)
    distances = distances / max_distance
    bool_mat = distances > 1
    #idx_mat = bool_mat.astype(int)
    distances[bool_mat == True] = 1  # distance higher than allowed
    # bool_ = np.where(distances > 1, 1, 0)
    # if bool_.shape[0] == 1 and bool_.shape[1] == 1:
    #
    # distances[np.where(distances > 1, 1, 0)] = 1

    return 1 - distances

def dist_GPU(trk_windows, detections, mean_x, mean_y, max_distance):

    distances = compute_dist_on_vec_GPU(trk_windows, detections)
    distances = distances.to('cpu').numpy() / max_distance
    distances[distances > 1] = 1  # distance higher than allowed

    return 1 - distances


def relativity_score(trk_boxes, detections, relatives=10):
    relatives = min(min(len(trk_boxes), len(detections)) - 1, relatives)
    best_trk_mag, best_trk_ang, trk_args, best_det_mag, best_det_ang, det_args = get_rel_fetures(trk_boxes,
                                                                                                 detections,
                                                                                                 relatives)

    res = calc_rel_score_GPU(best_trk_mag, best_trk_ang, best_det_mag, best_det_ang)

    return 1 - res

@jit(nopython=True)
def calc_rel_score(best_trk_mag, best_trk_ang, best_det_mag, best_det_ang):

    res = np.empty((best_trk_mag.shape[0], best_det_mag.shape[0]))
    for i in range(len(best_det_mag)):

        m = best_det_mag[i, 1:]
        a = best_det_ang[i, 1:]

        dm = best_trk_mag[:, 1:] - m
        da = best_trk_ang[:, 1:] - a


        relative_diff = np.sqrt(np.abs(dm * da))
        # relative_diff = np.sum(relative_diff[i, :])  # [:, :score_rel], axis=1)

        rel_vec = np.empty((relative_diff.shape[0]))
        for j in range(relative_diff.shape[0]):
            rel_vec[j] = np.sum(relative_diff[j, :])
        res[:, i] = rel_vec

    #res = np.stack(res, axis=1)
    res /= (res.max() + 1E-6)

    return res



def calc_rel_score_GPU(trk_mag, trk_ang, det_mag, det_ang):

    dm = trk_mag[:, 1:].unsqueeze(1) - det_mag[:, 1:].unsqueeze(0)
    da = trk_ang[:, 1:].unsqueeze(1) - det_ang[:, 1:].unsqueeze(0)

    relative_diff = torch.sqrt(torch.abs(dm * da))
    relative_diff = torch.sum(relative_diff, dim=2)

    res = relative_diff / (relative_diff.max() + 1E-6)

    return res.to('cpu').numpy()


@jit(nopython=True)
def get_features(bboxes, relatives):
    bboxes = bboxes.astype(np.float32)
    center_x = (bboxes[:, 2] + bboxes[:, 0]) / 2
    center_y = (bboxes[:, 3] + bboxes[:, 1]) / 2

    center_x = np.expand_dims(center_x, axis=0)
    center_y = np.expand_dims(center_y, axis=0)
    x_diffs = center_x.T - center_x
    y_diffs = center_y.T - center_y
    #magnitudes = np.sqrt(x_diffs**2 + y_diffs**2)
    magnitudes = np.hypot(x_diffs, y_diffs)
    angles = np.arctan2(y_diffs, x_diffs) * 180 / np.pi

    args = np.empty(magnitudes.shape, dtype=np.int8)
    for i in range(magnitudes.shape[0]):
        args[i, :] = np.argsort(magnitudes[i, :])

    ordered_mag, ordered_ang = take_along_axis_1(magnitudes, angles, args, relatives)

    return ordered_mag, ordered_ang, args


@jit(nopython=True, cache=True, nogil=True)
def take_along_axis_1(magintudes, angles, args, relatives):

    if relatives > args.shape[1]:
        relatives = args.shape[1]
    mag_mat = np.empty((args.shape[0], relatives), dtype=magintudes.dtype)
    ang_mat = np.empty((args.shape[0], relatives), dtype=angles.dtype)
    for i in range(mag_mat.shape[0]):
        for j in range(mag_mat.shape[1]):
            arg = args[i, j]
            mag_mat[i, j] = magintudes[i, arg]
            ang_mat[i, j] = angles[i, arg]

    return mag_mat, ang_mat


def get_features_GPU(bboxes, relatives):

    center_x = (bboxes[:, 2] + bboxes[:, 0]) / 2
    center_y = (bboxes[:, 3] + bboxes[:, 1]) / 2

    center_x = center_x.unsqueeze(0)
    center_y = center_y.unsqueeze(0)
    x_diffs = center_x.t() - center_x
    y_diffs = center_y.t() - center_y
    magnitudes = torch.sqrt(x_diffs**2 + y_diffs**2)
    angles = torch.atan2(y_diffs, x_diffs) * 180 / torch.tensor(np.pi)

    args = torch.argsort(magnitudes, dim=1)

    ordered_mag = torch.gather(magnitudes, 1, args)
    ordered_ang = torch.gather(angles, 1, args)

    ordered_mag = ordered_mag[:, :relatives]
    ordered_ang = ordered_ang[:, :relatives]

    return ordered_mag, ordered_ang, args


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


@jit(nopython=True, cache=True, nogil=True)
def compute_dist_on_vec(trk_windows, dets):
    trk_windows = trk_windows.astype(np.float32)
    dets = dets.astype(np.float32)
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


def compute_dist_on_vec_GPU(trk_windows, dets):
    trk_x = (trk_windows[:, 0] + trk_windows[:, 2]) / 2
    trk_y = (trk_windows[:, 1] + trk_windows[:, 3]) / 2

    det_x = (dets[:, 0] + dets[:, 2]) / 2
    det_y = (dets[:, 1] + dets[:, 3]) / 2

    trk_x = trk_x.unsqueeze(0)
    det_x = det_x.unsqueeze(0)
    squared_diff_x = (trk_x.t() - det_x) ** 2

    trk_y = trk_y.unsqueeze(0)
    det_y = det_y.unsqueeze(0)
    squared_diff_y = (trk_y.t() - det_y) ** 2

    return torch.sqrt(squared_diff_x + squared_diff_y)



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


def get_intersection(bboxes1, bboxes2):  # matches
    inter_aera = []

    if len(bboxes1) > 0 and len(bboxes2) > 0:
        inter_aera = calc_intersection(np.array(bboxes1), np.array(bboxes2))

    return inter_aera


@jit(nopython=True)
def calc_intersection(bboxes1: np.array, bboxes2: np.array):
    x11, y11, x12, y12 = bbox_to_coordinate_vectors(bboxes1)
    x21, y21, x22, y22 = bbox_to_coordinate_vectors(bboxes2)

    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    inter_aera = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

    return inter_aera

@jit(nopython=True)
def bbox_to_coordinate_vectors(bboxes):
    x1_vec = bboxes[:, 0].copy()
    y1_vec = bboxes[:, 1].copy()
    x2_vec = bboxes[:, 2].copy()
    y2_vec = bboxes[:, 3].copy()

    x1_vec = np.expand_dims(x1_vec, axis=1)
    y1_vec = np.expand_dims(y1_vec, axis=1)
    x2_vec = np.expand_dims(x2_vec, axis=1)
    y2_vec = np.expand_dims(y2_vec, axis=1)

    return x1_vec, y1_vec, x2_vec, y2_vec


def thread_get_features(bboxes, relatives, result_queue, name='get_features'):
    thread_results = get_features(bboxes, relatives)
    result_queue.put({name: thread_results})


def thread_get_features_GPU(bboxes, relatives, result_queue, name='get_features'):
    t_bboxes = torch.tensor(bboxes).to('cuda')
    thread_results = get_features_GPU(t_bboxes, relatives)
    result_queue.put({name: thread_results})


def get_rel_fetures(trk_boxes, detections, relatives):

    result_queue = queue.Queue()

    thread1 = threading.Thread(target=thread_get_features_GPU, args=(trk_boxes, relatives, result_queue, 'trk'))
    thread2 = threading.Thread(target=thread_get_features_GPU, args=(detections, relatives, result_queue, 'det'))

    # Start the threads
    thread1.start()
    thread2.start()

    # Wait for all threads to complete
    thread1.join()
    thread2.join()

    while not result_queue.empty():
        result = result_queue.get()
        func_result = list(result.keys())
        if 'trk' in func_result:
            trk_mag, trk_ang, trk_args = result['trk']
        elif 'det' in func_result:
            det_mag, det_ang, det_args = result['det']
        else:
            print('Error')

    return trk_mag, trk_ang, trk_args, det_mag, det_ang, det_args