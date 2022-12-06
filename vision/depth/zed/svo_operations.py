import pyzed.sl as sl
import cv2
import numpy as np


def get_frame(frame_mat, cam):
    cam.retrieve_image(frame_mat, sl.VIEW.LEFT)
    frame = frame_mat.get_data()[:, :, : 3]
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def get_depth(depth_mat, cam):
    cam_run_p = cam.get_init_parameters()
    cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
    depth = depth_mat.get_data()
    depth = (cam_run_p.depth_maximum_distance - np.clip(depth, 0, cam_run_p.depth_maximum_distance)) * 255 / cam_run_p.depth_maximum_distance
    bool_mask = np.where(np.isnan(depth), True, False)
    depth[bool_mask] = 0

    depth = cv2.medianBlur(depth, 5)

    return depth


def get_point_cloud(point_cloud_mat, cam):
    cam.retrieve_measure(point_cloud_mat, sl.MEASURE.XYZRGBA)
    point_cloud = point_cloud_mat.get_data()

    return point_cloud

# TODO Matan to go through implementation
def get_dist_vec(pc_mat, bbox2d,dim):
    x1 = int(bbox2d[0])
    y1 = int(bbox2d[1])
    x2 = int(bbox2d[2])
    y2 = int(bbox2d[3])

    if dim == 'width':
        mid_h = int((y2 + y1) / 2)

        if mid_h <= 0 or (x2 - x1) <= 0:
            mat = None
        else:
            mat = pc_mat[mid_h - 1: mid_h + 2, x1:x2, :-1].copy()
        return mat

    if dim == 'height':
        mid_w = int((x2 + x1) / 2)

        if mid_w <= 0 or (y2 - y1) <= 0:
            mat = None
        else:
            mat = pc_mat[y1:y2, mid_w - 1: mid_w + 2, :-1].copy()
        return mat

def get_cropped_point_cloud(bbox, point_cloud, margin=0.2):

    crop = point_cloud[int(bbox[1]):int(bbox[3]), int(bbox[0]): int(bbox[2]), :-1].copy()
    return crop

def get_width(crop, margin=0.2, fixed_z=True):
    h, w, c = crop.shape
    marginy = np.round(margin / 2 * h).astype(np.int16)
    vec = np.nanmean(crop[marginy:-marginy, :, :], axis=0)
    if fixed_z:
        width = np.sqrt(np.sum((vec[0, :-1] - vec[-1, :-1]) ** 2)) * 1000
    else:
        width = np.sqrt(np.sum((vec[0, :] - vec[-1, :]) ** 2)) * 1000
    return width

def get_height(crop, margin=0.2, fixed_z=True):
    h, w, c = crop.shape
    marginx = np.round(margin / 2 * w).astype(np.int16)
    vec = np.nanmean(crop[:, marginx:-marginx, :], axis=1)
    if fixed_z:
        height = np.sqrt(np.sum((vec[0, :-1] - vec[-1, :-1]) ** 2)) * 1000
    else:
        height = np.sqrt(np.sum((vec[0, :] - vec[-1, :]) ** 2)) * 1000
    return height


# TODO Matan to go through implementation
def get_measures(point_cloud, dets):

    for det in dets:
        # in case that is not a full fruit
        if det[-3] == 1:
            continue
        crop = get_cropped_point_cloud(det[:4], point_cloud)
        try:
            width = get_width(crop)
            det.append(width)
        except:
            pass

        #mat = get_dist_vec(point_cloud, bbox2d[:4], 'height')
        crop = get_cropped_point_cloud(det[:4], point_cloud)
        try:
            height = get_height(crop)
            det.append(height)
        except:
            pass

    return dets

def average_det_depth(crop, margin=0.2):
    h, w, c = crop.shape
    marginx = np.round(margin / 2 * w).astype(np.int16)
    marginy = np.round(margin / 2 * h).astype(np.int16)

    return np.nanmean(crop[marginy:-marginy, marginx:-marginx, 2])


def get_det_depth(point_cloud, dets):
    for det in dets:
        crop = get_cropped_point_cloud(det[:4], point_cloud)
        avg_depth = average_det_depth(crop)
        det.append(avg_depth)

    return dets
