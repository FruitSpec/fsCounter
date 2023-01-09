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


def get_dist_vec(pc_mat, bbox2d, dim):
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
    # TODO

    crop = point_cloud[max(int(bbox[1]), 0):int(bbox[3]), max(int(bbox[0]), 0): int(bbox[2]), :-1].copy()
    return crop


def get_distance(crop):
    h, w = crop.shape
    filter_ = 0.2
    if h < 4 or w < 4 or h == 0 or w == 0:
        return np.inf
    crop = crop[int(h * filter_): h - int(h * filter_), int(w * filter_):w - int(w * filter_)]
    return np.nanmedian(crop)


def get_width(crop, margin=0.2, fixed_z=True, max_z=1):
    h, w, c = crop.shape
    marginy = np.round(margin / 2 * h).astype(np.int16)
    crop_marg = crop[marginy:-marginy, :, :]
    crop_marg[crop_marg[:, :, 2] > max_z] = np.nan
    vec = np.nanmean(crop_marg, axis=0)
    vec = vec[np.isfinite(vec[:, 2])]
    if len(vec) < 2:
        return np.nan
    if fixed_z:
        width = np.sqrt(np.sum((vec[0, :-1] - vec[-1, :-1]) ** 2)) * 1000
    else:
        width = np.sqrt(np.sum((vec[0, :] - vec[-1, :]) ** 2)) * 1000
    return width


def get_height(crop, margin=0.2, fixed_z=True, max_z=1):
    h, w, c = crop.shape
    marginx = np.round(margin / 2 * w).astype(np.int16)
    crop_marg = crop[:, marginx:-marginx, :]
    crop_marg[crop_marg[:, :, 2] > max_z] = np.nan
    vec = np.nanmean(crop_marg, axis=1)
    vec = vec[np.isfinite(vec[:, 2])]
    if len(vec) < 2:
        return np.nan
    if fixed_z:
        height = np.sqrt(np.sum((vec[0, :-1] - vec[-1, :-1]) ** 2)) * 1000
    else:
        height = np.sqrt(np.sum((vec[0, :] - vec[-1, :]) ** 2)) * 1000
    return height


def get_dimensions(point_cloud, dets, dist_max):
    dims = []
    for det in dets:
        # in case that is not a full fruit
        if det[-3] == 1:
            continue
        crop = get_cropped_point_cloud(det[:4], point_cloud)
        width = get_width(crop, fixed_z=False, max_z=dist_max)
        height = get_height(crop, fixed_z=True, max_z=dist_max)
        distance = get_distance(crop[:, :, 2])

        dims.append([height, width, distance])

    return dims


def sl_get_dimensions(dets, wrapper):
    import matplotlib.pyplot as plt
    dims = []
    objects_in = []
    for det in dets:
        x1, x2, y1, y2 = max(int(det[1]), 0), int(det[3]), 1080 - min(int(det[2]), 1080), 1080 - max(int(det[0]), 0)
        mat = sl.Mat()
        wrapper.cam.retrieve_image(mat, sl.VIEW.LEFT)

        # img = mat.get_data()
        # img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 5)
        # plt.imshow(img)
        # plt.show()

        # try:
        bounding_box_2d = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        tmp = sl.CustomBoxObjectData()
        tmp.unique_object_id = sl.generate_unique_id()
        tmp.label = -1
        tmp.probability = det[4]
        tmp.bounding_box_2d = bounding_box_2d
        tmp.is_grounded = False
        objects_in.append(tmp)
        # except OverflowError:
        #     print("***************************************", det)
        #     print(x1, x2, y1, y2)

    wrapper.cam.ingest_custom_box_objects(objects_in)
    objects_out = sl.Objects()
    wrapper.cam.retrieve_objects(objects_out)
    for obj in objects_out.object_list:
        dims.append([obj.dimensions[1] * 1000, obj.dimensions[0] * 1000, obj.position[2]])

    return dims


def average_det_depth(crop, margin=0.2):
    h, w, c = crop.shape
    marginx = np.round(margin / 2 * w).astype(np.int16)
    marginy = np.round(margin / 2 * h).astype(np.int16)

    return np.nanmean(crop[marginy:-marginy, marginx:-marginx, 2])


def get_dets_ranges(point_cloud, dets):
    ranges = []
    for det in dets:
        crop = get_cropped_point_cloud(det[:4], point_cloud)
        ranges.append(average_det_depth(crop))

    return ranges
