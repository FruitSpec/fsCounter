import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from concurrent.futures import ThreadPoolExecutor

def kepp_dets_only(frame, detections):
    canvas = frame.copy()
    if len(detections) > 0:
        dets = np.array(detections)[:, :4]
    else:
        dets = []
    h, w = frame.shape[:2]
    bool_arr = np.zeros((h, w), dtype=np.bool)
    for det in dets:
        bool_arr[int(det[1]): int(det[3]), int(det[0]): int(det[2])] = True
    canvas[np.logical_not(bool_arr)] = 0

    return canvas

def get_ECCtranslation(img1, img2, number_of_iterations=10000, termination_eps=1e-10):

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # find transform
    score, M = cv2.findTransformECC(img1, img2, warp_matrix, motionType=warp_mode, criteria=criteria)

    return M, score, warp_matrix

def get_frames_overlap(frames_folder, resize_=640, method='hm', max_workers=8):
    """
    method can be: 1. 'at' for affine transform
                   2. 'hm' for homography
    """
    file_list = get_fsi_files(frames_folder)

    kp, des, heights, widths, rs = extract_keypoints(file_list, resize_, max_workers)

    M, _ = extract_matrix(kp, des, method, max_workers)

    if method == 'at':
        masks = list(map(translation_based, M, heights, widths, rs))
    elif method == 'hm':
        masks = list(map(find_overlapping, M, heights, widths))
    return masks


def load_and_extract_kp(file_path, resize_=640):
    img, r, h, w = load_img(file_path, resize_)
    kp, des = find_keypoints(img)

    res = [kp, des, r, h, w]
    return res

def height(img):

    return img.shape[0]

def width(img):
    return img.shape[1]

def extract_keypoints(file_list, resize_=640, max_workers=8):

    resize_list = [resize_ for _ in file_list]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(load_and_extract_kp, file_list, resize_list))

    return results_to_lists(results)

def get_fine_keypoints(img, max_workers=5):
    windows = get_windows(img)

    cr = []
    for w in windows:
        cr.append(img[w[1]: w[3], w[0]: w[2]].copy())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(find_keypoints, cr))

    return results

def get_fine_translation(key_des1, key_des2, max_workers=5):

    r = [1 for _ in key_des1]

    kp1 = [kp_des[0] for kp_des in key_des1]
    des1 = [kp_des[1] for kp_des in key_des1]
    kp2 = [kp_des[0] for kp_des in key_des2]
    des2 = [kp_des[1] for kp_des in key_des2]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(find_translation, kp1, des1, kp2, des2, r))

    x_values = list(filter(None, [r[0] for r in results]))
    y_values = list(filter(None, [r[1] for r in results]))
    if x_values:
        tx = np.median(x_values)
    else:
        tx = None
    if y_values:
        ty = np.median(y_values)
    else:
        ty = None

    return tx, ty
def  get_translation(img1, img2):

    kp1, des1 = find_keypoints(img1)
    kp2, des2 = find_keypoints(img2)

    tx, ty = find_translation(kp1, des1, kp2, des2, 1)

    return tx, ty
def get_windows(img1):
    h, w = img1.shape[:2]
    set1 = [int(w * 0.2), int(h * 0.2), int(w * 0.5), int(h * 0.4)]
    set2 = [int(w * 0.2), int(h * 0.6), int(w * 0.5), int(h * 0.8)]
    set3 = [int(w * 0.5), int(h * 0.2), int(w * 0.8), int(h * 0.4)]
    set4 = [int(w * 0.5), int(h * 0.6), int(w * 0.8), int(h * 0.8)]
    set5 = [int(w * 0.35), int(h * 0.35), int(w * 0.7), int(h * 0.7)]

    return [set1, set2, set3, set4, set5]

def results_to_lists(results):
    kp = []
    des = []
    rs = []
    heights = []
    widths = []
    for result in results:
        kp.append(result[0])
        des.append(result[1])
        rs.append(result[2])
        heights.append(result[3])
        widths.append(result[4])

    return kp, des, heights, widths, rs


def extract_matrix(kp, des, method='hm', max_workers=8):

    des1 = [d for d in des[:-1]]
    des2 = [d for d in des[1:]]

    kp1 = [k for k in kp[:-1]]
    kp2 = [k for k in kp[1:]]

    if method == 'hm':
        func_ = features_to_homography
    elif method == 'at':
        func_ = features_to_translation

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func_, kp1, kp2, des1, des2))

    M = []
    status = []
    for result in results:
        M.append(result[0])
        status.append(result[1])

    return M, status



def features_to_homography(kp1, kp2, des1, des2):
    good = match_descriptors(des1, des2)
    M, status = calc_homography(kp1, kp2, good)

    return M, status


def features_to_translation(kp1, kp2, des1, des2):
    good = match_descriptors(des1, des2)
    M, status = calc_affine_transform(kp1, kp2, good)

    return M, status

def find_keypoints(img):
    sift = cv2.SIFT_create()
    #sift = cv2.ORB()
    # find key points
    kp, des = sift.detectAndCompute(img, None)

    return kp, des


def match_descriptors(des1, des2, min_matches=10, threshold=0.7):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except:
        Warning('flann.knnMatch collapsed, return empty matches')
        matches = []
    # store all the good matches as per Lowe's ratio test.
    match = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            match.append(m)

    return match


def calc_affine_transform(kp1, kp2, match):
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in match]).reshape(-1, 1, 2)

    if dst_pts.__len__() > 0  and src_pts.__len__() > 0:  # not empty - there was a match
        M, status = cv2.estimateAffine2D(src_pts, dst_pts)
    else:
        M, status = None, [0]


    return M, status


def calc_homography(kp1, kp2, match):
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in match]).reshape(-1, 1, 2)

    M, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M, status


def trim(frame):
    # crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    # crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    # crop top
    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    # crop top
    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame


def get_fsi_files(folder_path):
    fsi_list = []
    fsi_frame = []
    tot_file_list = os.listdir(folder_path)
    for file in tot_file_list:
        if 'FSI' in file and ('jpg' in file or 'png' in file):
            fsi_list.append(file)
            fsi_frame.append(extract_frame_id(file))

    fsi_frame.sort()
    final_list = []
    for f_id in fsi_frame:
        for file in fsi_list:
            if str(f_id) in file:
                final_list.append(os.path.join(folder_path, file))
                break

    return final_list


def extract_frame_id(file_name):
    temp = file_name.split('.')[0]
    f_id = int(temp.split('_')[-1])
    return f_id


def load_img(file_path, resize_=640):
    r = None

    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h = height(img)
    w = width(img)

    if resize_ is not None:
        img, r = resize_img(img, resize_)

    return img, r, h, w


def stitch_folder(folder_path, resize=256, save=False):
    file_list = get_fsi_files(folder_path)

    stitcher = cv2.Stitcher_create()

    images = []
    for f in file_list:
        t_img, _ = load_img(os.path.join(fp, f))
        r = min(resize / t_img.shape[0], resize
                / t_img.shape[1])
        resized_img = cv2.resize(
            t_img,
            (int(t_img.shape[1] * r), int(t_img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        #resized_img = cv2.bilateralFilter(resized_img, 40, 15, 15)
        images.append(resized_img)

    s, r = stitcher.stitch(images)

    if s == 0:
        if save:
            r_save = cv2.cvtColor(r, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(folder_path, 'stitched.jpg'), r_save)

        return r
    else:
        print(f'Stitching operation failed; got status {s}')


def get_overlapping(folder_path):
    file_list = get_fsi_files(folder_path)

    sift = cv2.SIFT_create()
    res = []

    im_1, im_1_g = load_img(os.path.join(fp, file_list[0]))
    kp1, des1 = find_keypoints(sift, im_1_g)

    for f in file_list:
        im_2, im_2_g = load_img(os.path.join(fp, f))

        kp2, des2 = find_keypoints(sift, im_2_g)
        match = match_descriptors(des1, des2)
        M, mask = calc_homography(kp1, kp2, match)
        ovl = find_overlapping(im_1, im_2, M)

        res.append(remove_artifacts(ovl))

        im_1 = im_2.copy()
        kp1 = kp2
        des1 = des2

    return res

def find_overlapping(M, h, w):
    dummy = np.ones((h, w))

    dst = warp(dummy, dummy, M)

    dstMask = np.zeros((dst.shape[0], dst.shape[1]))
    dstMask[:h, :w] = 1

    boolMask = dstMask == 0
    dstMask[np.logical_not(boolMask)] = 1

    src = warp_back(dstMask, M, cols=w, rows=h)
    src = np.round(src).astype(np.uint8)

    return src

def remove_artifacts(src):

    labeled = measure.label(src, background=0)
    labeles_value = list(np.unique(labeled))
    area_dict = dict()
    for label_ in labeles_value:
        area_dict[label_] = np.sum(labeled == label_)
    area_dict = {k: v for k, v in sorted(area_dict.items(), key=lambda item: item[1])}

    keys_ = list(area_dict.keys())
    # biggest area is last
    for label_ in keys_[:-1]:
        boolMask = labeled == label_
        src[boolMask] = 0

    return src



def warp(im1, im2, M):
    cols = im2.shape[1] + im1.shape[1]
    rows = im1.shape[0]
    img_output = cv2.warpPerspective(im2, M, (cols, rows))
    output = trim(img_output)

    return output


def warp_back(im, M, cols, rows):
    inv_M = np.linalg.inv(M)
    img_output = cv2.warpPerspective(im, inv_M, (cols, rows))
    output = trim(img_output)

    return output


def translation_based(M, height, width, r):
    if r is None:
        r = 1

    tx = int(np.round(M[0, 2] / r))
    ty = int(np.round(M[1, 2] / r))

    mask = np.zeros((height, width))
    if tx < 0:
        if ty < 0:
            mask[-ty:, -tx:] = 1
        else:
            mask[:-ty, -tx:] = 1
    else:
        if ty < 0:
            mask[-ty:, :-tx] = 1
        else:
            mask[:-ty, :-tx] = 1

    return mask


def find_translation(kp1, des1, kp2, des2, r):
    M, status = features_to_translation(kp1, kp2, des1, des2)
    if 1 in np.unique(status) and True not in np.unique(np.isnan(M)) and True not in np.unique(np.isinf(M)):
        try:
            tx = int(np.round(M[0, 2] / r))
            ty = int(np.round(M[1, 2] / r))
        except:
            a=1
    else:
        tx, ty = None, None

    return tx, ty


def resize_img(input_, size):
    r = min(size / input_.shape[0], size / input_.shape[1])
    resized_img = cv2.resize(
        input_,
        (int(input_.shape[1] * r), int(input_.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)

    return resized_img, r

if __name__ == "__main__":
    #fp = r'C:\Users\Matan\Documents\Projects\Data\Slicer\wetransfer_ra_3_a_10-zip_2022-08-09_0816\15_20_A_16\15_20_A_16'
    fp = r'C:\Users\Matan\Documents\Projects\Data\Slicer\from Roi\RA_3_A_2\RA_3_A_2'
    res = get_frames_overlap(fp, max_workers=1)

    # if res is not None:
    #     plt.imshow(res)
    #     plt.show()