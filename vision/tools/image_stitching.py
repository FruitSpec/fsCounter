import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import RANSACRegressor
import kornia as K
import kornia.feature as KF
import torch
np.random.seed(123)
# cv2.setRNGSeed(123)

def keep_dets_only(frame, detections, margin=0.5):
    canvas = frame.copy()
    if len(detections) > 0:
        dets = np.array(detections)[:, :4]
    else:
        dets = []
    h, w = frame.shape[:2]
    bool_arr = np.zeros((h, w), dtype=np.bool)
    for det in dets:
        margin_h = ((det[3] - det[1]) * margin / 2)
        margin_w = ((det[2] - det[0]) * margin / 2)
        y1 = max(det[1] - margin_h, 0)
        y2 = min(det[3] + margin_h, frame.shape[0])
        x1 = max(det[0] - margin_w, 0)
        x2 = min(det[2] + margin_w, frame.shape[1])
        #bool_arr[int(det[1]): int(det[3]), int(det[0]): int(det[2])] = True
        bool_arr[int(y1): int(y2), int(x1): int(x2)] = True
    canvas[np.logical_not(bool_arr)] = 0

    return canvas


def plot_2_imgs(img1, img2, title="", save_to="", save_only=False):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    ax1.imshow(img1)
    ax2.imshow(img2)

    ax_1_range_x = np.arange(0, img1.shape[1], 25)
    ax_1_range_y = np.arange(0, img1.shape[0], 25)
    ax1.set_xticks(ax_1_range_x[::4])
    ax1.set_yticks(ax_1_range_y[::4])
    ax1.set_xticklabels(ax_1_range_x[::4])
    ax1.set_yticklabels(ax_1_range_y[::4])
    ax1.set_xticks(ax_1_range_x, minor=True)
    ax1.set_yticks(ax_1_range_y, minor=True)

    ax_2_range_x = np.arange(0, img2.shape[1], 25)
    ax_2_range_y = np.arange(0, img2.shape[0], 25)
    ax2.set_xticks(ax_2_range_x[::4])
    ax2.set_yticks(ax_2_range_y[::4])
    ax2.set_xticklabels(ax_2_range_x[::4])
    ax2.set_yticklabels(ax_2_range_y[::4])
    ax2.set_xticks(ax_2_range_x, minor=True)
    ax2.set_yticks(ax_2_range_y, minor=True)

    plt.title(title)
    if save_to != "":
        plt.savefig(save_to)
    if save_only:
        plt.close()
        return
    plt.show()


def load_color_img(file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def get_ECCtranslation(img1, img2, number_of_iterations=10000, termination_eps=1e-10):

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # find transform
    score, M = cv2.findTransformECC(img1, img2, warp_matrix, motionType=warp_mode, criteria=criteria)

    return M, score, warp_matrix


def get_frames_overlap(frames_folder=None, file_list=None, resize_=640, method='hm', max_workers=8):
    """
    method can be: 1. 'at' for affine transform
                   2. 'hm' for homography
    """
    if isinstance(file_list, type(None)):
        file_list = get_fsi_files(frames_folder)
    # file_list = [os.path.join(r"C:\Users\Nir\Downloads",image) for image in
    #              ["IMG_20220906_161320.jpg","IMG_20220906_161321.jpg",
    #               "IMG_20220906_161322.jpg","IMG_20220906_161323.jpg" ]]
    if method == "match":
        resize_list = [resize_ for _ in file_list]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(find_match_translation, file_list[:-1], file_list[1:], resize_list[:-1]))
        results = np.array(results).astype(int)
        txs, tys = results[:, 0], results[:, 1]
        widths = [file.shape[1] for file in file_list]
        heights = [file.shape[0] for file in file_list]
        masks = list(map(get_masks_from_tx_ty, txs, tys, heights, widths))
        return masks

    kp, des, heights, widths, rs = extract_keypoints(file_list, resize_, max_workers)

    M, _ = extract_matrix(kp, des, method, max_workers)

    if method == 'at':
        masks = list(map(translation_based, M, heights, widths, rs))
    elif method == 'hm':
        image_list = [load_color_img(img) for img in file_list]
        masks = list(map(find_overlapping, image_list[:-1], image_list[1:], M))
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

    tx, ty, _, _, _ = find_translation(kp1, des1, kp2, des2, 1)

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
    good, matches, matchesMask = match_descriptors(des1, des2)
    M, status = calc_affine_transform(kp1, kp2, good)

    return M, status, good, matches, matchesMask


def find_keypoints(img, matcher=None):
    if matcher is None:
        matcher = cv2.SIFT_create()
    kp, des = matcher.detectAndCompute(img, None)

    return kp, des


def get_fine_affine_translation(im_zed_stage2, im_jai):
    kp_des_zed, kp_des_jai = get_fine_keypoints(im_zed_stage2), get_fine_keypoints(im_jai)
    translation_res = get_fine_translation(kp_des_zed, kp_des_jai, max_workers=5)
    translation_res = np.array([list(affine_to_values(M)) for tx, ty, M in translation_res])
    tx, ty, sx, sy = np.nanmean(translation_res, axis=0)
    return tx, ty, sx, sy



def match_descriptors(des1, des2, min_matches=10, threshold=0.75):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except:
        Warning('flann.knnMatch collapsed, return empty matches')
        matches = []
    # store all the good matches as per Lowe's ratio test.
    match = []
    matchesMask = [[0, 0] for i in range(len(matches))]
    id = 0
    for m, n in matches:
        if m.distance < threshold * n.distance:
            match.append(m)
            matchesMask[id] = [1, 0]
        id += 1
    return match, matches, matchesMask


def calc_affine_transform(kp1, kp2, match, ransac=20):
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in match]).reshape(-1, 1, 2)

    if dst_pts.__len__() > 0  and src_pts.__len__() > 0:  # not empty - there was a match
        M, status = cv2.estimateAffine2D(src_pts, dst_pts, ransacReprojThreshold=ransac)
    else:
        M, status = None, [0]


    return M, status


def affine_to_values(M):
    if isinstance(M, type(None)):
        return np.nan, np.nan, np.nan, np.nan
    if np.isnan(M[0, 2]) or np.isnan(M[1, 2]):
        return np.nan, np.nan, np.nan, np.nan
    sx = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
    sy = np.sqrt(M[1, 0] ** 2 + M[1, 1] ** 2)
    tx = np.round(M[0, 2]).astype(np.int)
    ty = np.round(M[1, 2]).astype(np.int)
    return tx, ty, sx, sy

def get_affine_matrix(kp_zed, kp_jai, des_zed, des_jai, ransac=20, fixed_scaling=False):
    match, _, _ = match_descriptors(des_zed, des_jai) # more than 90$ of time
    M, st = calc_affine_transform(kp_zed, kp_jai, match, ransac)

    return M, st, match


def calc_homography(kp1, kp2, match):
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in match]).reshape(-1, 1, 2)

    M, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M, status


def get_affine_homography(kp1, kp2, des1, des2):
    match = match_descriptors(des1, des2)
    M, st = calc_homography(kp1, kp2, match)

    return M, st


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
    if isinstance(file_path, str):
        img = cv2.imread(file_path)
    else:
        img = file_path
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

    return get_masks_from_tx_ty(tx, ty, height, width)


def get_masks_from_tx_ty(tx, ty, height, width):
    mask = np.zeros((height, width))
    if np.abs(tx) > width or np.abs(ty) > height:
        print("tx or ty is too big")
        return None
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
    M, status, good, matches, matchesMask = features_to_translation(kp1, kp2, des1, des2)
    if isinstance(M, type(None)):
        return np.nan, np.nan, M
    if not np.isfinite(M[0, 2]) or not np.isfinite(M[1, 2]) :
        return np.nan, np.nan, M
    tx = int(np.round(M[0, 2] / r))
    ty = int(np.round(M[1, 2] / r))

    return tx, ty, good, matches, matchesMask


def resize_img(input_, size):
    r = min(size / input_.shape[0], size / input_.shape[1])
    resized_img = cv2.resize(
        input_,
        (int(input_.shape[1] * r), int(input_.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)

    return resized_img, r


def find_match_translation(last_frame, frame, size):
    frame, r = resize_img(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY), size)
    last_frame, r = resize_img(cv2.cvtColor(last_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY), size)
    try:
        # Apply template Matching
        res = cv2.matchTemplate(last_frame, frame[50:-50, 30:-30], cv2.TM_CCOEFF_NORMED)
        x_vec = np.mean(res, axis=0)
        y_vec = np.mean(res, axis=1)
        tx = (np.argmax(x_vec) - (res.shape[1] // 2 + 1)) / r
        ty = (np.argmax(y_vec) - (res.shape[0] // 2 + 1)) / r

    except:
        Warning('failed to match')
        tx = None
        ty = None

    return tx, ty


def find_loftr_translation(im_left, im_right, ret_keypoints=False, debug_path=""):
    im_right = K.image_to_tensor(im_right, True).float() / 255.
    im_left = K.image_to_tensor(im_left, True).float() / 255.

    input_dict = {"image0":  torch.unsqueeze(im_right, dim=0),  # LofTR works on grayscale images only
                  "image1": torch.unsqueeze(im_left, dim=0)}
    with torch.inference_mode():
        correspondences = KF.LoFTR(pretrained="outdoor")(input_dict)

    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    M, status = cv2.estimateAffine2D(mkpts0, mkpts1, ransacReprojThreshold=20, maxIters=5000)
    inliers = status > 0
    if debug_path != "":
       draw_loftr(im_right,im_left, mkpts0, mkpts1, inliers, debug_path)
    if ret_keypoints:
        return M, status, mkpts0, mkpts1
    return M, status

def draw_loftr(im_right, im_left, mkpts0, mkpts1, inliers, debug_path):

    draw_LAF_matches(
        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1, -1, 2),
                                     torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                                     torch.ones(mkpts0.shape[0]).view(1, -1, 1)),

        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1, -1, 2),
                                     torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                                     torch.ones(mkpts1.shape[0]).view(1, -1, 1)),
        torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
        K.tensor_to_image(im_left),
        K.tensor_to_image(im_right),
        inliers,
        debug_path = debug_path,
        draw_dict={'inlier_color': (0.2, 1, 0.2),
                   'tentative_color': None,
                   'feature_color': (0.2, 0.5, 1), 'vertical': False})


if __name__ == "__main__":
    #fp = r'C:\Users\Matan\Documents\Projects\Data\Slicer\wetransfer_ra_3_a_10-zip_2022-08-09_0816\15_20_A_16\15_20_A_16'
    fp = r'C:\Users\Matan\Documents\Projects\Data\Slicer\from Roi\RA_3_A_2\RA_3_A_2'
    res = get_frames_overlap(fp, max_workers=1)

    # if res is not None:
    #     plt.imshow(res)
    #     plt.show()