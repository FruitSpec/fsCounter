from MHS.F_model_training import *
from vision.feature_extractor.vegetation_indexes import num_deno_nan_divide, num_deno_nan_divide_np
import numpy as np
import matplotlib.pyplot as plt
from vision.tools.image_stitching import plot_2_imgs
from vision.feature_extractor.stat_tools import get_mode


def preprocess(frame, xyz_frame, resize_factor, show_imgs, vndvi_filter, vari_filter, channels_thresh):
    org_frame = frame
    xyz_frame = xyz_frame[:,:,:3]
    if resize_factor > 1:
        new_shape = (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor)
        frame = cv2.resize(frame, new_shape)
        xyz_frame = cv2.resize(xyz_frame, new_shape)
    else:
        frame, xyz_frame = np.copy(frame), np.copy(xyz_frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flatten_frame = gray_frame.flatten()
    gray_mode = get_mode(flatten_frame, kde_mode=False, bins=254)
    if show_imgs:
        plt.hist(flatten_frame, density=True, bins=50)
        plt.title(gray_mode)
        plt.show()
    if gray_mode < 25:  # Dark image
        vndvi_filter = vndvi_filter - 0.01
        if np.sum(vari_filter) > 0:
            vari_filter = (0.5, 100)
        channels_thresh[2] = (0, 255)
        channels_thresh[1] = (0, 255)
    return org_frame, frame, xyz_frame, vndvi_filter, vari_filter, channels_thresh


def apply_depth_filter(frame, xyz_frame, depth_filter, show_imgs):
    if np.sum(depth_filter) > 0:
        if show_imgs:
            frame_pre = frame.copy()
        depth_chanel = xyz_frame[:, :, 2]
        frame[depth_chanel < depth_filter[0]] = 0
        frame[depth_chanel > depth_filter[1]] = 0
        if show_imgs:
            plot_2_imgs(frame_pre, frame, title="pre depth filter,    post filter")
    return frame


def channel_filter(frame, channels_thresh, show_imgs):
    if show_imgs:
        frame_pre = frame.copy()
    mask = np.ones_like(frame[:, :, 0], dtype=bool)
    for channel, thresholds in channels_thresh.items():
        if not np.sum(thresholds):
            continue
        chan = frame[:, :, channel]
        channel_mask = cv2.inRange(chan, thresholds[0], thresholds[1])
        mask = np.bitwise_and(mask, channel_mask)
    frame[np.where(mask == 0)] = 0
    if show_imgs:
        plot_2_imgs(frame_pre, frame, title="prefilter channel, post filter")
    return frame


def gmr_filter(gree_minus_red_thres, frame, show_imgs):
    if np.sum(gree_minus_red_thres) != 0:
        diff = frame[:, :, 1] - frame[:, :, 0]
        valids = cv2.inRange(diff, gree_minus_red_thres[0], gree_minus_red_thres[1])
        if show_imgs:
            frame_pre = frame.copy()
        frame[np.where(valids == 0)] = 0
        if show_imgs:
            plot_2_imgs(frame_pre, frame, title="prefilter g minus r,    post filter")
    return frame


def gmb_filter(gree_minus_blue_thres, frame, show_imgs):
    if np.sum(gree_minus_blue_thres) != 0:
        diff = frame[:, :, 1] - frame[:, :, 2]
        valids = cv2.inRange(diff, gree_minus_blue_thres[0], gree_minus_blue_thres[1])
        if show_imgs:
            frame_pre = frame.copy()
        frame[np.where(valids == 0)] = 0
        if show_imgs:
            plot_2_imgs(frame_pre, frame, title="prefilter g minus b,    post filter")
    return frame


def apply_vari_filter(frame, Red, Green, Blue, vari_filter, show_imgs):
    vari = num_deno_nan_divide_np((Green - Red), (Green + Red - Blue))
    vari[~np.isfinite(vari)] = 0
    vari = np.clip(vari, -5, 15)
    if show_imgs:
        frame_pre = frame.copy()
    frame_copy = frame.copy()
    frame_copy[vari >= vari_filter[1]] = 0
    frame_copy[vari <= vari_filter[0]] = 0
    frame[-int(frame.shape[0] / 4):, :] = frame_copy[-int(frame.shape[0] / 4):, :]
    if show_imgs:
        plot_2_imgs(frame_pre, frame, title="prefilter vari,    post filter")
    return frame


def apply_vndvi_filter(frame, Red, Green, Blue, vndvi_filter, show_imgs):
    vndvi = 0.5268 * (Red ** (-0.1294) * Green ** (0.3389) * Blue ** (-0.3118))
    vndvi[~np.isfinite(vndvi)] = np.nanmin(vndvi)
    if show_imgs:
        frame_pre = frame.copy()
    frame[vndvi <= vndvi_filter] = np.nanmin(vndvi)
    if show_imgs:
        plot_2_imgs(frame_pre, frame, title="prefilter vndvi,    post filter")
    return frame


def show_vi_filter_res(org_frame, vari_filter, vndvi_filter):
    Red, Green, Blue = org_frame[:, :, 0], org_frame[:, :, 1], org_frame[:, :, 2]
    vari = num_deno_nan_divide_np((Green - Red), (Green + Red - Blue))
    vari[~np.isfinite(vari)] = 0
    vari = np.clip(vari, -5, 10)
    vndvi = 0.5268 * (Red ** (-0.1294) * Green ** (0.3389) * Blue ** (-0.3118))
    vndvi[~np.isfinite(vndvi)] = np.nanmin(vndvi)
    print(np.nanquantile(vndvi, 0.025), np.nanquantile(vndvi, 0.975))
    plt.hist(vndvi.flatten(), bins=50)
    plt.show()
    plot_2_imgs(vari, np.clip(vndvi, np.nanquantile(vndvi, 0.025), np.nanquantile(vndvi, 0.975)), title="vari,    ndvi")
    plot_2_imgs(cv2.inRange(vari, vari_filter[0], vari_filter[1]), vndvi > vndvi_filter,
                title="vari filtered,    ndvi filtred")


def debug_func(frame, Red, Green, Blue):
    vndvi = 0.5268 * (Red ** (-0.1294) * Green ** (0.3389) * Blue ** (-0.3118))
    vari = num_deno_nan_divide_np((Green - Red), (Green + Red - Blue))
    vari[~np.isfinite(vari)] = np.nanmin(vari)
    vari = np.clip(vari, -5, 10)
    plt.hist(vari.flatten(), bins=50)
    plt.show()
    vndvi[~np.isfinite(vndvi)] = np.nanmin(vndvi)
    plot_2_imgs(vari, vndvi, title="Vari, vNDVI")
    plot_2_imgs(vari < 0, vari < 0.5, title="Vari, vNDVI")
    plot_2_imgs(vari < 1, vari < 1.5, title="Vari, vNDVI")
    plot_2_imgs(vari < 2, vari < 2.5, title="Vari, vNDVI")
    plot_2_imgs(vari < 3, vari < 3.5, title="Vari, vNDVI")
    plot_2_imgs(vari < 4, vari < 4.5, title="Vari, vNDVI")
    plot_2_imgs(vari <= 5, vari < 5.5, title="Vari, vNDVI")


def calculate_gradients(xyz_frame, mid_height, include_y_grad):
    depth_channel = xyz_frame[mid_height:, :, 2]
    depth_grad = np.gradient(depth_channel)[0]

    y_grad = None
    if include_y_grad:
        y_channel = xyz_frame[mid_height:, :, 1]
        y_grad = np.gradient(y_channel)[0]

    return depth_grad, y_grad


def apply_thresholding(depth_grad, y_grad, min_grad=0.01):
    depth_grad_mask = depth_grad > min_grad

    y_grad_mask = None
    if y_grad is not None:
        y_grad_mask = y_grad > min_grad

    return depth_grad_mask, y_grad_mask


def combine_masks(depth_grad_mask, y_grad_mask):
    if y_grad_mask is not None:
        return depth_grad_mask.astype(np.uint8) | y_grad_mask.astype(np.uint8)
    else:
        return depth_grad_mask.astype(np.uint8)


def dilate_image(comb_img, img_h):
    comb_img = cv2.erode(comb_img, np.ones((img_h//200+1, img_h//200 +1)), iterations=1)
    return cv2.dilate(comb_img, np.ones((img_h//40+1, img_h//40 +1)), iterations=1)


def find_filtered_contours(comb_img_dilated, min_contour_area):
    contours, _ = cv2.findContours(comb_img_dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, [contour for contour in contours if cv2.contourArea(contour) >= min_contour_area * 10]


def create_output_image(xyz_frame, contours_filtered_drawed, mid_height):
    height, width = contours_filtered_drawed.shape
    output_img = np.zeros_like(xyz_frame)

    for x in range(width):
        column = contours_filtered_drawed[:, x]
        min_y = len(column) - np.argmax(column[::-1]) - 1 + mid_height
        output_img[min_y:, x] = 1

    return output_img


def segment_floor(xyz_frame, show_imgs=True, min_contour_area=100, include_y_grad=False):
    img_h = xyz_frame.shape[0]
    mid_height = img_h // 2

    depth_grad, y_grad = calculate_gradients(xyz_frame, mid_height, include_y_grad)
    depth_grad_mask, y_grad_mask = apply_thresholding(depth_grad, y_grad)

    if show_imgs and y_grad_mask is not None:
        plot_2_imgs(depth_grad_mask, y_grad_mask, title="masked grads (depth, y)")

    comb_img = combine_masks(depth_grad_mask, y_grad_mask)
    comb_img_dilated = dilate_image(comb_img, img_h)

    if show_imgs:
        plot_2_imgs(comb_img, comb_img_dilated, title="org gradient image, dilated image")

    contours, filtered_contours = find_filtered_contours(comb_img_dilated, min_contour_area)
    contours_filtered_drawed = cv2.drawContours(comb_img_dilated.copy(), filtered_contours, -1, (255, 0, 0), 1)

    if show_imgs:
        contours_drawed = cv2.drawContours(comb_img_dilated.copy(), contours, -1, (255, 0, 0), 1)
        plot_2_imgs(contours_drawed, contours_filtered_drawed, title="contours, filtered contours")

    output_img = create_output_image(xyz_frame, contours_filtered_drawed, mid_height)

    if show_imgs:
        plot_2_imgs(output_img, contours_filtered_drawed, title="final image, filtered contours")

    return output_img


def remove_floor(frame, xyz_frame, show_imgs=False):
    if show_imgs:
        frame_pre = frame.copy()
    frame[segment_floor(xyz_frame[:,:,:3], show_imgs) == 1] = 0
    if show_imgs:
        plot_2_imgs(frame_pre, frame, title="prefilter floor,    post floor filter")
    return frame


def thresh_channels(frame, xyz_frame, channels_thresh={0: (0, 175), 1: (10, 255), 2: (0, 200)},
                    gree_minus_red_thres=(0, 0), gree_minus_blue_thres=(0, 0), vari_filter=(0, 0), vndvi_filter=0.35,
                    depth_filter=(0.75, 5), show_imgs=False, resize_factor=4, full_tree_thresh=0.33):
    """
    Thresholds the input frame based on the given channel thresholds and green minus red thresholds.

    Args:
        frame (ndarray): The input frame to threshold.
        channels_thresh (dict): A dictionary containing the channel indices as keys and a tuple of thresholds as values.
        gree_minus_red_thres (tuple): A tuple containing the green minus red thresholds.

    Returns:
        ndarray: The thresholded frame.

    """
    org_frame, frame, xyz_frame, vndvi_filter, vari_filter, channels_thresh = preprocess(frame, xyz_frame,
                                                                                         resize_factor, show_imgs,
                                                                                         vndvi_filter, vari_filter,
                                                                                         channels_thresh)
    frame = remove_floor(frame, xyz_frame)  # , show_imgs)
    frame = apply_depth_filter(frame, xyz_frame, depth_filter, show_imgs)
    if np.sum(vari_filter) + vndvi_filter > 0:
        Red, Green, Blue = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    #     if show_imgs:
    #         debug_func(frame, Red, Green, Blue)

    frame = channel_filter(frame, channels_thresh, show_imgs)
    frame = gmr_filter(gree_minus_red_thres, frame, show_imgs)
    frame = gmb_filter(gree_minus_blue_thres, frame, show_imgs)
    if np.sum(vari_filter) > 0:
        frame = apply_vari_filter(frame, Red, Green, Blue, vari_filter, show_imgs)
    if vndvi_filter > 0:
        frame = apply_vndvi_filter(frame, Red, Green, Blue, vndvi_filter, show_imgs)
    if show_imgs:
        show_vi_filter_res(org_frame, vari_filter, vndvi_filter)
    threshed_img = cv2.threshold(frame[:, :, 1], 0, 255, cv2.THRESH_BINARY)[1] > 0
    full_tree = np.mean(threshed_img[:threshed_img.shape[0] // 15, :]) < full_tree_thresh
    return threshed_img, full_tree


def get_percent_h_seen(zed_threshed_cut, y1, y2):
    return round(np.mean(np.nansum(zed_threshed_cut[y1:y2, :], axis=0) / np.nansum(zed_threshed_cut, axis=0)), 2)


def get_percent_seen(zed_frame, xyz_frame, coors, return_threshed=False, show_imgs=False, resize_factor=4):
    x1, y1, x2, y2 = [int(cor / resize_factor) for cor in coors]
    zed_threshed, full_tree = thresh_channels(zed_frame, xyz_frame,
                                              show_imgs=show_imgs, resize_factor=resize_factor)
    zed_threshed_cut = zed_threshed[:, x1:x2]
    percent_seen = round(np.nansum(zed_threshed_cut[y1:y2, :]) / np.nansum(zed_threshed_cut), 2)
    percent_h_seen = get_percent_h_seen(zed_threshed_cut, y1, y2)
    percent_seen_top = round(np.nansum(zed_threshed_cut[y1:y2, :]) / np.nansum(zed_threshed_cut[:y2, :]), 2)
    no_tree_indicator = np.mean(zed_threshed) < 0.2
    if return_threshed:
        return percent_seen, percent_h_seen, percent_seen_top, no_tree_indicator, full_tree, zed_threshed_cut
    return percent_seen, percent_h_seen, percent_seen_top, no_tree_indicator, full_tree


def debug_percent_seen_batch(batch_fsi, batch_rgb_zed, batch_zed, frame_ids, b_align):
    for jai, zed, zed_xyz, f_id, cors in zip(batch_fsi, batch_rgb_zed, batch_zed, frame_ids, b_align):
        cors = [int(cor) for cor in cors[:4]]
        x1, y1, x2, y2 = cors
        zed_copy = zed.copy()
        color = (0, 0, 255)
        thickness = 2
        percent_seen, percent_h_seen, percent_seen_top, no_tree_indicator, full_tree, zed_threshed = get_percent_seen(
            zed_copy, zed_xyz, cors[:4], True, True)
        cv2.rectangle(zed_copy, (x1, y1), (x2, y2), color, thickness)
        plot_2_imgs(jai, zed_copy, title=f_id, save_to="", save_only=False, cv2_save=False, quick_save=False)
        plot_2_imgs(zed_threshed, zed_copy,
                    title=f"""frame: {f_id}, percent: {percent_seen}, percent_top: {percent_seen_top}, percent_h_seen: {percent_h_seen}
        full_tree: {full_tree}, no_tree_indicator: {no_tree_indicator}""",
                    save_to="", save_only=False, cv2_save=False, quick_save=False)
