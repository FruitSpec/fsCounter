import os
import sys
import cv2
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from vision.misc.help_func import get_repo_dir, load_json, validate_output_path

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.data.results_collector import ResultsCollector
from vision.tools.translation import translation as T
from vision.depth.slicer.slicer_flow import post_process
from vision.tools.sensors_alignment import SensorAligner
from vision.tools.camera import is_sturated, stretch_rgb
from vision.tools.image_stitching import resize_img
from vision.pipelines.ops.simulator import get_n_frames, init_cams, get_frame_drop



import kornia as k
import torch

matcher = k.feature.LoFTR(pretrained="outdoor")




def run(cfg, args, n_frames=None):
    print(f'Inferencing on {args.jai.movie_path}\n')

    sensor_aligner = SensorAligner(args=args.sensor_aligner, zed_shift=args.zed_shift)
    results_collector = ResultsCollector(rotate=args.rotate)
    detector = counter_detection(cfg, args)

    zed_cam, rgb_jai_cam, jai_cam = init_cams(args)
    frame_drop_jai = get_frame_drop(args)

    clahe = cv2.createCLAHE(2, (10, 10))

    det_scale_factor_x = sensor_aligner.roix / 1536
    det_scale_factor_y = sensor_aligner.roiy / 2046
    #sift = cv2.SIFT_create()
    # sift.setNOctaveLayers(6)
    # sift.setEdgeThreshold(20)
    # sift.setSigma(1)
    # sift.setContrastThreshold(0.03)

    #matcher = k.feature.LoFTR(pretrained="outdoor")
    #matcher = matcher.cuda()

    f_id = 0
    n_frames = zed_cam.get_number_of_frames() if n_frames is None else min(n_frames, zed_cam.get_number_of_frames())
    pbar = tqdm(total=n_frames)
    res = []
    while f_id < n_frames:
        pbar.update(1)
        if f_id in frame_drop_jai:
            sensor_aligner.zed_shift += 1

        zed_frame, _, fsi_ret, jai_frame, rgb_ret, rgb_jai_frame = get_frames(zed_cam, jai_cam,
                                                                              rgb_jai_cam, f_id,
                                                                              sensor_aligner)

        if not fsi_ret or not zed_cam.res or not rgb_ret:  # couldn't get frames, Break the loop
            break
        if is_sturated(rgb_jai_frame, 0.6) or is_sturated(zed_frame, 0.6):
            print(f'frame {f_id} is saturated, skipping')
            f_id += 1
            continue

        rgb_jai_frame = stretch_rgb(rgb_jai_frame, clahe=clahe)
        rgb_jai_frame = cv2.cvtColor(rgb_jai_frame, cv2.COLOR_RGB2BGR)
        # align sensors
        #corr, tx_a, ty_a, sx, sy, kp_z, kp_r, match, st = sensor_aligner.align_sensors(zed_frame, rgb_jai_frame)

        corr, tx_a, ty_a, kp_zed, kp_jai, gray_zed, gray_jai, match, st = sensor_aligner.align_sensors(zed_frame,
                                                                                                       rgb_jai_frame)

        det_outputs = detector.detect(jai_frame)
        jai_res_frame = results_collector.draw_dets(jai_frame, det_outputs)


        det_outputs_zed = np.array(det_outputs)
        if len(det_outputs_zed) > 0:
            det_outputs_zed[:, 0] *= det_scale_factor_x
            det_outputs_zed[:, 2] *= det_scale_factor_x
            det_outputs_zed[:, 1] *= det_scale_factor_y
            det_outputs_zed[:, 3] *= det_scale_factor_y

        zed_res_frame = zed_frame[int(corr[1]):int(corr[3]), int(corr[0]):int(corr[2]), :].copy()
        zed_res_frame = results_collector.draw_dets(zed_res_frame, det_outputs_zed)

        save_aligned(zed_res_frame, jai_res_frame, args.output_folder, f_id, sub_folder='dets')


        used_matches = np.sum(st.reshape(-1).astype(np.bool_))

        draw_matches(zed_frame, kp_zed, rgb_jai_frame, kp_jai, match, st, args.output_folder, f_id)
        save_aligned(zed_frame, rgb_jai_frame, args.output_folder, f_id, corr=corr)

        res.append({'x1': corr[0],
                    'y1': corr[1],
                    'x2': corr[2],
                    'y2': corr[3],
                    'tx': tx_a,
                    'ty': ty_a,
                    #'sx': sx,
                    #'sy': sy,
                    'umatches': used_matches,
                    'zed_shift': sensor_aligner.zed_shift})

        f_id += 1


    pbar.close()
    zed_cam.close()
    jai_cam.close()
    rgb_jai_cam.close()

    res_df = pd.DataFrame(data=res, columns=['x1', 'y1', 'x2', 'y2', 'tx', 'ty', 'sx', 'sy', 'umatches', 'zed_shift'])
    res_df.to_csv(os.path.join(args.output_folder, "res.csv"))

    return




def init_run_objects(cfg, args):
    """
    Initializes the necessary objects for the main run function.

    Args:
        cfg (obj): Config object containing necessary parameters.
        args (argparse.Namespace): Namespace object containing arguments.

    Returns:
        tuple: A tuple of initialized objects consisting of:
            - detector (counter_detection): Counter detection object.
            - results_collector (ResultsCollector): Results collector object.
            - translation (T): Translation object.
            - sensor_aligner (SensorAligner): Sensor aligner object.
            - zed_cam (ZEDCamera): ZED camera object.
            - rgb_jai_cam (JAI_Camera): RGB camera object.
            - jai_cam (JAI_Camera): JAI camera object.
    """
    detector = counter_detection(cfg, args)
    results_collector = ResultsCollector(rotate=args.rotate)
    translation = T(cfg.translation.translation_size, cfg.translation.dets_only, cfg.translation.mode)
    sensor_aligner = SensorAligner(args=args.sensor_aligner, zed_shift=args.zed_shift)
    zed_cam, rgb_jai_cam, jai_cam = init_cams(args)
    return detector, results_collector, translation, sensor_aligner, zed_cam, rgb_jai_cam, jai_cam


def get_frames(zed_cam, jai_cam, rgb_jai_cam, f_id, sensor_aligner):
    """
    Returns frames from the ZED, JAI, and RGB JAI cameras for a given frame ID.

    Args:
        zed_cam (ZEDCam): A ZED camera object.
        jai_cam (JAICam
): A JAI camera object.
        rgb_jai_cam (JAICam): A JAI camera object used for capturing RGB images.
        f_id (int): The frame ID for which frames are to be retrieved.
        sensor_aligner (SensorAligner): A SensorAligner object for synchronizing the cameras.

    Returns:
        tuple: A tuple containing the following:
            - `zed_frame`: The ZED camera frame for the given frame ID.
            - `point_cloud`: The point cloud data generated by the ZED camera for the given frame ID.
            - `fsi_ret`: The return value for the JAI camera frame capture.
            - `jai_frame`: The JAI camera frame for the given frame ID.
            - `rgb_ret`: The return value for the RGB JAI camera frame capture.
            - `rgb_jai_frame`: The RGB JAI camera frame for the given frame ID.
    """
    zed_frame, point_cloud = zed_cam.get_zed(f_id + sensor_aligner.zed_shift, exclude_depth=True)
    fsi_ret, jai_frame = jai_cam.get_frame()
    rgb_ret, rgb_jai_frame = rgb_jai_cam.get_frame()
    return zed_frame, point_cloud, fsi_ret, jai_frame, rgb_ret, rgb_jai_frame


def get_sift_des(img1, img2, sift=None):
    if sift is None:
        sift = cv2.SIFT_create()
    # # find key points
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    return kp1, des1, kp2, des2


def match_des(des1, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    if isinstance(des1, type(None)) or isinstance(des2, type(None)):
        print(f'match descriptor des is non')
    matches = flann.knnMatch(des1, des2, k=2)

    match = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            match.append(m)

    return match


def estimate_M(kp1, kp2, match, ransac=10):
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in match]).reshape(-1, 1, 2)
    M, status = cv2.estimateAffine2D(src_pts, dst_pts,
                                     ransacReprojThreshold=ransac, maxIters=5000)

    return M, status


def draw_matches(img1, kp1, img2, kp2, match, status, draw_output, id_):
    out_img = cv2.drawMatches(img1, kp1, img2, kp2, np.array(match)[status.reshape(-1).astype(np.bool_)],
                              None, (255, 0, 0), (0, 0, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(draw_output, f"alignment_f{id_}.jpg"), out_img)


def save_aligned(zed, jai, output_folder, f_id, corr=None, sub_folder='FOV'):
    if corr is not None:
        zed = zed[int(corr[1]):int(corr[3]), int(corr[0]):int(corr[2]), :]
    zed = cv2.resize(zed, (680, 960))
    jai = cv2.resize(jai, (680, 960))

    canvas = np.zeros((960, 680*2, 3))
    canvas[:, :680, :] = zed
    canvas[:, 680:, :] = jai

    fp = os.path.join(output_folder, sub_folder)
    validate_output_path(fp)
    cv2.imwrite(os.path.join(fp, f"aligned_f{f_id}.jpg"), canvas)







def zed_slicing_to_jai(slice_data_path, output_folder, rotate=False):
    slice_data = load_json(slice_data_path)
    slice_data = ResultsCollector().converted_slice_data(slice_data)
    slice_df = post_process(slice_data=slice_data)
    slice_df.to_csv(os.path.join(output_folder, 'all_slices.csv'))


def get_number_of_frames(jai_max_frames, metadata=None):
    max_cut_frame = int(metadata['max_cut_frame']) if metadata is not None else np.inf
    n_frames = get_n_frames(max_cut_frame,jai_max_frames, metadata)

    return n_frames

def match(zed_frame, rgb_frame, draw_output=None):
    kp1, des1, kp2, des2 = get_sift_des(zed_frame, rgb_frame)
    match = match_des(des1, des2)
    M, status = estimate_M(kp1, kp2, match)
    if draw_output is not None:
        draw_matches(zed_frame, kp1, rgb_frame, kp2, match, status, draw_output)
    used_matches = np.sum(status.reshape(-1).astype(np.bool_))
    print(f"number of used matches {used_matches}")

    return used_matches

def loftr_preprocess(zed_frame, rgb_frame, size=(360, 520)):
    z_frame = preprocess_frame(zed_frame, size)
    r_frame = preprocess_frame(rgb_frame, size)

    return z_frame, r_frame

def loftr_match(z_frame, r_frame, matcher):
    input_dict = {"image0": torch.unsqueeze(z_frame, dim=0),  # LofTR works on grayscale images only
                  "image1": torch.unsqueeze(r_frame, dim=0)}
    with torch.inference_mode():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()

    return mkpts0, mkpts1

def preprocess_frame(frame, size):
    out_frame = cv2.resize(frame, size)
    out_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2GRAY)
    out_frame = k.image_to_tensor(out_frame, True).float() / 255.
    out_frame = out_frame.cuda()

    return out_frame


def update_args(args, row_dict):
    row = row_dict['p']
    r = row_dict['r']
    side = 1 if row[-1] == 'A' else 2
    splited = row.split('/')
    row_id = splited[-1]
    plot_id = splited[-2]

    new_args = args.copy()
    new_args.zed.movie_path = os.path.join(row, f'ZED_{side}.svo')
    new_args.zed.rotate = r
    new_args.jai.movie_path = os.path.join(row, f'Result_FSI_{side}.mkv')
    new_args.rgb_jai.movie_path = os.path.join(row, f'Result_RGB_{side}.mkv')
    new_args.output_folder = os.path.join(args.output_folder, f"{plot_id}_{row_id}")

    validate_output_path(new_args.output_folder)

    return new_args




if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)

    validate_output_path(args.output_folder)
    #copy_configs(pipeline_config, runtime_config, args.output_folder)

    rows = [{'p':"/media/fruitspec-lab-3/cam172/customers/DEWAGD/190123/DWDBLE33/R11A", 'r': 2}] #,
            #{'p':"/media/fruitspec-lab-3/cam172/customers/DEWAGD/240123/DWDBNB07/R1A", 'r': 1},
            #{'p':"/media/fruitspec-lab-3/cam172/customers/DEWAGD/240123/DWDBNB05/R5A", 'r': 2},
            #{'p':"/media/fruitspec-lab-3/cam172/customers/DEWAGD/230123/DWDBVM20/R7A", 'r': 2}]

    for r_ in rows:
        t_args = update_args(args, r_)
        run(cfg, t_args, 1000)
