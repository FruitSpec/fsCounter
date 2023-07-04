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

from vision.pipelines.ops.frame_loader import FramesLoader
from vision.data.results_collector import ResultsCollector
from vision.tools.translation import translation as T
from vision.depth.slicer.slicer_flow import post_process
from vision.tools.sensors_alignment import SensorAligner
from vision.tools.image_stitching import draw_matches
from vision.pipelines.ops.simulator import get_n_frames, init_cams
from vision.pipelines.detection_flow import counter_detection


def run(cfg, args, n_frames=200):
    print(f'Inferencing on {args.jai.movie_path}\n')
    rc = ResultsCollector(rotate=args.rotate)
    sensor_aligner = SensorAligner(cfg=cfg.sensor_aligner, zed_shift=args.zed_shift, batch_size=cfg.batch_size)
 #   det = counter_detection(cfg, args)
    det_outputs = None
    res = []
    frame_loader = FramesLoader(cfg, args)
    crop = [sensor_aligner.x_s, sensor_aligner.y_s, sensor_aligner.x_e, sensor_aligner.y_e]

    keypoints_path = os.path.join(args.output_folder, 'kp_match')
    validate_output_path(keypoints_path)
    loaded_path = os.path.join(args.output_folder, 'loaded')
    validate_output_path(loaded_path)

    f_id = 0
    n_frames = len(frame_loader.sync_zed_ids) if n_frames is None else min(n_frames, len(frame_loader.sync_zed_ids))
    pbar = tqdm(total=n_frames)
    while f_id < n_frames:
        pbar.update(cfg.batch_size)
        zed_batch, depth_batch, jai_batch, rgb_batch = frame_loader.get_frames(f_id, 0)

        debug = []
        for i in range(cfg.batch_size):
            debug.append({'output_path': keypoints_path, 'f_id': f_id + i})
        alignment_results = sensor_aligner.align_on_batch(zed_batch, rgb_batch, debug=debug)
#        det_outputs = det.detect(jai_batch)
        rc.collect_alignment(alignment_results, f_id)
        if len(zed_batch) < cfg.batch_size:
            f_id += cfg.batch_size
            continue
        for id_ in range(cfg.batch_size):


            # save_aligned(zed_batch[id_],
            #              jai_batch[id_],
            #              args.output_folder,
            #              f_id + id_,
            #              #corr=alignment_results[id_][0], dets=det_outputs[id_])
            #              corr=alignment_results[id_][0], dets=det_outputs)
            # save_loaded_images(zed_batch[id_],
            #                    jai_batch[id_], f_id + id_, loaded_path, crop)

            res.append(alignment_results[id_])

        f_id += cfg.batch_size


    pbar.close()
    frame_loader.close_cameras()
    output_path = os.path.join(args.output_folder, "alignment.csv")
    rc.dump_to_csv(output_path, "alignment")

#    res_df = pd.DataFrame(data=res, columns=['x1', 'y1', 'x2', 'y2', 'tx', 'ty', 'sx', 'sy', 'umatches', 'zed_shift'])
#   res_df.to_csv(os.path.join(args.output_folder, "res.csv"))

    return


def save_loaded_images(zed_frame, jai_frame, f_id, output_fp, crop=[]):
    if len(crop) > 0:
        x1 = crop[0]
        y1 = crop[1]
        x2 = crop[2]
        y2 = crop[3]
        cropped = zed_frame[int(y1):int(y2), int(x1):int(x2)]
        cropped = cv2.resize(cropped, (480, 640))
    else:
        cropped = cv2.resize(zed_frame, (480, 640))

    jai_frame = cv2.resize(jai_frame, (480, 640))

    canvas = np.zeros((700, 1000, 3), dtype=np.uint8)
    canvas[10:10 + cropped.shape[0], 10:10 + cropped.shape[1], :] = cropped
    canvas[10:10 + jai_frame.shape[0], 510:510 + jai_frame.shape[1], :] = jai_frame

    file_name = os.path.join(output_fp, f"frame_{f_id}.jpg")
    cv2.imwrite(file_name, canvas)



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





def save_aligned(zed, jai, output_folder, f_id, corr=None, sub_folder='FOV',dets=None):
    if corr is not None and np.sum(np.isnan(corr)) == 0:
        zed = zed[int(corr[1]):int(corr[3]), int(corr[0]):int(corr[2]), :]

    gx = 680 / jai.shape[1]
    gy = 960 / jai.shape[0]
    zed = cv2.resize(zed, (680, 960))
    jai = cv2.resize(jai, (680, 960))

    if dets is not None:
        dets = np.array(dets)
        dets[:, 0] = dets[:, 0] * gx
        dets[:, 2] = dets[:, 2] * gx
        dets[:, 1] = dets[:, 1] * gy
        dets[:, 3] = dets[:, 3] * gy
        jai = ResultsCollector.draw_dets(jai, dets, t_index=7, text=False)
        zed = ResultsCollector.draw_dets(zed, dets, t_index=7, text=False)

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

def validate_from_files(alignment, tracks, cfg, args, jai_only=False):
    dets = track_to_det(tracks)
    jai_frames = list(dets.keys())
    a_hash = get_alignment_hash(alignment)

    frame_loader = FramesLoader(cfg, args)
    frame_loader.batch_size = 1

    for id_ in tqdm(jai_frames):
        zed_batch, depth_batch, jai_batch, rgb_batch = frame_loader.get_frames(int(id_), 0)
        if jai_only:
            jai = ResultsCollector.draw_dets(jai_batch[0], dets[id_], t_index=7, text=False)
            fp = os.path.join(args.output_folder, 'Dets')
            validate_output_path(fp)

            cv2.imwrite(os.path.join(fp, f"dets_f{id_}.jpg"), jai)
        else:
            save_aligned(zed_batch[0],
                         jai_batch[0],
                         args.output_folder,
                         id_,
                         corr=a_hash[id_], dets=dets[id_])


def track_to_det(tracks_df):
    dets = {}
    for i, row in tracks_df.iterrows():
        if row['frame_id'] in list(dets.keys()):
            dets[int(row['frame_id'])].append([row['x1'], row['y1'], row['x2'], row['y2'], row['obj_conf'], row['class_conf'], int(row['frame_id']), 0])
        else:
            dets[int(row['frame_id'])] = [[row['x1'], row['y1'], row['x2'], row['y1'], row['obj_conf'], row['class_conf'], int(row['frame_id']), 0]]


    return dets

def get_alignment_hash(alignment):
    data = alignment.to_numpy()
    frames = data[:, 6]
    corr = data[:, :4]

    hash = {}
    for i in range(len(frames)):
        hash[frames[i]] = list(corr[i, :])

    return hash


if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)

    folder = "/media/matans/My Book/FruitSpec/NWFMXX/G10000XX/070623/row_12/1"
    args.zed.movie_path = os.path.join(folder, "ZED.mkv")
    args.depth.movie_path = os.path.join(folder, "DEPTH.mkv")
    args.jai.movie_path = os.path.join(folder, "Result_FSI.mkv")
    args.rgb_jai.movie_path = os.path.join(folder, "Result_RGB.mkv")
    args.sync_data_log_path = os.path.join(folder, "jaized_timestamps.csv")
    args.output_folder = os.path.join("/media/matans/My Book/FruitSpec/compare_det", 'orig')
    validate_output_path(args.output_folder)

    #run(cfg, args, None)
    folder = "/media/matans/My Book/FruitSpec/NWFMXX/G10000XX/070623/row_12/1"
    #t_p = os.path.join(folder, "tracks.csv")
    # a_p = os.path.join(folder, "alignment.csv")
    t_p = "/media/matans/My Book/FruitSpec/WASHDE_data_results/plot/G10000XX/070623/row_12/1/tracks.csv"
    a_p= "/media/matans/My Book/FruitSpec/WASHDE_data_results/plot/G10000XX/070623/row_12/1/alignment.csv"

    tracks = pd.read_csv(t_p)
    alignment = pd.read_csv(a_p)
    args.output_folder = os.path.join("/media/matans/My Book/FruitSpec/compare_det", 'new')
    validate_output_path(args.output_folder)
    validate_from_files(alignment=alignment, tracks=tracks, cfg=cfg, args=args, jai_only=True)
