import os
import sys
import pyzed.sl as sl
import cv2
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

from vision.misc.help_func import get_repo_dir, scale_dets

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.data.results_collector import ResultsCollector, scale
from vision.depth.zed.clip_depth_viewer import init_cam
from vision.tracker.fsTracker.score_func import get_intersection

def run(cfg, args):
    detector = counter_detection(cfg, args)
    results_collector = ResultsCollector(rotate=args.rotate)

    cam, runtime = init_cam(args.movie_path, 0.1, 2.5)

    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Read until video is completed
    print(f'Inferencing on {args.movie_path}\n')
    number_of_frames = sl.Camera.get_svo_number_of_frames(cam)
    frame_mat = sl.Mat()
    depth_mat = sl.Mat()
    point_cloud_mat = sl.Mat()
    f_id = 0
    pbar = tqdm(total=number_of_frames)
    while True:
        pbar.update(1)
        res = cam.grab(runtime)

        if res == sl.ERROR_CODE.SUCCESS and f_id < number_of_frames:

            frame = get_frame(frame_mat, cam)
            depth = get_depth(depth_mat, cam)
            #point_cloud = get_point_cloud(point_cloud_mat, cam)

            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
                #point_cloud = cv2.rotate(point_cloud, cv2.ROTATE_90_CLOCKWISE)

            # detect:
            det_outputs = detector.detect(frame)
            scale_ = scale(detector.input_size, frame.shape)
            det_outputs = scale_dets(det_outputs, scale_)



            #filter
            filtered_outputs = filter_by_distance(det_outputs, depth)
            filtered_outputs = filter_by_height(filtered_outputs, depth)
            filtered_outputs = filter_by_size(filtered_outputs)
            filtered_outputs = filter_by_duplicates(filtered_outputs)



            # track:
            trk_outputs, trk_windows = detector.track(filtered_outputs, f_id, frame)

            # collect results:
            results_collector.collect_detections(det_outputs, f_id)
            results_collector.collect_tracks(trk_outputs)

            save_windows(trk_windows, trk_outputs, f_id, args)
            results_collector.draw_and_save(frame, trk_outputs, f_id, args.output_folder)
            results_collector.draw_and_save(depth, trk_outputs, f_id, os.path.join(args.output_folder, 'depth'))
     #       results_collector.draw_and_save(frame, trk_outputs, f_id, os.path.join(args.output_folder, 'filtered'))

            f_id += 1

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cam.close()

    results_collector.dump_to_csv(os.path.join(args.output_folder, 'detections.csv'))
    results_collector.dump_to_csv(os.path.join(args.output_folder, 'tracks.csv'), detections=False)

    #results_collector.write_results_on_movie(args.movie_path, args.output_folder, write_tracks=True, write_frames=True)


def get_id_and_categories(cfg):
    category = []
    category_ids = []
    for category, id_ in cfg.classes.items():
        category.append(category)
        category_ids.append(id_)

    return category, category_ids


def get_frame(frame_mat, cam):
    cam.retrieve_image(frame_mat, sl.VIEW.LEFT)
    frame = frame_mat.get_data()[:, :, : 3]
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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

def filter_by_distance(dets, depth, threshold=150, percentile=0.4, factor=2.5):
    filtered_dets = []
    range_ = []
    for det in dets:
        crop = depth[det[1]:det[3] - 1, det[0]:det[2] - 1]
        h,w = crop.shape
        if w == 0 or h == 0:
            range_.append(0)
        else:
            range_.append(np.nanmean(crop))



    if range_:  # not empty
        bool_vec = np.array(range_) > threshold
        #keep_index = [i for i, tf in enumerate(tf_vec) if tf is True]


        #det_range = range_.copy()
        #range_.sort(reverse=True)
        #threshold = np.round(len(range_) * percentile).astype(np.int32)
        #mean = np.mean(range_[:threshold])
        #std = np.std(range_[:threshold])

        #bool_vec = det_range >= mean - (factor * std)
        for d_id, bool_val in enumerate(bool_vec):
            if bool_val:
                filtered_dets.append(dets[d_id])

    return filtered_dets


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
        vec = iou_mat[:, i]
        vec[i] = 0  # remove the i bbox from vector
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
        y_threshold = np.mean(y_loc[y_loc > 0]) + y_crop - bias

        for det in det_outputs:
            if det[1] > y_threshold:
                filtered.append(det)

    return filtered











def save_windows(trk_windows, trk_outputs, f_id, args):
    canvas = np.zeros((args.frame_size[0], args.frame_size[1], 3)).astype(np.uint8)
    for w in trk_windows:
        canvas = cv2.rectangle(canvas, (int(w[0]), int(w[1])), (int(w[2]), int(w[3])), (255, 0, 0),
                               thickness=-1)
    for t in trk_outputs:
        canvas = cv2.rectangle(canvas, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (0, 0, 255),
                               thickness=-1)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    if not os.path.exists(os.path.join(args.output_folder, 'windows')):
        os.mkdir(os.path.join(args.output_folder, 'windows'))
    cv2.imwrite(os.path.join(args.output_folder, 'windows', f"windows_frame_{f_id}.jpg"), canvas)


def validate_output_path(args):

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    if not os.path.exists(os.path.join(args.output_folder, 'windows')):
        os.mkdir(os.path.join(args.output_folder, 'windows'))
    if not os.path.exists(os.path.join(args.output_folder, 'depth')):
        os.mkdir(os.path.join(args.output_folder, 'depth'))
    if not os.path.exists(os.path.join(args.output_folder, 'filtered')):
        os.mkdir(os.path.join(args.output_folder, 'filtered'))



if __name__ == "__main__":

    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/home/yotam/FruitSpec/Code/fsCounter/vision/pipelines/config/runtime_config.yaml"
    #config_file = "/config/pipeline_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(runtime_config)

    validate_output_path(args)
    run(cfg, args)
