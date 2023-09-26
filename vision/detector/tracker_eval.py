
import pandas as pd
import motmetrics as mm
import json
from vision.tracker.fsTracker.fs_tracker import FsTracker
from omegaconf import OmegaConf
from vision.misc.help_func import get_repo_dir
from vision.tools.translation import translation as T
import os
import cv2
from vision.data.results_collector import ResultsCollector
from vision.misc.help_func import validate_output_path
from vision.tools.utils_general import download_s3_files, find_subdirs_with_file


def compute_mot_metrics(ground_truth_df, predictions_df, max_iou=0.5):
    """
    Computes a comprehensive set of MOT metrics using py-motmetrics.

    Args:
    - ground_truth_df (pd.DataFrame): The ground truth data.
    - predictions_df (pd.DataFrame): The predicted tracking data.
    - max_iou (float): The IoU threshold for considering bounding boxes as a match.

    Returns:
    - pd.DataFrame: A summary table with MOT metrics.
    """

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    for frame in ground_truth_df['frame'].unique():
        print (frame)
        gt_frame = ground_truth_df[ground_truth_df['frame'] == frame]
        pred_frame = predictions_df[predictions_df['frame'] == frame]

        gt_boxes = gt_frame[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
        pred_boxes = pred_frame[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values

        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=max_iou)

        acc.update(
            gt_frame['track_id'].values.astype('int'),  # Ground truth objects in this frame
            pred_frame['track_id'].values.astype('int'),  # Predicted objects in this frame
            distances)

    # Compute a comprehensive set of MOT metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc)

    return summary



def convert_coco_to_mot_format(path_gt_tracker_json):
    """
    Converts COCO annotations format to the format expected by compute_mot_metrics().

    Args:
    - path_gt_tracker_json (str): path to a COCO-format annotations file.

    Returns:
    - pd.DataFrame: Data in the format suitable for compute_mot_metrics().
    """
    # Load the COCO-format annotations file
    with open(path_gt_tracker_json, 'r') as file:
        coco_data = json.load(file)

    # Map image IDs to file names and frame numbers
    id_to_frame_number = {img['id']: int(img['file_name'].split('_')[1]) for img in coco_data['images']}
    id_to_file_name = {img['id']: img['file_name'] for img in coco_data['images']}

    # Extract relevant annotation information and construct the DataFrame
    mot_data = []
    for anno in coco_data['annotations']:
        mot_data.append({
            'file_name': id_to_file_name[anno['image_id']],
            'image_id': anno['image_id'],
            'frame': id_to_frame_number[anno['image_id']],
            'bb_left': anno['bbox'][0],
            'bb_top': anno['bbox'][1],
            'bb_width': anno['bbox'][2],
            'bb_height': anno['bbox'][3],
            'track_id': anno['fruit_id'],
        })
    df = pd.DataFrame(mot_data)
    df.columns = ['file_name','image_id','frame', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'track_id']

    return df

def convert_fs_tracks_csv_to_mot_format(PATH_predicted_tracks):
    tracks_df = pd.read_csv(PATH_predicted_tracks)
    # convert bounding box format ['x1','y1','x2','y2'] to ['bb_left','bb_top','bb_width','bb_height'] format for motmetrics
    tracks_df['bb_left'] = tracks_df['x1']
    tracks_df['bb_top'] = tracks_df['y1']
    tracks_df['bb_width'] = tracks_df['x2'] - tracks_df['x1']
    tracks_df['bb_height'] = tracks_df['y2'] - tracks_df['y1']
    tracks_df = tracks_df.drop(columns=['x1', 'y1', 'x2', 'y2'])
    tracks_df = tracks_df.rename(columns={'frame_id': 'frame'})
    return tracks_df

def track(tracker, outputs, translations, frame_id=None):

    batch_results = []
    batch_windows = []
    for i, frame_output in enumerate(outputs):
        if frame_output is not None:
            tx, ty = translations[i]
            if frame_id is not None:
                id_ = frame_id + i
            else:
                id_ = None
            online_targets, track_windows = tracker.update(frame_output, tx, ty, id_)
            tracking_results = []
            for target in online_targets:
                target.append(id_)
                tracking_results.append(target)

        batch_results.append(tracking_results)
        batch_windows.append(track_windows)

    return batch_results, batch_windows


def _extract_ground_truth_detections(df_tracker_gt, image_name, det_outputs):

    detections = df_tracker_gt[df_tracker_gt['file_name'] == image_name]
    # convert detections to format recognised by tracker:
    converted_detections = []
    for _, row in detections.iterrows():
        bb_left = row['bb_left']
        bb_top = row['bb_top']
        x2 = bb_left + row['bb_width']
        y2 = bb_top + row['bb_height']
        converted_detections.append([bb_left, bb_top, x2, y2, 1, 1, 0.0])
    det_outputs.append(converted_detections)
    return det_outputs


def _load_image(path_images_dir, image_name, jai_batch):
    img_path = os.path.join(path_images_dir, image_name)
    img = cv2.imread(img_path)
    jai_batch.append(img)
    return jai_batch

def eval_tracker_from_tracks_csv(path_ground_truth, path_predicted_tracks, max_iou=0.5):
    '''
    This function evaluates tracker performance from fruitspec tracks.csv file.
    The ground truth is in coco format (json file).
    '''
    df_gt = convert_coco_to_mot_format(path_ground_truth)
    df_pred = convert_fs_tracks_csv_to_mot_format(path_predicted_tracks)
    tracker_eval_summary = compute_mot_metrics(df_gt, df_pred, max_iou=max_iou)
    return tracker_eval_summary

def extract_tracks_csv_from_GT_detections(DIR_IMAGES, PATH_gt_tracks, OUTPUT_DIR):

    cfg = OmegaConf.load(get_repo_dir() + "/vision/pipelines/config/pipeline_config.yaml")
    args = OmegaConf.load(get_repo_dir() + "/vision/pipelines/config/dual_runtime_config.yaml")

    #init tracker
    tracker = FsTracker(frame_size=args.frame_size,
                 minimal_max_distance=cfg.tracker.minimal_max_distance,
                 score_weights=cfg.tracker.score_weights,
                 match_type=cfg.tracker.match_type,
                 det_area=cfg.tracker.det_area,
                 max_losses=cfg.tracker.max_losses,
                 translation_size=cfg.tracker.translation_size,
                 major=cfg.tracker.major,
                 minor=cfg.tracker.minor,
                 compile_data=cfg.tracker.compile_data_path,
                 debug_folder=None)

    # init
    translation = T(cfg.batch_size, cfg.translation.translation_size, cfg.translation.dets_only, cfg.translation.mode)
    results_collector = ResultsCollector()

    df_tracker_gt = convert_coco_to_mot_format(PATH_gt_tracks)


    jai_batch =[]
    det_outputs = []
    for f_id, image_name in zip(df_tracker_gt['frame'].unique(), df_tracker_gt['file_name'].unique()):

        jai_batch = _load_image(DIR_IMAGES, image_name, jai_batch)
        det_outputs = _extract_ground_truth_detections(df_tracker_gt, image_name, det_outputs)


        if len(jai_batch) == cfg.batch_size:
            f_id = f_id - cfg.batch_size + 1
            translation_results = translation.batch_translation(batch=jai_batch, detections=det_outputs)

            trk_outputs, trk_windows  = track(tracker=tracker, outputs = det_outputs, translations=translation_results, frame_id=f_id)
            results_collector.collect_tracks(trk_outputs)

            # reset batch:
            jai_batch = []
            det_outputs = []

    res = pd.DataFrame(results_collector.tracks, columns = results_collector.tracks_header[:-1])
    validate_output_path(OUTPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, 'tracks_from_gt_dets.csv')
    res.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return res, output_path


def eval_single_video(df_gt, df_pred, video_name, global_acc=None, max_iou=0.5):
    # Create an accumulator for this video
    acc = mm.MOTAccumulator(auto_id=True)

    for frame in df_gt['frame'].unique():
        gt_frame = df_gt[df_gt['frame'] == frame]
        pred_frame = df_pred[df_pred['frame'] == frame]

        gt_boxes = gt_frame[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
        pred_boxes = pred_frame[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values

        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=max_iou)

        acc.update(
            gt_frame['track_id'].values.astype('int'),
            pred_frame['track_id'].values.astype('int'),
            distances)

        # Update the global accumulator
        if global_acc is not None:
            global_acc.update(
                gt_frame['track_id'].values.astype('int'),
                pred_frame['track_id'].values.astype('int'),
                distances)

    # Compute MOT metrics for this video
    mh = mm.metrics.create()
    summary = mh.compute(acc, name=video_name)
    summary.insert(0,'video_name', video_name)
    return summary, global_acc



if __name__ == '__main__':

    ###### Download files from s3:  ######################################################################

    s3_path = 's3://fruitspec.dataset/tagging/JAI TRACKING/'
    output_path = '/home/fruitspec-lab-3/FruitSpec/Data/tracker'
    # download_s3_files (s3_path, output_path, string_param=None, suffix='', skip_existing=True)

    ####### Evaluation of Tracker + detector from tracks.csv:################################################
    # procude_tracks_csv_from_GT_detections = False

    all_results = []
    global_acc = mm.MOTAccumulator(auto_id=True)  # For computing aggregate metrics

    # Itterate video dirs:
    subdirs = [f.path for f in os.scandir(output_path) if f.is_dir()]
    for subdir in subdirs:

        video_name = os.path.basename(os.path.normpath(subdir))
        # Get GT file path:
        path_df_gt = find_subdirs_with_file(folder_path=subdir, file_name = '.json', return_dirs=False, single_file=True)
        df_gt = convert_coco_to_mot_format(path_df_gt)

        df_predicted_tracks, path_predicted_tracks = extract_tracks_csv_from_GT_detections(DIR_IMAGES = subdir, PATH_gt_tracks = path_df_gt, OUTPUT_DIR = subdir)
        df_pred = convert_fs_tracks_csv_to_mot_format(path_predicted_tracks)

        try:
            summary , global_acc = eval_single_video(df_gt, df_pred, video_name, global_acc, max_iou=0.5)
            all_results.append(summary)

        except Exception:
            print (f'Skipping, error encountered at {subdir}' )


    # Compute aggregate MOT metrics across all videos
    mh_global = mm.metrics.create()
    global_summary = mh_global.compute(global_acc, name='aggregate')
    global_summary.insert(0, 'video_name', 'aggregate')
    #all_results.append(global_summary) # todo - there is a bug in the accumulated results since that in each video the objectID is starting from 1 again.

    res = pd.concat(all_results, axis=0).reset_index(drop=True)

    # save results:
    validate_output_path(output_path)
    output_path = os.path.join(output_path, 'tracker_eval_results.csv')
    res.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print ('Done')



#     ###############################33
#     PATH_GT_TRACKS = r'/home/lihi/FruitSpec/Data/tracker/batch_1_e/batch1e.json'
#     path_pred = r'/home/lihi/FruitSpec/Data/tracker/batch_1_e/tracks_from_gt_dets.csv'
#     df_gt = convert_coco_to_mot_format(PATH_GT_TRACKS)
#     df_pred = convert_fs_tracks_csv_to_mot_format(path_pred)
#
#     ground_truth_data = {
#         'video1': df_gt
#     }
#
#     predictions_data = {
#         'video1': df_pred
#     }
#
#     results = compute_mot_metrics_by_video(ground_truth_data, predictions_data,  max_iou=0.5)
#
#     print('ok')
# ##################################################################################
#
#
#     PATH_GT_TRACKS = r'/home/fruitspec-lab-3/FruitSpec/Data/tracker/batch_2_e/batch2e.json'
#     DIR_IMAGES = '/home/fruitspec-lab-3/FruitSpec/Data/tracker/batch_2_e/frames'
#     OUTPUT_DIR = r'/home/fruitspec-lab-3/FruitSpec/Data/tracker/batch_2_e'
#
#     # Extract tracks.csv from GT detections COCO format (json file):
#     df_predicted_tracks, path_predicted_tracks = extract_tracks_csv_from_GT_detections(DIR_IMAGES, PATH_GT_TRACKS, OUTPUT_DIR)
#
#     # Evaluate tracker from tracks.csv:
#     tracker_eval_summary = eval_tracker_from_tracks_csv(PATH_GT_TRACKS, path_predicted_tracks, max_iou=0.5)
#
#



