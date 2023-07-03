import os
import sys
import cv2
from omegaconf import OmegaConf
from tqdm import tqdm
import random

from vision.misc.help_func import get_repo_dir

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.tools.translation import translation as T
from vision.data.results_collector import ResultsCollector
from vision.misc.help_func import validate_output_path


def run(cfg, args):
    detector = counter_detection(cfg, args)
    results_collector_track = ResultsCollector(rotate=args.rotate)
    results_collector_annotate = ResultsCollector(rotate=args.rotate)
    translation = T(cfg.translation.translation_size, cfg.translation.dets_only, cfg.translation.mode)

    cap = cv2.VideoCapture(args.movie_path)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read until video is completed
    print(f'Inferencing on {args.movie_path}')
    f_id = -1
    count_tracks, count_annotate = 0,0
    pbar = tqdm(total=tot_frames)
    track_frames, annotate_frames = check_task(tot_frames, args.task)
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        f_id += 1
        if ret == True and (f_id in track_frames or f_id in annotate_frames):
            pbar.update(1)
            if args.rotate == 2:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif args.rotate == 1:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # detect:
            det_outputs = detector.detect(frame)

            # find translation
            tx, ty = translation.get_translation(frame, det_outputs)

            # track:
            trk_outputs, trk_windows = detector.track(det_outputs, tx, ty, f_id)

            if args.debug.is_debug and f_id in annotate_frames:
                # collect results:
                results_collector_annotate.collect_tracks(trk_outputs)
                args.output_folder = args.output_folder_annotate
                results_collector_annotate.det_to_coco(f_id, args, trk_outputs,frame)
                results_collector_annotate.debug(f_id, args, trk_outputs, det_outputs, frame, hists=None, trk_windows=trk_windows)

                count_annotate+=1

            if args.debug.is_debug and f_id in track_frames:
                # collect results:
                results_collector_track.collect_tracks(trk_outputs)
                args.output_folder = args.output_folder_track
                results_collector_track.det_to_coco(f_id, args, trk_outputs,frame)
                results_collector_track.debug(f_id, args, trk_outputs, det_outputs, frame, hists=None, trk_windows=trk_windows)

                count_tracks += 1


        # Break the loop
        elif ret == False:
            break

        elif count_annotate >= len(annotate_frames) and count_tracks >= len(track_frames):
            break

    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

    if len(annotate_frames) > 0:
        results_collector_annotate.dump_to_json(f"{args.output_folder_annotate}/coco.json")
    if len(track_frames) > 0:
        results_collector_track.dump_to_json(f"{args.output_folder_track}/coco.json")


def check_task(frames, task):
    f_ann = random.sample(range(frames), task['annotation'])
    start = random.choice(range(frames))
    end = start + task['tracking']
    f_track = range(frames)[start:end]
    return f_track, f_ann


if __name__ == "__main__":
    repo_dir = get_repo_dir()

    """user"""
    config_file = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/runtime_config.yaml"
    cfg = OmegaConf.load(repo_dir + config_file)
    args = OmegaConf.load(repo_dir + runtime_config)
    args.task = {'annotation': 0, 'tracking': 20}

    validate_output_path(args.output_folder)
    validate_output_path(args.output_folder_track, flag=args.task['tracking'])
    validate_output_path(args.output_folder_annotate, flag=args.task['annotation'])
    run(cfg, args)
