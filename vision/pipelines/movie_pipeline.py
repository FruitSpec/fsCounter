import os
import sys

import cv2
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

from vision.misc.help_func import get_repo_dir, scale_dets

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.pipelines.run_args import make_parser
from vision.data.results_collector import ResultsCollector, scale


def run(cfg, args):
    detector = counter_detection(cfg, args)
    results_collector = ResultsCollector(rotate=args.rotate)

    cap = cv2.VideoCapture(args.movie_path)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    tot_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Read until video is completed
    print(f'Inferencing on {args.movie_path}')
    f_id = 0
    ids = []
    pbar = tqdm(total=tot_frames)
    while (cap.isOpened()):


        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            pbar.update(1)
            if args.rotate:
                #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # detect:
            det_outputs = detector.detect(frame)
            scale_ = scale(detector.input_size, frame.shape)
            det_outputs = scale_dets(det_outputs, scale_)

            # track:
            trk_outputs, _ = detector.track(det_outputs, f_id, frame)

            # collect results:
            results_collector.collect_detections(det_outputs, f_id)
            results_collector.collect_tracks(trk_outputs)

            results_collector.draw_and_save(frame, trk_outputs, f_id, args.output_folder)
            ids.append(f_id)
            f_id += 1

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    #cv2.destroyAllWindows()

    results_collector.dump_to_csv(os.path.join(args.output_folder, 'detections.csv'))
    results_collector.dump_to_csv(os.path.join(args.output_folder, 'tracks.csv'), detections=False)

    results_collector.write_results_on_movie(args.movie_path, args.output_folder, write_tracks=True, write_frames=True)


def get_id_and_categories(cfg):
    category = []
    category_ids = []
    for category, id_ in cfg.classes.items():
        category.append(category)
        category_ids.append(id_)

    return category, category_ids



if __name__ == "__main__":

    repo_dir = get_repo_dir()
    config_file = "/vision/pipelines/config/pipeline_config.yaml"
    #config_file = "/config/pipeline_config.yaml"
    cfg = OmegaConf.load(repo_dir + config_file)


    args = make_parser()

    args.movie_path = '/home/yotam/FruitSpec/Data/Syngenta/JAI_blower/BLOWER_SIMPLE.mkv'
    args.output_folder = '/home/yotam/FruitSpec/Sandbox/Syngenta/blower_1'
    args.rotate = True
    args.frame_size = [2048, 1536]

    run(cfg, args)
