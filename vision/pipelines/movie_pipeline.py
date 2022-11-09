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
    detector = counter_detection(cfg)
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
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # detect:
            det_outputs = detector.detect(frame)
            scale_ = scale(detector.input_size, frame.shape)
            det_outputs = scale_dets(det_outputs, scale_)

            # track:
            trk_outputs, trk_windows = detector.track(det_outputs, f_id, frame)

            # collect results:
            results_collector.collect_detections(det_outputs, f_id)
            results_collector.collect_tracks(trk_outputs)


            ids.append(f_id)


            canvas = np.zeros((2048, 1536, 3)).astype(np.uint8)
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
            if not os.path.exists(os.path.join(args.output_folder, 'frames')):
                os.mkdir(os.path.join(args.output_folder, 'frames'))
            cv2.imwrite(os.path.join(args.output_folder, 'frames', f"frame_{f_id}.jpg"), frame)

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

    args.movie_path = '/home/fruitspec-lab/FruitSpec/Sandbox/merge_sensors/Result_FSI_1_20_FHD15.mkv'
    args.output_folder = '/home/fruitspec-lab/FruitSpec/Sandbox/tracker_optimization/Result_FSI_1_20_FHD15'
    args.rotate = True

    run(cfg, args)
