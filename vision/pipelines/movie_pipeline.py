import os
import sys

import cv2
from omegaconf import OmegaConf

from vision.misc.help_func import get_repo_dir

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.pipelines.run_args import make_parser
from vision.data.results_collector import ResultsCollector, scale
from vision.data import COCO_utils

def run(cfg, args):
    detector = counter_detection(cfg)
    results_collector = ResultsCollector()

    cap = cv2.VideoCapture(args.movie_path)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Read until video is completed
    print(f'Inferencing on {args.movie_path}')
    f_id = 0
    ids = []
    while (cap.isOpened()):


        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # detect:
            det_outputs = detector.detect(frame)

            # track:
            trk_outputs = detector.track(det_outputs, f_id)

            # collect results:
            scale_ = scale(detector.input_size, frame.shape)
            results_collector.collect_detections(det_outputs, f_id, scale_)
            results_collector.collect_tracks(trk_outputs)


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

    results_collector.write_results_on_movie(args.movie_path, args.output_folder, write_tracks=False, write_frames=True)

    #categories, class_ids = get_id_and_categories(cfg)
    #coco_data = COCO_utils.generate_coco_format(results_collector.detections,
    #                                            (height, width),
    #                                            ids,
    #                                            cfg.input_size,
    #                                            class_ids,
    #                                            categories,
    #                                            ids)
    #COCO_utils.write_coco_file(coco_data, os.path.join(args.output_folder, 'results_coco.json'))



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

    args.movie_path = '/home/fruitspec-lab/FruitSpec/Data/syngenta/1/Result_FSI_1.mkv'
    args.output_folder = '/home/fruitspec-lab/FruitSpec/Sandbox/Syngenta/1'

    run(cfg, args)
