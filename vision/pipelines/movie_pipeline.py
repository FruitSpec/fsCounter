import os
import sys

import cv2
from omegaconf import OmegaConf

cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.pipelines.run_args import make_parser
from vision.data.results_collector import ResultsCollector
from vision.data import COCO_utils

def run(cfg, args):
    detector = counter_detection(cfg)
    results_collector = ResultsCollector()

    cap = cv2.VideoCapture(args.movie_path)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    f_id = 0
    ids = []
    while (cap.isOpened()):

        width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # detect:
            det_outputs = detector.detect(frame)

            # track:
            trk_outputs, t2d_mapping = detector.track(det_outputs, f_id)

            # collect results:
            results_collector.collect_detections(det_outputs, t2d_mapping, f_id)


            ids.append(f_id)
            f_id += 1

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    results_collector.dump_to_csv(os.path.join(args.output_folder, 'detections.csv'))
    results_collector.dump_to_csv(os.path.join(args.output_folder, 'tracks.csv'), detections=False)

    categories, class_ids = get_id_and_categories(cfg)
    coco_data = COCO_utils.generate_coco_format(results_collector.detections,
                                                (height, width),
                                                ids,
                                                cfg.input_size,
                                                class_ids,
                                                categories,
                                                ids)
    COCO_utils.write_coco_file(coco_data, os.path.join(args.output_folder, 'results_coco.json'))



def get_id_and_categories(cfg):
    category = []
    category_ids = []
    for category, id_ in cfg.classes.items():
        category.append(category)
        category_ids.append(id_)

    return category, category_ids



if __name__ == "__main__":

    cwd = os.getcwd()
    config_file = "/vision/pipelines/config/pipeline_config.yaml"
    #config_file = "/config/pipeline_config.yaml"
    cfg = OmegaConf.load(cwd + config_file)


    args = make_parser()

    args.movie_path = '/home/yotam/Documents/FruitSpec/Data/JAI/Result_FSI_2_rot.mp4'
    args.output_folder = '/home/yotam/Documents/FruitSpec/Sandbox/sanity'

    run(cfg, args)
