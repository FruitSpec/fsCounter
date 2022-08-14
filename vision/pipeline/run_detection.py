import os

import cv2
from omegaconf import OmegaConf

from vision.pipeline.detection_flow import counter_detection
from vision.pipeline.run_args import make_parser

def run(cfg, args):
    detector = counter_detection(cfg)

    cap = cv2.VideoCapture(args.movie_path)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    f_id = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # detect:
            det_outputs = detector.detect(frame)

            # track:
            trk_outputs = detector.track(det_outputs, f_id)

            f_id += 1

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    pass

if __name__ == "__main__":

    cwd = os.getcwd()
    config_file = "/vision/pipeline/config/pipeline_config.yaml"
    cfg = OmegaConf.load(cwd + config_file)


    args = make_parser().parse_args()

    run(cfg)
