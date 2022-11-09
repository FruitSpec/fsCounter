import os
import cv2
import numpy as np
from tqdm import tqdm

from vision.tools.camera import jai_to_channels

def run(movie_path, output_path,  range=None, rotate=True):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    cap = cv2.VideoCapture(movie_path)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    tot_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Read until video is completed
    f_id = 0
    ids = []
    pbar = tqdm(total=tot_frames)
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            pbar.update(1)
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            BGR, channel_1, channel_2 = jai_to_channels(frame)
            fsi = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            fsi[:, :, 2] = BGR[:, :, 0]
            fsi[:, :, 1] = channel_1
            fsi[:, :, 0] = channel_2
            fsi = cv2.cvtColor(fsi, cv2.COLOR_RGB2BGR)
            if range:
                if f_id > range[1]:
                    break
                elif f_id > range[0]:
                    cv2.imwrite(os.path.join(output_path, f'rgb_{f_id}.jpg'), BGR)
                    cv2.imwrite(os.path.join(output_path, f'channel_1_{f_id}.jpg'), channel_1)
                    cv2.imwrite(os.path.join(output_path, f'channel_2_{f_id}.jpg'), channel_2)
                    cv2.imwrite(os.path.join(output_path, f'fsi_{f_id}.jpg'), fsi)
                    cv2.imwrite(os.path.join(output_path, f'frame_{f_id}.jpg'), frame)
            else:
                cv2.imwrite(os.path.join(output_path, f'rgb_{f_id}.jpg'), BGR)
                cv2.imwrite(os.path.join(output_path, f'channel_1_{f_id}.jpg'), channel_1)
                cv2.imwrite(os.path.join(output_path, f'channel_2_{f_id}.jpg'), channel_2)
                cv2.imwrite(os.path.join(output_path, f'frame_{f_id}.jpg'), frame)

        f_id += 1

    cap.release()


if __name__ == "__main__":
    movie_path = "/home/fruitspec-lab/FruitSpec/Sandbox/merge_sensors/Result_FSI_2_30_720_30.mkv"
    output_path = "/home/fruitspec-lab/FruitSpec/Sandbox/merge_sensors/FSI_2_30_720_30"
    range_ = None # [250, 300]
    run(movie_path, output_path, range=range_)



