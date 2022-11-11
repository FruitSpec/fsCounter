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


def slice_to_frames(movie_path, output_path, rotate=True):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    cap = cv2.VideoCapture(movie_path)
    temp = movie_path.split('.')[0]
    temp = temp.split('/')[-1]
    channel_id = temp.split('_')[1]

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
            cv2.imwrite(os.path.join(output_path, f"channel_{channel_id}_frame_{f_id}.jpg"), frame)
        f_id += 1

def mkv_to_fsi_and_rgb(folder, output_path):

    file_list = os.listdir(folder)
    for file in file_list:
        if 'mkv' in file:
            if 'RGB' in file:
                channel_rgb_p = os.path.join(folder, file)
            elif '800' in file:
                channel_800_p = os.path.join(folder, file)
            elif '975' in file:
                channel_975_p = os.path.join(folder, file)




    cap_rgb = cv2.VideoCapture(channel_rgb_p)
    cap_800 = cv2.VideoCapture(channel_800_p)
    cap_975 = cv2.VideoCapture(channel_975_p)

    print('rgb:',cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    print('800: ',cap_800.get(cv2.CAP_PROP_FRAME_COUNT))
    print('975: ',cap_975.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap_800.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_800.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_800.get(cv2.CAP_PROP_FPS))

    fsi_video_name = os.path.join(output_path, 'FSI.mkv')
    rgb_video_name = os.path.join(output_path, 'rgb.mkv')
    fsi = cv2.VideoWriter(fsi_video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   fps, (width, height))
    rgb = cv2.VideoWriter(rgb_video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          fps, (width, height))

    i = 0
    while cap_rgb.isOpened():
        #frame_rgb = get_frame_by_index(cap_rgb, i)
        #frame_800 = get_frame_by_index(cap_800, i)
        #frame_975 = get_frame_by_index(cap_975, i)

        _, frame_rgb = cap_rgb.read()
        _, frame_800 = cap_800.read()
        _, frame_975 = cap_975.read()

        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

        frame_fsi = frame_rgb.copy()
        frame_fsi[:, :, 0] = frame_800[:, :, 0]
        frame_fsi[:, :, 1] = frame_975[:, :, 0]

        frame_fsi = cv2.cvtColor(frame_fsi, cv2.COLOR_RGB2BGR)
        #frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        fsi.write(frame_fsi)
        rgb.write(frame_rgb)

        # i += 1
        # if i == 400:
        #     break

    fsi.release()
    rgb.release()
    cap_rgb.release()
    cap_800.release()
    cap_975.release()


def get_frame_by_index(cap, index_):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index_)
    ret, frame = cap.read()
    if ret == False:
        raise ValueError
    return frame

if __name__ == "__main__":
    movie_path = "/home/fruitspec-lab/FruitSpec/Sandbox/merge_sensors/Result_FSI_2_30_720_30.mkv"
    output_path = "/home/fruitspec-lab/FruitSpec/Sandbox/merge_sensors/FSI_2_30_720_30"
    range_ = None # [250, 300]
    run(movie_path, output_path, range=range_)



