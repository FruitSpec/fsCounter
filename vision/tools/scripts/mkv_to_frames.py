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

            if range:
                if f_id > range[1]:
                    break
                elif f_id > range[0]:

                    cv2.imwrite(os.path.join(output_path, f'frame_{f_id}.jpg'), frame)
            else:

                cv2.imwrite(os.path.join(output_path, f'frame_{f_id}.jpg'), frame)

            f_id += 1
        else:
            break

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
            cv2.imwrite(os.path.join(output_path, f"frame_{f_id}.jpg"), frame)

            f_id += 1
        else:
            break



def mkv_to_fsi_and_rgb(folder, output_path, suffix="", write_images=False):
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

    fsi_video_name = os.path.join(output_path, f'FSI{suffix}.mkv')
    rgb_video_name = os.path.join(output_path, f'rgb{suffix}.mkv')
    fsi = cv2.VideoWriter(fsi_video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   fps, (width, height))
    rgb = cv2.VideoWriter(rgb_video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          fps, (width, height))

    i = 0
    n = cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT)
    while cap_rgb.isOpened():
        print(f"\r{i}/{n - 1} ({i / (n - 1) * 100: .2f}%) frames", end="")
        #frame_rgb = get_frame_by_index(cap_rgb, i)
        #frame_800 = get_frame_by_index(cap_800, i)
        #frame_975 = get_frame_by_index(cap_975, i)

        _, frame_rgb = cap_rgb.read()
        _, frame_800 = cap_800.read()
        _, frame_975 = cap_975.read()
        if isinstance(frame_rgb, type(None)):
            break
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

        frame_fsi = frame_rgb.copy()
        frame_fsi[:, :, 0] = frame_800[:, :, 0]
        frame_fsi[:, :, 1] = frame_975[:, :, 0]

        frame_fsi = cv2.cvtColor(frame_fsi, cv2.COLOR_RGB2BGR)
        #frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        fsi.write(frame_fsi)
        rgb.write(frame_rgb)

        if write_images:
            frame_fsi = cv2.rotate(frame_fsi, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(os.path.join(output_path, f"FSI{suffix}_frame_{i}.jpg"), frame_fsi)
            cv2.imwrite(os.path.join(output_path, f"RGB{suffix}_frame_{i}.jpg"), frame_rgb)

        i += 1
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
    # mkv_to_fsi_and_rgb("/home/fruitspec-lab/FruitSpec/Sandbox/Run_9_nov/row_4",
    #                    "/home/fruitspec-lab/FruitSpec/Sandbox/Run_9_nov/row_4", write_images=True)
    # mkv_to_fsi_and_rgb("/home/fruitspec-lab/FruitSpec/Sandbox/Run_9_nov/row_3",
    #                    "/home/fruitspec-lab/FruitSpec/Sandbox/Run_9_nov/row_3", write_images=True)
    #mkv_to_fsi_and_rgb("/home/fruitspec-lab/FruitSpec/Sandbox/Run_9_nov/row_3","/home/fruitspec-lab/FruitSpec/Sandbox/Run_9_nov/preocessed")
    #slice_to_frames(movie_path, output_path, rotate=True)
    movie_path = "/home/matans/Documents/fruitspec/sandbox/debugging/060623/gc1/row_2/1/Result_FSI.mkv"
    output_path = "/home/matans/Documents/fruitspec/sandbox/debugging/060623/gc1_row_2_frames"
    range_ = [250, 350]
    run(movie_path, output_path, range=range_)



