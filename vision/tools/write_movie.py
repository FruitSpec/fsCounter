import cv2
import os
import collections

from tqdm import tqdm

def write_results_on_movie(frame_path, output_path, range=[0,500]):
    """
        the function draw results on each frame
        - Each frame will be saved with results if 'write_frames' is True
        - 'write_tracks' indicate if the tracker results will be drawn or detections
    """
    file_list = os.listdir(frame_path)
    file_dict = {}
    for file in file_list:
        splited = file.split('.')
        if 'jpg' in splited[-1]:
            words = splited[0].split('_')
            frame_id = int(words[-2])
            file_path = os.path.join(output_path, file)
            file_dict[frame_id] = file_path

    file_dict = collections.OrderedDict(sorted(file_dict.items()))

    width, height = 1536, 2048

    output_video_name = os.path.join(output_path, 'result_video.avi')
    output_video = cv2

    for id_, f_path in tqdm(file_dict.items()):
        if id_ < range[0]:
            continue
        elif id_ > range[1]:
            break
        frame = cv2.imread(f_path)
        output_video.write(frame)

    output_video.release()

    print('Done')

if __name__ == "__main__":

    frame_path = "/home/yotam/FruitSpec/Data/MIC_03112022/res"
    output_path = "/home/yotam/FruitSpec/Data/MIC_03112022/res"
    range = [363, 513]
    write_results_on_movie(frame_path, output_path, range)