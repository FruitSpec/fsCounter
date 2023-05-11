"""
This script is based on the 'vision/depth/slicer/slicer_flow.py' script.
It used to find if the camera is facing a tree line by estimating the % of pixels in depth image that
consider far away. It can save raw depth video and depth video with drawings
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from vision.misc.help_func import validate_output_path
from vision.tools.video_wrapper import video_wrapper
from vision.tools.utils_general import find_subdirs_with_file


def slice_clip(filepath, output_dir, output_name, rotate=0, signal_thrs=0.5, start_frame=1, end_frame=None, save_video = False, save_draw= False):
    print(f'working on {filepath}')
    validate_output_path(output_dir)
    depth_minimum = 1
    depth_maximum = 3.5
    cam = video_wrapper(filepath, rotate=rotate, depth_minimum=depth_minimum, depth_maximum=depth_maximum)
    number_of_frames = cam.get_number_of_frames()
    if end_frame is not None:
        end_frame = min(number_of_frames, end_frame)
    else:
        end_frame = number_of_frames

    index = start_frame
    pbar = tqdm(total=end_frame-start_frame)

    if save_video:
        writer1 = cv2.VideoWriter(os.path.join(output_dir, 'depth_draw_'+ output_name +".mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 15, (cam.get_width(), cam.get_height()))
    if save_draw:
        writer2 = cv2.VideoWriter(os.path.join(output_dir, 'depth_'+output_name + ".mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 15,(cam.get_width() * 2, cam.get_height()))

    df = pd.DataFrame()

    while True:
        #print(f'done {index} / {end_frame-start_frame}')
        pbar.update(1)
        # TODO: add validation for start / end row and saturation
        if index > end_frame:
            break
        ########################
        if index%30==0:   # todo: remove. its just a fix for a local bug
            index+=1
            continue
        #############################
        # Capture frame-by-frame
        frame, depth = cam.get_zed(index, exclude_point_cloud=True)
        depth = depth.astype(np.uint8)

        if not cam.res:  # couldn't get frames
            print (f'cam.res {cam.res}, Break the loop')
            # Break the loop
            break
        if index == 103:
            a = 149

        # shadow sky noise:
        b = frame[:, :, 0].copy()
        depth[b > 240] = 0

        y_start = int(depth.shape[0]*0.25)
        y_end = int(depth.shape[0]*0.75)

        score = is_wide_gap(depth, start=y_start, end=y_end, signal_thrs=signal_thrs, percentile=0.9) # % of far black (below threshold) pixels

        row = pd.DataFrame([{'frame': int(index), 'score': round(score,2)}])
        df = pd.concat([df, row], axis=0, ignore_index=True)

        if (depth is not None) and save_video:
            depth_3d = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
            writer1.write(depth_3d)

        if (depth is not None) and save_draw:
            depth_3d = draw_lines_text(depth_3d, y_end, y_start, text =f'Score {score}')
            merged = cv2.hconcat([frame, depth_3d])
            writer2.write(merged)

        index += 1

    file_path_df = os.path.join(output_dir,output_name + "_.csv")
    df.to_csv(file_path_df)
    print (f'Saved df to {file_path_df}')

    if save_video:
        writer1.release()


def draw_lines_text(img, y_end, y_start, text):

    depth_3d = cv2.putText(img=img, text=text, org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale=2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    depth_3d = cv2.line(depth_3d, pt1=(0, y_start), pt2=(img.shape[1], y_start), color=(0, 255, 0), thickness=5)
    depth_3d = cv2.line(depth_3d, pt1=(0, y_end), pt2=(img.shape[1], y_end), color=(0, 255, 0), thickness=5)
    return depth_3d


def is_wide_gap(depth, start, end, signal_thrs=20, percentile=0.9):
    width = depth.shape[1]

    search_area = depth[start: end, :].copy()
    score = np.sum(search_area < signal_thrs) / ((end - start) * width) # % pixels blow threshold
    score = round(score, 2)
    return score
    #return score >= percentile


def create_args_list(path_list, output_list, window_thrs, neighbours_thrs, signal_thrs, dist_thrs, start_frame, debug):
    args_list = []
    for path, output in zip(path_list, output_list):
        args_list.append({"file_name": path,
                          "output_path": output,
                          "window_thrs": window_thrs,
                          "neighbours_thrs": neighbours_thrs,
                          "dist_thrs": dist_thrs,
                          "signal_thrs": signal_thrs,
                          "start_frame": start_frame,
                          "debug": debug})
    return args_list


if __name__ == "__main__":

    # itterate over all svo files in the folder, analyse and save depth videos
    folder_path = r'/home/lihi/FruitSpec/Data/customers/EinVered/SUMERGOL'

    dirs = find_subdirs_with_file(folder_path, file_name='ZED_1.svo')

    for folder in dirs:
        svo_path = os.path.join(folder,'ZED_1.svo')
        output_dir = os.path.join(folder,'rows_detection')
        output_name = "_".join(folder.split('/')[-4:])

        slice_clip(svo_path,
                   output_dir,
                   output_name,
                   rotate=2,
                   signal_thrs=0.2,
                   start_frame=0,
                   end_frame=None,
                   save_video = True,
                   save_draw = True)

