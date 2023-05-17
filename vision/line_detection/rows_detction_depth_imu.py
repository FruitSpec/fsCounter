"""
This script is based on the 'vision/depth/slicer/slicer_flow.py' script.
It used to find if the camera is facing a tree line by estimating the % of pixels in depth image that
consider far away.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import collections
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
#import copy
from vision.depth.slicer.slicer import slice_frame, print_lines
from vision.depth.slicer.slicer_validatior import slicer_validator
from vision.misc.help_func import validate_output_path
from vision.tools.video_wrapper import video_wrapper
from vision.depth.slicer.slicer_validatior import load_json, write_json
from vision.tools.camera import is_saturated



def slice_frames(depth_folder_path, output_path, rgb_folder_path=None, debug=False, start_frame=None, end_frame=None):

    depth_frames_dict, frames_dict = init_frames(depth_folder_path, rgb_folder_path)
    slice_data = {}
    rgb = None

    start_frame, end_frame = init_start_end(depth_frames_dict, start_frame, end_frame)
    print(f'params {start_frame, end_frame}')
    for id_, f_path in tqdm(depth_frames_dict.items()):

        if id_ < start_frame:
            continue
        if id_ > end_frame:
            break

        depth = cv2.imread(f_path)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        res = slice_frame(depth, window_thrs=0.4)
        slice_data[id_] = res

    slice_data = smooth_slicing(slice_data)

    if debug:
        for id_, f_path in tqdm(depth_frames_dict.items()):

            if id_ < start_frame:
                continue
            if id_ > end_frame:
                break

            depth = cv2.imread(f_path)
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
            res = slice_data[id_]
            if frames_dict is not None:
                rgb = cv2.imread(frames_dict[id_])
            save_reults(res, depth, rgb, output_path, id_)

    return slice_data


def slice_clip(svo_path, log_path, output_dir, output_name, EMA_alpha_AV ,EMA_alpha_depth,ground_truth = None, rotate=0, angular_velocity_thresh = 10, depth_percentile_thresh=0.5, signal_thrs=0.5,  start_frame=1, depth_minimum = 1, depth_maximum = 3.5, end_frame=None, save_video = False):

    validate_output_path(output_dir)

    # load log file to csv:
    print (f'Extracting Sensors data {log_path} from ')
    log_contents = read_log_file(log_path)
    columns_names = ['date', 'timestamp', 'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
     'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z']
    df = read_imu_log(log_contents, columns_names)


    # Extract video
    print(f'Extracting video from {svo_path}')
    cam = video_wrapper(svo_path, rotate=rotate, depth_minimum=depth_minimum, depth_maximum=depth_maximum)
    end_frame = validate_end_frame(cam, end_frame)

    index = start_frame
    pbar = tqdm(total=end_frame-start_frame)

    if save_video:
        output_video_path = output_dir + output_name + ".mp4"
        writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (cam.get_width() * 2, cam.get_height()))

    in_line = False

    while True:

        pbar.update(1)
        # TODO: add validation for start / end row and saturation
        if index > end_frame:
            break
        ########################
        # skip every 30 frames
        if index%30==0:   # todo: remove. its just a fix for a local bug
            index+=1
            continue
        #############################
        # Capture frame-by-frame
        frame, depth = cam.get_zed(index, exclude_point_cloud=True)

        if not cam.res:  # couldn't get frames
            print (f'cam.res {cam.res}, Break the loop')
            # Break the loop
            break


        depth_ema_previous = df.loc[index - 1, ['score_ema']][0] if ('score_ema' in df.columns) else None
        is_near, score, score_ema, saturated, y_end, y_start = near_objects (frame, depth, depth_percentile_thresh, signal_thrs, EMA_alpha_depth, depth_ema_previous)

        # decide if turning from imu sensors (angular_velocity_x):
        av_ema_previous = df.loc[index - 1, ['AV_ema']][0] if 'AV_ema' in df.columns else None
        current_sensors_data = df.loc[index].angular_velocity_x
        turning, angular_velocity, angular_velocity_ema = is_turning (current_sensors_data, angular_velocity_thresh, av_ema_previous, EMA_alpha_AV)


        # decide if in line:
        in_line = in_line_decision(in_line, is_near, turning)

        # Update dataframe:
        new_columns = ['turning','AV_thresh','is_near', 'score', 'score_ema', 'depth_thresh','saturated', 'line_pred', 'line_GT']
        df.loc[index, new_columns] = [turning, angular_velocity_thresh, is_near, score, score_ema, depth_percentile_thresh,saturated, in_line, ground_truth]

        # update video output
        if (depth is not None) and save_video:
            depth_3d = draw_lines_text(depth, y_end, y_start, in_line, turning, angular_velocity,is_near, score)
            merged = cv2.hconcat([frame, depth_3d])
            writer.write(merged)

        index += 1

    # Save csv:
    file_path_df = output_dir + "/depth_ein_vered_SUMERGOL_230423_row_3XXX.csv"
    df.to_csv(file_path_df)
    print (f'Saved df to {file_path_df}')

    if save_video:
        writer.release()
        print (f'Saved video to {output_video_path}')


    return df


# def moving_average_smoothing(df_column, index, window_size=30):
#     idx_start = max(0,index-window_size+1)
#     idx_end = min (index + 1, len(df_column))
#     moving_average = df_column.iloc[idx_start:idx_end].mean()
#     return moving_average
#
# ma_score = moving_average_smoothing(df['score'], index, window_size=30)

def near_objects(frame, depth, depth_percentile_thresh, signal_thrs, EMA_alpha_depth, ema_previous):
    saturated = is_saturated(frame)
    depth = remove_sky_noise(depth, frame)

    y_start = int(depth.shape[0] * 0.25)
    y_end = int(depth.shape[0] * 0.75)
    score = near_objects_around(depth, start=y_start, end=y_end,
                         signal_thrs=signal_thrs)  # % of far black (below threshold) pixels

    # smooth depth score with exponential moving average:
    score_ema = ema(score, ema_previous=ema_previous, alpha=EMA_alpha_depth)
    is_near = score_ema <= depth_percentile_thresh

    return is_near, score, score_ema, saturated, y_end, y_start,

def ema(score, ema_previous, alpha):
    if ema_previous is None:
        return score
    else:
        ema_curr = (1 - alpha) * ema_previous + alpha * score
        return ema_curr

def validate_end_frame(cam, end_frame):
    number_of_frames = cam.get_number_of_frames()
    if end_frame is not None:
        end_frame = min(number_of_frames, end_frame)
    else:
        end_frame = number_of_frames
    return end_frame


def remove_sky_noise(depth, frame):
    # shadow sky noise:
    b = frame[:, :, 0].copy()
    depth[b > 240] = 0
    return depth


def in_line_decision(in_line, is_near, turning):
    # todo -decide for initial frame
    '''
    Start a new row when depth is near,
    End the row if turning
    '''
    if (in_line == False) and is_near:  # Start line if near
        in_line = True
    elif (in_line == True) and turning:  # End line if turning
        in_line = False
    return in_line


def draw_lines_text(depth_img, y_end, y_start, in_line, turning, angular_velocity, is_near, score, ground_truth=None):

    text1 = f'Line Pred: {in_line}'
    text2 = f'Near obj:  {is_near}, {score}'
    text3 = f'Turning:   {turning}, {angular_velocity}'

    depth_3d = cv2.cvtColor(depth_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    depth_3d = cv2.putText(img=depth_3d, text=text1, org=(80, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale=2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    depth_3d = cv2.putText(img=depth_3d, text=text2, org=(80, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale=2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    depth_3d = cv2.putText(img=depth_3d, text=text3, org=(80, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale=2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    if ground_truth is not None:
        depth_3d = cv2.putText(img=depth_3d, text=f'GT: {ground_truth}', org=(700, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                               fontScale=2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    depth_3d = cv2.line(depth_3d, pt1=(0, y_start), pt2=(depth_img.shape[1], y_start), color=(0, 255, 0), thickness=5)
    depth_3d = cv2.line(depth_3d, pt1=(0, y_end), pt2=(depth_img.shape[1], y_end), color=(0, 255, 0), thickness=5)
    return depth_3d


def worker(cam, index):
    frame, depth, point_cloud = cam.get_zed(index)
    if not cam.res:  # couldn't get frames
        # Break the loop
        return None

    # shadow sky noise:
    b = frame[:, :, 0].copy()
    depth[b > 240] = 0
    # slice frame
    if near_objects(depth) or is_saturated(frame):
        index += 1
        return []
    res = slice_frame(depth.astype(np.uint8), window_thrs=window_thrs, neighbours_thrs=neighbours_thrs)
    return res


def is_turning(angular_velocity_x, angular_velocity_thresh, AV_ema_previous, EMA_alpha_AV):
    '''
    This function returns True is the Exponential moving average (EMA) of the angular_velocity_x is above thresh
    '''
    AV_ema = ema(angular_velocity_x, ema_previous=AV_ema_previous, alpha=EMA_alpha_AV)
    turning = abs(AV_ema) > angular_velocity_thresh
    return turning, round(angular_velocity_x, 2), round(AV_ema,2)

def post_process(slice_data, output_path=None, save_csv=False, save_csv_trees=False):

    trees_data = parse_data_to_trees(slice_data)
    #hash = {}
    #for tree_id, frames in trees_data.items():
    #    for frame in frames:
    #        if frame['frame_id'] in list(hash.keys()):
    #            hash[frame['frame_id']].append(frame)
    #        else:
    #            hash[frame['frame_id']] = [frame]

    df_all = []
    for tree_id, tree in trees_data.items():
        df = pd.DataFrame(data=tree, columns=['frame_id', 'tree_id', 'start', 'end'])
        if save_csv_trees:
            df.to_csv(os.path.join(output_path, f"T{tree_id}_slices.csv"))
        df_all.append(df)

    df_all = pd.concat(df_all)
    if save_csv:
        df_all.to_csv(os.path.join(output_path, f"slices.csv"))

    return df_all


def init_frames(depth_folder, rgb_folder):
    depth_frames_dict = get_frames_dict(depth_folder)
    if rgb_folder is not None:
        frames_dict = get_frames_dict(rgb_folder)
    else:
        frames_dict = None

    return depth_frames_dict, frames_dict


def init_start_end(depth_frames_dict, start_frame, end_frame):
    f_ids = list(depth_frames_dict.keys())
    min_frame_id = min(f_ids)
    max_frame_id = max(f_ids)

    if start_frame is not None:
        start_frame = max(min_frame_id, start_frame)
    else:
        start_frame = min_frame_id

    if end_frame is not None:
        end_frame = min(max_frame_id, end_frame)
    else:
        end_frame = max_frame_id

    return start_frame, end_frame


def smooth_slicing(slice_data, dist_thrs=300):
    ids = list(slice_data.keys())
    min_id = min(ids)
    max_id = max(ids)
    frames_back = 2

    for id_, res in slice_data.items():
        if id_ < min_id + frames_back:
            continue
        back_ind = id_ - 2
        mid_ind = id_ - 1
        if back_ind not in ids or mid_ind not in ids or id_ not in ids:
            continue
        res_back = slice_data[id_ - 2]
        res_middle = slice_data[id_ - 1]
        res_current = slice_data[id_]
        if len(res_middle) == 2:
            # maximal slices per frame
            continue
        for r_c in res_current:
            for r_b in res_back:
                if len(res_middle) == 2:
                    # maximal slices per frame
                    continue
                if np.abs(r_c - r_b) < dist_thrs:  # match
                    r_m = (r_c + r_b) // 2
                    if len(res_middle) == 0:
                        res_middle = [r_m]
                    elif np.abs(res_middle[0] - r_m) > dist_thrs:  # new slice
                        if res_middle[0] - r_m > 0:
                            res_middle = [r_m, res_middle[0]]

                        else:
                            res_middle = [res_middle[0],r_m]

        slice_data[id_ - 1] = res_middle

    return slice_data


def save_reults(res, depth, rgb, output_path, frame_id):
    depth_output = os.path.join(output_path, 'depth')
    validate_output_path(depth_output)
    rgb_output = os.path.join(output_path, 'rgb')
    validate_output_path(rgb_output)

    output = print_lines(depth, depth, res)
    fp = os.path.join(depth_output, f"depth_slice_{frame_id}.jpg")
    cv2.imwrite(fp, output)

    if rgb is not None:
        output = print_lines(rgb, depth, res)
        fp = os.path.join(rgb_output, f"rgb_slice_{frame_id}.jpg")
        cv2.imwrite(fp, output)



def get_frames_dict(folder_path):

    file_list = os.listdir(folder_path)
    file_dict = {}
    for file in file_list:
        splited = file.split('.')
        if 'jpg' in splited[-1]:
            words = splited[0].split('_')
            frame_id = int(words[-1])
            file_path = os.path.join(folder_path, file)
            file_dict[frame_id] = file_path

    return collections.OrderedDict(sorted(file_dict.items()))

def parse_data_to_trees(data, dist_thrs=150):
    tree_id = 0
    last_state = 0
    trees_data = {tree_id:[]}
    data = collections.OrderedDict(sorted(data.items()))
    for frame_id, loc in data.items():
        trees = list(trees_data.keys())
        state = get_state(loc)

        if last_state == 0:
            if state == 0:
                if len(trees) == 1:
                    continue
                else:
                    trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': -1})
            elif state == 1:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc[0]})
                trees_data[tree_id + 1] = [{'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': loc[0], 'end': -1}]
            elif state == 2:
                tree_id += 1
                trees_data[tree_id] = [{'frame_id': frame_id, 'tree_id': tree_id, 'start': loc[0], 'end': loc[1]}]
                #trees_data[tree_id + 1] = [{'frame_id': frame_id, 'tree_id': tree_id, 'start': loc[1], 'end': -1}]


        elif last_state == 1:
            if state == 0:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': -1})
            elif state == 1:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc[0]})
                trees_data[tree_id + 1].append({'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': loc[0], 'end': -1})
            elif state == 2:
                tree_id += 1
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': loc[0], 'end': loc[1]})
                #trees_data[tree_id + 1] = [{'frame_id': frame_id, 'tree_id': tree_id, 'start': loc[0], 'end': loc[1]}]

        elif last_state == 2:
            if state == 0:
                tree_id += 1
                trees_data[tree_id] = [{'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': -1}]
            elif state == 1:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc[0]})
                trees_data[tree_id + 1] = [{'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': loc[0], 'end': -1}]
            elif state == 2:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': loc[0], 'end': loc[1]})


        last_state = state

    return trees_data


def get_state(loc):
    if len(loc) == 0:
        state = 0
    elif len(loc) == 1:
        state = 1
    elif len(loc) == 2:
        state = 2
    return state


def near_objects_around(depth, start=750, end=1250, signal_thrs=20):

    width = depth.shape[1]
    search_area = depth[start: end, :].copy()
    score = np.sum(search_area < signal_thrs) / ((end - start) * width) # % pixels blow threshold
    score = round(score, 2)
    return score

    #return score >= percentile

def slice_folder_mp(folder_path, file_name, output_path, window_thrs, neighbours_thrs, dist_thrs, signal_thrs, start_frame, debug, max_workers=3):
    folder_list = os.listdir(folder_path)
    output_list = []
    path_list = []
    for folder in folder_list:
        temp_path = os.path.join(folder_path, folder)
        if not os.path.isdir(temp_path):
            continue

        temp_output = os.path.join(output_path, folder)
        validate_output_path(temp_output)
        output_list.append(temp_output)
        path_list.append(os.path.join(temp_path, file_name))

    args_list = create_args_list(path_list, output_list, window_thrs, neighbours_thrs, signal_thrs, dist_thrs, start_frame, debug)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        #results = list(executor.map(slice_clip, args_list))
        results = list(executor.map(lambda kwargs: slice_clip(**kwargs), args_list))


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



def slice_folder(folder_path, file_name, output_path, window_thrs, neighbours_thrs, dist_thrs, signal_thrs, start_frame, debug):
    folder_list = os.listdir(folder_path)

    for folder in folder_list:
        temp_path = os.path.join(folder_path, folder)
        if not os.path.isdir(temp_path):
            continue
        print(f'working on row {folder}')
        temp_output = os.path.join(output_path, folder)
        validate_output_path(temp_output)

        file_path = os.path.join(temp_path, file_name)
        data = slice_clip(file_path,
                          temp_output,
                          rotate=2,
                          window_thrs=window_thrs,
                          neighbours_thrs=neighbours_thrs,
                          dist_thrs=dist_thrs,
                          signal_thrs=signal_thrs,
                          start_frame=start_frame,
                          end_frame=None,
                          debug=debug)





if __name__ == "__main__":

    svo_path = "/home/lihi/FruitSpec/Data/customers/EinVered/SUMERGOL/250423/row_1/ZED_1.svo"
    log_path = f'/home/lihi/FruitSpec/Data/customers/EinVered/SUMERGOL/250423/row_1/Summer_Gold1/imu_1.log'
    output_dir = "/home/lihi/FruitSpec/Data/customers/EinVered/SUMERGOL/250423/row_1/rows_detection"

    output_name = 'EinVered_230423_SUMERGOL_230423_row_2'

    data = slice_clip(svo_path,log_path,output_dir,output_name,
                      rotate=2,
                      angular_velocity_thresh=10,
                      EMA_alpha_AV = 0.05,
                      depth_percentile_thresh=0.5,
                      signal_thrs=0.2,
                      EMA_alpha_depth=0.01,
                      depth_minimum=1,
                      depth_maximum=3.5,
                      start_frame=0,
                      end_frame=None,
                      save_video = True)


