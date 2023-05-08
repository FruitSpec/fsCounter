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


def slice_clip(filepath, output_path, rotate=0, window_thrs=0.4, neighbours_thrs=250, dist_thrs=150, signal_thrs=0.5,start_frame=0, end_frame=None, smooth=True, debug=False):
    print(f'working on {filepath}')
    depth_minimum = 1
    depth_maximum = 3.5
    cam = video_wrapper(filepath, rotate=rotate, depth_minimum=depth_minimum, depth_maximum=depth_maximum)
    number_of_frames = cam.get_number_of_frames()
    if end_frame is not None:
        end_frame = min(number_of_frames, end_frame)
    else:
        end_frame = number_of_frames

    slice_data = {}
    index = start_frame
    pbar = tqdm(total=end_frame-start_frame)

    while True:
        #print(f'done {index} / {end_frame-start_frame}')
        pbar.update(1)
        # TODO: add validation for start / end row and saturation
        if index > end_frame:
            break

        # Capture frame-by-frame
        frame, depth, point_cloud = cam.get_zed(index, exclude_point_cloud=True)
        if not cam.res:  # couldn't get frames
            # Break the loop
            break
        if index == 103:
            a = 149
        # shadow sky noise:
        b = frame[:, :, 0].copy()
        depth[b > 240] = 0
        # slice frame
        if is_wide_gap(depth):
            index += 1
            continue
        res = slice_frame(depth.astype(np.uint8),
                          window_thrs=window_thrs,
                          neighbours_thrs=neighbours_thrs,
                          signal_thrs=signal_thrs)
        slice_data[index] = res

        index += 1

    if smooth:
        slice_data = smooth_slicing(slice_data, dist_thrs=dist_thrs)

    if debug:
        print('saving debug data')
        for id_, res in tqdm(slice_data.items()):

            frame, depth, point_cloud = cam.get_zed(id_)
            if not cam.res:  # couldn't get frames
                # Break the loop
                break
            b = frame[:, :, 0].copy()
            depth[b > 240] = 0
            save_reults(res, depth.astype(np.uint8), frame, output_path, id_)

    write_json({"filepath": file_path, "output_path": output_path, "data": slice_data})

    return slice_data

def worker(cam, index):
    frame, depth, point_cloud = cam.get_zed(index)
    if not cam.res:  # couldn't get frames
        # Break the loop
        return None

    # shadow sky noise:
    b = frame[:, :, 0].copy()
    depth[b > 240] = 0
    # slice frame
    if is_wide_gap(depth) or is_saturated(frame):
        index += 1
        return []
    res = slice_frame(depth.astype(np.uint8), window_thrs=window_thrs, neighbours_thrs=neighbours_thrs)
    return res


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


def is_wide_gap(depth, start=750, end=1250, signal_thrs=20, percentile=0.9):
    width = depth.shape[1]

    search_area = depth[start: end, :].copy()
    score = np.sum(search_area < signal_thrs) / ((end - start) * width)

    return score >= percentile

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
    # for single file run
    file_path = "/media/fruitspec-lab/cam172/DEWAGB/210123/DWDBNW25/R5/ZED_1.svo"

    # for folder run
    folder_path = "/media/fruitspec-lab/cam172/DEWAGB/210123/DWDBNW25"
    file_name = "ZED_1.svo"

    # output validation
    output_path = "/home/fruitspec-lab/FruitSpec/Sandbox/DWDB_2023/Slicing/DWDBNW25"
    validate_output_path(output_path)

    neighbours_thrs = 350
    signal_thrs=0.5
    window_thrs = 0.7
    dist_thrs = 300
    start_frame = 0
    debug = False

    #slice_folder_mp(folder_path, file_name, output_path, window_thrs, neighbours_thrs, dist_thrs, signal_thrs, start_frame,
    #                debug)
    slice_folder(folder_path, file_name, output_path, window_thrs, neighbours_thrs, dist_thrs, signal_thrs, start_frame,
                 debug)

    # data = slice_clip(file_path,
    #                  output_path,
    #                  rotate=2,
    #                  window_thrs=window_thrs,
    #                  neighbours_thrs=neighbours_thrs,
    #                  dist_thrs=dist_thrs,
    #                  signal_thrs=signal_thrs,
    #                  start_frame=0,
    #                  end_frame=400,
    #                  debug=debug)


    #slicer_validator(file_path, output_path, rotate=2)
    # slice_data = load_json(file_path, output_path)
    # post_process(slice_data=slice_data, output_path=output_path)
    # slice_frames(depth_folder_path=depth_folder_path,
    #             output_path=output_path,
    #             rgb_folder_path=rgb_folder_path,
    #             start_frame=start_frame,
    #             debug=debug)