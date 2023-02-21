import os
import numpy as np
import cv2
import json
import collections
import pandas as pd
from tqdm import tqdm

from vision.visualization.drawer import draw_highlighted_test
from vision.tools.image_stitching import get_fine_keypoints, resize_img, get_fine_translation
from vision.tools.video_wrapper import video_wrapper
from vision.misc.help_func import validate_output_path


def mouse_callback(event, x, y, flags, params):
    """
    draws / cleans bounding box from showing image
    :param event: click event
    :param x: x of mouse
    :param y: y of mouse
    :param flags:
    :param params: params for image showing
    :return:
    """

    if event == cv2.EVENT_LBUTTONDOWN:
        if flags == cv2.EVENT_FLAG_CTRLKEY or flags == cv2.EVENT_FLAG_CTRLKEY + 1:  # second part is due to bug of cv2
            params["data"][params['index']]['start'] = x
            params["data"][params['index']]['end'] = min(x + 10, int(params['width'] // params['resize_factor']))
        else:
            params["data"][params['index']]['start'] = x
        frame = print_lines(params)
        frame = print_text(frame, params)
        cv2.imshow(params['headline'], frame)
    if event == cv2.EVENT_RBUTTONDOWN:
        if flags == cv2.EVENT_FLAG_CTRLKEY or flags == cv2.EVENT_FLAG_CTRLKEY + 2:  # second part is due to bug of cv2
            params["data"][params['index']]['start'] = x
            params["data"][params['index']]['end'] = max(x - 10, 0)
        else:
            params["data"][params['index']]['end'] = x
        frame = print_lines(params)
        frame = print_text(frame, params)
        cv2.imshow(params['headline'], frame)


def print_lines(params):
    frame = params['frame'].copy()
    y = int(params['height'] // params['resize_factor'])

    if params['data'][params['index']]['start'] is not None:
        x = int(params['data'][params['index']]['start'])
        frame = cv2.line(frame, (x, 0), (x, y), (255, 0, 0), 2)
    if params['data'][params['index']]['end'] is not None:
        x = int(params['data'][params['index']]['end'])
        frame = cv2.line(frame, (x, 0), (x, y), (255, 0, 255), 2)

    return frame


def print_text(frame, params, text=None):
    if text is None:
        text = f'Frame {params["index"]}'
    orange = (48, 130, 245)
    white = (255, 255, 255)

    height = int(params['height'] // params['resize_factor'])
    text_width = 150
    start_point = (10, height - 10)
    frame = draw_highlighted_test(frame, text, start_point, text_width, orange, white, above=True, font_scale=8,
                                  thickness=2, factor=10, box_h_margin=3)

    return frame


def update_index(k, params):
    """
    :param k: current key value
    :param params: run parameters
    :return: updated run parameters
    """
    index = params['index']
    if k == 122:
        index = max(index - 1, 0)
        params['find_translation'] = False
    elif k == 115:
        index = max(index + 100, 0)
        params['find_translation'] = False
    elif k == 120:
        index = max(index - 100, 0)
        params['find_translation'] = False
    elif k == 32:
        index = index
        params["data"][params['index']]['start'] = None
        params["data"][params['index']]['end'] = None
        params['find_translation'] = False
    else:
        index = max(index + 1, 0)
        params['find_translation'] = True

    params['index'] = index
    return params


def manual_slicer(filepath, output_path, data=None, rotate=0, index=0, draw_start=None, draw_end=None, resize_factor=3):
    """
    this is where the magic happens, palys the video
    """
    if data is None:
        data = load_json(filepath, output_path)
    params = {"filepath": filepath,
              "output_path": output_path,
              "data": data,
              "rotate": rotate,
              "index": index,
              "draw_start": draw_start,
              "draw_end": draw_end,
              'resize_factor': resize_factor,
              'last_kp_des': None,
              'find_translation': False}

    headline = f'clip {params["filepath"]}'
    params['headline'] = headline
    cv2.namedWindow(headline, cv2.WINDOW_GUI_NORMAL)

    cam = video_wrapper(filepath, rotate=rotate)
    number_of_frames = cam.get_number_of_frames()
    width = cam.get_width()
    height = cam.get_height()

    params['height'] = height
    params['width'] = width

    # Read until video is completed
    while True:
        # Capture frame-by-frame
        print(params["index"])
        if cam.mode == 'svo':
            cam.grab(params["index"])
        ret, frame = cam.get_frame(params["index"])
        if ret != True:
            break  # couldn't load frame

        # preprocess: resize and rotate if needed
        frame, params = preprocess_frame(frame, params)

        params = init_data_index(params)
        # params = get_updated_location_in_index(frame, params)

        frame = print_lines(params)
        frame = print_text(frame, params)

        cv2.imshow(headline, frame)

        cv2.setMouseCallback(headline, mouse_callback, params)
        k = cv2.waitKey()
        # Press Q on keyboard to  exit
        if cv2.waitKey(k) & 0xFF == ord('q'):
            write_json(params)
            break
        params = update_index(k, params)
        write_json(params)

    # When everything done, release the video capture object
    cam.close()

    # Closes all the frames
    cv2.destroyAllWindows()


def preprocess_frame(frame, params):
    height = int(params['height'] // params['resize_factor'])

    frame, r = resize_img(frame, height)
    # if params['rotate']:
    #       frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    params['frame'] = frame.copy()
    params['r'] = r

    return frame, params


def get_updated_location_in_index(frame, params):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    kp_des = get_fine_keypoints(gray)

    if params['find_translation']:
        last_locations = params['data'][params['index'] - 1]
        start = last_locations['start']
        end = last_locations['end']
        if (start is not None) or (end is not None):  # no start or end locations marked
            last_kp_des = params['last_kp_des']
            res = get_fine_translation(last_kp_des, kp_des)
            translation = np.array(res)
            translation = np.mean(translation, axis=0)
            if start:
                start = min(max(start + (int(translation[0]) // 2), 0), int(params['height'] // params['resize_factor']))
                params['data'][params['index']]['start'] = start
            if end:
                end = min(max(end + (int(translation[0]) // 2), 0), int(params['height'] // params['resize_factor']))
                params['data'][params['index']]['end'] = end

    params['last_kp_des'] = kp_des

    return params


def init_data_index(params):
    data_indexes = list(params['data'].keys())
    if params['index'] not in data_indexes:
        params['data'][params['index']] = {'start': None, 'end': None}

    return params


def write_json(params):
    temp_str = params['filepath']
    temp_str = temp_str.split('.')[0]
    clip_name = temp_str.split('/')[-1]

    output_file_name = os.path.join(params['output_path'], f'{clip_name}_slice_data.json')
    with open(output_file_name, 'w') as f:
        json.dump(params['data'], f)


def load_json(filepath, output_path):
    temp_str = filepath
    temp_str = temp_str.split('.')[0]
    clip_name = temp_str.split('/')[-1]

    input_file_name = os.path.join(output_path, f'{clip_name}_slice_data.json')
    if os.path.exists(input_file_name):
        with open(input_file_name, 'r') as f:
            loaded_data = json.load(f)
        data = {}
        for k, v in loaded_data.items():
            data[int(k)] = v

    else:
        data = {}
    return data


def slice_to_trees(data_file, file_path, output_path, resize_factor=3, h=2048, w=1536, on_fly=True):
    size = int(w // resize_factor)
    r = min(size / h, size / w)

    with open(data_file, 'r') as f:
        loaded_data = json.load(f)
    data = {}
    for k, v in loaded_data.items():
        data[int(k)] = v
    data = collections.OrderedDict(sorted(data.items()))

    trees_data = parse_data_to_trees(data)
    df_out = pd.DataFrame([item for sublist in list(trees_data.values()) for item in sublist])
    df_out[["start", "end"]] = df_out[["start", "end"]]/r
    df_out[["start", "end"]] = df_out[["start", "end"]].replace((-1)/r, -1)
    df_out.to_csv(os.path.join(output_path, "all_slices.csv"))
    return df_out
    hash = {}
    for tree_id, frames in trees_data.items():
        for frame in frames:
            if frame['frame_id'] in list(hash.keys()):
                hash[frame['frame_id']].append(frame)
            else:
                hash[frame['frame_id']] = [frame]

    if not on_fly:
        cap = cv2.VideoCapture(file_path)
        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        # Read until video is completed
        f_id = 0
        hash_ids = list(hash.keys())
        pbar = tqdm(total=len(hash_ids))
        while (cap.isOpened()):

            ret, frame = cap.read()
            if ret == True:
                pbar.update(1)
                if f_id in hash_ids:
                    for frame_data in hash[f_id]:
                        if not os.path.exists(os.path.join(output_path, f"T{frame_data['tree_id']}")):
                            os.mkdir(os.path.join(output_path, f"T{frame_data['tree_id']}"))
                        cv2.imwrite(os.path.join(output_path, f"T{frame_data['tree_id']}", f"frame_{f_id}.jpg"), frame)
                f_id += 1
            # Break the loop
            else:
                break
        # When everything done, release the video capture object
        cap.release()
    df_all = []
    for tree_id, tree in trees_data.items():
        df = pd.DataFrame(data=tree, columns=['frame_id', 'tree_id', 'start', 'end'])
        df.loc[df['start'] != -1, 'start'] = df.loc[df['start'] != -1, 'start'] // r
        df.loc[df['end'] != -1, 'end'] = df.loc[df['end'] != -1, 'end'] // r
        if on_fly:
            df_all.append(df)
        else:
            if not os.path.exists(os.path.join(output_path, f"T{tree_id}")):
                os.mkdir(os.path.join(output_path, f"T{tree_id}"))
            df.to_csv(os.path.join(output_path, f"T{tree_id}", f"slices.csv"))
    if on_fly:
        return pd.concat(df_all, axis=0)


def parse_data_to_trees(data):
    tree_id = 0
    # started_tree = False
    # start_and_end_tree = False

    last_state = 0
    trees_data = {}
    for frame_id, loc in data.items():
        if frame_id == 668:
            a = 1

        trees = list(trees_data.keys())
        state = get_state(loc)

        if last_state == 0:
            if state == 0:
                if len(trees) == 0:
                    continue
                elif tree_id != trees[-1]:
                    continue
                else:
                    trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': -1})
            elif state == 1:
                tree_id += 1
                trees_data[tree_id] = [{'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': -1}]
            elif state == 2:
                if tree_id == trees[-1]:
                    trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                else:
                    print(f"{frame_id}: {tree_id}")
                    raise ValueError("Got tree closing before tree opening")
            elif state == 3:
                print(f"{frame_id}: {tree_id}")
                raise ValueError("Got tree closing before tree opening")
            elif state == 4:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                # start new tree
                trees_data[tree_id + 1] = [{'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': loc['start'], 'end': -1}]
            elif state == 5:
                tree_id += 1
                trees_data[tree_id] = [
                    {'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': loc['end']}]
            elif state == 6:
                print(f"{frame_id}: {tree_id}")
                raise ValueError("Got tree closing before tree opening")


        elif last_state == 1:
            if state == 0:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': -1})
            elif state == 1:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': -1})
            elif state == 2:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
            elif state == 3:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': -1})
                # start new tree
                trees_data[tree_id + 1] = [{'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': -1, 'end': loc['end']}]

            elif state == 4:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                # start new tree
                trees_data[tree_id + 1] = [{'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': loc['start'], 'end': -1}]
            elif state == 5:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': loc['end']})
            elif state == 6:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                trees_data[tree_id + 1] = [{'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': loc['start'], 'end': -1}]

        elif last_state == 2:
            if state == 0:
                continue
            elif state == 1:
                tree_id += 1
                trees_data[tree_id] = [{'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': -1}]
            elif state == 2:
                if tree_id == trees[-1]:
                    trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
            elif state == 3:
                print(f"{frame_id}: {tree_id}")
                raise ValueError("Got wrong state 3 after state 2")
            elif state == 4:
                trees_data[tree_id].append(
                    {'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                # start new tree
                trees_data[tree_id + 1] = [
                    {'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': loc['start'], 'end': -1}]
            elif state == 5:
                tree_id += 1
                trees_data[tree_id] = [
                    {'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': loc['end']}]
            elif state == 6:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                trees_data[tree_id + 1] = [{'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': loc['start'], 'end': -1}]

        # start - end
        elif last_state == 3:
            if state == 0:
                tree_id += 1
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': -1})
            elif state == 1:
                tree_id += 1
                trees_data[tree_id] = [{'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': -1}]
            elif state == 2:
                if tree_id == trees[-1]:
                    trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                else:
                    print(f"{frame_id}: {tree_id}")
                    raise ValueError("Got tree closing before tree opening")
            elif state == 3:  # start - end
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': -1})
                # add to next tree
                trees_data[tree_id + 1].append({'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': -1, 'end': loc['end']})
            elif state == 4:  # end - start
                print(f"{frame_id}: {tree_id}")
                raise ValueError("Got wrong state 4 after state 3")
            elif state == 5:
                tree_id += 1
                trees_data[tree_id] = [
                    {'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': loc['end']}]
            elif state == 6:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                tree_id += 1
                trees_data[tree_id] = [{'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': -1}]

        # end-start
        elif last_state == 4:
            if state == 0:
                continue
            elif state == 1:
                tree_id += 1
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': -1})
            elif state == 2:
                if tree_id == trees[-1]:
                    trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                else:
                    tree_id += 1
                    trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
            elif state == 3:  # start - end
                print(f"{frame_id}: {tree_id}")
                raise ValueError("Got tree closing before tree opening")
            elif state == 4:  # end - start
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                # add to next tree
                trees_data[tree_id + 1].append(
                    {'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': loc['start'], 'end': -1})
            elif state == 5:
                tree_id += 1
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': loc['end']})
            elif state == 6:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                trees_data[tree_id + 1].append({'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': loc['start'], 'end': -1})

        # whole tree
        elif last_state == 5:
            if state == 0:
                continue
            elif state == 1:
                tree_id += 1
                trees_data[tree_id] = [{'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': -1}]
            elif state == 2:
                if tree_id == trees[-1]:
                    trees_data[tree_id].append(
                        {'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                else:
                    print(f"{frame_id}: {tree_id}")
                    raise ValueError("Got tree closing before tree opening")
            elif state == 3:  # start - end
                print(f"{frame_id}: {tree_id}")
                raise ValueError("Got tree closing before tree opening")
            elif state == 4:  # end - start
                trees_data[tree_id].append(
                    {'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                # add to next tree
                trees_data[tree_id + 1] = [{'frame_id': frame_id, 'tree_id': tree_id+1, 'start': loc['start'], 'end': -1}]
            elif state == 5:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': loc['end']})
            elif state == 6:
                trees_data[tree_id].append(
                    {'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                trees_data[tree_id + 1] = [{'frame_id': frame_id, 'tree_id': tree_id+1, 'start': loc['start'], 'end': -1}]

        elif last_state == 6:
            if state == 0:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': -1})
            elif state == 1:
                tree_id += 1
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': -1})
            elif state == 2:
                if tree_id == trees[-1]:
                    trees_data[tree_id].append(
                        {'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                else:
                    print(f"{frame_id}: {tree_id}")
                    raise ValueError("Got tree closing before tree opening")
            elif state == 3:  # start - end
                print(f"{frame_id}: {tree_id}")
                raise ValueError("Got tree closing before tree opening")
            elif state == 4:  # end - start
                trees_data[tree_id].append(
                    {'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                # add to next tree
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': -1})
            elif state == 5:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': loc['end']})
            elif state == 6:
                trees_data[tree_id].append(
                    {'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                trees_data[tree_id + 1].append({'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': loc['start'], 'end': -1})
        last_state = state

    return trees_data


def get_state(loc):
    if loc['start'] is None and loc['end'] is None:
        state = 0
    elif loc['start'] is not None and loc['end'] is not None:
        if np.abs(loc['end'] - loc['start']) < 20:
            if loc['end'] > loc['start']:  # assuming moving right
                state = 3  # start-end
            else:
                state = 4  # end-start
        elif loc['end'] > loc['start']:
            state = 5  # whole tree
        else:
            state = 6  # end - start
    elif loc['end'] is not None:
        state = 2
    else:
        state = 1

    return state


def slice_to_csv(data_file, output_path, resize_factor=3, h=2048, w=1536):
    size = int(h // resize_factor)
    r = min(size / h, size / w)

    with open(data_file, 'r') as f:
        loaded_data = json.load(f)
    data = {}
    for k, v in loaded_data.items():
        data[int(k)] = v
    data = collections.OrderedDict(sorted(data.items()))

    trees_data = parse_data_to_trees(data)
    return trees_data


def change_slice_format(frame_slice, factor=2048/1536):
    """
    changes a slicing foramt from old one with rotation bug to new one
    :param frame_slice: a subdict per frame
    :param factor: the resize factor
    :return: resized dict
    """
    if not isinstance(frame_slice["start"], type(None)):
        frame_slice["start"] = frame_slice["start"]*factor
    if not isinstance(frame_slice["end"], type(None)):
        frame_slice["end"] = frame_slice["end"]*factor
    return frame_slice


def fix_slices(customer_folder = "/media/fruitspec-lab/cam175/DEWAGD"):
    """
    iterates over the subfolders of customer_folder and all the slicing jasons jsons in them and reformats them
    :param customer_folder: customer folder path (could be any master folder)
    :return: None
    """
    for root, dirs, files in os.walk(customer_folder):
        for file in files:
            if file.endswith('.json') and "slice_data" in file and "ZED" not in file:
                data_file = os.path.join(root, file)
                with open(data_file) as json_file:
                    data = json.load(json_file)
                try:
                    data = {int(key): change_slice_format(item) for key, item in data.items()}
                except:
                    print("problem with ", json_file)
                with open(data_file, 'w') as json_file:
                    json.dump(data, json_file)


if __name__ == "__main__":
    fp = '/media/fruitspec-lab/Expansion/Tomato_20_deg/pre/TOMATO_14022320_deg_pre_b_20_ZED_1.svo'
    output_path = '/media/fruitspec-lab/Expansion/Tomato_20_deg/pre'
    # validate_output_path(output_path)
    manual_slicer(fp, output_path, rotate=2)
