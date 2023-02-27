import os
import numpy as np
import cv2
import json
import pandas as pd
import collections
from vision.visualization.drawer import draw_highlighted_test
from vision.tools.image_stitching import get_fine_keypoints, resize_img, get_fine_translation
from vision.tools.video_wrapper import video_wrapper
from vision.misc.help_func import validate_output_path
from vision.visualization.drawer import draw_rectangle


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
            params["data"][params['index']]['end'] = max(x - 10, 0)
            #params["data"][params['index']]['end'] = min(x + 10, int(params['width'] // params['resize_factor']))
        if flags == cv2.EVENT_FLAG_ALTKEY + 1: # or flags == cv2.EVENT_FLAG_ALTKEY + 1:  # second part is due to bug of cv2
            params['left_clusters'] = True
            count = params["data"][params['index']]['left_clusters']['count']
            params["data"][params['index']]['left_clusters'][count] = [x, y, x, y]
        else:
            params["data"][params['index']]['start'] = x

    if event == cv2.EVENT_RBUTTONDOWN:
        if flags == cv2.EVENT_FLAG_CTRLKEY or flags == cv2.EVENT_FLAG_CTRLKEY + 2:  # second part is due to bug of cv2
            params["data"][params['index']]['start'] = x
            params["data"][params['index']]['end'] = max(x - 10, 0)
        elif flags == cv2.EVENT_FLAG_ALTKEY + 2:
            params['right_clusters'] = True
            count = params["data"][params['index']]['right_clusters']['count']
            params["data"][params['index']]['right_clusters'][count] = [x, y, x, y]
        else:
            params["data"][params['index']]['end'] = x

    if event == cv2.EVENT_LBUTTONUP and flags == cv2.EVENT_FLAG_ALTKEY + 1:
        params['left_clusters'] = False
        count = params["data"][params['index']]['left_clusters']['count']
        if count in list(params["data"][params['index']]['right_clusters'].keys()):
            params["data"][params['index']]['left_clusters'][count][2] = x
            params["data"][params['index']]['left_clusters'][count][3] = y
            params["data"][params['index']]['left_clusters']['count'] += 1
    if event == cv2.EVENT_RBUTTONUP and flags == cv2.EVENT_FLAG_ALTKEY + 2:
        params['right_clusters'] = False
        count = params["data"][params['index']]['right_clusters']['count']
        if count in list(params["data"][params['index']]['right_clusters'].keys()):
            params["data"][params['index']]['right_clusters'][count][2] = x
            params["data"][params['index']]['right_clusters'][count][3] = y
            params["data"][params['index']]['right_clusters']['count'] += 1

    if event == cv2.EVENT_MOUSEMOVE and (flags == cv2.EVENT_FLAG_ALTKEY + 1 or flags == cv2.EVENT_FLAG_ALTKEY + 2):
        count = params["data"][params['index']]['right_clusters']['count']

        if params['left_clusters']:
            count = params["data"][params['index']]['left_clusters']['count']
            if count in list(params["data"][params['index']]['left_clusters'].keys()):
                count = params["data"][params['index']]['left_clusters']['count']
                params["data"][params['index']]['left_clusters'][count][2] = x
                params["data"][params['index']]['left_clusters'][count][3] = y
        if params['right_clusters']:
            count = params["data"][params['index']]['right_clusters']['count']
            if count in list(params["data"][params['index']]['right_clusters'].keys()):
                count = params["data"][params['index']]['right_clusters']['count']
                params["data"][params['index']]['right_clusters'][count][2] = x
                params["data"][params['index']]['right_clusters'][count][3] = y

    frame = print_lines(params)
    frame = print_text(frame, params)
    frame = print_rectangles(frame, params)
    cv2.imshow(params['headline'], frame)


def print_lines(params):
    frame = params['frame'].copy()
    y = int(params['height'] // params['resize_factor'])

    if params['data'][params['index']]['start'] is not None:
        x = params['data'][params['index']]['start']
        frame = cv2.line(frame, (x, 0), (x, y), (255, 0, 0), 2)
    if params['data'][params['index']]['end'] is not None:
        x = params['data'][params['index']]['end']
        frame = cv2.line(frame, (x, 0), (x, y), (255, 0, 255), 2)

    return frame


def print_rectangles(frame, params):
    left_clusters = params['data'][params['index']]['left_clusters']
    right_clusters = params['data'][params['index']]['right_clusters']

    for key, values in left_clusters.items():
        if key != 'count':
            start_point = (values[0], values[1])
            end_point = (values[2], values[3])
            frame = draw_rectangle(frame, start_point, end_point, (255, 0, 255),)

    for key, values in right_clusters.items():
        if key != 'count':
            start_point = (values[0], values[1])
            end_point = (values[2], values[3])
            frame = draw_rectangle(frame, start_point, end_point, (255, 0, 0),)

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
    params['find_translation'] = False

    if k == 122:
        index = max(index - 1, 0)

    if k == 99:
        index += 1

    elif k == 115:
        index += 100

    elif k == 120:
        index = max(index - 100, 0)

    elif k == 32:
        index = index
        params["data"][params['index']]['start'] = None
        params["data"][params['index']]['end'] = None
        params["data"][params['index']]['left_clusters'] = {'count':0}
        params["data"][params['index']]['right_clusters'] = {'count': 0}
        params['find_translation'] = False

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
              'find_translation': False,
              'right_clusters': False,
              'left_clusters': False}

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
        frame = print_rectangles(frame, params)

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
        #params['data'][params['index']] = {'start': None, 'end': None}
        params['data'][params['index']] = {'start': None,
                                           'end': None,
                                           "left_clusters": {'count': 0},
                                           "right_clusters": {'count': 0}}

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
    size = int(h // resize_factor)
    r = min(size / h, size / w)
    #r = 1

    with open(data_file, 'r') as f:
        loaded_data = json.load(f)
    data = {}
    for k, v in loaded_data.items():
        data[int(k)] = v
    data = collections.OrderedDict(sorted(data.items()))

    trees_data, border_data = parse_data_to_trees(data)
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

    border_df = pd.DataFrame(data=border_data, columns=['frame_id', 'tree_id', 'x1', 'y1', 'x2', 'y2'])
    border_df['x1'] /= r
    border_df['y1'] /= r
    border_df['x2'] /= r
    border_df['y2'] /= r
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
        return pd.concat(df_all, axis=0), border_df


def parse_data_to_trees(data):
    tree_id = 0
    # started_tree = False
    # start_and_end_tree = False

    last_state = 0
    trees_data = {}
    border_data = []
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
                    raise ValueError(f"Got tree closing before tree opening frame: {frame_id}")
            elif state == 3:
                raise ValueError(f"Got tree closing before tree opening frame: {frame_id}")
            elif state == 4:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                # start new tree
                trees_data[tree_id + 1] = [{'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': loc['start'], 'end': -1}]
                border_data = update_border_data(border_data, loc, frame_id, tree_id)
            elif state == 5:
                tree_id += 1
                trees_data[tree_id] = [
                    {'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': loc['end']}]
            elif state == 6:
                raise ValueError(f"Got tree closing before tree opening frame: {frame_id}")


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
                border_data = update_border_data(border_data, loc, frame_id, tree_id)
            elif state == 5:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': loc['end']})
            elif state == 6:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                trees_data[tree_id + 1] = [{'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': loc['start'], 'end': -1}]
                border_data = update_border_data(border_data, loc, frame_id, tree_id)

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
                raise ValueError(f"Got wrong state 3 after state 2 frame: {frame_id}")
            elif state == 4:
                trees_data[tree_id].append(
                    {'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                # start new tree
                trees_data[tree_id + 1] = [
                    {'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': loc['start'], 'end': -1}]
                border_data = update_border_data(border_data, loc, frame_id, tree_id)
            elif state == 5:
                tree_id += 1
                trees_data[tree_id] = [
                    {'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': loc['end']}]
            elif state == 6:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                trees_data[tree_id + 1] = [{'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': loc['start'], 'end': -1}]
                border_data = update_border_data(border_data, loc, frame_id, tree_id)

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
                    raise ValueError(f"Got tree closing before tree opening frame: {frame_id}")
            elif state == 3:  # start - end
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': -1})
                # add to next tree
                trees_data[tree_id + 1].append({'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': -1, 'end': loc['end']})
            elif state == 4:  # end - start
                raise ValueError(f"Got wrong state 4 after state 3 frame: {frame_id}")
            elif state == 5:
                tree_id += 1
                trees_data[tree_id] = [
                    {'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': loc['end']}]
            elif state == 6:
                border_data = update_border_data(border_data, loc, frame_id, tree_id)
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                tree_id += 1
                trees_data[tree_id] = [{'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': -1}]


        # end-start
        elif last_state == 4:
            if state == 0:
                tree_id += 1
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': -1})
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
                raise ValueError(f"Got tree closing before tree opening frame: {frame_id}")
            elif state == 4:  # end - start
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                # add to next tree
                trees_data[tree_id + 1].append(
                    {'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': loc['start'], 'end': -1})
                border_data = update_border_data(border_data, loc, frame_id, tree_id)
            elif state == 5:
                tree_id += 1
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': loc['end']})
            elif state == 6:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                trees_data[tree_id + 1].append({'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': loc['start'], 'end': -1})
                border_data = update_border_data(border_data, loc, frame_id, tree_id)

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
                    raise ValueError(f"Got tree closing before tree opening frame: {frame_id}")
            elif state == 3:  # start - end
                raise ValueError(f"Got tree closing before tree opening frame: {frame_id}")
            elif state == 4:  # end - start
                trees_data[tree_id].append(
                    {'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                # add to next tree
                trees_data[tree_id + 1] = [{'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': -1}]
                border_data = update_border_data(border_data, loc, frame_id, tree_id)
            elif state == 5:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': loc['end']})
            elif state == 6:
                trees_data[tree_id].append(
                    {'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                trees_data[tree_id + 1] = [{'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': -1}]
                border_data = update_border_data(border_data, loc, frame_id, tree_id)

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
                    raise ValueError(f"Got tree closing before tree opening frame: {frame_id}")
            elif state == 3:  # start - end
                raise ValueError(f"Got tree closing before tree opening frame: {frame_id}")
            elif state == 4:  # end - start
                trees_data[tree_id].append(
                    {'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                # add to next tree
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': -1})
                border_data = update_border_data(border_data, loc, frame_id, tree_id)
            elif state == 5:
                trees_data[tree_id].append({'frame_id': frame_id, 'tree_id': tree_id, 'start': loc['start'], 'end': loc['end']})
            elif state == 6:
                trees_data[tree_id].append(
                    {'frame_id': frame_id, 'tree_id': tree_id, 'start': -1, 'end': loc['end']})
                trees_data[tree_id + 1].append({'frame_id': frame_id, 'tree_id': tree_id + 1, 'start': loc['start'], 'end': -1})
                border_data = update_border_data(border_data, loc, frame_id, tree_id)
        last_state = state

    return trees_data, border_data


def update_border_data(border_data, loc, frame_id, tree_id):
    loc_keys = list(loc.keys())
    if 'left_clusters' not in loc_keys:
        return border_data
    if len(loc['left_clusters']) > 1:  # not only count value
        for k, v in loc['left_clusters'].items():
            if k != 'count':
                x1 = min(v[0], v[2])
                x2 = max(v[0], v[2])
                y1 = min(v[1], v[3])
                y2 = max(v[1], v[3])
                border_data.append(
                    {'frame_id': frame_id, 'tree_id': tree_id, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    if len(loc['right_clusters']) > 1:  # not only count value
        for k, v in loc['right_clusters'].items():
            if k != 'count':
                border_data.append(
                    {'frame_id': frame_id, 'tree_id': tree_id + 1, 'x1': v[0], 'y1': v[1], 'x2': v[2], 'y2': v[3]})
    return border_data



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
        state = 2  # end
    else:
        state = 1  # start

    return state


if __name__ == "__main__":
    fp = '/home/fruitspec-lab-3/FruitSpec/Data/Syngenta/tomato/230123/post/10/ZED_1.svo'
    output_path = '/home/fruitspec-lab-3/FruitSpec/Sandbox/Syngenta/testing'
    validate_output_path(output_path)
    manual_slicer(fp, output_path, rotate=2)

    data_file = "/home/fruitspec-lab-3/FruitSpec/Sandbox/Syngenta/testing/ZED_1_slice_data.json"
    #slice_to_trees(data_file, None, None, h=1920, w=1080)


