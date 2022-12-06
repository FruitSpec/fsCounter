import os
import numpy as np
import cv2
import json

from vision.visualization.drawer import draw_highlighted_test
from vision.tools.image_stitching import find_keypoints, get_fine_keypoints, find_translation, resize_img, get_fine_translation


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
        x = params['data'][params['index']]['start']
        frame = cv2.line(frame, (x, 0), (x, y), (255, 0, 0), 2)
    if params['data'][params['index']]['end'] is not None:
        x = params['data'][params['index']]['end']
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


def manual_slicer(filepath, output_path, data=None, rotate=False, index=0, draw_start=None, draw_end=None, resize_factor=3):
    """
    this is where the magic happens, palys the video
    """
    if data is None:
        data = {}
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

    cap = cv2.VideoCapture(filepath)
    number_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    params['height'] = height
    params['width'] = width

    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        print(params["index"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, params["index"])
        ret, frame = cap.read()
        if ret == True:

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
                # write_json(params)
                break
            params = update_index(k, params)
            # write_json(params)
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def preprocess_frame(frame, params):
    height = int(params['height'] // params['resize_factor'])

    frame, r = resize_img(frame, height)
    if params['rotate']:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
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
    clip_name = temp_str('/')[-1]

    output_file_name = os.path.join(params['output_path'], f'{clip_name}_silces_data.json')
    with open(output_file_name, 'w') as fp:
        json.dump(params['data'], fp)


if __name__ == "__main__":
    fp = "/home/yotam/Downloads/Result_FSI_1.mkv"#'/media/yotam/Extreme Pro/JAIZED_CaraCara_151122/R_4/Result_FSI_4.mkv'
    output_path = '/home/yotam/FruitSpec/Sandbox/detection_caracara'
    manual_slicer(fp, output_path, rotate=True)
