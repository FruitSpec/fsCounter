import os
import numpy as np
import cv2
import json

from vision.tools.video_wrapper import video_wrapper
from vision.visualization.drawer import draw_highlighted_test, get_color
from vision.tools.image_stitching import resize_img
from vision.misc.help_func import validate_output_path
from vision.tools.sensors_alignment import align_sensors


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
        params = on_click(flags, x, y, params)
        frame = print_dot(params)
        cv2.imshow(params['headline'], frame)
    if event == cv2.EVENT_RBUTTONDOWN:
        params = on_click(flags, x, y, params)
        frame = print_dot(params)
        cv2.imshow(params['headline'], frame)

def on_click(flags, x, y, params):
    sample_id = params['sample']
    if x < params['canvas_half_width']:
        if flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_RBUTTONDOWN or flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_LBUTTONDOWN:  # second part is due to bug of cv2

            params["data"][sample_id]['left'] = {'x': x - params['zed_offset'], 'y': y}
    else:
        if flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_RBUTTONDOWN  or flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_LBUTTONDOWN:  # second part is due to bug of cv2
            params["data"][sample_id]['right'] = {'x': x - params['jai_offset'], 'y': y}

    return params


def print_dot(params):

    frame = params['frame'].copy()
    for id_, sample in params['data'].items():
        left_x = sample['left']['x']
        left_y = sample['left']['y']

        right_x = sample['right']['x']
        right_y = sample['right']['y']

        if left_x is not None and left_y is not None:
            frame = cv2.circle(frame, (left_x + params['zed_offset'], left_y), 3, get_color(id_), 2)
        if right_x is not None and right_y is not None:
            frame = cv2.circle(frame, (right_x + params['jai_offset'], right_y), 3, get_color(id_), 2)

    return frame


def print_text(frame, params, text=None):
    if text is None:
        text = f'Frame {params["index"]}'
    orange = (48, 130, 245)
    white = (255, 255, 255)

    size = params['size']
    text_width = 150
    start_point = (10, size - 10)
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

    elif k == 115:
        index = max(index + 100, 0)

    elif k == 120:
        index = max(index - 100, 0)

    elif k == 32:
        index = index
        params['sample'] += 1
        params = init_data_sample(params)


    elif k == 114:  # r key to remove sample
        samples_ids = list(params['data'].keys())
        if len(samples_ids) >= 2:
            del params['data'][samples_ids[-2]]
        else:
            del params['data'][samples_ids[0]]


    else:
        index = max(index + 1, 0)


    params['index'] = index
    return params


def manual_calibration(zed_filepath, jai_filepath, output_path, data=None, zed_rotate=2, jai_rotate=1, index=0, draw_start=None, draw_end=None, size=960):
    """
    this is where the magic happens, palys the video
    """
    if data is None:
        data = {}
    params = {"zed_filepath": zed_filepath,
              "jai_filepath": jai_filepath,
              "output_path": output_path,
              "data": data,
              "zed_rotate": zed_rotate,
              "jai_rotate": jai_rotate,
              "index": index,
              "draw_start": draw_start,
              "draw_end": draw_end,
              'size': size,
              'last_kp_des': None,
              'find_translation': False,
              'sample': 0}

    headline = f'clip {params["zed_filepath"]}'
    params['headline'] = headline
    cv2.namedWindow(headline, cv2.WINDOW_GUI_NORMAL)

    zed_cam = video_wrapper(zed_filepath, zed_rotate)
    jai_cam = video_wrapper(jai_filepath, jai_rotate)
    number_of_frames = jai_cam.get_number_of_frames()

    tx = []
    ty = []
    # Read until video is completed
    while True:
        # Capture frame-by-frame
        print(params["index"])
        zed_cam.grab(params["index"])
        _, zed_frame = zed_cam.get_frame()
        #jai_cam.grab(params["index"])
        #_, jai_frame = jai_cam.get_frame()
        #ret1, zed_frame = zed_cam.get_frame(params["index"])
        ret, jai_frame = jai_cam.get_frame(params["index"])
        if not ret and not zed_cam.res:  # couldn't get frames
        #if not ret and not ret1:  # couldn't get frames
            break
        zed_orig = zed_frame.copy()
        jai_orig = jai_frame.copy()
        # preprocess: resize and rotate if needed
        jai_frame, params = preprocess_frame(jai_frame, params)
        zed_frame, params = preprocess_frame(zed_frame, params)

        #jai_frame = cv2.cvtColor(jai_frame, cv2.COLOR_RGB2GRAY)
        #zed_frame = cv2.cvtColor(zed_frame, cv2.COLOR_RGB2GRAY)

        canvas_half_width = (max(zed_frame.shape[1], jai_frame.shape[1]) + 50)
        if len(jai_frame.shape) > 2:
            canvas = np.zeros((size, canvas_half_width * 2, 3)).astype(np.uint8)
            canvas[:zed_frame.shape[0], 25:zed_frame.shape[1]+25, :] = zed_frame
            canvas[:jai_frame.shape[0], canvas_half_width + 25: canvas_half_width + 25 + jai_frame.shape[1], :] = jai_frame
        else:
            canvas = np.zeros((size, canvas_half_width * 2)).astype(np.uint8)
            canvas[:, 25:zed_frame.shape[1] + 25] = zed_frame
            canvas[:, canvas_half_width + 25: canvas_half_width + 25 + jai_frame.shape[1]] = jai_frame
        params['frame'] = canvas
        params['canvas_half_width'] = canvas_half_width
        params['zed_offset'] = 25
        params['jai_offset'] = canvas_half_width + 25



        params = init_data_sample(params)
        # #params = get_updated_location_in_index(frame, params)
        #
        canvas = print_dot(params)
        #frame = print_text(frame, params)

        cv2.imshow(headline, canvas)

        cv2.setMouseCallback(headline, mouse_callback, params)
        k = cv2.waitKey()
        # Press Q on keyboard to  exit
        if cv2.waitKey(k) & 0xFF == ord('q'):
            write_json(params)
            break
        if k==114:
            x1, y1, x2, y2 = align_sensors(zed_orig, jai_orig)
            tx.append(x1)
            ty.append(y1)

            zed_c = zed_orig[int(y1): int(y2), int(x1): int(x2)]
            jai_c, params = preprocess_frame(jai_orig, params)
            zed_c, params = preprocess_frame(zed_c, params)

            canvas_half_width = (max(zed_c.shape[1], jai_c.shape[1]) + 50)
            canvas1 = np.zeros((size, canvas_half_width * 2, 3)).astype(np.uint8)
            canvas1[:zed_c.shape[0], 25:zed_c.shape[1] + 25, :] = zed_c
            canvas1[:jai_c.shape[0], canvas_half_width + 25: canvas_half_width + 25 + jai_c.shape[1], :] = jai_c
            cv2.imshow(headline, canvas1)
            k = cv2.waitKey()
        params = update_index(k, params)
        write_json(params)

    # When everything done, release the video capture object
    zed_cam.close()
    jai_cam.close()

    # Closes all the frames
    cv2.destroyAllWindows()

def preprocess_frame(frame, params):

    frame, r = resize_img(frame, params['size'])
    params['frame'] = frame.copy()
    params['r'] = r

    return frame, params

def init_data_sample(params):
    sample = params['sample']
    sample_ids = list(params['data'].keys())
    if sample not in sample_ids:
        params['data'][sample] = {'left': {'x': None, 'y': None}, 'right': {'x': None, 'y': None}}

    return params


def write_json(params):

    output_file_name = os.path.join(params['output_path'], f'calibration_data.json')
    with open(output_file_name, 'w') as fp:
        json.dump(params['data'], fp)




if __name__ == "__main__":
    #zed_fp = '/home/yotam/FruitSpec/Sandbox/sync_test/ZED_1.svo'
    #jai_fp = '/home/yotam/FruitSpec/Sandbox/sync_test/Result_RGB_1.mkv'
    #zed_fp = "/home/yotam/FruitSpec/Sandbox/JAIZED EXPERIMENT BITRATE/FSI_b 51200.mkv"
    jai_fp = "/home/yotam/FruitSpec/Data/Scan_3011/wetransfer_new-scan_2022-11-30_1639/r2in/Result_RGB_1.mkv"
    #zed_fp = "/home/yotam/FruitSpec/Sandbox/JAIZED EXPERIMENT BITRATE/ZED_b 20K.svo"
    zed_fp = "/home/yotam/FruitSpec/Data/Scan_3011/wetransfer_new-scan_2022-11-30_1639/r2in/ZED_1.svo"

    output_path = '/home/yotam/FruitSpec/Sandbox/Syngenta/pepper/dual/'
    validate_output_path(output_path)
    manual_calibration(zed_fp, jai_fp, output_path, zed_rotate=2, jai_rotate=1)

