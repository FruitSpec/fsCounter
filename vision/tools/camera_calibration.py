import os
import numpy as np
import cv2
import json

from vision.tools.video_wrapper import video_wrapper
from vision.visualization.drawer import draw_highlighted_test, get_color
from vision.tools.image_stitching import resize_img
from vision.misc.help_func import validate_output_path
from image_stitching import plot_2_imgs
from sensors_alignment import SensorAligner

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

            params["data"][sample_id]['left'] = {'x': (x - params['zed_offset']), 'y': y}
    else:
        if flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_RBUTTONDOWN or flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_LBUTTONDOWN:  # second part is due to bug of cv2
            params["data"][sample_id]['right'] = {'x': x - params['jai_offset'], 'y': y}

    return params


def return_dot_to_original_size(dots, r_jai, r_zed):
    return {key: {"left": {"x": int(value["left"]["x"]/r_zed), "y": int(value["left"]["y"]/r_zed)},
                  "right": {"x": int(value["right"]["x"]/r_jai), "y": int(value["right"]["y"]/r_jai)}}
                        for key, value in dots.items()
                        if np.all(np.array([value["left"]["x"], value["left"]["y"],
                                            value["right"]["x"], value["right"]["y"]]) != None)}


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
        index = max(index + 10, 0)

    elif k == 120:
        index = max(index - 10, 0)

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


def manual_calibration(zed_filepath, jai_filepath, output_path, data=None, zed_rotate=2, jai_rotate=1,
                       index=0, draw_start=None, draw_end=None, size=1440, zed_shift=0, jai_83=False):
    """
    this is where the magic happens, plays the video
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
              'sample': len(data)}

    headline = f'clip {params["zed_filepath"]}'
    params['headline'] = headline
    cv2.namedWindow(headline, cv2.WINDOW_GUI_NORMAL)

    zed_cam = video_wrapper(zed_filepath, zed_rotate)
    jai_cam = video_wrapper(jai_filepath, jai_rotate)
    number_of_frames = jai_cam.get_number_of_frames()

    # Read until video is completed
    while True:
        # Capture frame-by-frame
        print(params["index"])
        zed_cam.grab(max(params["index"] + zed_shift,1))
        _, zed_frame = zed_cam.get_frame()
        ret, jai_frame = jai_cam.get_frame(params["index"])
        if jai_83:
            jai_frame = jai_frame[0:-180, 265: 1285, :]
        if not ret and not zed_cam.res:  # couldn't get frames
            break

        # preprocess: resize and rotate if needed
        jai_frame, params = preprocess_frame(jai_frame, params, r_suffix="jai")
        zed_frame, params = preprocess_frame(zed_frame, params, r_suffix="zed")

        canvas_half_width = (max(zed_frame.shape[1], jai_frame.shape[1]) + 50)
        canvas = np.zeros((size, canvas_half_width * 2, 3)).astype(np.uint8)
        canvas[:, 25:zed_frame.shape[1]+25, :] = zed_frame
        canvas[:, canvas_half_width + 25: canvas_half_width + 25 + jai_frame.shape[1], :] = jai_frame
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
        if k == ord('q'):
            write_json(params)
            break
        params = update_index(k, params)
        write_json(params)

    params['data'] = return_dot_to_original_size(params['data'], params['r_jai'], params['r_zed'])
    write_json(params, real_coords=True)
    write_coords(params, zed_cam.get_frame()[1], jai_cam.get_frame(params["index"])[1])


    # When everything done, release the video capture object
    zed_cam.close()
    jai_cam.close()

    # Closes all the frames
    cv2.destroyAllWindows()


def preprocess_frame(frame, params, r_suffix):
    frame, r = resize_img(frame, params['size'])
    params['frame'] = frame.copy()
    params[f'r_{r_suffix}'] = r

    return frame, params


def init_data_sample(params):
    sample = params['sample']
    sample_ids = list(params['data'].keys())
    if sample not in sample_ids:
        params['data'][sample] = {'left': {'x': None, 'y': None}, 'right': {'x': None, 'y': None}}

    return params


def write_json(params, real_coords=False):
    output_file_name = os.path.join(params['output_path'], f'calibration_data.json')
    if real_coords:
        output_file_name = os.path.join(params['output_path'], f'calibration_data_real.json')
    with open(output_file_name, 'w') as fp:
        json.dump(params['data'], fp)


def check_correctness(zed_path, jai_path, data, jai_83=False):
    zed = cv2.imread(zed_path)[:, :, ::-1]
    jai = cv2.imread(jai_path)[:, :, ::-1]
    zed = zed.astype(np.uint8)
    jai = jai.astype(np.uint8)
    if jai_83:
        jai = jai[0:-180, 265: 1285, :]
    for id_, sample in data.items():
        left_x = sample['left']['x']
        left_y = sample['left']['y']

        right_x = sample['right']['x']
        right_y = sample['right']['y']

        if left_x is not None and left_y is not None:
            zed = cv2.circle(zed, (left_x, left_y), 5, get_color(id_), 5)
        if left_x is not None and left_y is not None:
            jai = cv2.circle(jai, (right_x, right_y), 5, get_color(id_), 5)
    plot_2_imgs(zed, jai)
    return zed, jai

def get_scale_translation(M):
    tx = int(np.round(M[0, 2]))
    ty = int(np.round(M[1, 2]))
    sx = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
    sy = np.sqrt(M[1, 0] ** 2 + M[1, 1] ** 2)
    return tx, ty, sx, sy


def data_to_transformation(data):
    src = []
    dst = []
    for key in data.keys():
        left = data[key]["left"]
        right = data[key]["right"]
        if left["x"] == None or left["y"] == None or right["x"] == None or right["y"] == None:
            continue
        dst.append([left["x"], left["y"]])
        src.append([right["x"], right["y"]])
    #M = cv2.getAffineTransform(np.array(src, dtype=np.float32)[[0, 2, 4]], np.array(dst, dtype=np.float32)[[0, 2, 4]])
    M = cv2.estimateAffine2D(np.array(src), np.array(dst), ransacReprojThreshold=100
                             , maxIters=5000, confidence=0.999, refineIters=10)[0]
    return M


def data_to_coords(zed, jai, data):
    M = data_to_transformation(data)
    tx, ty, sx, sy = get_scale_translation(M)
    x1, y1, x2, y2 = SensorAligner.get_coordinates_in_zed(zed, jai, tx, ty, sx, sy)
    return x1, y1, x2, y2


def write_coords(params, zed_frame, jai_frame):
    x1, y1, x2, y2 = data_to_coords(zed_frame, jai_frame, params['data'])
    output_file_name = os.path.join(params['output_path'], f'coords.json')
    with open(output_file_name, 'w') as fp:
        json.dump(dict(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)), fp)


if __name__ == "__main__":
    jai_im_path = "/media/fruitspec-lab/TEMP SSD/83_alignemt_calibration/jai_frame_67.jpg"
    zed_im_path = "/media/fruitspec-lab/TEMP SSD/83_alignemt_calibration/zed_frame_67.jpg"
    row = '/media/fruitspec-lab/TEMP SSD/Tomato/Size/PRE/90Rep1'
    side = 1
    zed_fp = os.path.join(row, f'ZED_{side}.svo')
    jai_fp = os.path.join(row, f'Result_FSI_{side}.mkv')
    output_path = row
    json_path = "/media/fruitspec-lab/TEMP SSD/Tomato/FCountDeleaf/window_trail/10_5_post/Result_FSI_1.mkv"
    json_path_real = "/media/fruitspec-lab/TEMP SSD/TOMATO_SA_BYER_COLOR/pre/1/calibration_data_real.json"
    jai_83 = False
    # #
    # with open(json_path) as json_file:
    #     data = json.load(json_file)
    # data = {int(key): value for key, value in data.items()}
    manual_calibration(zed_fp, jai_fp, output_path, zed_rotate=2, jai_rotate=1, index=65, zed_shift=0, jai_83=jai_83)

    with open(json_path_real) as json_file_real:
        real_data = json.load(json_file_real)
    real_data = {int(key): value for key, value in real_data.items()}

    zed_w_chess, jai_w_chess = check_correctness(zed_im_path, jai_im_path, real_data, jai_83=jai_83)
    validate_output_path(output_path)
    zed = cv2.imread(zed_im_path)[:, :, ::-1]
    jai = cv2.imread(jai_im_path)[:, :, ::-1]
    x1, y1, x2, y2 = data_to_coords(zed, jai, real_data)
    M = data_to_transformation(real_data)
    zed_wrapped = cv2.warpAffine(zed, M, zed.shape[:2][::-1])
    new_points = [M @ np.array([value["left"]["x"], value["left"]["y"], 1]).T for key, value in real_data.items()]
    right_points = [[int(value["right"]["x"]), int(value["right"]["y"])] for key, value in real_data.items()]
    wraped_data = {i: {"left": {"x": int(new_points[i][0]), "y": int(new_points[i][1])},
                       "right": {"x": right_points[i][0], "y": right_points[i][1]}} for i in range(len(new_points))}
    x1_wr, y1_wr, x2_wr, y2_wr = data_to_coords(zed_wrapped, jai, wraped_data)
    zed_wrapped_wp = zed_wrapped.copy()
    for i, point in enumerate(new_points):
        left_x, left_y = point.astype(int)
        zed_wrapped_wp = cv2.circle(zed_wrapped_wp, (left_x, left_y), 5, get_color(i), 5)
    plot_2_imgs(zed, zed_wrapped_wp)

    plot_2_imgs(cv2.resize(zed[y1:y2, x1:x2], (600, 900)), cv2.resize(jai, (600, 900)))
    plot_2_imgs(cv2.resize(zed_wrapped[y1_wr:y2_wr, x1_wr:x2_wr], (600, 900)), cv2.resize(jai, (600, 900)))
    # plot_2_imgs(cv2.resize(cut_black(cv2.warpAffine(zed, M, zed.shape[:2][::-1])), (600, 900)),
    #             cv2.resize(jai, (600, 900)))
    # plot_2_imgs(cut_black(cv2.warpAffine(zed, M, zed.shape[:2][::-1])), zed)
    #manual_calibration(zed_fp, jai_fp, output_path, zed_rotate=2, jai_rotate=1)

