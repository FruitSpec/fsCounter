import numpy as np
import pyzed.sl as sl
import cv2

rotatred_vid = True
if not rotatred_vid:
    resizing_factor_x = 1.0714
    resizing_factor_y = 1.0714
    x_pic_rang = 1920
else:
    x_pic_rang = 1080
    y_pic_rang = 1920
    resizing_factor_x = x_pic_rang/1792
    resizing_factor_y = y_pic_rang / 1008


def run(filepath, cam=None, runtime=None, size=(1792//2, 1008//2), mode="depth", index=0,
              x_0=None, y_0=None, x_1=None, y_1=None, left_click=False, group_1=None, group_2=None, groups = None):
    """
    this is where the magic happens, palys the video
    :param filepath:
    :param cam:
    :param runtime:
    :param size:
    :param mode:
    :param index:
    :param x_0:
    :param y_0:
    :param x_1:
    :param y_1:
    :param left_click:
    :return:
    """
    params = {"filepath": filepath, "cam": cam, "runtime": runtime, "size": size, "mode": mode, "index": index,
              "x_0": x_0, "y_0": y_0, "x_1": x_1, "y_1": y_1,"left_click": left_click, "group_1":  group_1, "group_2":  group_2}
    if isinstance(cam, type(None)):
        cam, runtime = init_cam(filepath)
        params["cam"] = cam
        params["runtime"] = runtime
    cv2.namedWindow('img')
    cv2.setMouseCallback('img', draw_rect, params)
    number_of_frames = sl.Camera.get_svo_number_of_frames(cam)
    while True:
        print(f'reading frame number: {index}')
        cam.set_svo_position(index)
        res = cam.grab(runtime)
        cam_run_p = cam.get_init_parameters()
        if res == sl.ERROR_CODE.SUCCESS and index < number_of_frames:
            mat = sl.Mat()
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            img = mat.get_data()[:, :, : 3]

            depth_map = sl.Mat()
            cam.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            depth_img = depth_map.get_data()
            depth_img = (cam_run_p.depth_maximum_distance - np.clip(depth_img, 0, cam_run_p.depth_maximum_distance)) * 255 / cam_run_p.depth_maximum_distance
            bool_mask = np.where(np.isnan(depth_img), True, False)
            depth_img[bool_mask] = 0
            #if remove_high_blues:
            if True:
                mat = sl.Mat()
                cam.retrieve_image(mat, sl.VIEW.LEFT)
                depth_img[mat.get_data()[:, :, 0] > 190] = 0
            depth_img = cv2.medianBlur(depth_img, 5)

            if len(depth_img.shape) < 3:
                depth_img = np.dstack([depth_img, depth_img, depth_img])
            img_display = make_one_image(depth_img, img, size)

            cv2.imshow('img', img_display)
            k = cv2.waitKey()
            index, cont = update_index(k, index)
            params["index"] = index
            if cont:
                continue
        else:
            break
    cam.close()

def init_cam(filepath, depth_minimum=0.5, depth_maximum=10):
    """
    inits camera and runtime
    :param filepath: path to svo file
    :return:
    """
    input_type = sl.InputType()
    input_type.set_from_svo_file(filepath)
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    #init_params.depth_mode = sl.DEPTH_MODE.QUALITY
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = depth_minimum
    init_params.depth_maximum_distance = depth_maximum
    init_params.depth_stabilization = True
    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 100
    #runtime.sensing_mode = sl.SENSING_MODE.STANDARD
    runtime.sensing_mode = sl.SENSING_MODE.FILL
    cam = sl.Camera()
    status = cam.open(init_params)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    cam.enable_positional_tracking(positional_tracking_parameters)
    detection_parameters = sl.ObjectDetectionParameters()
    detection_parameters.detection_model = sl.DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    detection_parameters.enable_tracking = True
    detection_parameters.enable_mask_output = True
    cam.enable_object_detection(detection_parameters)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
    return cam, runtime


def make_one_image(depth_img, img, size, rotatred_vid=True):
    """
    combines depth image and rgb image
    :param depth_img: depth_img
    :param img: rgb image
    :param size: size rgb/depth image
    :return: a combined image of rgb and depth side by side
    """
    img_display = np.zeros((int(size[1]), size[0] * 2, 3), dtype='uint8')
    if rotatred_vid:
        depth_img = cv2.rotate(depth_img, cv2.cv2.ROTATE_90_CLOCKWISE)
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)

    im_depth_display = cv2.resize(depth_img, size)
    im_left_display = cv2.resize(img, size)
    img_display[:, :size[0]] = im_left_display
    img_display[:, size[0]:] = im_depth_display
    return img_display

def update_index(k, index):
    """
    :param k: current key value
    :param index: current index
    :return: next index
    """
    cont = False
    if k == 83 or k ==100:
        index = max(index - 1, 0)
        cont = True
    if k == 82 or k == 119:
        index = max(index + 100, 0)
        cont = True
    if k == 84 or k == 115:
        index = max(index - 100, 0)
        cont = True
    if k == 32:
        global x_0, y_0, x_1, y_1
        x_0, y_0, x_1, y_1 = None, None, None, None
    return index, cont

def draw_rect(event, x, y, flags, params):
    """
    draws / cleans bounding box from showing image
    :param event: click event
    :param x: x of mouse
    :param y: y of mouse
    :param flags:
    :param params: params for image showing
    :return:
    """
    cam = params["cam"]
    restart = False
    if event == cv2.EVENT_LBUTTONDOWN:
        x_0 = int(x * 2 * resizing_factor_x)
        y_0 = int(y * 2 * resizing_factor_y)
        params["x_0"] = x_0
        params["y_0"] = y_0
        params["x_1"] = None
        params["y_1"] = None
        restart = True
        params["left_click"] = False
    if event == cv2.EVENT_RBUTTONDOWN:
        params["group_1"] = None
        params["group_2"] = None
        params["x_0"] = None
        params["y_0"] = None
        params["x_1"] = None
        params["y_1"] = None
        restart = True
        params["left_click"] = False
    if event == cv2.EVENT_LBUTTONUP:
        x_1 = int(x * 2 * resizing_factor_x)
        y_1 = int(y * 2 * resizing_factor_y)
        params["x_1"] = x_1
        params["y_1"] = y_1
        point_cloud = sl.Mat()
        cam.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        point_cloud = point_cloud.get_data()
        if rotatred_vid:
            point_cloud = cv2.rotate(point_cloud, cv2.cv2.ROTATE_90_CLOCKWISE)
        print(point_cloud[params["y_0"], params["x_0"]])
        print(point_cloud[params["y_1"], params["x_1"]])
        print(params["x_0"])
        print(params["x_1"])
        print(params["y_0"])
        print(params["y_1"])
        restart = True
        params["left_click"] = True
    if event == cv2.EVENT_MOUSEMOVE and not isinstance(params["x_0"],type(None)) and not params["left_click"]:
        x_1 = int(x * 2 * resizing_factor_x)
        y_1 = int(y * 2 * resizing_factor_y)
        params["x_1"] = x_1
        params["y_1"] = y_1
        restart = True
    if event == cv2.EVENT_MBUTTONDOWN:
        x_real_pixel = int(x * 2 * resizing_factor_x)
        y_real_pixel = int(y * 2 * resizing_factor_y)
        x_axis = np.arange(x_pic_rang)
        depth_map = sl.Mat()
        cam.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
        depth_img = depth_map.get_data()
        y_axis = depth_img[y_real_pixel, :][:, 0]

        x_axis_2 = np.arange(0,x_pic_rang)

        params["x_0"] = None
        params["y_0"] = None
        params["x_1"] = None
        params["y_1"] = None
        restart = True
        params["left_click"] = False
    if restart:
        run(**params)


if __name__ == "__main__":

    #fp = '/media/fruitspec-lab/cam172/JAI_ZED_Scan_25_10_22/R2_SH_S_FHD15/HD1080_SN39018199_13-59-44.svo'
    fp = "/home/lihi/FruitSpec/Data/customers/DEWAGD/190123/DWDBLE33/R11A/ZED_1.svo"
    run(fp)