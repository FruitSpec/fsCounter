import cv2
import pyzed.sl as sl
import numpy as np
import copy


class video_wrapper():

    def __init__(self, filepath, rotate=0, depth_minimum=0.1, depth_maximum=2.5):
        if 'svo' in filepath.split('.')[-1]:
            self.mode = 'svo'
            self.cam, self.runtime = self.init_cam(filepath, depth_minimum, depth_maximum)
            self.mat = sl.Mat()
            self.res = None


        else:
            self.mode = 'other'
            self.cam = cv2.VideoCapture(filepath)
            self.runtime = None
            self.mat = None
            self.res = None
        self.to_rotate = rotate
        self.rotation = self.get_rotation(rotate)

    @staticmethod
    def get_rotation(rotate):
        if rotate == 1:
            rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif rotate == 2:
            rotation = cv2.ROTATE_90_CLOCKWISE
        else:
            rotation = None

        return rotation

    def get_zed(self, frame_number=None, exclude_depth=False):

        if self.mode == 'svo':
            self.grab(frame_number)
            _, frame = self.get_frame()
            point_cloud = self.get_point_cloud()
            if exclude_depth:
                return frame, point_cloud
            depth = self.get_depth()
        else:
            Warning('Not implemented for file type')
            frame = None
            depth = None

        return frame, depth, point_cloud

    def grab(self, frame_number=None):
        if self.mode == 'svo':
            if frame_number is not None:
                self.cam.set_svo_position(frame_number)
            res = self.cam.grab(self.runtime)
            if res == sl.ERROR_CODE.SUCCESS:
                self.res = True
            else:
                self.res = False

        else:
            Warning('Grab Not implemented for file type')

    def get_frame(self, frame_number=None):
        if self.mode == 'svo':
            if self.res:
                self.cam.retrieve_image(self.mat, sl.VIEW.LEFT)
                frame = self.mat.get_data()[:, :, : 3]
                ret = self.res
            else:
                frame = None
                ret = False
        else:
            if frame_number is not None:
                self.cam.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cam.read()

        if ret:
            frame = self.rotate(frame)

        return ret, frame

    def get_depth(self):
        depth = None
        if self.mode == 'svo':
            if self.res:
                cam_run_p = self.cam.get_init_parameters()
                self.cam.retrieve_measure(self.mat, sl.MEASURE.DEPTH)
                depth = self.mat.get_data()
                depth = (cam_run_p.depth_maximum_distance - np.clip(depth, 0, cam_run_p.depth_maximum_distance)) * 255 / (cam_run_p.depth_maximum_distance - cam_run_p.depth_minimum_distance)
                bool_mask = np.where(np.isnan(depth), True, False)
                depth[bool_mask] = 0

                depth = cv2.medianBlur(depth, 5)
                depth = self.rotate(depth)

        else:
            Warning('Depth Not implemented for file type')

        return depth

    def get_point_cloud(self):
        point_cloud = None
        if self.mode == 'svo':
            if self.res:
                self.cam.retrieve_measure(self.mat, sl.MEASURE.XYZRGBA)
                point_cloud = self.mat.get_data()
                point_cloud = self.rotate(point_cloud)
        else:
            Warning('point_cloud Not implemented for file type')

        return point_cloud

    def get_number_of_frames(self):
        if self.mode == 'svo':
            number_of_frames = sl.Camera.get_svo_number_of_frames(self.cam)
        else:
            number_of_frames = self.cam.get(cv2.CAP_PROP_FRAME_COUNT)

        return number_of_frames

    def get_fps(self):
        if self.mode == 'svo':
            fps = None
            Warning('FPS Not implemented for svo file')
        else:
            fps = self.cam.get(cv2.CAP_PROP_FPS)
        return fps

    def get_height(self):
        if self.mode == 'svo':
            ci = self.cam.get_camera_information()
            # camera is rotated - width is height
            if self.to_rotate:
                height = ci.camera_configuration.camera_resolution.width
            else:
                height = ci.camera_configuration.camera_resolution.height
        else:
            if self.to_rotate:
                height = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
            else:
                height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

        return height

    def get_width(self):
        if self.mode == 'svo':
            ci = self.cam.get_camera_information()
            # camera is rotated - height is width
            if self.to_rotate:
                width = ci.camera_configuration.camera_resolution.height
            else:
                width = ci.camera_configuration.camera_resolution.width
        else:
            if self.to_rotate:
                width = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
            else:
                width = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        return width

    def rotate(self, frame):
        if self.to_rotate:
            frame = cv2.rotate(frame, self.rotation)
        return frame

    def close(self):
        if self.mode == 'svo':
            self.cam.close()
        else:
            self.cam.release()

    def is_open(self):
        pass

    def copy(self):
        return copy.copy(self)


    @staticmethod
    def init_cam(filepath, depth_minimum=0.1, depth_maximum=2.5):
        """
        inits camera and runtime
        :param filepath: path to svo file
        :return:
        """
        input_type = sl.InputType()
        input_type.set_from_svo_file(filepath)
        init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        # init_params.depth_mode = sl.DEPTH_MODE.QUALITY
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_minimum_distance = depth_minimum
        init_params.depth_maximum_distance = depth_maximum
        init_params.depth_stabilization = True
        runtime = sl.RuntimeParameters()
        runtime.confidence_threshold = 100
        # runtime.sensing_mode = sl.SENSING_MODE.STANDARD
        runtime.sensing_mode = sl.SENSING_MODE.FILL
        cam = sl.Camera()
        status = cam.open(init_params)
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        cam.enable_positional_tracking(positional_tracking_parameters)
        detection_parameters = sl.ObjectDetectionParameters()
        detection_parameters.detection_model = sl.DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        detection_parameters.enable_tracking = False
        detection_parameters.enable_mask_output = True
        cam.enable_object_detection(detection_parameters)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()
        return cam, runtime
